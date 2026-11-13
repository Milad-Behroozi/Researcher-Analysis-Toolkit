# LSV_Analysis.py
# Researcher Analysis Toolkit - Corva-like LSV GUI (REAL, not demo)
# Features implemented:
# - Load CSV/TXT/XLSX (XLSX via openpyxl; CSV/TXT via pandas or csv)
# - Column detection + user selection
# - Area normalization (I -> j), optional sign invert
# - Reference conversion to RHE/NHE/Both (common refs + user-defined E° vs NHE)
# - Optional iR correction using device compensation (%) + Rs (ohm)
# - Onset potential from threshold current density (mA/cm^2)
# - Report E@j_ref and j@E_ref (optional)
# - Results shown in center (Summary / Key Results / Formulas) + Plot (Matplotlib)
# - Export: PNG + PDF + processed CSV + metadata JSON in an output folder
#
# Install:
#   pip install PyQt6 matplotlib numpy
# Optional (recommended):
#   pip install pandas openpyxl
#
# Run:
#   python LSV_Analysis.py

from __future__ import annotations

import os
import sys
import json
import datetime as _dt
import ctypes
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import re
import numpy as np

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QAction, QIcon
from PyQt6.QtGui import QFont, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFileDialog,
    QGroupBox, QComboBox, QLineEdit, QCheckBox,
    QScrollArea, QSplitter, QTabWidget, QTextEdit,
    QTableWidget, QTableWidgetItem, QMessageBox, QSizePolicy,
    QDoubleSpinBox
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages


# -------------------------------
# Output folder policy (Desktop/Researcher Output/LSV Analysis/<Raw File Name>/run_00X)
# -------------------------------
def get_default_output_dir() -> str:
    """Default output folder: Desktop/Researcher Output/LSV Analysis"""
    home = os.path.expanduser("~")
    desktop = os.path.join(home, "Desktop")
    if not os.path.isdir(desktop):
        desktop = os.path.join(os.environ.get("USERPROFILE", home), "Desktop")
    return os.path.join(desktop, "Researcher Output", "LSV Analysis")


def _safe_folder_name(name: str) -> str:
    """Sanitize a name for a folder (cross-platform friendly)."""
    name = (name or "").strip()
    if not name:
        return "untitled"
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name


def _next_run_folder(parent_dir: str) -> str:
    """Return next run_00X folder path inside parent_dir, creating parent if needed."""
    os.makedirs(parent_dir, exist_ok=True)
    pat = re.compile(r"^run_(\d{3})$")
    max_n = 0
    try:
        for d in os.listdir(parent_dir):
            m = pat.match(d)
            if m:
                try:
                    max_n = max(max_n, int(m.group(1)))
                except Exception:
                    pass
    except Exception:
        pass
    return os.path.join(parent_dir, f"run_{max_n + 1:03d}")


def build_run_folder(raw_file_path: str) -> str:
    """Build and create the output run folder for a given raw file path."""
    raw_base = os.path.splitext(os.path.basename(raw_file_path))[0]
    base_dir = get_default_output_dir()
    file_dir = os.path.join(base_dir, _safe_folder_name(raw_base))
    run_dir = _next_run_folder(file_dir)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


@dataclass
class LSVResult:
    summary_text: str
    key_values: Dict[str, Dict[str, Any]]  # {param: {"value":..., "unit":...}}
    formulas_used: List[str]

    # Base LSV (vs RHE) for plotting
    E_plot: np.ndarray
    j_plot: np.ndarray
    eta_plot: np.ndarray
    logj_plot: np.ndarray

    # Labels
    E_label: str
    j_label: str

    # Optional annotations
    onset_V: Optional[float]
    onset_fit: Optional[Dict[str, Any]]
    j_ref_target: Optional[float]
    E_at_jref: Optional[float]




def write_pdf_report(pdf_path: str,
                     inputs: Dict[str, Any],
                     formulas: List[str],
                     key_vals: Dict[str, Dict[str, Any]],
                     fig1_png: str,
                     fig2_png: str,
                     fig3_png: str) -> None:
    """Create an A4 PDF report (commercial-looking) using Matplotlib only."""
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle

# Extract reference details from formulas (if present)
    ref_name = inputs.get("ref", "Unknown")
    pH = float(inputs.get("pH", 7.0))
    E0 = None

    import re
    for line in formulas:
        m = re.search(r"E°\(ref vs NHE\)\s*=\s*([+\-]?\d+\.\d+)", line)
        if m:
            try:
                E0 = float(m.group(1))
            except Exception:
                pass


    today = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    with PdfPages(pdf_path) as pdf:
        # ---------------- Cover (Page 1) ----------------
        fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        # Header band
        ax.add_patch(Rectangle((0, 0.87), 1, 0.13, transform=ax.transAxes, clip_on=False, alpha=0.12))
        fig.text(0.08, 0.94, "Researcher Analysis Toolkit", fontsize=18, fontweight="bold")
        fig.text(0.08, 0.905, "Electrochemical LSV Analysis Report", fontsize=14, fontstyle="italic")

        # Meta box
        fig.text(0.08, 0.84, f"Generated: {today}", fontsize=10)
        fig.text(0.08, 0.815, f"Sample/File: {os.path.basename(inputs['file'])}", fontsize=10)
        fig.text(0.08, 0.79, f"Mode: {inputs['mode']}   |   Reference: {ref_name}   |   pH: {pH:.2f}", fontsize=10)
        fig.text(0.08, 0.765, f"Output scale: {inputs['scale']}   |   Area: {float(inputs['area_cm2']):g} cm²", fontsize=10)

        # Notice
        notice = ("This report is prepared as a pre-publication electrochemical analysis document "
                  "and may be submitted as supporting data to a peer-reviewed journal.")
        fig.text(0.08, 0.71, notice, fontsize=10)

        # Citation
        doi = "10.1039/D6RA08618G"
        fig.text(
            0.08, 0.64,
            "Generated with Researcher Analysis Toolkit v1.0.0",
            fontsize=10,
        )
        fig.text(
            0.08, 0.615,
            f"Suggested citation · Behroozi et al., RSC Adv. (2026). {doi}",
            fontsize=10,
            color="#1F4E79",
            url=f"https://doi.org/{doi}",
        )

        # Footer
        fig.text(0.92, 0.03, "Researcher Analysis Toolkit", ha="right", fontsize=9, alpha=0.7)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Formulas (Page 2) ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        fig.text(0.08, 0.95, "Formulas & Conversions", fontsize=16, fontweight="bold")

        # Pretty equations
        yy = 0.89
        fig.text(0.10, yy, r"$E_{\mathrm{NHE}} = E_{\mathrm{measured}} + E^\circ_{\mathrm{ref}}$", fontsize=14); yy -= 0.06
        fig.text(0.10, yy, r"$E_{\mathrm{RHE}} = E_{\mathrm{NHE}} + 0.05916 \times pH$", fontsize=14); yy -= 0.06
        fig.text(0.10, yy, r"$E_{\mathrm{RHE}} = E_{\mathrm{measured}} + E^\circ_{\mathrm{ref}} + 0.05916 \times pH$", fontsize=14); yy -= 0.07

        # Exact values used
        fig.text(0.08, yy, "Values used in this analysis:", fontsize=12, fontweight="bold"); yy -= 0.035
        fig.text(0.10, yy, f"• Reference electrode: {ref_name}", fontsize=11); yy -= 0.03
        if E0 is not None:
            fig.text(0.10, yy, f"• Reference potential: E°(ref vs NHE) = {E0:+.6f} V", fontsize=11); yy -= 0.03
        fig.text(0.10, yy, f"• pH = {pH:.2f}", fontsize=11); yy -= 0.05

        # If iR applied
        if inputs.get("ir_apply"):
            Rs = inputs.get("Rs_ohm")
            frac = float(inputs.get("ir_device_pct", 0.0)) / 100.0
            fig.text(0.08, yy, "iR compensation:", fontsize=12, fontweight="bold"); yy -= 0.035
            fig.text(0.10, yy, r"$E_{\mathrm{corrected}} = E_{\mathrm{measured}} - I \times R_s \times (1-f)$", fontsize=13); yy -= 0.04
            fig.text(0.10, yy, f"• Device compensation: f = {frac:.2f}", fontsize=11); yy -= 0.03
            fig.text(0.10, yy, f"• Rs = {Rs} Ω", fontsize=11); yy -= 0.04

        # Overpotential definition
        fig.text(0.08, yy, "Overpotential:", fontsize=12, fontweight="bold"); yy -= 0.035
        if inputs.get("mode") == "OER":
            fig.text(0.10, yy, r"$\eta = E_{\mathrm{RHE}} - 1.23$", fontsize=14); yy -= 0.03
        else:
            fig.text(0.10, yy, r"$\eta = 0 - E_{\mathrm{RHE}}$", fontsize=14); yy -= 0.03

        fig.text(0.92, 0.03, "Researcher Analysis Toolkit", ha="right", fontsize=9, alpha=0.7)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Key Results (Page 3) ----------------
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
        fig.text(0.08, 0.95, "Key Results", fontsize=16, fontweight="bold")

        yy = 0.90
        for k, v in key_vals.items():
            val = v.get("value", "")
            unit = v.get("unit", "")
            fig.text(0.10, yy, f"{k}", fontsize=11, fontweight="bold")
            fig.text(0.55, yy, f"{val}", fontsize=11)
            fig.text(0.80, yy, f"{unit}", fontsize=11)
            yy -= 0.03
            if yy < 0.08:
                pdf.savefig(fig); plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
                yy = 0.95

        fig.text(0.92, 0.03, "Researcher Analysis Toolkit", ha="right", fontsize=9, alpha=0.7)
        pdf.savefig(fig); plt.close(fig)

        # ---------------- Figures (Pages 4+) ----------------
        for title, p in [("Figure 1 — LSV curve", fig1_png),
                         ("Figure 2 — Overpotential plot", fig2_png),
                         ("Figure 3 — Tafel-ready plot", fig3_png)]:
            fig = plt.figure(figsize=(8.27, 11.69))
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            fig.text(0.08, 0.95, title, fontsize=15, fontweight="bold")

            ax2 = fig.add_axes([0.10, 0.12, 0.80, 0.78])
            ax2.axis("off")
            img = mpimg.imread(p)
            ax2.imshow(img)

            fig.text(0.92, 0.03, "Researcher Analysis Toolkit", ha="right", fontsize=9, alpha=0.7)
            pdf.savefig(fig); plt.close(fig)

def _try_import_pandas():
    try:
        import pandas as pd  # type: ignore
        return pd
    except Exception:
        return None


def load_table(path: str) -> Tuple[List[str], Dict[str, np.ndarray]]:
    """Load data file into columns.

    Returns:
        (column_names, dict[name] -> np.ndarray)

    Supports:
        csv, txt, xlsx, xls
    """
    ext = os.path.splitext(path)[1].lower().strip(".")
    pd = _try_import_pandas()

    if ext in ("csv", "txt"):
        if pd is not None:
            try:
                df = pd.read_csv(path, sep=None, engine="python")
            except Exception:
                df = pd.read_csv(path)
            cols = [str(c) for c in df.columns]
            data = {str(c): df[c].to_numpy() for c in df.columns}
            return cols, data

        import csv
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            raise ValueError("Empty file.")
        header = rows[0]
        cols = [h.strip() for h in header]
        col_vals = {c: [] for c in cols}
        for r in rows[1:]:
            if not r or len(r) < len(cols):
                continue
            for i, c in enumerate(cols):
                try:
                    col_vals[c].append(float(r[i]))
                except Exception:
                    col_vals[c].append(float("nan"))
        data = {c: np.array(v, dtype=float) for c, v in col_vals.items()}
        return cols, data

    if ext in ("xlsx", "xls"):
        try:
            import openpyxl  # type: ignore
        except Exception as e:
            raise RuntimeError("To read XLSX/XLS, install openpyxl: pip install openpyxl") from e

        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            raise ValueError("Empty XLSX/XLS.")
        header = [str(x).strip() if x is not None else "" for x in rows[0]]
        cols = [h if h else f"col_{i}" for i, h in enumerate(header)]
        col_vals = {c: [] for c in cols}
        for r in rows[1:]:
            if r is None:
                continue
            for i, c in enumerate(cols):
                v = r[i] if i < len(r) else None
                try:
                    col_vals[c].append(float(v))
                except Exception:
                    col_vals[c].append(float("nan"))
        data = {c: np.array(v, dtype=float) for c, v in col_vals.items()}
        return cols, data

    raise ValueError(f"Unsupported file type: {ext}")


def _clean_xy(E: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(E) & np.isfinite(I)
    E2 = E[mask]
    I2 = I[mask]
    if E2.size < 5:
        raise ValueError("Not enough numeric data points after cleaning.")
    idx = np.argsort(E2)
    return E2[idx], I2[idx]


REF_E0_VS_NHE = {
    "Ag/AgCl (sat KCl)": 0.197,
    "Ag/AgCl (3M KCl)": 0.210,
    "Ag/AgCl (1M KCl)": 0.235,
    "SCE": 0.241,
    "Hg/HgO": 0.098,
    "RHE": 0.0,
    "NHE": 0.0,
}


def convert_potential(E_ref: np.ndarray, ref_name: str, pH: float, out_scale: str, custom_E0: Optional[float]) -> Tuple[np.ndarray, str, List[str]]:
    formulas: List[str] = []

    if ref_name == "User-defined (enter E°)":
        if custom_E0 is None:
            raise ValueError("Custom E° vs NHE is required for user-defined reference.")
        E0 = float(custom_E0)
    else:
        if ref_name not in REF_E0_VS_NHE:
            raise ValueError(f"Unknown reference: {ref_name}")
        E0 = float(REF_E0_VS_NHE[ref_name])

    
    # --- Document the exact conversion used (for report + GUI) ---
    # Base equations:
    #   E_NHE = E_measured + E0_ref(vs NHE)
    #   E_RHE = E_NHE + 0.05916*pH
    if ref_name == "User-defined (enter E°)":
        formulas.append(f"Reference: User-defined; E°(ref vs NHE) = {E0:+.6f} V")
    else:
        formulas.append(f"Reference: {ref_name}; E°(ref vs NHE) = {E0:+.6f} V")

    formulas.append("E_NHE = E_measured + E°(ref vs NHE)")
    formulas.append("E_RHE = E_NHE + 0.05916 × pH (25°C)")
    formulas.append(f"pH used = {pH:.2f}")
    formulas.append(f"Combined: E_RHE = E_measured + ({E0:+.6f}) + 0.05916×({pH:.2f})")
    
    E_nhe = E_ref + E0
    if out_scale == "NHE":
        return E_nhe, "E (V vs NHE)", formulas

    E_rhe = E_nhe + 0.05916 * pH
    if out_scale == "RHE":
        return E_rhe, "E (V vs RHE)", formulas

    return E_rhe, "E (V vs RHE)", formulas  # default


def apply_ir_correction(E_scale: np.ndarray, I_amp: np.ndarray, Rs_ohm: float, device_percent: float) -> Tuple[np.ndarray, List[str], float]:
    f = max(0.0, min(100.0, device_percent)) / 100.0
    Rs_res = Rs_ohm * (1.0 - f)
    formulas = ["E_corrected = E_measured − I × Rs × (1 − f)"]
    E_corr = E_scale - I_amp * Rs_res
    return E_corr, formulas, Rs_res


def onset_potential(E: np.ndarray, j_mAcm2: np.ndarray, thr_mAcm2: float, mode: str) -> Optional[float]:
    if E.size == 0:
        return None
    if mode == "OER":
        mask = j_mAcm2 >= thr_mAcm2
    else:
        mask = np.abs(j_mAcm2) >= thr_mAcm2
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    return float(E[int(idx[0])])


def tangent_onset_fit(E: np.ndarray, j_mAcm2: np.ndarray, mode: str,
                      smooth_win: int = 11, fit_halfwin: int = 8) -> Optional[Dict[str, Any]]:
    """Tangent extrapolation onset (intersection of tangent of active region with j=0 baseline).

    Returns dict with keys: E_onset, m, b, i1, i2, E_sorted, j_smooth_sorted
    """
    import numpy as np

    if E.size < 5:
        return None

    order = np.argsort(E)
    Es = E[order]
    js = j_mAcm2[order]

    # Simple moving-average smoothing (keeps dependencies minimal)
    if smooth_win >= 3 and Es.size > smooth_win:
        kernel = np.ones(int(smooth_win), dtype=float)
        kernel /= kernel.sum()
        j_s = np.convolve(js, kernel, mode="same")
    else:
        j_s = js

    dj = np.gradient(j_s, Es)

    # --- Choose an "activation" region instead of the global steepest-slope point ---
    # The global max |dj/dE| is often at the end of the scan (mass-transport limits / breakdown),
    # which gives a non-physical onset. We constrain the search to early-to-mid region where
    # current is still a fraction of the maximum.
    left = max(2, int(fit_halfwin))
    right = min(Es.size - 3, Es.size - int(fit_halfwin) - 1)
    if right <= left:
        return None

    # Define activation current (always positive) for robust selection
    if mode == "OER":
        j_act = j_s
    else:  # HER
        j_act = -j_s

    # If activation current isn't positive anywhere, cannot determine onset
    if not np.isfinite(np.nanmax(j_act)) or np.nanmax(j_act) <= 0:
        return None

    jmax = float(np.nanmax(j_act))
    # Use 2%..30% of max as typical "activation" band
    lo = 0.02 * jmax
    hi = 0.30 * jmax

    valid = np.where((j_act >= lo) & (j_act <= hi))[0]
    valid = valid[(valid >= left) & (valid <= right)]
    if valid.size < 5:
        # relax upper bound if needed
        valid = np.where((j_act >= lo) & (j_act <= 0.50 * jmax))[0]
        valid = valid[(valid >= left) & (valid <= right)]
    # For OER, onset cannot be in the low-potential region (thermodynamic limit is 1.23 V vs RHE).
    # We keep a small margin to tolerate noisy/shifted data.
    if mode == "OER":
        valid = valid[Es[valid] >= 1.0]

    if valid.size < 5:
        return None

    # Prefer the point with the largest *positive* slope in activation sense
    dj_act = np.gradient(j_act, Es)
    idx0 = int(valid[np.nanargmax(dj_act[valid])])

    i1 = max(0, idx0 - int(fit_halfwin))
    i2 = min(Es.size, idx0 + int(fit_halfwin) + 1)
    if i2 - i1 < 3:
        return None

    m, b = np.polyfit(Es[i1:i2], j_s[i1:i2], 1)
    if abs(m) < 1e-12:
        return None

    E_on = -b / m

    # Optional sanity: onset should be within scan range
    if not (Es.min() <= E_on <= Es.max()):
        # Still return; caller may decide to fallback
        pass

    return {
        "E_onset": float(E_on),
        "m": float(m),
        "b": float(b),
        "i1": int(i1),
        "i2": int(i2),
        "E_sorted": Es,
        "j_smooth_sorted": j_s,
    }


def onset_potential_tangent(E: np.ndarray, j_mAcm2: np.ndarray, mode: str) -> Optional[float]:
    fit = tangent_onset_fit(E, j_mAcm2, mode)
    if fit is None:
        return None
    return float(fit["E_onset"])


def interpolate_E_at_j(E: np.ndarray, j: np.ndarray, j_target: float) -> Optional[float]:
    if E.size < 2:
        return None
    order = np.argsort(j)
    js = j[order]
    Es = E[order]
    if j_target < js[0] or j_target > js[-1]:
        return None
    return float(np.interp(j_target, js, Es))


def interpolate_j_at_E(E: np.ndarray, j: np.ndarray, E_target: float) -> Optional[float]:
    if E.size < 2:
        return None
    if E_target < E[0] or E_target > E[-1]:
        return None
    return float(np.interp(E_target, E, j))


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure()
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)


class DropArea(QLabel):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(140)
        self.setText("Drop your data file here\n(CSV / TXT / XLSX)\n\nor click Browse")
        self.setStyleSheet(
            """
            QLabel {
                border: 2px dashed #9a9a9a;
                border-radius: 10px;
                color: #444;
                background: #fafafa;
                font-size: 14px;
            }
            """
        )

    def set_path(self, path: Optional[str]):
        if path:
            base = os.path.basename(path)
            self.setText(f"Selected file:\n{base}\n\n(Drag & drop to replace)")
        else:
            self.setText("Drop your data file here\n(CSV / TXT / XLSX)\n\nor click Browse")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        p = urls[0].toLocalFile()
        if not p:
            return
        self.set_path(p)
        top = self.window()
        if hasattr(top, "on_file_selected"):
            top.on_file_selected(p)  # type: ignore[attr-defined]


class CorvaLSVWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Researcher Analysis Toolkit — Linear Sweep Voltammetry (LSV)")
        self.setWindowIcon(QIcon("Researcherlogo.png"))
        self.setMinimumSize(QSize(1180, 720))

        self.current_file: Optional[str] = None
        self._last_result: Optional[LSVResult] = None
        self._cols: List[str] = []
        self._data: Dict[str, np.ndarray] = {}

        root = QWidget()
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = self._build_left_panel()
        right = self._build_right_panel()

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([820, 360])

        layout = QVBoxLayout(root)
        layout.addWidget(splitter)

        self._build_menu()

    def _build_menu(self):
        m = self.menuBar()
        help_menu = m.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self.on_about)
        help_menu.addAction(act_about)

    def on_about(self):
        QMessageBox.information(
            self,
            "About",
            "LSV Analysis (REAL)\n"
            "Exports: PNG/PDF + processed CSV + metadata JSON\n"
            "Brand: Researcher Analysis Toolkit"
        )

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        self.drop = DropArea(parent=w)
        outer.addWidget(self.drop)

        btn_row = QHBoxLayout()
        self.btn_browse = QPushButton("Browse...")
        self.btn_load_cols = QPushButton("Load columns")
        btn_row.addWidget(self.btn_browse)
        btn_row.addWidget(self.btn_load_cols)
        outer.addLayout(btn_row)

        self.btn_run = QPushButton("Run LSV Analysis + Export PNG/PDF")
        self.btn_run.setMinimumHeight(38)
        outer.addWidget(self.btn_run)

        results_box = QGroupBox("Results")
        rb = QVBoxLayout(results_box)
        rb.setContentsMargins(10, 10, 10, 10)

        self.tabs = QTabWidget()
        self.tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.tabs.addTab(self.txt_summary, "Summary")

        self.tbl = QTableWidget(0, 3)
        self.tbl.setHorizontalHeaderLabels(["Parameter", "Value", "Unit"])
        self.tbl.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.tbl, "Key Results")

        self.txt_formulas = QTextEdit()
        self.txt_formulas.setReadOnly(True)
        self.tabs.addTab(self.txt_formulas, "Formulas")

        plot_wrap = QWidget()
        pw = QVBoxLayout(plot_wrap)
        pw.setContentsMargins(0, 0, 0, 0)

        # Plot selector (show all extracted figures)
        self.plot_selector = QComboBox()
        self.plot_selector.addItems(["LSV curve", "Overpotential plot", "Tafel-ready plot"])
        self.plot_selector.currentIndexChanged.connect(self._on_plot_selector_changed)
        self.plot_selector.setEnabled(False)
        pw.addWidget(self.plot_selector)

        self.canvas = MplCanvas(plot_wrap)
        pw.addWidget(self.canvas)

        self.tabs.addTab(plot_wrap, "Plot")

        rb.addWidget(self.tabs)
        outer.addWidget(results_box, stretch=1)

        self.btn_browse.clicked.connect(self.on_browse)
        self.btn_load_cols.clicked.connect(self.on_load_columns)
        self.btn_run.clicked.connect(self.on_run_lsv)

        return w

    def _build_right_panel(self) -> QWidget:
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(10, 10, 10, 10)
        inner_layout.setSpacing(10)

        gb_mode = QGroupBox("Reaction Mode")
        gm = QVBoxLayout(gb_mode)
        gm.setContentsMargins(10, 10, 10, 10)
        gm.addWidget(QLabel("Select mode: OER or HER:"))
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["OER", "HER"])
        gm.addWidget(self.cb_mode)
        inner_layout.addWidget(gb_mode)

        gb_cols = QGroupBox("Column Selection")
        gc = QGridLayout(gb_cols)
        gc.setContentsMargins(10, 10, 10, 10)
        gc.setHorizontalSpacing(8)
        gc.setVerticalSpacing(8)

        gc.addWidget(QLabel("Potential column (E):"), 0, 0, 1, 2)
        self.cb_E = QComboBox()
        self.cb_E.addItem("")
        gc.addWidget(self.cb_E, 1, 0, 1, 2)

        gc.addWidget(QLabel("Current column (I or j):"), 2, 0, 1, 2)
        self.cb_I = QComboBox()
        self.cb_I.addItem("")
        gc.addWidget(self.cb_I, 3, 0, 1, 2)
        inner_layout.addWidget(gb_cols)

        gb_norm = QGroupBox("Electrode Normalization")
        gn = QGridLayout(gb_norm)
        gn.setContentsMargins(10, 10, 10, 10)
        gn.setHorizontalSpacing(8)
        gn.setVerticalSpacing(8)

        gn.addWidget(QLabel("Area (cm^2):"), 0, 0)
        self.ed_area = QLineEdit("1.0")
        gn.addWidget(self.ed_area, 0, 1)

        self.ck_invert = QCheckBox("Invert current sign (if needed)")
        gn.addWidget(self.ck_invert, 1, 0, 1, 2)
        inner_layout.addWidget(gb_norm)

        gb_ref = QGroupBox("Reference Conversion")
        gr = QGridLayout(gb_ref)
        gr.setContentsMargins(10, 10, 10, 10)
        gr.setHorizontalSpacing(8)
        gr.setVerticalSpacing(8)

        gr.addWidget(QLabel("Reference electrode:"), 0, 0)
        self.cb_ref = QComboBox()
        self.cb_ref.addItems([
            "Ag/AgCl (sat KCl)",
            "Ag/AgCl (3M KCl)",
            "Ag/AgCl (1M KCl)",
            "SCE",
            "Hg/HgO",
            "RHE",
            "NHE",
            "User-defined (enter E°)"
        ])
        gr.addWidget(self.cb_ref, 0, 1)

        gr.addWidget(QLabel("pH:"), 1, 0)
        self.sp_pH = QDoubleSpinBox()
        self.sp_pH.setRange(0.0, 14.0)
        self.sp_pH.setDecimals(2)
        self.sp_pH.setValue(7.00)
        gr.addWidget(self.sp_pH, 1, 1)

        gr.addWidget(QLabel("Output scale:"), 2, 0)
        self.cb_scale = QComboBox()
        self.cb_scale.addItems(["RHE", "NHE", "Both"])
        gr.addWidget(self.cb_scale, 2, 1)

        gr.addWidget(QLabel("Custom E° vs NHE (V):"), 3, 0)
        self.ed_custom_E0 = QLineEdit()
        self.ed_custom_E0.setPlaceholderText("Only if User-defined")
        gr.addWidget(self.ed_custom_E0, 3, 1)

        inner_layout.addWidget(gb_ref)

        gb_ir = QGroupBox("iR Compensation (optional)")
        gi = QGridLayout(gb_ir)
        gi.setContentsMargins(10, 10, 10, 10)
        gi.setHorizontalSpacing(8)
        gi.setVerticalSpacing(8)

        self.ck_ir = QCheckBox("Apply iR correction")
        gi.addWidget(self.ck_ir, 0, 0, 1, 2)

        gi.addWidget(QLabel("Device compensation (%):"), 1, 0)
        self.ed_ir_percent = QLineEdit("0")
        self.ed_ir_percent.setPlaceholderText("e.g., 85")
        gi.addWidget(self.ed_ir_percent, 1, 1)

        gi.addWidget(QLabel("Rs (Ω) [from EIS]:"), 2, 0)
        self.ed_rs = QLineEdit()
        self.ed_rs.setPlaceholderText("optional")
        gi.addWidget(self.ed_rs, 2, 1)

        hint = QLabel("Tip: If you don't know Rs, obtain it from EIS as high-frequency intercept (Nyquist).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#555; font-size: 11px;")
        gi.addWidget(hint, 3, 0, 1, 2)

        inner_layout.addWidget(gb_ir)

        gb_on = QGroupBox("Onset Potential")
        go = QGridLayout(gb_on)
        go.setContentsMargins(10, 10, 10, 10)
        go.setHorizontalSpacing(8)
        go.setVerticalSpacing(8)

        go.addWidget(QLabel("Onset threshold (mA/cm^2):"), 0, 0)
        self.sp_onset = QDoubleSpinBox()
        self.sp_onset.setRange(0.01, 1000.0)
        self.sp_onset.setDecimals(2)
        self.sp_onset.setValue(1.00)
        go.addWidget(self.sp_onset, 0, 1)

        inner_layout.addWidget(gb_on)

        gb_eval = QGroupBox("Evaluation (optional)")
        ge = QGridLayout(gb_eval)
        ge.setContentsMargins(10, 10, 10, 10)
        ge.setHorizontalSpacing(8)
        ge.setVerticalSpacing(8)

        ge.addWidget(QLabel("Report E @ j_ref (mA/cm^2):"), 0, 0)
        self.ed_jref = QLineEdit("10")
        ge.addWidget(self.ed_jref, 0, 1)

        ge.addWidget(QLabel("Report j @ E_ref (V):"), 1, 0)
        self.ed_eref = QLineEdit()
        self.ed_eref.setPlaceholderText("e.g., 1.60")
        ge.addWidget(self.ed_eref, 1, 1)

        inner_layout.addWidget(gb_eval)

        inner_layout.addStretch(1)

        scroll.setWidget(inner)
        container_layout.addWidget(scroll)

        container.setMinimumWidth(360)
        container.setMaximumWidth(420)
        return container

    def on_file_selected(self, path: str):
        self.current_file = path

    def on_browse(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select LSV data file",
            "",
            "Data files (*.csv *.txt *.xlsx *.xls);;All files (*.*)"
        )
        if p:
            self.drop.set_path(p)
            self.on_file_selected(p)

    def on_load_columns(self):
        if not self.current_file:
            QMessageBox.warning(self, "No file", "Please select a file first.")
            return
        try:
            cols, data = load_table(self.current_file)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        self._cols = cols
        self._data = data

        self.cb_E.clear()
        self.cb_I.clear()
        self.cb_E.addItems(cols)
        self.cb_I.addItems(cols)

        def pick(cands):
            cols_low = [c.strip().lower() for c in cols]
            for cand in cands:
                if cand in cols_low:
                    return cols[cols_low.index(cand)]
            return cols[0] if cols else ""

        self.cb_E.setCurrentText(pick(["potential/v", "ewe/v", "potential", "e (v)"]))
        self.cb_I.setCurrentText(pick(["current/a", "i/a", "current", "j/macm2", "i/ma"]))

        QMessageBox.information(self, "Columns loaded", f"Loaded {len(cols)} columns.")

    def on_run_lsv(self):
        if not self.current_file:
            QMessageBox.warning(self, "No file", "Please select a file first.")
            return
        if not self._data:
            QMessageBox.warning(self, "Columns not loaded", "Please click 'Load columns' first.")
            return
        try:
            inputs = self._collect_inputs()
            res, out_dir = self._run_lsv_engine(inputs)
            self._show_results(res)
            QMessageBox.information(self, "Done", f"LSV analysis finished.\nOutputs saved to:\n{out_dir}")
        except Exception as e:
            QMessageBox.critical(self, "Analysis error", str(e))

    def _collect_inputs(self) -> Dict[str, Any]:
        def ffloat(s: str) -> Optional[float]:
            s = (s or "").strip()
            if not s:
                return None
            return float(s)

        return {
            "file": self.current_file,
            "mode": self.cb_mode.currentText(),
            "col_E": self.cb_E.currentText(),
            "col_I": self.cb_I.currentText(),
            "area_cm2": float(self.ed_area.text().strip() or "1.0"),
            "invert_current": self.ck_invert.isChecked(),
            "ref": self.cb_ref.currentText(),
            "pH": float(self.sp_pH.value()),
            "scale": self.cb_scale.currentText(),
            "custom_E0": ffloat(self.ed_custom_E0.text()),
            "ir_apply": self.ck_ir.isChecked(),
            "ir_percent": float(self.ed_ir_percent.text().strip() or "0"),
            "Rs_ohm": ffloat(self.ed_rs.text()),
            "onset_thr": float(self.sp_onset.value()),
            "j_ref": ffloat(self.ed_jref.text()),
            "E_ref": ffloat(self.ed_eref.text()),
        }

    def _run_lsv_engine(self, inputs: Dict[str, Any]) -> Tuple[LSVResult, str]:
        E_raw = np.asarray(self._data[inputs["col_E"]], dtype=float)
        I_raw = np.asarray(self._data[inputs["col_I"]], dtype=float)
        E_raw, I_raw = _clean_xy(E_raw, I_raw)

        # --- Auto current sign standardization based on selected reaction mode ---
        # Many CHI-style exports report anodic currents as negative. For publication-style plots:
        #   OER -> j should be positive (anodic)
        #   HER -> j should be negative (cathodic)
        mode = inputs["mode"]

        try:
            n = len(I_raw)
            tail = I_raw[-max(5, n // 10):] if n > 0 else I_raw
            tail_med = float(np.nanmedian(tail)) if tail is not None else 0.0
        except Exception:
            tail_med = 0.0

        if mode == "OER" and tail_med < 0:
            I_raw = -I_raw
        if mode == "HER" and tail_med > 0:
            I_raw = -I_raw

        # User override (manual invert)
        if inputs["invert_current"]:
            I_raw = -I_raw

        area = float(inputs["area_cm2"])
        if area <= 0:
            raise ValueError("Area must be > 0.")
        j_mAcm2 = (I_raw / area) * 1000.0

        E_scale, E_label, formulas = convert_potential(E_raw, inputs["ref"], inputs["pH"], inputs["scale"], inputs["custom_E0"])

        if "RHE" in E_label:
            E_rhe = E_scale
        else:
            E_rhe = E_scale + 0.05916 * inputs["pH"]

        if inputs["ir_apply"]:
            Rs = inputs["Rs_ohm"]
            if Rs is None:
                raise ValueError("iR correction is enabled but Rs is empty.\n\nTip: You can obtain Rs from EIS as the high-frequency intercept of the Nyquist plot.")
            E_rhe, f2, Rs_res = apply_ir_correction(E_rhe, I_raw, float(Rs), float(inputs["ir_percent"]))
            formulas.extend(f2)
        else:
            Rs_res = None

        mode = inputs["mode"]
        if mode == "OER":
            eta = E_rhe - 1.23
            formulas.append("η_OER = E(RHE) − 1.23 V")
        else:
            eta = 0.0 - E_rhe
            formulas.append("η_HER = 0 − E(RHE)")

        # log10(|j|) for Tafel-ready outputs (avoid log(0))
        j_abs = np.abs(j_mAcm2)
        j_abs_safe = np.where(j_abs > 1e-12, j_abs, np.nan)
        logj = np.log10(j_abs_safe)

        fit_onset = tangent_onset_fit(E_rhe, j_mAcm2, mode)


        onset = None if fit_onset is None else float(fit_onset["E_onset"])


        if onset is None:


            onset = onset_potential(E_rhe, j_mAcm2, float(inputs["onset_thr"]), mode)

        E_at_j = None
        j_ref_target = None
        if inputs["j_ref"] is not None:
            # In standard convention: OER uses +j, HER uses -j
            j_ref_target = float(inputs["j_ref"])
            if mode == "HER":
                j_ref_target = -abs(j_ref_target)
            else:
                j_ref_target = abs(j_ref_target)
            E_at_j = interpolate_E_at_j(E_rhe, j_mAcm2, j_ref_target)

        j_at_E = None
        if inputs["E_ref"] is not None:
            j_at_E = interpolate_j_at_E(E_rhe, j_mAcm2, float(inputs["E_ref"]))

        # Output folder policy:
        # Desktop/Researcher Output/LSV Analysis/<Raw File Name>/run_00X
        out_dir = build_run_folder(inputs["file"])

        base = os.path.splitext(os.path.basename(inputs["file"]))[0]
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

        fig_png = os.path.join(out_dir, "Figure1_LSV_curve.png")
        fig_pdf = os.path.join(out_dir, "Figure1_LSV_curve.pdf")

        self.canvas.ax.clear()
        self.canvas.ax.plot(E_rhe, j_mAcm2)
        self.canvas.ax.set_xlabel("E (V vs RHE)")
        self.canvas.ax.set_ylabel("j (mA/cm$^2$)")
        self.canvas.ax.set_title(f"LSV ({mode})")

        # If j_ref was reached, mark it with dashed guide lines (article style)
        if E_at_j is not None and j_ref_target is not None:
            self.canvas.ax.axhline(y=j_ref_target, linestyle="--", linewidth=1.0)
            self.canvas.ax.axvline(x=E_at_j, linestyle="--", linewidth=1.0)
            self.canvas.ax.plot([E_at_j], [j_ref_target], marker="o", markersize=5)
            self.canvas.ax.annotate(
                f"({E_at_j:.3f} V, {j_ref_target:.1f} mA/cm²)",
                xy=(E_at_j, j_ref_target),
                xytext=(10, -20),
                textcoords="offset points",
                fontsize=9,
                arrowprops=dict(arrowstyle="->", lw=1.0),
            )


        # small signature
        self.canvas.ax.text(0.99, 0.01, "Researcher Analysis Toolkit",
                            transform=self.canvas.ax.transAxes,
                            ha="right", va="bottom", fontsize=8, alpha=0.7)


        # Onset visualization (tangent extrapolation, like typical journal figures)
        self.canvas.ax.axhline(0.0, linestyle="--", linewidth=1)

        if fit_onset is not None and onset is not None:
            Es = fit_onset["E_sorted"]
            j_s = fit_onset["j_smooth_sorted"]
            i1 = fit_onset["i1"]
            i2 = fit_onset["i2"]
            m_fit = fit_onset["m"]
            b_fit = fit_onset["b"]

            # show the local fit region
            self.canvas.ax.plot(Es[i1:i2], j_s[i1:i2], linewidth=2)

            # tangent line from onset to end of fit window
            E_line = np.array([onset, Es[i2-1]])
            j_line = m_fit * E_line + b_fit
            self.canvas.ax.plot(E_line, j_line, linestyle="--", linewidth=1.5)

            # onset marker at j=0
            self.canvas.ax.plot([onset], [0.0], marker="o", markersize=5)

            # annotate in mV (matches typical reporting)
            onset_mV = onset * 1000.0
            y_text = (0.15 * float(np.nanmax(j_mAcm2)) if mode == 'OER' else 0.25 * float(np.nanmin(j_mAcm2)))
            self.canvas.ax.annotate(f"onset potential {onset_mV:.0f} mV",
                                    xy=(onset, 0.0),
                                    xytext=(onset, y_text),
                                    textcoords="data",
                                    arrowprops=dict(arrowstyle="->", linewidth=1),
                                    ha="left", va="center")
        elif onset is not None:
            # fallback: threshold-based onset
            self.canvas.ax.axvline(onset, linestyle="--")
            ymax = float(np.nanmax(j_mAcm2)) if np.isfinite(np.nanmax(j_mAcm2)) else 1.0
            self.canvas.ax.text(onset, ymax * 0.9, f"Onset={onset:.3f} V", rotation=90, va="top")

        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.canvas.figure.savefig(fig_png, dpi=300)
        self.canvas.figure.savefig(fig_pdf)

        # ----- Figure2: Overpotential plot (η vs j)
        fig2_png = os.path.join(out_dir, "Figure2_Overpotential_plot.png")
        fig2_pdf = os.path.join(out_dir, "Figure2_Overpotential_plot.pdf")

        self.canvas.ax.clear()
        self.canvas.ax.plot(eta, j_mAcm2)
        self.canvas.ax.set_xlabel("Overpotential η (V)")
        self.canvas.ax.set_ylabel("j (mA/cm$^2$)")
        self.canvas.ax.set_title(f"Overpotential Plot ({mode})")
        self.canvas.ax.text(0.99, 0.01, "Researcher Analysis Toolkit",
                            transform=self.canvas.ax.transAxes,
                            ha="right", va="bottom", fontsize=8, alpha=0.7)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.canvas.figure.savefig(fig2_png, dpi=300)
        self.canvas.figure.savefig(fig2_pdf)

        # ----- Figure3: Tafel-ready plot (η vs log10|j|)
        fig3_png = os.path.join(out_dir, "Figure3_Tafel_ready_plot.png")
        fig3_pdf = os.path.join(out_dir, "Figure3_Tafel_ready_plot.pdf")

        self.canvas.ax.clear()
        self.canvas.ax.plot(logj, eta)
        self.canvas.ax.set_xlabel("log$_{10}$(|j|) (mA/cm$^2$)")
        self.canvas.ax.set_ylabel("Overpotential η (V)")
        self.canvas.ax.set_title(f"Tafel-ready Plot ({mode})")
        self.canvas.ax.text(0.99, 0.01, "Researcher Analysis Toolkit",
                            transform=self.canvas.ax.transAxes,
                            ha="right", va="bottom", fontsize=8, alpha=0.7)
        self.canvas.figure.tight_layout()
        self.canvas.draw()
        self.canvas.figure.savefig(fig3_png, dpi=300)
        self.canvas.figure.savefig(fig3_pdf)

        # ----- PDF report
        pdf_report = os.path.join(out_dir, "RAT_LSV_Analysis_Report.pdf")
        _key_for_pdf: Dict[str, Dict[str, Any]] = {}
        _key_for_pdf["Onset potential"] = {"value": None if onset is None else round(onset, 6), "unit": "V vs RHE"}
        if E_at_j is not None:
            _key_for_pdf[f"E @ {inputs['j_ref']} mA/cm²"] = {"value": round(E_at_j, 6), "unit": "V vs RHE"}
            eta_at_j = (E_at_j - 1.23) if mode == "OER" else (0.0 - E_at_j)
            _key_for_pdf[f"η @ {inputs['j_ref']} mA/cm²"] = {"value": round(float(eta_at_j), 6), "unit": "V"}
        if j_at_E is not None:
            _key_for_pdf[f"j @ {inputs['E_ref']} V"] = {"value": round(j_at_E, 6), "unit": "mA/cm²"}
        _key_for_pdf["Max j"] = {"value": round(float(np.nanmax(j_mAcm2)), 6), "unit": "mA/cm²"}

        try:
            write_pdf_report(pdf_report, inputs, formulas, _key_for_pdf, fig_png, fig2_png, fig3_png)
        except Exception:
            pass

        csv_path = os.path.join(out_dir, "RAT_LSV_processed_data.csv")
        np.savetxt(
            csv_path,
            np.column_stack([E_raw, E_rhe, j_mAcm2, eta, logj]),
            delimiter=",",
            header="E_raw_V, E_RHE_V, j_mAcm2, eta_V, log10_abs_j",
            comments=""
        )

        key_csv = os.path.join(out_dir, "RAT_LSV_key_results.csv")
        with open(key_csv, "w", encoding="utf-8") as f:
            f.write("Parameter,Value,Unit\n")

            onset_txt = "" if onset is None else f"{onset:.6f}"
            f.write(f"Onset potential,{onset_txt},V vs RHE\n")

            if E_at_j is not None:
                f.write(f"E @ {inputs['j_ref']} mA/cm2,{E_at_j:.6f},V vs RHE\n")
                eta_at_j = (E_at_j - 1.23) if mode == "OER" else (0.0 - E_at_j)
                f.write(f"η @ {inputs['j_ref']} mA/cm2,{eta_at_j:.6f},V\n")

            if j_at_E is not None:
                f.write(f"j @ {inputs['E_ref']} V,{j_at_E:.6f},mA/cm2\n")

            f.write(f"Max j,{np.nanmax(j_mAcm2):.6f},mA/cm2\n")

        meta = {
            "software": {"name": "Researcher Analysis Toolkit", "module": "LSV Analysis", "version": "v2"},
            "input_file": inputs["file"],
            "mode": mode,
            "columns": {"E": inputs["col_E"], "I": inputs["col_I"]},
            "area_cm2": area,
            "invert_current": bool(inputs["invert_current"]),
            "reference_electrode": inputs["ref"],
            "pH": float(inputs["pH"]),
            "output_scale": inputs["scale"],
            "custom_E0_vs_NHE": inputs["custom_E0"],
            "ir": {
                "enabled": bool(inputs["ir_apply"]),
                "device_comp_percent": float(inputs["ir_percent"]),
                "Rs_ohm": inputs["Rs_ohm"],
                "Rs_residual_ohm": Rs_res,
            },
            "onset_threshold_mAcm2": float(inputs["onset_thr"]),
            "results": {
                "onset_V_vs_RHE": onset,
                "E_at_jref_V_vs_RHE": E_at_j,
                "eta_at_jref_V": (None if E_at_j is None else ((E_at_j - 1.23) if mode == "OER" else (0.0 - E_at_j))),
                "j_at_Eref_mAcm2": j_at_E,
                "max_j_mAcm2": float(np.nanmax(j_mAcm2)),
            },
            "formulas_used": formulas,
            "exports": {
    "plot1_png": os.path.basename(fig_png),
    "plot1_pdf": os.path.basename(fig_pdf),
    "plot2_png": os.path.basename(fig2_png),
    "plot2_pdf": os.path.basename(fig2_pdf),
    "plot3_png": os.path.basename(fig3_png),
    "plot3_pdf": os.path.basename(fig3_pdf),
    "report_pdf": "RAT_LSV_Analysis_Report.pdf",
    "processed_csv": os.path.basename(csv_path),
    "key_results_csv": os.path.basename(key_csv),
},
        }
        json_path = os.path.join(out_dir, "RAT_LSV_analysis_metadata.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        summary = (
            f"File: {os.path.basename(inputs['file'])}\n"
            f"Mode: {mode}\n"
            f"Columns: E='{inputs['col_E']}', I='{inputs['col_I']}'\n"
            f"Reference: {inputs['ref']} | pH={inputs['pH']:.2f}\n"
            f"Area: {area:.4g} cm² | Invert current: {'ON' if inputs['invert_current'] else 'OFF'}\n"
            f"iR correction: {'ON' if inputs['ir_apply'] else 'OFF'}\n"
            f"Onset threshold: {inputs['onset_thr']:.2f} mA/cm²\n\n"
            f"Outputs saved in:\n{out_dir}\n"
        )

        key_vals: Dict[str, Dict[str, Any]] = {}
        key_vals["Onset potential"] = {"value": None if onset is None else round(onset, 6), "unit": "V vs RHE"}
        if E_at_j is not None:
            key_vals[f"E @ {inputs['j_ref']} mA/cm²"] = {"value": round(E_at_j, 6), "unit": "V vs RHE"}
            eta_at_j = (E_at_j - 1.23) if mode == "OER" else (0.0 - E_at_j)
            key_vals[f"η @ {inputs['j_ref']} mA/cm²"] = {"value": round(float(eta_at_j), 6), "unit": "V"}
        if j_at_E is not None:
            key_vals[f"j @ {inputs['E_ref']} V"] = {"value": round(j_at_E, 6), "unit": "mA/cm²"}
        key_vals["Max j"] = {"value": round(float(np.nanmax(j_mAcm2)), 6), "unit": "mA/cm²"}

        return LSVResult(
            summary_text=summary,
            key_values=key_vals,
            formulas_used=formulas,
            E_plot=E_rhe,
            j_plot=j_mAcm2,
            eta_plot=eta,
            logj_plot=logj,
            E_label="E (V vs RHE)",
            j_label="j (mA/cm²)",
            onset_V=onset,
            onset_fit=fit_onset,
            j_ref_target=j_ref_target,
            E_at_jref=E_at_j,
        ), out_dir

    def _show_results(self, result: LSVResult):
        self.txt_summary.setPlainText(result.summary_text)

        self.tbl.setRowCount(0)
        for r, (k, v) in enumerate(result.key_values.items()):
            self.tbl.insertRow(r)
            self.tbl.setItem(r, 0, QTableWidgetItem(str(k)))
            self.tbl.setItem(r, 1, QTableWidgetItem("" if v.get("value") is None else str(v.get("value"))))
            self.tbl.setItem(r, 2, QTableWidgetItem(str(v.get("unit", ""))))

        self.txt_formulas.setPlainText("\n".join(result.formulas_used))
        self.tabs.setCurrentIndex(0)

        # Enable plot selector and render selected plot
        self._last_result = result
        try:
            self.plot_selector.setEnabled(True)
            self._render_plot(self.plot_selector.currentIndex(), result)
        except Exception:
            pass


    def _on_plot_selector_changed(self, idx: int) -> None:
        if self._last_result is None:
            return
        self._render_plot(idx, self._last_result)

    def _render_plot(self, idx: int, result: LSVResult) -> None:
        """Render selected plot in the Plot tab without re-running analysis."""
        self.canvas.ax.clear()

        if idx == 0:
            # LSV curve
            self.canvas.ax.plot(result.E_plot, result.j_plot)
            self.canvas.ax.set_xlabel(result.E_label)
            self.canvas.ax.set_ylabel(result.j_label)
            self.canvas.ax.set_title(f"LSV ({self.mode_box.currentText()})")

            # baseline
            self.canvas.ax.axhline(0.0, linestyle="--", linewidth=1.0)

            # j_ref guide lines (if available)
            if result.E_at_jref is not None and result.j_ref_target is not None:
                self.canvas.ax.axhline(y=result.j_ref_target, linestyle="--", linewidth=1.0)
                self.canvas.ax.axvline(x=result.E_at_jref, linestyle="--", linewidth=1.0)
                self.canvas.ax.plot([result.E_at_jref], [result.j_ref_target], marker="o", markersize=5)

            # onset tangent (if available)
            fit = result.onset_fit or {}
            onset = result.onset_V
            if onset is not None and fit.get("m") is not None and fit.get("b") is not None and fit.get("E_sorted") is not None and fit.get("j_smooth_sorted") is not None:
                Es_all = np.asarray(fit["E_sorted"])
                js_all = np.asarray(fit["j_smooth_sorted"])
                i1 = int(fit.get("i1", 0))
                i2 = int(fit.get("i2", 0))
                if i2 > i1 and i2 <= Es_all.size:
                    Es = Es_all[i1:i2]
                    js = js_all[i1:i2]
                    self.canvas.ax.plot(Es, js, linewidth=2)
                    E_line = np.linspace(float(np.nanmin(Es)), float(np.nanmax(Es)), 100)
                    j_line = float(fit["m"]) * E_line + float(fit["b"])
                self.canvas.ax.plot(E_line, j_line, linestyle="--", linewidth=1.5)
                self.canvas.ax.plot([onset], [0.0], marker="o", markersize=6)
                self.canvas.ax.annotate(
                    f"onset potential {onset*1000:.0f} mV",
                    xy=(onset, 0.0),
                    xytext=(15, -25),
                    textcoords="offset points",
                    fontsize=10,
                    arrowprops=dict(arrowstyle="->", lw=1.2),
                )

        elif idx == 1:
            # Overpotential plot
            self.canvas.ax.plot(result.eta_plot, result.j_plot)
            self.canvas.ax.set_xlabel("Overpotential η (V)")
            self.canvas.ax.set_ylabel(result.j_label)
            self.canvas.ax.set_title("Overpotential plot")

        else:
            # Tafel-ready plot
            self.canvas.ax.plot(result.logj_plot, result.eta_plot)
            self.canvas.ax.set_xlabel("log$_{10}$(|j|) (mA/cm$^2$)")
            self.canvas.ax.set_ylabel("Overpotential η (V)")
            self.canvas.ax.set_title("Tafel-ready plot")

        # signature
        self.canvas.ax.text(0.99, 0.01, "Researcher Analysis Toolkit",
                            transform=self.canvas.ax.transAxes,
                            ha="right", va="bottom", fontsize=8, alpha=0.7)

        self.canvas.figure.tight_layout()
        self.canvas.draw()

def main():
    if os.name == 'nt':  
        myappid = 'researcher.analysistoolkit.lsv.1.0.0' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)
    
    app.setWindowIcon(QIcon("Researcherlogo.png"))

    f = QFont()
    f.setPointSize(10)
    app.setFont(f)

    win = CorvaLSVWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
