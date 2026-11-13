# -*- coding: utf-8 -*-
"""
Corva · CV Analysis (Drag&Drop + Multi-files + Scan-rate Table + PDF Report)

✅ Drag & Drop appends multiple files (not limited to one)
✅ Browse + Add more files
✅ Remove selected / Clear all files
✅ Accepts many formats: .csv .txt .dat .tsv .xlsx .xls (auto delimiter sniff for text)
✅ Scan-rate table: shows each file + editable scan rate (mV/s) BEFORE analysis
   - pre-fills from filename if possible
   - updates Parsed? YES/NO automatically when you type
   - overrides filename parsing during analysis
✅ Output scale: Measured / NHE / RHE (ref electrode + pH)
✅ Outputs saved to: Desktop/Researcher Output/CV Analysis/<run_name>[_1,_2,...]/
✅ Exports: CSV + PNG + OFFICIAL PDF report

Install:
pip install pyqt6 numpy pandas matplotlib scipy reportlab openpyxl
"""

from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
import ctypes
from PyQt6.QtGui import QIcon

from scipy.signal import savgol_filter, find_peaks
from scipy.stats import linregress

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QFormLayout,
    QSplitter, QGroupBox,
    QPushButton, QLabel, QFileDialog,
    QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QMessageBox, QLineEdit, QTextEdit,
    QTableWidget, QTableWidgetItem
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# PDF
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import cm


# -----------------------------
# Reference electrode offsets to NHE (V)
# -----------------------------
REF_TABLE = [
    ("Ag/AgCl (sat KCl)", +0.197),
    ("Ag/AgCl (3.0 M KCl)", +0.210),
    ("Ag/AgCl (1.0 M KCl)", +0.235),
    ("SCE (sat KCl)", +0.241),
    ("Hg/HgO (1.0 M KOH)", +0.098),
    ("Hg/HgO (0.1 M KOH)", +0.140),
    ("Ag/Ag2SO4 (sat K2SO4)", +0.650),
    ("Custom offset to NHE", None),
]


# -----------------------------
# Output path rules
# -----------------------------
def desktop_path() -> str:
    return os.path.join(os.path.expanduser("~"), "Desktop")


def default_cv_output_root() -> str:
    return os.path.join(desktop_path(), "Researcher Output", "CV Analysis")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]+', "_", name)
    name = name.strip().strip(".")
    return name if name else "CV_Run"


def unique_dir(path: str) -> str:
    if not os.path.exists(path):
        return path
    k = 1
    while True:
        p = f"{path}_{k}"
        if not os.path.exists(p):
            return p
        k += 1


def unique_path(path: str) -> str:
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    k = 1
    while True:
        p = f"{base}_{k}{ext}"
        if not os.path.exists(p):
            return p
        k += 1


def build_run_outdir(files: List[str]) -> str:
    root = default_cv_output_root()
    if len(files) == 1:
        base = os.path.splitext(os.path.basename(files[0]))[0]
    else:
        base = "CV_Batch"
    outdir = os.path.join(root, safe_name(base))
    return unique_dir(outdir)


# -----------------------------
# Read any tabular file
# -----------------------------
def read_any_table(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path, sheet_name=0, nrows=nrows)
    # sep=None + engine python -> auto-detect delimiter (comma/tab/space/semicolon, etc.)
    return pd.read_csv(path, nrows=nrows, sep=None, engine="python")


# -----------------------------
# Utilities
# -----------------------------
def safe_odd_window(n: int) -> int:
    n = int(n)
    if n < 5:
        n = 5
    if n % 2 == 0:
        n += 1
    return n


def parse_scanrate_from_filename(fname: str) -> Optional[Tuple[float, str]]:
    """
    Robust scan-rate parser from filename.
    Accepts:
      50mV/s, 50mVs, 0.05V/s, SR50, scanrate50, scanrate=50 ...
    Returns (V/s, label)
    """
    s = os.path.basename(fname)
    s2 = s.replace(" ", "").replace("-", "").replace("_", "").lower()

    # explicit V/s
    m = re.search(r"(\d+(?:\.\d+)?)v\/s", s2) or re.search(r"(\d+(?:\.\d+)?)vs", s2)
    if m:
        val = float(m.group(1))
        return val, f"{val:g} V/s"

    # explicit mV/s
    m = re.search(r"(\d+(?:\.\d+)?)mv\/s", s2) or re.search(r"(\d+(?:\.\d+)?)mvs", s2) or re.search(r"(\d+(?:\.\d+)?)mvs1", s2)
    if m:
        val = float(m.group(1))
        return val * 1e-3, f"{val:g} mV/s"

    # SR50 / scanrate50 / scanrate=50  (assume mV/s)
    m = re.search(r"(?:sr|scanrate|scanr|scanrate=|scanrate:)(\d+(?:\.\d+)?)", s2)
    if m:
        val = float(m.group(1))
        return val * 1e-3, f"{val:g} mV/s"

    return None


def split_forward_reverse(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split into forward and reverse using the first sign-change in dE."""
    dE = np.diff(E)
    sign = np.sign(dE)
    sign[sign == 0] = 1
    idx_change = np.where(np.diff(sign) != 0)[0]
    if len(idx_change) == 0:
        mid = len(E) // 2
        return np.arange(0, mid), np.arange(mid, len(E))
    k = int(idx_change[0] + 1)
    return np.arange(0, k), np.arange(k, len(E))


def top_peaks(x: np.ndarray, y: np.ndarray, kind: str, max_peaks: int,
              prominence: Optional[float], distance: int) -> List[Tuple[float, float, int]]:
    """Find top peaks in forward (anodic) or reverse (cathodic)."""
    if len(x) < 10:
        return []
    if kind == "anodic":
        yy = y
        peaks, _ = find_peaks(yy, prominence=prominence, distance=distance)
        scores = yy[peaks] if len(peaks) else np.array([])
    else:
        yy = -y
        peaks, _ = find_peaks(yy, prominence=prominence, distance=distance)
        scores = yy[peaks] if len(peaks) else np.array([])

    if len(peaks) == 0:
        return []

    order = np.argsort(scores)[::-1]
    out: List[Tuple[float, float, int]] = []
    for i in order[:max_peaks]:
        p = int(peaks[i])
        out.append((float(x[p]), float(y[p]), int(p)))
    return out


def power_law_fit(v: np.ndarray, ip: np.ndarray) -> Dict[str, float]:
    """ip = a*v^b (fit in log space)."""
    v = np.array(v, float)
    ip = np.array(ip, float)
    mask = (v > 0) & (np.abs(ip) > 0) & np.isfinite(v) & np.isfinite(ip)
    v2 = v[mask]
    ip2 = np.abs(ip[mask])
    if len(v2) < 2:
        return {"a": np.nan, "b": np.nan, "r2": np.nan}
    logv = np.log10(v2)
    logip = np.log10(ip2)
    res = linregress(logv, logip)
    return {"a": float(10 ** res.intercept), "b": float(res.slope), "r2": float(res.rvalue ** 2)}


def dunn_fit(v: np.ndarray, ip: np.ndarray) -> Dict[str, float]:
    """Dunn: i = k1*v + k2*v^0.5."""
    v = np.array(v, float)
    ip = np.array(ip, float)
    mask = np.isfinite(v) & np.isfinite(ip) & (v > 0)
    v = v[mask]
    ip = ip[mask]
    if len(v) < 3:
        return {"k1": np.nan, "k2": np.nan, "r2": np.nan}
    X = np.column_stack([v, np.sqrt(v)])
    beta, *_ = np.linalg.lstsq(X, ip, rcond=None)
    k1, k2 = beta
    yhat = X @ beta
    ss_res = float(np.sum((ip - yhat) ** 2))
    ss_tot = float(np.sum((ip - np.mean(ip)) ** 2)) if len(ip) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"k1": float(k1), "k2": float(k2), "r2": float(r2)}


# -----------------------------
# PDF table (full-width + wrap + NaN -> N/A)
# -----------------------------
def df_to_rl_table(df: pd.DataFrame, max_rows: int = 35, page_width_pts: float = 500) -> Table:
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    style.fontSize = 7
    style.leading = 8

    if df is None or len(df) == 0:
        return Table([["No data"]])

    d = df.copy()
    if len(d) > max_rows:
        d = d.iloc[:max_rows].copy()

    def fmt(x):
        if x is None:
            return "N/A"
        if isinstance(x, (float, np.floating)):
            if np.isnan(x):
                return "N/A"
            return f"{x:.6g}"
        s = str(x)
        return s if s.strip() else "N/A"

    header = [Paragraph(f"<b>{c}</b>", style) for c in d.columns]
    data = [header]
    for _, row in d.iterrows():
        data.append([Paragraph(fmt(v), style) for v in row.values])

    ncol = len(d.columns)
    col_w = float(page_width_pts) / max(ncol, 1)
    colWidths = [col_w] * ncol

    tbl = Table(data, colWidths=colWidths, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#efefef")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return tbl


def build_pdf(pdf_path: str, meta: Dict[str, str], df_sr: pd.DataFrame, df_main: pd.DataFrame,
              df_scan: pd.DataFrame, df_cdl: pd.DataFrame, fig_paths: List[str], notes: List[str]) -> None:
    styles = getSampleStyleSheet()

    left_m = 1.4 * cm
    right_m = 1.4 * cm
    top_m = 1.4 * cm
    bottom_m = 1.4 * cm

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        leftMargin=left_m, rightMargin=right_m,
        topMargin=top_m, bottomMargin=bottom_m
    )
    usable_w = A4[0] - left_m - right_m

    story = []
    story.append(Paragraph("<b>Official CV Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))
    story.append(Paragraph(
        "Generated with Researcher Analysis Toolkit v1.0.0",
        styles["BodyText"],
    ))
    story.append(Paragraph(
        '<b>Suggested citation</b> · Behroozi et al., <i>RSC Adv.</i> (2026). '
        '<link href="https://doi.org/10.1039/D6RA08618G" color="#1F4E79">'
        '<u>10.1039/D6RA08618G</u></link>',
        styles["BodyText"],
    ))
    story.append(Spacer(1, 12))
    story.append(Paragraph("<b>Settings & Metadata</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph("<br/>".join([f"<b>{k}:</b> {v}" for k, v in meta.items()]), styles["BodyText"]))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Scan-rate Detection / Inputs</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(df_to_rl_table(df_sr, page_width_pts=usable_w))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Main Results (E1/2 & Ep1/2 included)</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(df_to_rl_table(df_main, page_width_pts=usable_w))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Scan-rate Kinetics</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(df_to_rl_table(df_scan, page_width_pts=usable_w))
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>Cdl & ECSA</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(df_to_rl_table(df_cdl, page_width_pts=usable_w))
    story.append(Spacer(1, 10))

    story.append(PageBreak())
    story.append(Paragraph("<b>Figures</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    for fp in fig_paths:
        if fp and os.path.exists(fp):
            story.append(Paragraph(os.path.basename(fp), styles["BodyText"]))
            story.append(Spacer(1, 4))
            story.append(RLImage(fp, width=usable_w, height=0.58 * usable_w))
            story.append(Spacer(1, 12))

    if notes:
        story.append(PageBreak())
        story.append(Paragraph("<b>Notes</b>", styles["Heading2"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph("<br/>".join(notes), styles["BodyText"]))

    doc.build(story)


# -----------------------------
# Conversion
# -----------------------------
def convert_potential(E_meas: np.ndarray, offset_to_nhe: float, out_scale: str, pH: float) -> Tuple[np.ndarray, str]:
    if out_scale == "Measured":
        return E_meas.astype(float), "E (V vs Ref)"
    E_nhe = E_meas.astype(float) + float(offset_to_nhe)
    if out_scale == "NHE":
        return E_nhe, "E (V vs NHE)"
    E_rhe = E_nhe + 0.05916 * float(pH)
    return E_rhe, "E (V vs RHE)"


def convert_current(I_A: np.ndarray, unit: str, use_area: bool, area_cm2: float) -> Tuple[np.ndarray, str]:
    I = I_A.astype(float)
    density = False
    if use_area:
        I = I / float(area_cm2)
        density = True
    if unit.lower() == "ma":
        I = I * 1000.0
        u = "mA"
    else:
        u = "A"
    ylabel = f"j ({u}/cm$^2$)" if density else f"I ({u})"
    return I, ylabel


@dataclass
class LoadedData:
    file: str
    df: pd.DataFrame


@dataclass
class CVRecord:
    file: str
    scanrate_Vs: float
    scanrate_label: str
    E: np.ndarray
    I: np.ndarray
    fwd_idx: np.ndarray
    rev_idx: np.ndarray
    peaks_an: List[Tuple[float, float, int]]
    peaks_ca: List[Tuple[float, float, int]]


# -----------------------------
# Drag&Drop Widget (APPEND MODE)
# -----------------------------
class DropBox(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Drag & Drop data files here (.csv .txt .xlsx ...)\n(or use Browse / Add files)")
        self.setStyleSheet("QLabel{border:2px dashed #999; padding:18px; font-size:14px;}")

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths: List[str] = []
        for url in e.mimeData().urls():
            p = url.toLocalFile()
            if p and os.path.isfile(p):
                paths.append(os.path.abspath(p))
        if not paths:
            return
        win = self.window()
        if hasattr(win, "add_files_received"):
            win.add_files_received(paths)


# -----------------------------
# Main GUI
# -----------------------------
class CorvaCV(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Researcher Analysis Toolkit — Cyclic Voltammetry (CV)")
        self.resize(1450, 860)

        self.loaded: List[LoadedData] = []
        self.files: List[str] = []
        self.records: List[CVRecord] = []
        self.outdir: Optional[str] = None
        self._sr_override: Dict[str, float] = {}

        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main.addWidget(splitter)

        # LEFT
        left = QWidget()
        left_l = QVBoxLayout(left)

        self.drop = DropBox(left)
        left_l.addWidget(self.drop)

        self.btn_browse = QPushButton("Browse data file(s)")
        self.btn_browse.clicked.connect(self.browse_files)
        left_l.addWidget(self.btn_browse)

        self.btn_add = QPushButton("Add more data file(s)")
        self.btn_add.clicked.connect(self.add_more_files)
        left_l.addWidget(self.btn_add)

        btn_row = QHBoxLayout()
        self.btn_remove_sel = QPushButton("Remove selected")
        self.btn_remove_sel.clicked.connect(self.remove_selected_files)
        self.btn_clear = QPushButton("Clear all")
        self.btn_clear.clicked.connect(self.clear_all_files)
        btn_row.addWidget(self.btn_remove_sel)
        btn_row.addWidget(self.btn_clear)
        left_l.addLayout(btn_row)

        self.lbl_files = QLabel("Files: (none)")
        self.lbl_files.setWordWrap(True)
        left_l.addWidget(self.lbl_files)

        # Scan-rate table
        self.tbl_sr = QTableWidget(0, 3)
        self.tbl_sr.setHorizontalHeaderLabels(["File", "Scan rate (mV/s)", "Parsed?"])
        self.tbl_sr.setColumnWidth(0, 280)
        self.tbl_sr.setColumnWidth(1, 130)
        self.tbl_sr.setColumnWidth(2, 70)
        self.tbl_sr.setMinimumHeight(170)
        left_l.addWidget(self.tbl_sr)

        # ✅ update YES/NO when user types
        self.tbl_sr.itemChanged.connect(self.on_scanrate_cell_changed)

        grp_cols = QGroupBox("Select Columns (after loading)")
        form_cols = QFormLayout(grp_cols)

        self.cmb_file = QComboBox()
        self.cmb_file.currentIndexChanged.connect(self.on_file_changed)
        form_cols.addRow("File:", self.cmb_file)

        self.cmb_E = QComboBox()
        self.cmb_I = QComboBox()
        form_cols.addRow("Potential column (E):", self.cmb_E)
        form_cols.addRow("Current column (I):", self.cmb_I)

        self.cmb_cycle = QComboBox()
        self.cmb_cycle.addItem("(none)")
        form_cols.addRow("Cycle/Segment column (optional):", self.cmb_cycle)

        left_l.addWidget(grp_cols)

        self.btn_load = QPushButton("Load Selected File Columns")
        self.btn_load.clicked.connect(self.load_current_selection)
        self.btn_load.setEnabled(False)
        left_l.addWidget(self.btn_load)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(220)
        left_l.addWidget(self.log)

        splitter.addWidget(left)

        # CENTER plot
        center = QWidget()
        center_l = QVBoxLayout(center)
        self.fig = plt.Figure()
        self.canvas = FigureCanvas(self.fig)
        center_l.addWidget(self.canvas, 1)
        self.lbl_status = QLabel("Status: Ready")
        center_l.addWidget(self.lbl_status)
        splitter.addWidget(center)

        # RIGHT settings
        right = QWidget()
        right_l = QVBoxLayout(right)

        grp_ref = QGroupBox("Reference / Output Scale")
        form_ref = QFormLayout(grp_ref)

        self.cmb_ref = QComboBox()
        for name, _ in REF_TABLE:
            self.cmb_ref.addItem(name)
        self.cmb_ref.currentIndexChanged.connect(self.on_ref_changed)
        form_ref.addRow("Reference electrode:", self.cmb_ref)

        self.spn_custom_offset = QDoubleSpinBox()
        self.spn_custom_offset.setDecimals(6)
        self.spn_custom_offset.setRange(-2.0, 2.0)
        self.spn_custom_offset.setValue(0.0)
        self.spn_custom_offset.setEnabled(False)
        form_ref.addRow("Custom offset to NHE (V):", self.spn_custom_offset)

        self.cmb_scale = QComboBox()
        self.cmb_scale.addItems(["Measured", "NHE", "RHE"])
        form_ref.addRow("Output scale:", self.cmb_scale)

        self.spn_pH = QDoubleSpinBox()
        self.spn_pH.setDecimals(2)
        self.spn_pH.setRange(0.0, 14.0)
        self.spn_pH.setValue(7.0)
        form_ref.addRow("pH (for RHE):", self.spn_pH)

        right_l.addWidget(grp_ref)

        grp_set = QGroupBox("Data Settings")
        form_set = QFormLayout(grp_set)

        self.cmb_current_unit = QComboBox()
        self.cmb_current_unit.addItems(["A", "mA"])
        self.cmb_current_unit.setCurrentText("mA")
        form_set.addRow("Current unit:", self.cmb_current_unit)

        self.chk_use_area = QCheckBox("Use area (current density)")
        self.chk_use_area.setChecked(True)
        form_set.addRow(self.chk_use_area)

        self.spn_area = QDoubleSpinBox()
        self.spn_area.setDecimals(4)
        self.spn_area.setRange(0.0001, 1000.0)
        self.spn_area.setValue(1.0)
        form_set.addRow("Area (cm²):", self.spn_area)

        self.chk_smooth = QCheckBox("Smooth current (Savitzky–Golay)")
        self.chk_smooth.setChecked(True)
        form_set.addRow(self.chk_smooth)

        self.spn_win = QSpinBox()
        self.spn_win.setRange(5, 501)
        self.spn_win.setSingleStep(2)
        self.spn_win.setValue(21)
        form_set.addRow("Savgol window:", self.spn_win)

        self.spn_poly = QSpinBox()
        self.spn_poly.setRange(2, 7)
        self.spn_poly.setValue(3)
        form_set.addRow("Savgol poly:", self.spn_poly)

        self.spn_peak_dist = QSpinBox()
        self.spn_peak_dist.setRange(5, 5000)
        self.spn_peak_dist.setValue(20)
        form_set.addRow("Peak distance (pts):", self.spn_peak_dist)

        self.spn_max_peaks = QSpinBox()
        self.spn_max_peaks.setRange(1, 10)
        self.spn_max_peaks.setValue(1)
        form_set.addRow("Max peaks/branch:", self.spn_max_peaks)

        right_l.addWidget(grp_set)

        # Requirements label
        self.lbl_req = QLabel(
            "Requirements:\n"
            "• Scan-rate kinetics (b-value/Dunn): need ≥ 2 scan rates\n"
            "• Cdl/ECSA: need ≥ 3 scan rates\n"
            "Tip: add multiple files with different scan rates (e.g., 10, 20, 50, 100 mV/s)."
        )
        self.lbl_req.setWordWrap(True)
        self.lbl_req.setStyleSheet("QLabel{padding:6px; border:1px solid #ddd; border-radius:6px;}")
        right_l.addWidget(self.lbl_req)

        grp_cdl = QGroupBox("Cdl / ECSA")
        form_cdl = QFormLayout(grp_cdl)

        self.chk_cdl = QCheckBox("Enable Cdl/ECSA")
        self.chk_cdl.setChecked(True)
        form_cdl.addRow(self.chk_cdl)

        self.spn_E1 = QDoubleSpinBox()
        self.spn_E1.setDecimals(4)
        self.spn_E1.setRange(-10.0, 10.0)
        self.spn_E1.setValue(0.10)
        form_cdl.addRow("Non-faradaic E1 (V):", self.spn_E1)

        self.spn_E2 = QDoubleSpinBox()
        self.spn_E2.setDecimals(4)
        self.spn_E2.setRange(-10.0, 10.0)
        self.spn_E2.setValue(0.20)
        form_cdl.addRow("Non-faradaic E2 (V):", self.spn_E2)

        self.chk_ecsa = QCheckBox("Compute ECSA = Cdl / Cs")
        self.chk_ecsa.setChecked(False)
        form_cdl.addRow(self.chk_ecsa)

        self.spn_Cs = QDoubleSpinBox()
        self.spn_Cs.setDecimals(4)
        self.spn_Cs.setRange(0.0001, 1.0)
        self.spn_Cs.setValue(0.040)
        form_cdl.addRow("Cs (F/cm²):", self.spn_Cs)

        right_l.addWidget(grp_cdl)

        grp_out = QGroupBox("Export")
        form_out = QFormLayout(grp_out)
        self.edt_out = QLineEdit("")
        self.edt_out.setReadOnly(True)
        form_out.addRow("Output folder:", self.edt_out)

        self.btn_start = QPushButton("Start Analysis + Export (CSV+PNG+PDF)")
        self.btn_start.clicked.connect(self.start_analysis_export)
        self.btn_start.setEnabled(False)

        # Do NOT create output folders yet (only create on Start/Export)
        self.outdir = None
        self.edt_out.setText(default_cv_output_root() + "  (will be created on export)")
        form_out.addRow(self.btn_start)

        right_l.addWidget(grp_out)
        right_l.addStretch(1)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 820, 420])

    # ---------- UI helpers ----------
    def logp(self, s: str) -> None:
        self.log.append(s)

    def on_ref_changed(self) -> None:
        self.spn_custom_offset.setEnabled(self.cmb_ref.currentText().startswith("Custom"))

    def browse_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CV data file(s)", "", "Data Files (*.csv *.txt *.dat *.tsv *.xlsx *.xls);;All Files (*.*)"
        )
        if files:
            self.on_files_received(files)

    def add_more_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add data file(s)", "", "Data Files (*.csv *.txt *.dat *.tsv *.xlsx *.xls);;All Files (*.*)"
        )
        if not files:
            return
        new_files = [os.path.abspath(f) for f in files if os.path.isfile(f)]
        if not new_files:
            return
        existing = set(self.files) if self.files else set()
        merged = list(existing.union(new_files))
        merged.sort()
        self.on_files_received(merged)

    # used by Drag&Drop to APPEND
    def add_files_received(self, new_files: List[str]) -> None:
        new_files = [os.path.abspath(f) for f in new_files if os.path.isfile(f)]
        if not new_files:
            return
        existing = set(self.files) if self.files else set()
        merged = list(existing.union(new_files))
        merged.sort()
        self.on_files_received(merged)

    def clear_all_files(self) -> None:
        self.files = []
        self.loaded = []
        self.records = []
        self._sr_override = {}
        self.cmb_file.clear()
        self.cmb_E.clear()
        self.cmb_I.clear()
        self.cmb_cycle.clear()
        self.cmb_cycle.addItem("(none)")
        self.tbl_sr.setRowCount(0)
        self.lbl_files.setText("Files: (none)")
        self.btn_load.setEnabled(False)
        self.btn_start.setEnabled(False)

        # Do NOT create output folders yet (only create on Start/Export)
        self.outdir = None
        self.edt_out.setText(default_cv_output_root() + "  (will be created on export)")
        self.fig.clear()
        self.canvas.draw()
        self.lbl_status.setText("Status: Ready")
        self.logp("🧹 Cleared all files. You can add new file(s) now.")

    def remove_selected_files(self) -> None:
        if self.tbl_sr.rowCount() == 0:
            return
        rows = sorted({i.row() for i in self.tbl_sr.selectedItems()}, reverse=True)
        if not rows:
            QMessageBox.information(self, "Remove", "Select row(s) in the scan-rate table to remove.")
            return
        to_remove = set()
        for r in rows:
            it = self.tbl_sr.item(r, 0)
            if it:
                to_remove.add(it.text().strip())
        kept = [f for f in self.files if os.path.basename(f) not in to_remove]
        if kept:
            self.on_files_received(kept)
        else:
            self.clear_all_files()

    def on_scanrate_cell_changed(self, item: QTableWidgetItem) -> None:
        # Only when scan-rate column changes (col=1)
        if item is None or item.column() != 1:
            return
        row = item.row()
        txt = item.text().strip()
        status_item = self.tbl_sr.item(row, 2)
        if status_item is None:
            status_item = QTableWidgetItem("")
            status_item.setFlags(status_item.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.tbl_sr.setItem(row, 2, status_item)

        try:
            val = float(txt)
            status_item.setText("YES" if val > 0 else "NO")
        except Exception:
            status_item.setText("NO")

    def on_files_received(self, files: List[str]) -> None:
        files = [os.path.abspath(f) for f in files if os.path.isfile(f)]
        if not files:
            return

        self.files = files
        self.loaded = []
        self.records = []

        self.cmb_file.clear()
        self.cmb_E.clear()
        self.cmb_I.clear()
        self.cmb_cycle.clear()
        self.cmb_cycle.addItem("(none)")

        self.lbl_files.setText("Files:\n" + "\n".join([os.path.basename(f) for f in files]))
        self.logp(f"Received {len(files)} file(s). Reading headers...")

        # Fill scan-rate table (block signal while populating)
        self.tbl_sr.blockSignals(True)
        self.tbl_sr.setRowCount(0)
        for f in self.files:
            base = os.path.basename(f)
            sr = parse_scanrate_from_filename(f)

            row = self.tbl_sr.rowCount()
            self.tbl_sr.insertRow(row)

            it0 = QTableWidgetItem(base)
            it0.setFlags(it0.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.tbl_sr.setItem(row, 0, it0)

            if sr is not None and np.isfinite(sr[0]) and sr[0] > 0:
                mv = sr[0] * 1000.0
                it1 = QTableWidgetItem(f"{mv:g}")
                it2 = QTableWidgetItem("YES")
            else:
                it1 = QTableWidgetItem("")
                it2 = QTableWidgetItem("NO")

            self.tbl_sr.setItem(row, 1, it1)
            it2.setFlags(it2.flags() ^ Qt.ItemFlag.ItemIsEditable)
            self.tbl_sr.setItem(row, 2, it2)

        self.tbl_sr.blockSignals(False)

        ok = 0
        for f in files:
            try:
                df = read_any_table(f, nrows=5)
                self.loaded.append(LoadedData(f, df))
                self.cmb_file.addItem(os.path.basename(f))
                ok += 1
            except Exception as e:
                self.logp(f"❌ Failed to read {os.path.basename(f)}: {e}")

        if ok == 0:
            QMessageBox.critical(self, "Error", "No valid files loaded.")
            return

        self.cmb_file.setCurrentIndex(0)
        self.on_file_changed(0)
        self.btn_load.setEnabled(True)
        self.btn_start.setEnabled(False)

        # Do NOT create output folders yet (only create on Start/Export)
        self.outdir = None
        self.edt_out.setText(default_cv_output_root() + "  (will be created on export)")

        valid_sr = 0
        for r in range(self.tbl_sr.rowCount()):
            txt = self.tbl_sr.item(r, 1).text().strip() if self.tbl_sr.item(r, 1) else ""
            try:
                mv = float(txt) if txt else np.nan
            except Exception:
                mv = np.nan
            if np.isfinite(mv) and mv > 0:
                valid_sr += 1

        self.logp(
            f"Scan-rate values available: {valid_sr}/{len(self.files)}\n"
            f"• Kinetics needs ≥2 scan rates\n"
            f"• Cdl/ECSA needs ≥3 scan rates"
        )

        self.lbl_status.setText("Status: Select columns, then click 'Load Selected File Columns'.")

    def on_file_changed(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.loaded):
            return
        df = self.loaded[idx].df
        cols = list(df.columns)

        self.cmb_E.clear()
        self.cmb_I.clear()
        self.cmb_cycle.clear()
        self.cmb_cycle.addItem("(none)")

        for c in cols:
            self.cmb_E.addItem(c)
            self.cmb_I.addItem(c)
            self.cmb_cycle.addItem(c)

        def guess(targets: List[str]) -> Optional[str]:
            cols_l = [c.lower() for c in cols]
            for t in targets:
                t = t.lower()
                for c, cl in zip(cols, cols_l):
                    if t in cl:
                        return c
            return None

        gE = guess(["potential", "ewe", "voltage", "potential/v"])
        gI = guess(["current", "current/a", "i/a"])
        gC = guess(["cycle", "segment", "loop"])

        if gE:
            self.cmb_E.setCurrentText(gE)
        if gI:
            self.cmb_I.setCurrentText(gI)
        if gC:
            self.cmb_cycle.setCurrentText(gC)

    def load_current_selection(self) -> None:
        idx = self.cmb_file.currentIndex()
        if idx < 0 or idx >= len(self.loaded):
            return

        f = self.loaded[idx].file
        colE = self.cmb_E.currentText()
        colI = self.cmb_I.currentText()

        try:
            df = read_any_table(f)
        except Exception as e:
            QMessageBox.critical(self, "Read error", str(e))
            return

        if colE not in df.columns or colI not in df.columns:
            QMessageBox.warning(self, "Columns", "Selected columns not found in file.")
            return

        self.loaded[idx] = LoadedData(f, df)
        self.logp(f"✅ Loaded full data: {os.path.basename(f)} | E='{colE}' I='{colI}'")

        self.btn_start.setEnabled(True)
        self.lbl_status.setText("Status: Columns selected. Click Start Analysis.")
        self.preview_single(idx, colE, colI)

    def preview_single(self, idx: int, colE: str, colI: str) -> None:
        df = self.loaded[idx].df
        try:
            E_meas = df[colE].to_numpy(float)
            I_A = df[colI].to_numpy(float)
        except Exception:
            return

        E, xlabel = convert_potential(E_meas, self.get_offset(), self.cmb_scale.currentText(), float(self.spn_pH.value()))
        I, ylabel = convert_current(I_A, self.cmb_current_unit.currentText(), self.chk_use_area.isChecked(), float(self.spn_area.value()))

        mask = np.isfinite(E) & np.isfinite(I)
        E = E[mask]
        I = I[mask]

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.plot(E, I)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Preview (selected columns)")
        ax.grid(True, alpha=0.2)
        self.fig.tight_layout()
        self.canvas.draw()

    # ---------- Analysis ----------
    def get_offset(self) -> float:
        name = self.cmb_ref.currentText()
        for n, off in REF_TABLE:
            if n == name:
                return float(self.spn_custom_offset.value()) if off is None else float(off)
        return 0.0

    def get_scanrates_from_table(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for r in range(self.tbl_sr.rowCount()):
            base_item = self.tbl_sr.item(r, 0)
            val_item = self.tbl_sr.item(r, 1)
            if not base_item or not val_item:
                continue
            base = base_item.text().strip()
            txt = val_item.text().strip()
            if not txt:
                continue
            try:
                mv = float(txt)
                if mv > 0:
                    out[base] = mv * 1e-3  # V/s
            except Exception:
                pass
        return out

    def read_record(self, df: pd.DataFrame, file_path: str, colE: str, colI: str) -> CVRecord:
        E_meas = df[colE].to_numpy(float)
        I_A = df[colI].to_numpy(float)

        E, _ = convert_potential(E_meas, self.get_offset(), self.cmb_scale.currentText(), float(self.spn_pH.value()))
        I, _ = convert_current(I_A, self.cmb_current_unit.currentText(), self.chk_use_area.isChecked(), float(self.spn_area.value()))

        mask = np.isfinite(E) & np.isfinite(I)
        E = E[mask]
        I = I[mask]

        I_for_peaks = I.copy()
        if self.chk_smooth.isChecked():
            w = safe_odd_window(int(self.spn_win.value()))
            if w < len(I_for_peaks):
                try:
                    I_for_peaks = savgol_filter(I_for_peaks, window_length=w, polyorder=int(self.spn_poly.value()))
                except Exception:
                    pass

        fwd, rev = split_forward_reverse(E)

        span = float(np.nanmax(I_for_peaks) - np.nanmin(I_for_peaks)) if len(I_for_peaks) else 0.0
        prom = 0.05 * span if span > 0 else None

        peaks_an = top_peaks(E[fwd], I_for_peaks[fwd], "anodic",
                             int(self.spn_max_peaks.value()), prom, int(self.spn_peak_dist.value()))
        peaks_ca = top_peaks(E[rev], I_for_peaks[rev], "cathodic",
                             int(self.spn_max_peaks.value()), prom, int(self.spn_peak_dist.value()))

        # fallback (never empty)
        if (not peaks_an) and (len(fwd) > 10):
            i_local = I_for_peaks[fwd]
            e_local = E[fwd]
            k = int(np.argmax(i_local))
            peaks_an = [(float(e_local[k]), float(i_local[k]), int(fwd[k]))]
        if (not peaks_ca) and (len(rev) > 10):
            i_local = I_for_peaks[rev]
            e_local = E[rev]
            k = int(np.argmin(i_local))
            peaks_ca = [(float(e_local[k]), float(i_local[k]), int(rev[k]))]

        # scan rate: table override first
        base = os.path.basename(file_path)
        override = getattr(self, "_sr_override", {})
        if base in override:
            scanrate_Vs = float(override[base])
            scanrate_label = f"{scanrate_Vs*1000:g} mV/s"
        else:
            sr = parse_scanrate_from_filename(file_path)
            if sr is None:
                scanrate_Vs = np.nan
                scanrate_label = base
            else:
                scanrate_Vs, scanrate_label = sr

        return CVRecord(file_path, scanrate_Vs, scanrate_label, E, I, fwd, rev, peaks_an, peaks_ca)

    def start_analysis_export(self) -> None:
        if not self.files or not self.loaded:
            QMessageBox.warning(self, "No files", "Please load file(s) first.")
            return

        colE = self.cmb_E.currentText()
        colI = self.cmb_I.currentText()
        cyc_col = self.cmb_cycle.currentText()
        cyc_col = None if cyc_col == "(none)" else cyc_col

        # scan-rate override from table
        self._sr_override = self.get_scanrates_from_table()

        outdir = build_run_outdir(self.files)
        ensure_dir(outdir)
        self.outdir = outdir
        self.edt_out.setText(outdir)

        processed_rows: List[Dict[str, float]] = []
        main_rows: List[Dict[str, float]] = []
        self.records = []

        for ld in self.loaded:
            f = ld.file
            try:
                df = read_any_table(f)
            except Exception as e:
                self.logp(f"❌ Failed reading {os.path.basename(f)}: {e}")
                continue

            if colE not in df.columns or colI not in df.columns:
                self.logp(f"❌ Skipped {os.path.basename(f)} (columns not found).")
                continue

            r = self.read_record(df, f, colE, colI)
            self.records.append(r)

            for Ei, Ii in zip(r.E, r.I):
                processed_rows.append({
                    "file": os.path.basename(r.file),
                    "scanrate_V_per_s": r.scanrate_Vs,
                    "scanrate_label": r.scanrate_label,
                    "E_V": Ei,
                    "I_out": Ii
                })

        if not self.records:
            QMessageBox.critical(self, "No valid records", "No files could be analyzed with selected columns.")
            return

        # sort by scanrate (NaN last)
        self.records.sort(key=lambda x: (np.nan_to_num(x.scanrate_Vs, nan=1e9)))

        df_processed = pd.DataFrame(processed_rows)
        df_processed.to_csv(unique_path(os.path.join(outdir, "CV_processed_data.csv")), index=False)

        df_sr = pd.DataFrame([{
            "file": os.path.basename(r.file),
            "scanrate_V_per_s": r.scanrate_Vs,
            "scanrate_label": r.scanrate_label,
            "source": "USER" if os.path.basename(r.file) in self._sr_override else ("FILENAME" if np.isfinite(r.scanrate_Vs) else "NONE"),
            "valid": "YES" if (np.isfinite(r.scanrate_Vs) and r.scanrate_Vs > 0) else "NO"
        } for r in self.records])
        df_sr.to_csv(unique_path(os.path.join(outdir, "ScanRate_Detection.csv")), index=False)

        annotations: Dict[str, Dict[str, float]] = {}

        for r in self.records:
            Epa = Ipa = Epc = Ipc = np.nan
            if r.peaks_an:
                Epa, Ipa, _ = r.peaks_an[0]
            if r.peaks_ca:
                Epc, Ipc, _ = r.peaks_ca[0]

            E12 = np.nan
            if np.isfinite(Epa) and np.isfinite(Epc):
                E12 = 0.5 * (Epa + Epc)

            Ep12_an = np.nan
            Ep12_ca = np.nan
            try:
                if np.isfinite(Epa) and np.isfinite(Ipa):
                    idx = r.fwd_idx
                    Eb = r.E[idx]
                    Ib = r.I[idx]
                    k0 = int(np.argmin(np.abs(Eb - Epa)))
                    target = Ipa / 2.0
                    if k0 > 5:
                        Ep12_an = float(Eb[np.argmin(np.abs(Ib[:k0 + 1] - target))])
                if np.isfinite(Epc) and np.isfinite(Ipc):
                    idx = r.rev_idx
                    Eb = r.E[idx]
                    Ib = r.I[idx]
                    k0 = int(np.argmin(np.abs(Eb - Epc)))
                    target = Ipc / 2.0
                    if k0 > 5:
                        Ep12_ca = float(Eb[np.argmin(np.abs(Ib[:k0 + 1] - target))])
            except Exception:
                pass

            dEp = float(Epa - Epc) if np.isfinite(Epa) and np.isfinite(Epc) else np.nan

            status = []
            if not (np.isfinite(r.scanrate_Vs) and r.scanrate_Vs > 0):
                status.append("Scan rate missing/invalid")
            status_s = " | ".join(status) if status else "OK"

            fname = os.path.basename(r.file)
            annotations[fname] = {"E12": E12, "Ep12_an": Ep12_an, "Ep12_ca": Ep12_ca}

            main_rows.append({
                "file": fname,
                "scanrate_V_per_s": r.scanrate_Vs,
                "scanrate_label": r.scanrate_label,
                "Epa_V": Epa, "Ipa_out": Ipa,
                "Epc_V": Epc, "Ipc_out": Ipc,
                "DeltaEp_V": dEp,
                "E1_2_V": E12,
                "Ep1_2_anodic_V": Ep12_an,
                "Ep1_2_cathodic_V": Ep12_ca,
                "status": status_s,
            })

        df_main = pd.DataFrame(main_rows)
        df_main.to_csv(unique_path(os.path.join(outdir, "CV_main_results.csv")), index=False)

        # Kinetics
        v = np.array([r.scanrate_Vs for r in self.records], float)
        ipa = np.array([row["Ipa_out"] for row in main_rows], float)
        ipc = np.array([row["Ipc_out"] for row in main_rows], float)

        mask_v = np.isfinite(v) & (v > 0)
        mask_ipa = mask_v & np.isfinite(ipa)
        mask_ipc = mask_v & np.isfinite(ipc)

        v_ipa = v[mask_ipa]
        ipa_ok = ipa[mask_ipa]
        v_ipc = v[mask_ipc]
        ipc_ok = ipc[mask_ipc]

        fit_an = power_law_fit(v_ipa, ipa_ok) if len(v_ipa) >= 2 else {"a": np.nan, "b": np.nan, "r2": np.nan}
        fit_ca = power_law_fit(v_ipc, ipc_ok) if len(v_ipc) >= 2 else {"a": np.nan, "b": np.nan, "r2": np.nan}
        dunn_an = dunn_fit(v_ipa, ipa_ok) if len(v_ipa) >= 3 else {"k1": np.nan, "k2": np.nan, "r2": np.nan}
        dunn_ca = dunn_fit(v_ipc, ipc_ok) if len(v_ipc) >= 3 else {"k1": np.nan, "k2": np.nan, "r2": np.nan}

        df_scan = pd.DataFrame([
            {"peak": "anodic_primary",
             "powerlaw_a": fit_an["a"], "powerlaw_b": fit_an["b"], "powerlaw_r2": fit_an["r2"],
             "dunn_k1": dunn_an["k1"], "dunn_k2": dunn_an["k2"], "dunn_r2": dunn_an["r2"]},
            {"peak": "cathodic_primary",
             "powerlaw_a": fit_ca["a"], "powerlaw_b": fit_ca["b"], "powerlaw_r2": fit_ca["r2"],
             "dunn_k1": dunn_ca["k1"], "dunn_k2": dunn_ca["k2"], "dunn_r2": dunn_ca["r2"]},
        ])
        df_scan.to_csv(unique_path(os.path.join(outdir, "ScanRate_Analysis.csv")), index=False)

        pd.DataFrame([
            {"model": "powerlaw_anodic", **fit_an},
            {"model": "powerlaw_cathodic", **fit_ca},
            {"model": "dunn_anodic", **dunn_an},
            {"model": "dunn_cathodic", **dunn_ca},
        ]).to_csv(unique_path(os.path.join(outdir, "Kinetic_Model_Fit.csv")), index=False)

        # Axis labels
        if self.cmb_scale.currentText() == "NHE":
            xlabel = "E (V vs NHE)"
        elif self.cmb_scale.currentText() == "RHE":
            xlabel = "E (V vs RHE)"
        else:
            xlabel = "E (V vs Ref)"

        if self.chk_use_area.isChecked():
            ylabel = f"j ({self.cmb_current_unit.currentText()}/cm$^2$)"
        else:
            ylabel = f"I ({self.cmb_current_unit.currentText()})"

        fig_paths: List[str] = []

        # CV_with_Peaks.png
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        for r in self.records:
            fname = os.path.basename(r.file)
            ax1.plot(r.E, r.I, label=r.scanrate_label)

            if r.peaks_an:
                Epa, Ipa, _ = r.peaks_an[0]
                ax1.scatter([Epa], [Ipa], marker="o")
            if r.peaks_ca:
                Epc, Ipc, _ = r.peaks_ca[0]
                ax1.scatter([Epc], [Ipc], marker="o")

            ann = annotations.get(fname, {})
            E12 = ann.get("E12", np.nan)
            if np.isfinite(E12):
                ax1.axvline(E12, linestyle="--", alpha=0.20)

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title("CV with peaks + E1/2")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)
        fig1.tight_layout()
        cv_png = unique_path(os.path.join(outdir, "CV_with_Peaks.png"))
        fig1.savefig(cv_png, dpi=300)
        plt.close(fig1)
        fig_paths.append(cv_png)

        # Peak current vs scan rate
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        if len(v_ipa) == 0 and len(v_ipc) == 0:
            ax2.text(0.5, 0.5, "No valid scan rates.\nFill 'Scan rate (mV/s)' in the table.",
                     ha="center", va="center", transform=ax2.transAxes)
        else:
            if len(v_ipa) > 0:
                ax2.scatter(v_ipa, ipa_ok, label="Anodic peak")
            if len(v_ipc) > 0:
                ax2.scatter(v_ipc, ipc_ok, label="Cathodic peak")

            if len(v_ipa) >= 2 and np.isfinite(fit_an["a"]) and np.isfinite(fit_an["b"]):
                vline = np.linspace(np.min(v_ipa), np.max(v_ipa), 200)
                ax2.plot(vline, fit_an["a"] * (vline ** fit_an["b"]), label=f"Anodic b={fit_an['b']:.3f}")

            if len(v_ipc) >= 2 and np.isfinite(fit_ca["a"]) and np.isfinite(fit_ca["b"]):
                vline = np.linspace(np.min(v_ipc), np.max(v_ipc), 200)
                ax2.plot(vline, fit_ca["a"] * (vline ** fit_ca["b"]), label=f"Cathodic b={fit_ca['b']:.3f}")

        ax2.set_xlabel("Scan rate (V/s)")
        ax2.set_ylabel(ylabel)
        ax2.set_title("Peak current vs scan rate")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)
        fig2.tight_layout()
        pk_png = unique_path(os.path.join(outdir, "Peak_Current_vs_ScanRate.png"))
        fig2.savefig(pk_png, dpi=300)
        plt.close(fig2)
        fig_paths.append(pk_png)

        # Cdl/ECSA
        df_cdl = pd.DataFrame([{"note": "Cdl/ECSA disabled."}])
        valid_sr = int(np.sum(mask_v))
        if self.chk_cdl.isChecked():
            if valid_sr < 3:
                df_cdl = pd.DataFrame([{"note": "Not enough scan rates for Cdl fit (need ≥3)."}])
            else:
                E1w = float(self.spn_E1.value())
                E2w = float(self.spn_E2.value())
                lo, hi = min(E1w, E2w), max(E1w, E2w)

                v_cdl: List[float] = []
                dj2: List[float] = []
                for r in self.records:
                    if not (np.isfinite(r.scanrate_Vs) and r.scanrate_Vs > 0):
                        continue
                    fmask = (r.E[r.fwd_idx] >= lo) & (r.E[r.fwd_idx] <= hi)
                    rmask = (r.E[r.rev_idx] >= lo) & (r.E[r.rev_idx] <= hi)
                    if np.sum(fmask) < 5 or np.sum(rmask) < 5:
                        continue
                    If = float(np.nanmean(r.I[r.fwd_idx][fmask]))
                    Ir = float(np.nanmean(r.I[r.rev_idx][rmask]))
                    v_cdl.append(r.scanrate_Vs)
                    dj2.append(0.5 * (If - Ir))

                v_cdl_arr = np.array(v_cdl, float)
                dj2_arr = np.array(dj2, float)

                if len(v_cdl_arr) >= 3 and np.all(np.isfinite(v_cdl_arr)) and np.all(np.isfinite(dj2_arr)):
                    res = linregress(v_cdl_arr, dj2_arr)
                    Cdl = float(res.slope)
                    r2 = float(res.rvalue ** 2)

                    ECSA = np.nan
                    Cs = float(self.spn_Cs.value())
                    if self.chk_ecsa.isChecked():
                        ECSA = Cdl / Cs

                    df_cdl = pd.DataFrame([{
                        "window_E1_V": E1w,
                        "window_E2_V": E2w,
                        "Cdl_value": Cdl,
                        "fit_r2": r2,
                        "Cs_used_F_per_cm2": Cs if self.chk_ecsa.isChecked() else np.nan,
                        "ECSA_or_factor": ECSA if self.chk_ecsa.isChecked() else np.nan,
                        "note": "If using current density: ECSA is factor; if absolute current: cm^2."
                    }])

                    figC = plt.figure()
                    axC = figC.add_subplot(111)
                    axC.scatter(v_cdl_arr, dj2_arr, label="Δj/2")
                    vv = np.linspace(np.min(v_cdl_arr), np.max(v_cdl_arr), 100)
                    axC.plot(vv, res.intercept + res.slope * vv, label=f"Slope=Cdl, R²={r2:.3f}")
                    axC.set_xlabel("Scan rate (V/s)")
                    axC.set_ylabel("Δj/2")
                    axC.set_title("Cdl fit")
                    axC.grid(True, alpha=0.2)
                    axC.legend(fontsize=8)
                    figC.tight_layout()
                    cdl_png = unique_path(os.path.join(outdir, "Cdl_Fit.png"))
                    figC.savefig(cdl_png, dpi=300)
                    plt.close(figC)
                    fig_paths.append(cdl_png)
                else:
                    df_cdl = pd.DataFrame([{"note": "Cdl fit failed: not enough valid points in selected window."}])

        df_cdl.to_csv(unique_path(os.path.join(outdir, "Cdl_ECSA_results.csv")), index=False)

        # Cycle stability (optional)
        if cyc_col:
            try:
                df0 = read_any_table(self.records[0].file)
                if cyc_col in df0.columns:
                    cyc = df0[cyc_col].to_numpy()
                    I0 = df0[colI].to_numpy(float)
                    I0, _ = convert_current(I0, self.cmb_current_unit.currentText(), self.chk_use_area.isChecked(), float(self.spn_area.value()))
                    mask0 = np.isfinite(cyc) & np.isfinite(I0)
                    tmp = pd.DataFrame({"cycle": cyc[mask0], "I": I0[mask0]})
                    g = tmp.groupby("cycle")["I"].max()

                    figS = plt.figure()
                    axS = figS.add_subplot(111)
                    axS.plot(g.index.values, g.values)
                    axS.set_xlabel("Cycle")
                    axS.set_ylabel(ylabel)
                    axS.set_title("Cycle stability (max I per cycle)")
                    axS.grid(True, alpha=0.2)
                    figS.tight_layout()
                    stab_png = unique_path(os.path.join(outdir, "Cycle_Stability.png"))
                    figS.savefig(stab_png, dpi=300)
                    plt.close(figS)
                    fig_paths.append(stab_png)
            except Exception:
                pass

        # Meta + PDF
        meta = {
            "Date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "Files": ", ".join([os.path.basename(f) for f in self.files]),
            "Selected E column": colE,
            "Selected I column": colI,
            "Reference electrode": self.cmb_ref.currentText(),
            "Offset to NHE (V)": f"{self.get_offset():+.6f}",
            "Output scale": self.cmb_scale.currentText(),
            "pH (for RHE)": f"{self.spn_pH.value():.2f}" if self.cmb_scale.currentText() == "RHE" else "N/A",
            "Current density": "Yes" if self.chk_use_area.isChecked() else "No",
            "Area (cm²)": f"{self.spn_area.value():.4f}" if self.chk_use_area.isChecked() else "N/A",
            "Cdl/ECSA": "Enabled" if self.chk_cdl.isChecked() else "Disabled",
            "Output folder": outdir,
        }
        notes = [
            "E_NHE = E_measured + offset_to_NHE",
            "E_RHE = E_NHE + 0.05916×pH (25°C)",
            "E1/2 = (Epa + Epc)/2 ; Ep1/2 from Ipeak/2 on each branch (when meaningful)",
            "Kinetics: ip = a·v^b and Dunn: i = k1·v + k2·v^0.5",
            "Minimum requirements: kinetics needs ≥2 valid scan rates; Cdl needs ≥3 valid scan rates.",
            "Tip: Fill scan rates in the table (mV/s) before starting."
        ]
        pdf_path = unique_path(os.path.join(outdir, "CV_Report.pdf"))
        build_pdf(pdf_path, meta, df_sr, df_main, df_scan, df_cdl, fig_paths, notes)

        self.lbl_status.setText(f"Status: Done. Saved in: {outdir}")
        QMessageBox.information(self, "Done", f"Export completed.\nSaved in:\n{outdir}\n\nPDF:\n{pdf_path}")


import os
import sys
import ctypes
from PyQt6.QtGui import QIcon

def main() -> None:
    try:
        myappid = 'researcher.toolkit.cvanalysis.1_0_0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        print(f"Taskbar icon fix skipped: {e}")

    app = QApplication(sys.argv)

    basedir = os.path.dirname(os.path.abspath(__file__))
    
    icon_path = os.path.join(basedir, 'Researcherlogo.ico') 

    if os.path.exists(icon_path):
        app_icon = QIcon(icon_path)
        app.setWindowIcon(app_icon)
    else:
        print(f"Warning: Icon file not found at {icon_path}")

    w = CorvaCV()
    
    if os.path.exists(icon_path):
        w.setWindowIcon(app_icon)

    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

