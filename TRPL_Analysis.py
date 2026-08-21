import sys
import os
import traceback
import ctypes
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QDragEnterEvent, QDropEvent, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QFormLayout,
    QTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QScrollArea,
    QAbstractScrollArea,
    QHeaderView,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from matplotlib.backends.backend_pdf import PdfPages


APP_NAME = "Researcher Analysis Toolkit — Time-Resolved Photoluminescence (TRPL)"
APP_VERSION = "1.0.0"


# =========================
# Fitting models
# =========================
def mono_exp(t, A, tau, y0):
    return A * np.exp(-t / tau) + y0


def bi_exp(t, A1, tau1, A2, tau2, y0):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + y0


def tri_exp(t, A1, tau1, A2, tau2, A3, tau3, y0):
    return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + A3 * np.exp(-t / tau3) + y0


MODEL_CONFIG = {
    "Mono": {"func": mono_exp, "params": ["A", "tau", "y0"]},
    "Bi": {"func": bi_exp, "params": ["A1", "tau1", "A2", "tau2", "y0"]},
    "Tri": {"func": tri_exp, "params": ["A1", "tau1", "A2", "tau2", "A3", "tau3", "y0"]},
}


# =========================
# Metrics
# =========================
def rss(y, yhat):
    return float(np.sum((y - yhat) ** 2))


def r_squared(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return np.nan
    return float(1 - ss_res / ss_tot)


def rmse(y, yhat):
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def chi_square_basic(y, yhat):
    denom = np.clip(np.abs(yhat), 1e-12, None)
    return float(np.sum(((y - yhat) ** 2) / denom))


def aic(n, rss_value, k):
    rss_value = max(rss_value, 1e-18)
    return float(n * np.log(rss_value / n) + 2 * k)


def bic(n, rss_value, k):
    rss_value = max(rss_value, 1e-18)
    return float(n * np.log(rss_value / n) + k * np.log(n))


def calc_tau_avg(model_name, params):
    if model_name == "Mono":
        return float(params[1])

    if model_name == "Bi":
        A1, tau1, A2, tau2, _ = params
        denom = A1 * tau1 + A2 * tau2
        if abs(denom) < 1e-18:
            return np.nan
        return float((A1 * tau1**2 + A2 * tau2**2) / denom)

    if model_name == "Tri":
        A1, tau1, A2, tau2, A3, tau3, _ = params
        denom = A1 * tau1 + A2 * tau2 + A3 * tau3
        if abs(denom) < 1e-18:
            return np.nan
        return float((A1 * tau1**2 + A2 * tau2**2 + A3 * tau3**2) / denom)

    return np.nan


# =========================
# Equation text / math for report
# =========================
def get_model_equation_math(model_name):
    if model_name == "Mono":
        return {
            "title": "Mono-exponential model",
            "equation": r"$I(t)=A e^{-t/\tau}+y_0$",
            "details": [
                r"$A$: amplitude",
                r"$\tau$: lifetime",
                r"$y_0$: baseline offset",
            ],
            "tau_avg": None,
        }

    if model_name == "Bi":
        return {
            "title": "Bi-exponential model",
            "equation": r"$I(t)=A_1 e^{-t/\tau_1}+A_2 e^{-t/\tau_2}+y_0$",
            "details": [
                r"$A_1, A_2$: amplitudes",
                r"$\tau_1, \tau_2$: lifetimes",
                r"$y_0$: baseline offset",
            ],
            "tau_avg": r"$\tau_{\mathrm{avg}}=\dfrac{A_1\tau_1^2+A_2\tau_2^2}{A_1\tau_1+A_2\tau_2}$",
        }

    if model_name == "Tri":
        return {
            "title": "Tri-exponential model",
            "equation": r"$I(t)=A_1 e^{-t/\tau_1}+A_2 e^{-t/\tau_2}+A_3 e^{-t/\tau_3}+y_0$",
            "details": [
                r"$A_1, A_2, A_3$: amplitudes",
                r"$\tau_1, \tau_2, \tau_3$: lifetimes",
                r"$y_0$: baseline offset",
            ],
            "tau_avg": r"$\tau_{\mathrm{avg}}=\dfrac{A_1\tau_1^2+A_2\tau_2^2+A_3\tau_3^2}{A_1\tau_1+A_2\tau_2+A_3\tau_3}$",
        }

    return None


def draw_reference_equations_page(fig):
    ax = fig.add_subplot(111)
    ax.axis("off")

    y = 0.95
    ax.text(0.05, y, "Reference fitting models", fontsize=16, fontweight="bold", va="top")
    y -= 0.08

    ax.text(0.05, y, "Mono-exponential", fontsize=13, fontweight="bold", va="top")
    y -= 0.05
    ax.text(0.08, y, r"$I(t)=A e^{-t/\tau}+y_0$", fontsize=18, va="top")
    y -= 0.10

    ax.text(0.05, y, "Bi-exponential", fontsize=13, fontweight="bold", va="top")
    y -= 0.05
    ax.text(0.08, y, r"$I(t)=A_1 e^{-t/\tau_1}+A_2 e^{-t/\tau_2}+y_0$", fontsize=18, va="top")
    y -= 0.07
    ax.text(
        0.08,
        y,
        r"$\tau_{\mathrm{avg}}=\dfrac{A_1\tau_1^2+A_2\tau_2^2}{A_1\tau_1+A_2\tau_2}$",
        fontsize=18,
        va="top",
    )
    y -= 0.12

    ax.text(0.05, y, "Tri-exponential", fontsize=13, fontweight="bold", va="top")
    y -= 0.05
    ax.text(
        0.08,
        y,
        r"$I(t)=A_1 e^{-t/\tau_1}+A_2 e^{-t/\tau_2}+A_3 e^{-t/\tau_3}+y_0$",
        fontsize=18,
        va="top",
    )
    y -= 0.07
    ax.text(
        0.08,
        y,
        r"$\tau_{\mathrm{avg}}=\dfrac{A_1\tau_1^2+A_2\tau_2^2+A_3\tau_3^2}{A_1\tau_1+A_2\tau_2+A_3\tau_3}$",
        fontsize=18,
        va="top",
    )


def get_parameter_table_rows(fit_result):
    rows = []
    for name, value in zip(fit_result["param_names"], fit_result["params"]):
        rows.append((name, f"{value:.10g}"))

    rows.extend(
        [
            ("tau_avg", f"{fit_result['tau_avg']:.10g}"),
            ("R²", f"{fit_result['r2']:.10g}"),
            ("χ²", f"{fit_result['chi2']:.10g}"),
            ("RMSE", f"{fit_result['rmse']:.10g}"),
            ("RSS", f"{fit_result['rss']:.10g}"),
            ("AIC", f"{fit_result['aic']:.10g}"),
            ("BIC", f"{fit_result['bic']:.10g}"),
        ]
    )
    return rows


# =========================
# Data utilities
# =========================
def find_numeric_columns(df: pd.DataFrame):
    numeric_cols = []
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().sum() > max(5, int(0.5 * len(df))):
            numeric_cols.append(col)
    return numeric_cols


def guess_time_intensity_columns(df: pd.DataFrame):
    numeric_cols = find_numeric_columns(df)
    if len(numeric_cols) < 2:
        raise ValueError("Could not detect at least two numeric columns.")

    lowered = {col: str(col).strip().lower() for col in numeric_cols}
    time_col = None
    inten_col = None

    for col, low in lowered.items():
        if time_col is None and any(k in low for k in ["time", "ns", "delay", "ps"]):
            time_col = col
        if inten_col is None and any(k in low for k in ["int", "counts", "pl", "signal", "emission"]):
            inten_col = col

    if time_col is None:
        time_col = numeric_cols[0]
    if inten_col is None:
        inten_col = numeric_cols[1] if numeric_cols[1] != time_col else numeric_cols[0]

    if time_col == inten_col:
        others = [c for c in numeric_cols if c != time_col]
        if not others:
            raise ValueError("Could not separate time and intensity columns.")
        inten_col = others[0]

    return time_col, inten_col


def maybe_convert_time_to_ns(t):
    t = np.asarray(t, dtype=float)
    if len(t) == 0:
        return t, False
    if np.nanmax(t) > 1000:
        return t / 1000.0, True
    return t, False


def load_data(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".csv", ".txt"]:
        try:
            df = pd.read_csv(file_path)
        except Exception:
            df = pd.read_csv(file_path, sep=None, engine="python")
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    time_col, intensity_col = guess_time_intensity_columns(df)

    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[intensity_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    order = np.argsort(t)
    t = t[order]
    y = y[order]

    t, converted = maybe_convert_time_to_ns(t)
    return df, time_col, intensity_col, t, y, converted


def suggest_fit_region(t, y):
    if len(t) < 20:
        return float(t.min()), float(t.max())

    y_clean = np.array(y, dtype=float)
    idx_peak = int(np.argmax(y_clean))
    y_peak = max(y_clean[idx_peak], 1e-18)

    start_idx = idx_peak
    for i in range(idx_peak, min(idx_peak + 10, len(y_clean))):
        if y_clean[i] <= 0.98 * y_peak:
            start_idx = i
            break

    tail_threshold = 0.02 * y_peak
    end_idx = len(y_clean) - 1
    candidates = np.where(y_clean[start_idx:] <= tail_threshold)[0]
    if len(candidates) > 0:
        end_idx = start_idx + int(candidates[0])
        end_idx = max(end_idx, start_idx + 10)

    return float(t[start_idx]), float(t[end_idx])


def build_initial_guess(model_name, t, y, fix_y0=False):
    y0_guess = 0.0 if fix_y0 else float(np.min(y[-max(5, len(y) // 10):]))
    amp_guess = float(max(np.max(y) - y0_guess, 1e-6))
    span = float(max(t.max() - t.min(), 1e-6))

    if model_name == "Mono":
        p0 = [amp_guess, max(span / 5, 0.2), y0_guess]
        lower = [0, 1e-12, -np.inf]
        upper = [np.inf, np.inf, np.inf]

    elif model_name == "Bi":
        p0 = [0.7 * amp_guess, max(span / 20, 0.05), 0.3 * amp_guess, max(span / 4, 0.5), y0_guess]
        lower = [0, 1e-12, 0, 1e-12, -np.inf]
        upper = [np.inf, np.inf, np.inf, np.inf, np.inf]

    else:
        p0 = [
            0.5 * amp_guess,
            max(span / 30, 0.02),
            0.3 * amp_guess,
            max(span / 8, 0.2),
            0.2 * amp_guess,
            max(span / 2, 1.0),
            y0_guess,
        ]
        lower = [0, 1e-12, 0, 1e-12, 0, 1e-12, -np.inf]
        upper = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

    if fix_y0:
        p0[-1] = 0.0
        lower[-1] = -1e-15
        upper[-1] = 1e-15

    return p0, (lower, upper)


def fit_model(model_name, t, y, fix_y0=False):
    func = MODEL_CONFIG[model_name]["func"]
    p0, bounds = build_initial_guess(model_name, t, y, fix_y0=fix_y0)
    params, cov = curve_fit(func, t, y, p0=p0, bounds=bounds, maxfev=100000)
    yhat = func(t, *params)

    return {
        "model": model_name,
        "params": params,
        "param_names": MODEL_CONFIG[model_name]["params"],
        "cov": cov,
        "yhat": yhat,
        "rss": rss(y, yhat),
        "r2": r_squared(y, yhat),
        "rmse": rmse(y, yhat),
        "chi2": chi_square_basic(y, yhat),
        "aic": aic(len(y), rss(y, yhat), len(params)),
        "bic": bic(len(y), rss(y, yhat), len(params)),
        "tau_avg": calc_tau_avg(model_name, params),
        "residuals": y - yhat,
    }


# =========================
# Matplotlib canvas
# =========================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 6), tight_layout=True)
        self.ax_main = self.fig.add_subplot(211)
        self.ax_res = self.fig.add_subplot(212, sharex=self.ax_main)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()


# =========================
# Main window
# =========================
class TRPLMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setWindowIcon(QIcon("Researcherlogo.png"))
        self.resize(1500, 920)
        self.setMinimumSize(1100, 700)
        self.setAcceptDrops(True)

        self.file_path = None
        self.raw_df = None
        self.time_col = None
        self.intensity_col = None
        self.time_converted_to_ns = False
        self.t = None
        self.y = None
        self.t_plot = None
        self.y_plot = None
        self.fit_result = None
        self.selected_range = None

        self._build_ui()
        self._connect_plot_selector()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setMinimumWidth(380)
        left_scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(8, 8, 8, 8)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        drop_group = QGroupBox("Input")
        drop_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        drop_layout = QVBoxLayout(drop_group)

        self.drop_label = QLabel("Drag and drop a CSV/TXT/XLSX file here\nor click 'Open File'")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setWordWrap(True)
        self.drop_label.setStyleSheet("border: 2px dashed #777; padding: 20px; font-size: 14px;")

        self.file_info = QLabel("No file loaded")
        self.file_info.setWordWrap(True)

        open_btn = QPushButton("Open File")
        open_btn.clicked.connect(self.open_file)

        drop_layout.addWidget(self.drop_label)
        drop_layout.addWidget(open_btn)
        drop_layout.addWidget(self.file_info)
        left_layout.addWidget(drop_group)

        control_group = QGroupBox("Fitting Controls")
        control_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        form = QFormLayout(control_group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self.x_column_combo = QComboBox()
        self.y_column_combo = QComboBox()
        self.x_column_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.y_column_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        apply_columns_btn = QPushButton("Apply X/Y Columns")
        apply_columns_btn.clicked.connect(self.apply_selected_columns)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Mono", "Bi", "Tri"])
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.normalize_box = QCheckBox("Normalize intensity")
        self.normalize_box.setChecked(True)
        self.normalize_box.stateChanged.connect(self.refresh_plot_only)

        self.fix_y0_box = QCheckBox("Fix baseline (y0 = 0)")
        self.fix_y0_box.setChecked(True)

        self.range_start = QDoubleSpinBox()
        self.range_start.setDecimals(6)
        self.range_start.setMaximum(1e12)
        self.range_start.setMinimum(-1e12)
        self.range_start.setKeyboardTracking(False)

        self.range_end = QDoubleSpinBox()
        self.range_end.setDecimals(6)
        self.range_end.setMaximum(1e12)
        self.range_end.setMinimum(-1e12)
        self.range_end.setKeyboardTracking(False)

        suggest_btn = QPushButton("Suggest Range")
        suggest_btn.clicked.connect(self.auto_suggest_range)

        fit_btn = QPushButton("Run Fit")
        fit_btn.clicked.connect(self.run_fit)

        compare_btn = QPushButton("Compare All Models")
        compare_btn.clicked.connect(self.compare_all_models)

        export_btn = QPushButton("Export PDF Report")
        export_btn.clicked.connect(self.export_pdf_report)

        form.addRow("X column (Time)", self.x_column_combo)
        form.addRow("Y column (Intensity)", self.y_column_combo)
        form.addRow("", apply_columns_btn)
        form.addRow("Model", self.model_combo)
        form.addRow("", self.normalize_box)
        form.addRow("", self.fix_y0_box)
        form.addRow("Fit start", self.range_start)
        form.addRow("Fit end", self.range_end)
        form.addRow("", suggest_btn)
        form.addRow("", fit_btn)
        form.addRow("", compare_btn)
        form.addRow("", export_btn)
        left_layout.addWidget(control_group)

        results_group = QGroupBox("Fit Results")
        results_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.results_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.results_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.results_table.setWordWrap(True)
        self.results_table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        self.results_table.setMinimumHeight(220)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(140)
        self.results_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        results_layout.addWidget(self.results_table)
        results_layout.addWidget(self.results_text)
        left_layout.addWidget(results_group)

        compare_group = QGroupBox("Model Comparison")
        compare_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        compare_layout = QVBoxLayout(compare_group)
        self.compare_list = QListWidget()
        self.compare_list.setWordWrap(True)
        self.compare_list.setMinimumHeight(150)
        compare_layout.addWidget(self.compare_list)
        left_layout.addWidget(compare_group)
        left_layout.addStretch(1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(right)

        self.canvas = MplCanvas()
        right_layout.addWidget(self.canvas)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1080])

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")

        open_action = QAction("Open", self)
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        export_action = QAction("Export PDF Report", self)
        export_action.triggered.connect(self.export_pdf_report)
        file_menu.addAction(export_action)

    def _connect_plot_selector(self):
        def onselect(xmin, xmax):
            if self.t is None:
                return
            xmin, xmax = sorted([xmin, xmax])
            self.range_start.setValue(float(xmin))
            self.range_end.setValue(float(xmax))
            self.selected_range = (float(xmin), float(xmax))
            self.refresh_plot_only()

        self.span = SpanSelector(
            self.canvas.ax_main,
            onselect,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.25),
            interactive=True,
            drag_from_anywhere=True,
        )

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return
        file_path = urls[0].toLocalFile()
        if file_path:
            self.load_file(file_path)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open TRPL Data File",
            "",
            "Data Files (*.csv *.txt *.xlsx *.xls);;All Files (*)",
        )
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path):
        try:
            self.raw_df, self.time_col, self.intensity_col, self.t, self.y, self.time_converted_to_ns = load_data(file_path)
            self.file_path = file_path

            self.x_column_combo.clear()
            self.y_column_combo.clear()
            for col in self.raw_df.columns:
                self.x_column_combo.addItem(str(col))
                self.y_column_combo.addItem(str(col))

            self.x_column_combo.setCurrentText(self.time_col)
            self.y_column_combo.setCurrentText(self.intensity_col)

            converted_note = " | Time converted: ps → ns" if self.time_converted_to_ns else ""
            self.file_info.setText(
                f"Loaded: {os.path.basename(file_path)}\n"
                f"X: {self.time_col} | Y: {self.intensity_col} | Points: {len(self.t)}{converted_note}"
            )

            xmin, xmax = suggest_fit_region(self.t, self.y)
            self.range_start.setValue(xmin)
            self.range_end.setValue(xmax)
            self.selected_range = (xmin, xmax)

            self.fit_result = None
            self.compare_list.clear()
            self.results_text.clear()
            self.results_table.setRowCount(0)
            self.refresh_plot_only()

        except Exception as e:
            self.show_error("Failed to load file", e)

    def apply_selected_columns(self):
        try:
            if self.raw_df is None:
                return

            x_col = self.x_column_combo.currentText()
            y_col = self.y_column_combo.currentText()

            t = pd.to_numeric(self.raw_df[x_col], errors="coerce").to_numpy(dtype=float)
            y = pd.to_numeric(self.raw_df[y_col], errors="coerce").to_numpy(dtype=float)

            mask = np.isfinite(t) & np.isfinite(y)
            t = t[mask]
            y = y[mask]

            order = np.argsort(t)
            t = t[order]
            y = y[order]

            t, converted = maybe_convert_time_to_ns(t)

            self.t = t
            self.y = y
            self.time_col = x_col
            self.intensity_col = y_col
            self.time_converted_to_ns = converted

            converted_note = " | Time converted: ps → ns" if converted else ""
            self.file_info.setText(
                f"Loaded: {os.path.basename(self.file_path) if self.file_path else '-'}\n"
                f"X: {self.time_col} | Y: {self.intensity_col} | Points: {len(self.t)}{converted_note}"
            )

            xmin, xmax = suggest_fit_region(self.t, self.y)
            self.range_start.setValue(xmin)
            self.range_end.setValue(xmax)

            self.fit_result = None
            self.compare_list.clear()
            self.results_text.clear()
            self.results_table.setRowCount(0)
            self.refresh_plot_only()

        except Exception as e:
            self.show_error("Column selection failed", e)

    def get_plot_data(self):
        if self.t is None or self.y is None:
            return None, None

        y_plot = self.y.copy().astype(float)
        if self.normalize_box.isChecked():
            ymax = np.max(np.abs(y_plot))
            if ymax > 0:
                y_plot = y_plot / ymax

        return self.t.copy().astype(float), y_plot

    def refresh_plot_only(self):
        self.canvas.ax_main.clear()
        self.canvas.ax_res.clear()

        t_plot, y_plot = self.get_plot_data()
        if t_plot is None:
            self.canvas.draw_idle()
            return

        self.t_plot = t_plot
        self.y_plot = y_plot

        self.canvas.ax_main.plot(t_plot, y_plot, marker="o", linestyle="", markersize=3, label="Raw data")
        self.canvas.ax_main.set_title("TRPL Raw Data and Fit")
        self.canvas.ax_main.set_xlabel(f"{self.time_col} (ns)")
        self.canvas.ax_main.set_ylabel(f"{self.intensity_col}" + (" (normalized)" if self.normalize_box.isChecked() else ""))
        self.canvas.ax_main.grid(True, alpha=0.3)

        start = self.range_start.value()
        end = self.range_end.value()

        if end > start:
            self.canvas.ax_main.axvspan(start, end, alpha=0.15, label="Selected fit range")

        if self.fit_result is not None:
            fit_mask = (t_plot >= start) & (t_plot <= end)
            if np.any(fit_mask):
                self.canvas.ax_main.plot(
                    t_plot[fit_mask],
                    self.fit_result["yhat"],
                    linewidth=2.0,
                    label=f"{self.fit_result['model']} fit",
                )
                self.canvas.ax_res.plot(
                    t_plot[fit_mask],
                    self.fit_result["residuals"],
                    marker="o",
                    linestyle="",
                    markersize=3,
                )
                self.canvas.ax_res.axhline(0, linewidth=1)
                self.canvas.ax_res.set_ylabel("Residual")
                self.canvas.ax_res.set_xlabel(f"{self.time_col} (ns)")
                self.canvas.ax_res.grid(True, alpha=0.3)

        self.canvas.ax_main.legend(loc="best")
        self.canvas.fig.tight_layout()
        self.canvas.draw_idle()

    def auto_suggest_range(self):
        if self.t is None or self.y is None:
            return
        xmin, xmax = suggest_fit_region(self.t, self.y)
        self.range_start.setValue(xmin)
        self.range_end.setValue(xmax)
        self.selected_range = (xmin, xmax)
        self.refresh_plot_only()

    def get_selected_segment(self):
        if self.t_plot is None or self.y_plot is None:
            raise ValueError("No data loaded.")

        start = self.range_start.value()
        end = self.range_end.value()

        if end <= start:
            raise ValueError("Fit end must be greater than fit start.")

        mask = (self.t_plot >= start) & (self.t_plot <= end)
        t_seg = self.t_plot[mask]
        y_seg = self.y_plot[mask]

        if len(t_seg) < 5:
            raise ValueError("Selected range is too small for fitting.")

        return t_seg, y_seg

    def run_fit(self):
        try:
            model_name = self.model_combo.currentText()
            t_seg, y_seg = self.get_selected_segment()
            self.fit_result = fit_model(model_name, t_seg, y_seg, fix_y0=self.fix_y0_box.isChecked())
            self.update_results_view(self.fit_result)
            self.refresh_plot_only()
        except Exception as e:
            self.show_error("Fit failed", e)

    def compare_all_models(self):
        try:
            t_seg, y_seg = self.get_selected_segment()
            self.compare_list.clear()
            results = []

            for model in ["Mono", "Bi", "Tri"]:
                try:
                    res = fit_model(model, t_seg, y_seg, fix_y0=self.fix_y0_box.isChecked())
                    results.append(res)
                except Exception as ex:
                    self.compare_list.addItem(QListWidgetItem(f"{model}: failed ({str(ex)})"))

            if not results:
                raise RuntimeError("All models failed.")

            results_sorted = sorted(results, key=lambda r: (r["bic"], r["rmse"]))
            best = results_sorted[0]

            for r in results_sorted:
                item = QListWidgetItem(
                    f"{r['model']} | R²={r['r2']:.6f} | χ²={r['chi2']:.6g} | "
                    f"RMSE={r['rmse']:.6g} | AIC={r['aic']:.3f} | BIC={r['bic']:.3f} | "
                    f"τavg={r['tau_avg']:.6g}"
                )
                self.compare_list.addItem(item)

            self.fit_result = best
            self.model_combo.setCurrentText(best["model"])
            self.update_results_view(best, comparison_note="Best model selected automatically using BIC and RMSE.")
            self.refresh_plot_only()

        except Exception as e:
            self.show_error("Model comparison failed", e)

    def update_results_view(self, result, comparison_note=""):
        self.results_table.setRowCount(0)

        rows = get_parameter_table_rows(result)
        self.results_table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.results_table.setItem(i, 0, QTableWidgetItem(k))
            self.results_table.setItem(i, 1, QTableWidgetItem(v))
        self.results_table.resizeRowsToContents()

        math_info = get_model_equation_math(result["model"])
        lines = [
            f"Model: {result['model']}",
            f"File: {os.path.basename(self.file_path) if self.file_path else '-'}",
            f"X column: {self.time_col}",
            f"Y column: {self.intensity_col}",
            f"Range: {self.range_start.value():.6g} to {self.range_end.value():.6g} ns",
            f"Normalize: {'Yes' if self.normalize_box.isChecked() else 'No'}",
            f"Fix y0: {'Yes' if self.fix_y0_box.isChecked() else 'No'}",
            "",
            f"Equation title: {math_info['title']}",
            f"Equation: {math_info['equation']}",
        ]

        for line in math_info["details"]:
            lines.append(line)

        if math_info["tau_avg"] is not None:
            lines.append("")
            lines.append(f"Average lifetime formula: {math_info['tau_avg']}")

        if self.time_converted_to_ns:
            lines.append("")
            lines.append("Time axis was auto-converted from ps to ns.")

        if comparison_note:
            lines.append("")
            lines.append(comparison_note)

        note = self._parameter_sanity_note(result)
        if note:
            lines.append("")
            lines.append(note)

        self.results_text.setPlainText("\n".join(lines))

    def _parameter_sanity_note(self, result):
        params = result["params"]
        names = result["param_names"]
        issues = []

        for n, v in zip(names, params):
            if "tau" in n and v <= 0:
                issues.append(f"Non-physical {n} <= 0")
            if n.startswith("A") and v < 0:
                issues.append(f"Negative amplitude {n}")

        return "Parameter check: OK" if not issues else "Parameter check: " + "; ".join(issues)

    def _draw_text_block(self, ax, y, text, x=0.03, fontsize=10, family="monospace", weight=None, line_h=0.028, extra_gap=0.012):
        ax.text(
            x,
            y,
            text,
            va="top",
            ha="left",
            fontsize=fontsize,
            family=family,
            fontweight=weight,
        )
        n_lines = text.count("\n") + 1
        used = n_lines * line_h + extra_gap
        return y - used

    def export_pdf_report(self):
        if self.t_plot is None or self.y_plot is None:
            QMessageBox.information(self, "No Data", "Load and fit data before exporting a report.")
            return

        if self.fit_result is None:
            QMessageBox.information(self, "No Fit", "Run a fit before exporting a report.")
            return

        default_name = f"TRPL_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        save_path, _ = QFileDialog.getSaveFileName(self, "Save PDF Report", default_name, "PDF Files (*.pdf)")
        if not save_path:
            return

        try:
            start = self.range_start.value()
            end = self.range_end.value()

            mask = (self.t_plot >= start) & (self.t_plot <= end)
            t_seg = self.t_plot[mask]
            yhat = self.fit_result["yhat"]
            residuals = self.fit_result["residuals"]
            param_rows = get_parameter_table_rows(self.fit_result)
            model_math = get_model_equation_math(self.fit_result["model"])

            with PdfPages(save_path) as pdf:
                # Page 1: dynamic layout
                fig1 = Figure(figsize=(8.27, 11.69))
                ax1 = fig1.add_subplot(111)
                ax1.axis("off")

                y = 0.98

                summary_text = "\n".join([
                    APP_NAME,
                    f"Version: {APP_VERSION}",
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "",
                    f"Input file: {os.path.basename(self.file_path)}",
                    f"X column: {self.time_col}",
                    f"Y column: {self.intensity_col}",
                    f"Selected range: {start:.6g} to {end:.6g} ns",
                    f"Model: {self.fit_result['model']}",
                    f"Normalization: {'Yes' if self.normalize_box.isChecked() else 'No'}",
                    f"Fix baseline: {'Yes' if self.fix_y0_box.isChecked() else 'No'}",
                    f"Auto ps→ns conversion: {'Yes' if self.time_converted_to_ns else 'No'}",
                ])
                y = self._draw_text_block(ax1, y, summary_text, x=0.03, fontsize=10, family="monospace")

                y = self._draw_text_block(ax1, y, model_math["title"], x=0.03, fontsize=14, family="sans-serif", weight="bold")

                ax1.text(0.08, y, model_math["equation"], va="top", ha="left", fontsize=20)
                y -= 0.075

                y = self._draw_text_block(ax1, y, "Parameter definitions:", x=0.03, fontsize=12, family="sans-serif", weight="bold")

                for line in model_math["details"]:
                    ax1.text(0.08, y, line, va="top", ha="left", fontsize=12)
                    y -= 0.045

                if model_math["tau_avg"] is not None:
                    y -= 0.01
                    y = self._draw_text_block(ax1, y, "Average lifetime:", x=0.03, fontsize=12, family="sans-serif", weight="bold")
                    ax1.text(0.08, y, model_math["tau_avg"], va="top", ha="left", fontsize=18)
                    y -= 0.08

                y = self._draw_text_block(ax1, y, "Parameter table:", x=0.03, fontsize=12, family="sans-serif", weight="bold")

                table_text = ""
                for key, val in param_rows:
                    table_text += f"{key:<12} : {val}\n"
                y = self._draw_text_block(ax1, y, table_text.rstrip(), x=0.03, fontsize=10, family="monospace")

                sanity_note = self._parameter_sanity_note(self.fit_result)
                y = self._draw_text_block(ax1, y, sanity_note, x=0.03, fontsize=10, family="monospace")

                pdf.savefig(fig1)

                # Page 2: fit and residual
                fig2 = Figure(figsize=(10, 7))
                ax2 = fig2.add_subplot(211)
                ax3 = fig2.add_subplot(212, sharex=ax2)

                ax2.plot(
                    self.t_plot,
                    self.y_plot,
                    marker="o",
                    linestyle="",
                    markersize=3,
                    alpha=0.6,
                    label="Raw data",
                )
                ax2.axvspan(start, end, alpha=0.15, label="Selected range")
                ax2.plot(t_seg, yhat, linewidth=2, label=f"{self.fit_result['model']} fit")
                ax2.set_title("TRPL Raw Data and Fit")
                ax2.set_xlabel(f"{self.time_col} (ns)")
                ax2.set_ylabel(
                    f"{self.intensity_col}" + (" (normalized)" if self.normalize_box.isChecked() else "")
                )
                ax2.grid(True, alpha=0.3)
                ax2.legend(loc="best")

                ax3.plot(t_seg, residuals, marker="o", linestyle="", markersize=3)
                ax3.axhline(0, linewidth=1)
                ax3.set_xlabel(f"{self.time_col} (ns)")
                ax3.set_ylabel("Residual")
                ax3.set_title("Residual Plot")
                ax3.grid(True, alpha=0.3)
                fig2.tight_layout()
                pdf.savefig(fig2)

                # Page 3: all equations
                fig3 = Figure(figsize=(8.27, 11.69))
                draw_reference_equations_page(fig3)
                pdf.savefig(fig3)

            QMessageBox.information(self, "Export Complete", f"PDF report saved to:\n{save_path}")

        except Exception as e:
            self.show_error("Failed to export PDF report", e)

    def show_error(self, title, error):
        details = traceback.format_exc()
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setWindowTitle(title)
        msg.setText(str(error))
        msg.setDetailedText(details)
        msg.exec()


def main():
    if os.name == 'nt':
        try:
            myappid = f'mycompany.trpl_analysis_toolkit.{APP_VERSION}' 
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
        except Exception:
            pass

    app = QApplication(sys.argv)
    
    app.setWindowIcon(QIcon("Researcherlogo.png"))
    
    window = TRPLMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()