import sys
import os
import json
import traceback
import ctypes
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QGroupBox,
    QLineEdit,
    QComboBox,
    QDoubleSpinBox,
    QTabWidget,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QScrollArea,
    QCheckBox,
    QFrame,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas as pdf_canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


APP_NAME = "Researcher Analysis Toolkit - Tauc Plot"
OUTPUT_TECHNIQUE = "Tauc Plot Analysis"
APP_VERSION = "1.0.0"
ARTICLE_DOI = "10.1039/D6RA08618G"
ARTICLE_URL = f"https://doi.org/{ARTICLE_DOI}"
ARTICLE_CITATION = f"Suggested citation · Behroozi et al., RSC Adv. (2026). {ARTICLE_DOI}"


def resource_path(filename: str) -> str:
    if getattr(sys, "_MEIPASS", None):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


@dataclass
class TaucInputs:
    sample_name: str
    data_mode: str
    transition_type: str
    custom_exponent: float
    wavelength_column: str
    signal_column: str
    file_path: str
    tangent_min_eV: float
    tangent_max_eV: float
    use_baseline: bool
    baseline_min_eV: float
    baseline_max_eV: float
    notes: str


class OutputManager:
    @staticmethod
    def desktop_dir() -> Path:
        desktop = Path.home() / "Desktop"
        return desktop if desktop.exists() else Path.home()

    @classmethod
    def sanitize_name(cls, text: str) -> str:
        text = (text or "Unnamed_Sample").strip()
        for ch in '<>:"/\\|?*':
            text = text.replace(ch, "_")
        return text.replace(" ", "_")

    @classmethod
    def create_run_folder(cls, sample_name: str) -> Path:
        root = cls.desktop_dir() / "Researcher Output" / OUTPUT_TECHNIQUE / cls.sanitize_name(sample_name)
        root.mkdir(parents=True, exist_ok=True)

        existing = []
        for item in root.iterdir():
            if item.is_dir() and item.name.startswith("run_"):
                try:
                    existing.append(int(item.name.split("_")[1]))
                except Exception:
                    pass

        next_idx = max(existing) + 1 if existing else 1
        run_dir = root / f"run_{next_idx:03d}"

        for sub in ["data", "csv", "json", "png", "report"]:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        return run_dir


class DataUtils:
    @staticmethod
    def read_table(file_path: str) -> pd.DataFrame:
        if not file_path:
            raise ValueError("No file selected.")

        ext = Path(file_path).suffix.lower()

        if ext in [".csv", ".txt", ".dat"]:
            for sep in [",", "\t", ";", r"\s+"]:
                try:
                    df = pd.read_csv(file_path, sep=sep, engine="python")
                    if df.shape[1] >= 2:
                        return df
                except Exception:
                    continue
            raise ValueError("Could not parse text file.")

        if ext in [".xlsx", ".xls"]:
            try:
                return pd.read_excel(file_path)
            except Exception as e:
                raise ValueError(
                    "Failed to read Excel file. If this is .xls, install xlrd:\n"
                    "python -m pip install xlrd"
                ) from e

        raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def clean_numeric_columns(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Selected columns were not found in the file.")

        out = df[[x_col, y_col]].copy()
        out.columns = ["wavelength_nm", "signal"]
        out["wavelength_nm"] = pd.to_numeric(out["wavelength_nm"], errors="coerce")
        out["signal"] = pd.to_numeric(out["signal"], errors="coerce")
        out = out.dropna()
        out = out[out["wavelength_nm"] > 0]
        out = out.sort_values("wavelength_nm").drop_duplicates(subset="wavelength_nm")

        if len(out) < 3:
            raise ValueError("Not enough valid numeric points.")

        return out.reset_index(drop=True)

    @staticmethod
    def wavelength_to_energy_eV(wavelength_nm: np.ndarray) -> np.ndarray:
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        if np.any(wavelength_nm <= 0):
            raise ValueError("Wavelength must be greater than zero.")
        return 1240.0 / wavelength_nm

    @staticmethod
    def alpha_from_transmittance_percent(t_percent: np.ndarray) -> np.ndarray:
        t_percent = np.asarray(t_percent, dtype=float)
        if np.any(t_percent <= 0) or np.any(t_percent > 100):
            raise ValueError("Transmittance (%) must be between 0 and 100.")
        return np.log10(100.0 / t_percent)

    @staticmethod
    def alpha_from_reflectance_percent(r_percent: np.ndarray) -> np.ndarray:
        r_percent = np.asarray(r_percent, dtype=float)
        if np.any(r_percent <= 0) or np.any(r_percent > 100):
            raise ValueError("Reflectance (%) must be between 0 and 100.")
        r = r_percent / 100.0
        return ((1.0 - r) ** 2) / (2.0 * r)

    @staticmethod
    def alpha_from_absorbance(absorbance: np.ndarray) -> np.ndarray:
        absorbance = np.asarray(absorbance, dtype=float)
        return absorbance.copy()

    @staticmethod
    def get_exponent(transition_type: str, custom_exponent: float) -> float:
        if transition_type == "Direct allowed":
            return 2.0
        if transition_type == "Indirect allowed":
            return 0.5
        if transition_type == "Custom":
            if custom_exponent <= 0:
                raise ValueError("Custom exponent must be greater than zero.")
            return custom_exponent
        raise ValueError("Invalid transition type.")

    @staticmethod
    def linear_fit(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float) -> Tuple[float, float, np.ndarray, np.ndarray]:
        mask = (x >= x_min) & (x <= x_max)
        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < 2:
            raise ValueError("Not enough points in selected region.")

        coeffs = np.polyfit(x_fit, y_fit, 1)
        slope, intercept = coeffs[0], coeffs[1]
        y_pred = slope * x_fit + intercept
        return float(slope), float(intercept), x_fit, y_pred

    @staticmethod
    def intersection_x(m1: float, b1: float, m2: float, b2: float) -> float:
        if abs(m1 - m2) < 1e-15:
            raise ValueError("Tangent and baseline are nearly parallel. Intersection cannot be determined.")
        return (b2 - b1) / (m1 - m2)


class PlotCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.canvas.draw_idle()


class TaucPlotWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.last_result = None

        self.tangent_line_x = None
        self.tangent_line_y = None
        self.baseline_line_x = None
        self.baseline_line_y = None

        self.selection_mode = None   # None / "tangent" / "baseline"
        self.selection_points = []
        self.selection_connection_id = None

        self.tangent_region = []
        self.baseline_region = []

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1550, 980)
        self.set_app_icon()
        self.build_ui()
        self.create_menu()
        self.apply_styles()
        self.update_ui_states()

    def set_app_icon(self):
        for name in ["Researcherlogo.ico", "ResearcherLogo.ico"]:
            path = resource_path(name)
            if os.path.exists(path):
                icon = QIcon(path)
                self.setWindowIcon(icon)
                app = QApplication.instance()
                if app is not None:
                    app.setWindowIcon(icon)
                break

    def create_menu(self):
        menu = self.menuBar().addMenu("File")

        open_file = QAction("Open Data File", self)
        open_file.triggered.connect(self.load_data_file)
        menu.addAction(open_file)

        menu.addSeparator()

        save_results = QAction("Save Results", self)
        save_results.triggered.connect(self.save_results)
        menu.addAction(save_results)

        menu.addSeparator()

        exit_app = QAction("Exit", self)
        exit_app.triggered.connect(self.close)
        menu.addAction(exit_app)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setSpacing(10)

        title_frame = QFrame()
        title_layout = QVBoxLayout(title_frame)
        title = QLabel("Tauc Plot Module")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Macro-style logic: Transmittance / Reflectance / Absorbance + Tangent/Baseline")
        subtitle.setObjectName("subTitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        left_layout.addWidget(title_frame)

        left_layout.addWidget(self.build_general_group())
        left_layout.addWidget(self.build_data_group())
        left_layout.addWidget(self.build_tangent_group())
        left_layout.addWidget(self.build_baseline_group())
        left_layout.addWidget(self.build_action_group())
        left_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_container)
        scroll.setMinimumWidth(560)
        scroll.setMaximumWidth(760)

        self.tabs = QTabWidget()

        self.raw_plot = PlotCanvas()
        self.tauc_plot = PlotCanvas()

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_tabs = QTabWidget()
        preview_tabs.addTab(self.raw_plot, "Raw Data")
        preview_tabs.addTab(self.tauc_plot, "Tauc Plot")
        preview_layout.addWidget(preview_tabs)

        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        self.result_summary = QPlainTextEdit()
        self.result_summary.setReadOnly(True)
        self.result_summary.setMinimumHeight(260)

        self.result_table = QTableWidget(0, 2)
        self.result_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        result_layout.addWidget(self.result_summary)
        result_layout.addWidget(self.result_table)

        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        log_layout.addWidget(self.log_box)

        self.tabs.addTab(preview_widget, "Preview")
        self.tabs.addTab(result_widget, "Results")
        self.tabs.addTab(log_widget, "Log")

        main_layout.addWidget(scroll, 0)
        main_layout.addWidget(self.tabs, 1)

    def build_general_group(self):
        box = QGroupBox("General Settings")
        layout = QGridLayout(box)

        self.sample_edit = QLineEdit("Sample_01")

        self.data_mode_combo = QComboBox()
        self.data_mode_combo.addItems([
            "Transmittance (%)",
            "Reflectance (%)",
            "Absorbance",
        ])

        self.transition_combo = QComboBox()
        self.transition_combo.addItems([
            "Direct allowed",
            "Indirect allowed",
            "Custom",
        ])
        self.transition_combo.currentTextChanged.connect(self.update_ui_states)

        self.custom_exponent_spin = QDoubleSpinBox()
        self.custom_exponent_spin.setDecimals(6)
        self.custom_exponent_spin.setMaximum(1000)
        self.custom_exponent_spin.setValue(2.0)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)

        btn_file = QPushButton("Browse Data File")
        btn_file.clicked.connect(self.load_data_file)

        btn_file_clear = QPushButton("Refresh")
        btn_file_clear.clicked.connect(self.clear_file)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setFixedHeight(90)
        self.notes_edit.setPlaceholderText("Optional notes...")

        layout.addWidget(QLabel("Sample Name:"), 0, 0)
        layout.addWidget(self.sample_edit, 0, 1, 1, 3)

        layout.addWidget(QLabel("Data Mode:"), 1, 0)
        layout.addWidget(self.data_mode_combo, 1, 1, 1, 3)

        layout.addWidget(QLabel("Transition Type:"), 2, 0)
        layout.addWidget(self.transition_combo, 2, 1)
        layout.addWidget(QLabel("Custom Exponent:"), 2, 2)
        layout.addWidget(self.custom_exponent_spin, 2, 3)

        layout.addWidget(QLabel("Data File:"), 3, 0)
        layout.addWidget(self.file_path_edit, 3, 1)
        layout.addWidget(btn_file, 3, 2)
        layout.addWidget(btn_file_clear, 3, 3)

        layout.addWidget(QLabel("Notes:"), 4, 0)
        layout.addWidget(self.notes_edit, 4, 1, 1, 3)

        return box

    def build_data_group(self):
        box = QGroupBox("Data Columns")
        layout = QGridLayout(box)

        self.wavelength_col_combo = QComboBox()
        self.signal_col_combo = QComboBox()

        layout.addWidget(QLabel("Wavelength Column (nm):"), 0, 0)
        layout.addWidget(self.wavelength_col_combo, 0, 1)

        layout.addWidget(QLabel("Signal Column:"), 1, 0)
        layout.addWidget(self.signal_col_combo, 1, 1)

        return box

    def build_tangent_group(self):
        box = QGroupBox("Tangent Region")
        layout = QGridLayout(box)

        self.tangent_min_spin = QDoubleSpinBox()
        self.tangent_min_spin.setDecimals(6)
        self.tangent_min_spin.setMaximum(1000)
        self.tangent_min_spin.setValue(2.7)
        self.tangent_min_spin.setSuffix(" eV")

        self.tangent_max_spin = QDoubleSpinBox()
        self.tangent_max_spin.setDecimals(6)
        self.tangent_max_spin.setMaximum(1000)
        self.tangent_max_spin.setValue(3.2)
        self.tangent_max_spin.setSuffix(" eV")

        self.select_tangent_btn = QPushButton("Select Tangent Region Graphically")
        self.select_tangent_btn.clicked.connect(self.enable_tangent_selection)

        layout.addWidget(QLabel("Tangent Min:"), 0, 0)
        layout.addWidget(self.tangent_min_spin, 0, 1)
        layout.addWidget(QLabel("Tangent Max:"), 0, 2)
        layout.addWidget(self.tangent_max_spin, 0, 3)

        layout.addWidget(self.select_tangent_btn, 1, 0, 1, 4)

        return box

    def build_baseline_group(self):
        box = QGroupBox("Baseline Region")
        layout = QGridLayout(box)

        self.use_baseline_checkbox = QCheckBox("Use baseline correction")
        self.use_baseline_checkbox.setChecked(True)
        self.use_baseline_checkbox.toggled.connect(self.update_ui_states)

        self.baseline_min_spin = QDoubleSpinBox()
        self.baseline_min_spin.setDecimals(6)
        self.baseline_min_spin.setMaximum(1000)
        self.baseline_min_spin.setValue(1.8)
        self.baseline_min_spin.setSuffix(" eV")

        self.baseline_max_spin = QDoubleSpinBox()
        self.baseline_max_spin.setDecimals(6)
        self.baseline_max_spin.setMaximum(1000)
        self.baseline_max_spin.setValue(2.3)
        self.baseline_max_spin.setSuffix(" eV")

        self.select_baseline_btn = QPushButton("Select Baseline Region Graphically")
        self.select_baseline_btn.clicked.connect(self.enable_baseline_selection)

        layout.addWidget(self.use_baseline_checkbox, 0, 0, 1, 4)

        layout.addWidget(QLabel("Baseline Min:"), 1, 0)
        layout.addWidget(self.baseline_min_spin, 1, 1)
        layout.addWidget(QLabel("Baseline Max:"), 1, 2)
        layout.addWidget(self.baseline_max_spin, 1, 3)

        layout.addWidget(self.select_baseline_btn, 2, 0, 1, 4)

        return box

    def build_action_group(self):
        box = QGroupBox("Actions")
        layout = QHBoxLayout(box)

        self.calc_btn = QPushButton("Calculate Tauc Plot")
        self.calc_btn.clicked.connect(self.calculate_tauc)

        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)

        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self.clear_results)

        for btn in [self.calc_btn, self.save_btn, self.clear_btn]:
            btn.setMinimumHeight(42)

        layout.addWidget(self.calc_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.clear_btn)

        return box

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #f5f7fb; }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cfd8e3;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 12px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }
            QPushButton {
                background: #e8eef9;
                border: 1px solid #c6d3eb;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #dbe8fb; }
            QLineEdit, QComboBox, QDoubleSpinBox, QPlainTextEdit, QTableWidget {
                background: white;
                border: 1px solid #cdd6e1;
                border-radius: 8px;
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid #cfd8e3;
                background: white;
                border-radius: 10px;
            }
            #mainTitle { font-size: 20px; font-weight: 700; color: #183153; }
            #subTitle { font-size: 11px; color: #5d6b82; }
        """)

    def update_ui_states(self):
        custom_mode = self.transition_combo.currentText() == "Custom"
        self.custom_exponent_spin.setEnabled(custom_mode)

        baseline_on = self.use_baseline_checkbox.isChecked()
        self.baseline_min_spin.setEnabled(baseline_on)
        self.baseline_max_spin.setEnabled(baseline_on)
        self.select_baseline_btn.setEnabled(baseline_on)

    def log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{stamp}] {text}")
        self.statusBar().showMessage(text, 5000)

    def clear_file(self):
        self.raw_df = None
        self.processed_df = None
        self.last_result = None
        self.tangent_line_x = None
        self.tangent_line_y = None
        self.baseline_line_x = None
        self.baseline_line_y = None
        self.tangent_region = []
        self.baseline_region = []
        self.file_path_edit.clear()
        self.wavelength_col_combo.clear()
        self.signal_col_combo.clear()
        self.raw_plot.clear()
        self.tauc_plot.clear()
        self.result_summary.clear()
        self.result_table.setRowCount(0)
        self.log("Data file cleared.")

    def load_data_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "Data Files (*.csv *.txt *.dat *.xlsx *.xls)"
        )
        if not path:
            return

        try:
            self.raw_df = DataUtils.read_table(path)
            self.file_path_edit.setText(path)

            cols = list(self.raw_df.columns)
            self.wavelength_col_combo.clear()
            self.signal_col_combo.clear()
            self.wavelength_col_combo.addItems(cols)
            self.signal_col_combo.addItems(cols)

            if len(cols) >= 2:
                self.wavelength_col_combo.setCurrentIndex(0)
                self.signal_col_combo.setCurrentIndex(1)

            self.preview_raw_data()
            self.log(f"Loaded data file: {Path(path).name}")
        except Exception as e:
            self.show_error(f"Failed to load data file:\n{e}")

    def preview_raw_data(self):
        if self.raw_df is None:
            return
        try:
            x_col = self.wavelength_col_combo.currentText() or self.raw_df.columns[0]
            y_col = self.signal_col_combo.currentText() or self.raw_df.columns[1]
            df = DataUtils.clean_numeric_columns(self.raw_df, x_col, y_col)

            self.raw_plot.ax.clear()
            self.raw_plot.ax.plot(df["wavelength_nm"], df["signal"], linewidth=1.8)
            self.raw_plot.ax.set_title("Raw Data")
            self.raw_plot.ax.set_xlabel("Wavelength (nm)")
            self.raw_plot.ax.set_ylabel("Signal")
            self.raw_plot.ax.grid(True, alpha=0.3)
            self.raw_plot.figure.tight_layout()
            self.raw_plot.canvas.draw_idle()
        except Exception as e:
            self.log(f"Raw preview warning: {e}")

    def collect_inputs(self) -> TaucInputs:
        return TaucInputs(
            sample_name=self.sample_edit.text().strip() or "Sample_01",
            data_mode=self.data_mode_combo.currentText(),
            transition_type=self.transition_combo.currentText(),
            custom_exponent=float(self.custom_exponent_spin.value()),
            wavelength_column=self.wavelength_col_combo.currentText(),
            signal_column=self.signal_col_combo.currentText(),
            file_path=self.file_path_edit.text().strip(),
            tangent_min_eV=float(self.tangent_min_spin.value()),
            tangent_max_eV=float(self.tangent_max_spin.value()),
            use_baseline=bool(self.use_baseline_checkbox.isChecked()),
            baseline_min_eV=float(self.baseline_min_spin.value()),
            baseline_max_eV=float(self.baseline_max_spin.value()),
            notes=self.notes_edit.toPlainText().strip(),
        )

    def get_alpha(self, signal: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
        if mode == "Transmittance (%)":
            return DataUtils.alpha_from_transmittance_percent(signal), "log(100/T%)"
        if mode == "Reflectance (%)":
            return DataUtils.alpha_from_reflectance_percent(signal), "F(R)"
        if mode == "Absorbance":
            return DataUtils.alpha_from_absorbance(signal), "Absorbance"
        raise ValueError(f"Unknown data mode: {mode}")

    def calculate_tauc(self):
        try:
            inp = self.collect_inputs()

            if self.raw_df is None:
                raise ValueError("Please load a data file first.")
            if inp.tangent_max_eV <= inp.tangent_min_eV:
                raise ValueError("Tangent max must be greater than tangent min.")
            if inp.use_baseline and inp.baseline_max_eV <= inp.baseline_min_eV:
                raise ValueError("Baseline max must be greater than baseline min.")

            raw_clean = DataUtils.clean_numeric_columns(
                self.raw_df,
                inp.wavelength_column,
                inp.signal_column
            )

            wavelength_nm = raw_clean["wavelength_nm"].to_numpy(dtype=float)
            signal = raw_clean["signal"].to_numpy(dtype=float)
            hv = DataUtils.wavelength_to_energy_eV(wavelength_nm)
            alpha, alpha_label = self.get_alpha(signal, inp.data_mode)
            exponent = DataUtils.get_exponent(inp.transition_type, inp.custom_exponent)
            tauc_y = (alpha * hv) ** exponent

            processed = pd.DataFrame({
                "Wavelength_nm": wavelength_nm,
                "Energy_eV": hv,
                "Signal_raw": signal,
                "Alpha": alpha,
                "Tauc_Y": tauc_y,
            }).sort_values("Energy_eV").reset_index(drop=True)

            x = processed["Energy_eV"].to_numpy(dtype=float)
            y = processed["Tauc_Y"].to_numpy(dtype=float)

            m_tan, b_tan, x_tan, y_tan = DataUtils.linear_fit(
                x, y, inp.tangent_min_eV, inp.tangent_max_eV
            )

            if inp.use_baseline:
                m_base, b_base, x_base, y_base = DataUtils.linear_fit(
                    x, y, inp.baseline_min_eV, inp.baseline_max_eV
                )
            else:
                m_base, b_base = 0.0, 0.0
                x_base = np.array([x.min(), x.max()])
                y_base = np.zeros_like(x_base)

            bandgap = DataUtils.intersection_x(m_tan, b_tan, m_base, b_base)

            self.processed_df = processed
            self.tangent_line_x = x_tan
            self.tangent_line_y = y_tan
            self.baseline_line_x = x_base
            self.baseline_line_y = y_base
            self.tangent_region = [inp.tangent_min_eV, inp.tangent_max_eV]
            self.baseline_region = [inp.baseline_min_eV, inp.baseline_max_eV] if inp.use_baseline else []

            self.last_result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "inputs": asdict(inp),
                "computed": {
                    "points": int(len(processed)),
                    "data_mode": inp.data_mode,
                    "transition_type": inp.transition_type,
                    "exponent": float(exponent),
                    "alpha_label": alpha_label,
                    "tangent_min_eV": float(inp.tangent_min_eV),
                    "tangent_max_eV": float(inp.tangent_max_eV),
                    "tangent_slope": float(m_tan),
                    "tangent_intercept": float(b_tan),
                    "baseline_enabled": bool(inp.use_baseline),
                    "baseline_min_eV": float(inp.baseline_min_eV) if inp.use_baseline else None,
                    "baseline_max_eV": float(inp.baseline_max_eV) if inp.use_baseline else None,
                    "baseline_slope": float(m_base),
                    "baseline_intercept": float(b_base),
                    "bandgap_eV": float(bandgap),
                    "energy_min_eV": float(processed["Energy_eV"].min()),
                    "energy_max_eV": float(processed["Energy_eV"].max()),
                    "signal_min": float(np.min(signal)),
                    "signal_max": float(np.max(signal)),
                },
                "warnings": self.build_warnings(bandgap),
            }

            self.update_result_views()
            self.update_tauc_plot()
            self.tabs.setCurrentIndex(0)
            self.log(f"Tauc plot calculated successfully. Estimated band gap = {bandgap:.6f} eV")

        except Exception as e:
            self.show_error(f"Calculation failed:\n{e}")
            self.log(f"Calculation failed: {e}")

    def enable_tangent_selection(self):
        if self.processed_df is None or self.processed_df.empty:
            self.show_error("Please calculate the Tauc plot first.")
            return
        self.selection_mode = "tangent"
        self.selection_points = []
        self._activate_plot_selection()
        self.log("Tangent selection enabled. Click two points on the Tauc plot.")
        self.tabs.setCurrentIndex(0)

    def enable_baseline_selection(self):
        if self.processed_df is None or self.processed_df.empty:
            self.show_error("Please calculate the Tauc plot first.")
            return
        if not self.use_baseline_checkbox.isChecked():
            self.show_error("Enable baseline correction first.")
            return
        self.selection_mode = "baseline"
        self.selection_points = []
        self._activate_plot_selection()
        self.log("Baseline selection enabled. Click two points on the Tauc plot.")
        self.tabs.setCurrentIndex(0)

    def _activate_plot_selection(self):
        if self.selection_connection_id is not None:
            try:
                self.tauc_plot.canvas.mpl_disconnect(self.selection_connection_id)
            except Exception:
                pass
        self.update_tauc_plot()
        self.selection_connection_id = self.tauc_plot.canvas.mpl_connect(
            "button_press_event", self.on_tauc_plot_clicked
        )

    def on_tauc_plot_clicked(self, event):
        if self.selection_mode is None:
            return
        if event.inaxes != self.tauc_plot.ax:
            return
        if event.xdata is None:
            return

        self.selection_points.append(float(event.xdata))

        if len(self.selection_points) == 1:
            self.update_tauc_plot()
            self.tauc_plot.ax.axvline(self.selection_points[0], linestyle="--", linewidth=1.2)
            self.tauc_plot.canvas.draw_idle()
            return

        if len(self.selection_points) == 2:
            x1, x2 = sorted(self.selection_points[:2])

            if self.selection_mode == "tangent":
                self.tangent_min_spin.setValue(x1)
                self.tangent_max_spin.setValue(x2)
                self.log(f"Tangent region selected: {x1:.4f} to {x2:.4f} eV")
            elif self.selection_mode == "baseline":
                self.baseline_min_spin.setValue(x1)
                self.baseline_max_spin.setValue(x2)
                self.log(f"Baseline region selected: {x1:.4f} to {x2:.4f} eV")

            if self.selection_connection_id is not None:
                try:
                    self.tauc_plot.canvas.mpl_disconnect(self.selection_connection_id)
                except Exception:
                    pass
                self.selection_connection_id = None

            self.selection_mode = None
            self.selection_points = []
            self.recalculate_from_selected_regions()

    def recalculate_from_selected_regions(self):
        if self.processed_df is None or self.processed_df.empty:
            return
        try:
            x = self.processed_df["Energy_eV"].to_numpy(dtype=float)
            y = self.processed_df["Tauc_Y"].to_numpy(dtype=float)

            tmin = float(self.tangent_min_spin.value())
            tmax = float(self.tangent_max_spin.value())
            if tmax <= tmin:
                raise ValueError("Tangent max must be greater than tangent min.")

            m_tan, b_tan, x_tan, y_tan = DataUtils.linear_fit(x, y, tmin, tmax)

            use_baseline = self.use_baseline_checkbox.isChecked()
            if use_baseline:
                bmin = float(self.baseline_min_spin.value())
                bmax = float(self.baseline_max_spin.value())
                if bmax <= bmin:
                    raise ValueError("Baseline max must be greater than baseline min.")
                m_base, b_base, x_base, y_base = DataUtils.linear_fit(x, y, bmin, bmax)
            else:
                m_base, b_base = 0.0, 0.0
                x_base = np.array([x.min(), x.max()])
                y_base = np.zeros_like(x_base)

            bandgap = DataUtils.intersection_x(m_tan, b_tan, m_base, b_base)

            self.tangent_line_x = x_tan
            self.tangent_line_y = y_tan
            self.baseline_line_x = x_base
            self.baseline_line_y = y_base
            self.tangent_region = [tmin, tmax]
            self.baseline_region = [bmin, bmax] if use_baseline else []

            self.last_result["computed"]["tangent_min_eV"] = float(tmin)
            self.last_result["computed"]["tangent_max_eV"] = float(tmax)
            self.last_result["computed"]["tangent_slope"] = float(m_tan)
            self.last_result["computed"]["tangent_intercept"] = float(b_tan)
            self.last_result["computed"]["baseline_enabled"] = bool(use_baseline)
            self.last_result["computed"]["baseline_min_eV"] = float(bmin) if use_baseline else None
            self.last_result["computed"]["baseline_max_eV"] = float(bmax) if use_baseline else None
            self.last_result["computed"]["baseline_slope"] = float(m_base)
            self.last_result["computed"]["baseline_intercept"] = float(b_base)
            self.last_result["computed"]["bandgap_eV"] = float(bandgap)
            self.last_result["warnings"] = self.build_warnings(bandgap)

            self.update_result_views()
            self.update_tauc_plot()
            self.tabs.setCurrentIndex(0)
            self.log(f"Band gap updated from selected regions: {bandgap:.6f} eV")

        except Exception as e:
            self.show_error(f"Region-based recalculation failed:\n{e}")
            self.log(f"Region-based recalculation failed: {e}")

    def build_warnings(self, bandgap: float):
        warnings = []
        if bandgap < 0:
            warnings.append("Estimated band gap is negative. Check tangent/baseline regions.")
        if bandgap > 10:
            warnings.append("Estimated band gap is unusually high. Check data mode and selected regions.")
        return warnings

    def update_result_views(self):
        if not self.last_result:
            return

        comp = self.last_result["computed"]
        inp = self.last_result["inputs"]
        warnings = self.last_result["warnings"]

        lines = [
            f"Sample: {inp.get('sample_name', '')}",
            f"Data Mode: {inp.get('data_mode', '')}",
            f"Transition Type: {inp.get('transition_type', '')}",
            f"Estimated Band Gap (eV): {comp.get('bandgap_eV', 0):.6f}",
            f"Tangent Range (eV): {comp.get('tangent_min_eV')} to {comp.get('tangent_max_eV')}",
            f"Baseline Enabled: {comp.get('baseline_enabled')}",
            f"Baseline Range (eV): {comp.get('baseline_min_eV')} to {comp.get('baseline_max_eV')}",
        ]

        if warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in warnings:
                lines.append(f"- {w}")

        self.result_summary.setPlainText("\n".join(lines))

        rows = list(comp.items())
        self.result_table.setRowCount(len(rows))
        for r, (k, v) in enumerate(rows):
            self.result_table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.result_table.setItem(r, 1, QTableWidgetItem(str(v)))

    def update_tauc_plot(self):
        if self.processed_df is None or self.processed_df.empty or self.last_result is None:
            return

        exponent = self.last_result["computed"]["exponent"]
        alpha_label = self.last_result["computed"]["alpha_label"]

        self.tauc_plot.ax.clear()

        self.tauc_plot.ax.plot(
            self.processed_df["Energy_eV"],
            self.processed_df["Tauc_Y"],
            linewidth=1.8,
            label="Tauc data"
        )

        if len(self.tangent_region) == 2:
            x1, x2 = sorted(self.tangent_region)
            self.tauc_plot.ax.axvspan(x1, x2, alpha=0.25, color="orange", label="Tangent region")

        if len(self.baseline_region) == 2:
            x1, x2 = sorted(self.baseline_region)
            self.tauc_plot.ax.axvspan(x1, x2, alpha=0.20, color="green", label="Baseline region")

        if self.tangent_line_x is not None and self.tangent_line_y is not None:
            self.tauc_plot.ax.plot(
                self.tangent_line_x,
                self.tangent_line_y,
                linestyle="--",
                linewidth=1.8,
                label="Tangent fit"
            )

        if self.baseline_line_x is not None and self.baseline_line_y is not None:
            self.tauc_plot.ax.plot(
                self.baseline_line_x,
                self.baseline_line_y,
                linestyle="--",
                linewidth=1.8,
                label="Baseline fit"
            )

        bandgap = self.last_result["computed"]["bandgap_eV"]
        self.tauc_plot.ax.axvline(
            bandgap,
            linestyle=":",
            linewidth=1.6,
            label=f"Eg = {bandgap:.3f} eV"
        )

        self.tauc_plot.ax.set_title("Tauc Plot")
        self.tauc_plot.ax.set_xlabel("Photon Energy (eV)")
        self.tauc_plot.ax.set_ylabel(f"({alpha_label}·hν)^{exponent}")
        self.tauc_plot.ax.grid(True, alpha=0.3)
        self.tauc_plot.ax.legend()
        self.tauc_plot.figure.tight_layout()
        self.tauc_plot.canvas.draw_idle()

    def clear_results(self):
        self.processed_df = None
        self.last_result = None
        self.tangent_line_x = None
        self.tangent_line_y = None
        self.baseline_line_x = None
        self.baseline_line_y = None
        self.tangent_region = []
        self.baseline_region = []
        self.selection_mode = None
        self.selection_points = []

        if self.selection_connection_id is not None:
            try:
                self.tauc_plot.canvas.mpl_disconnect(self.selection_connection_id)
            except Exception:
                pass
            self.selection_connection_id = None

        self.result_summary.clear()
        self.result_table.setRowCount(0)
        self.raw_plot.clear()
        self.tauc_plot.clear()
        self.log("Results cleared.")

    def save_results(self):
        if not self.last_result:
            self.show_error("No results to save. Please calculate Tauc plot first.")
            return

        try:
            sample_name = self.last_result["inputs"]["sample_name"]
            run_dir = OutputManager.create_run_folder(sample_name)

            with open(run_dir / "json" / "tauc_summary.json", "w", encoding="utf-8") as f:
                json.dump(self.last_result, f, indent=2, ensure_ascii=False)

            if self.raw_df is not None:
                self.raw_df.to_csv(run_dir / "csv" / "raw_data.csv", index=False)

            if self.processed_df is not None:
                self.processed_df.to_csv(run_dir / "csv" / "processed_tauc.csv", index=False)

            self.raw_plot.figure.savefig(run_dir / "png" / "raw_plot.png", dpi=300, bbox_inches="tight")
            self.tauc_plot.figure.savefig(run_dir / "png" / "tauc_plot.png", dpi=300, bbox_inches="tight")

            txt_path = run_dir / "report" / "tauc_report.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(APP_NAME + "\n")
                f.write("=" * len(APP_NAME) + "\n\n")
                f.write(f"Generated: {self.last_result['timestamp']}\n")
                f.write(f"Software version: {APP_VERSION}\n")
                f.write(f"{ARTICLE_CITATION}\n{ARTICLE_URL}\n\n")
                f.write("INPUTS\n")
                for k, v in self.last_result["inputs"].items():
                    f.write(f"- {k}: {v}\n")
                f.write("\nCOMPUTED\n")
                for k, v in self.last_result["computed"].items():
                    f.write(f"- {k}: {v}\n")
                if self.last_result["warnings"]:
                    f.write("\nWARNINGS\n")
                    for w in self.last_result["warnings"]:
                        f.write(f"- {w}\n")

            if REPORTLAB_AVAILABLE:
                self.create_pdf_report(run_dir / "report" / "tauc_report.pdf")

            meta = {
                "software": APP_NAME,
                "saved_at": datetime.now().isoformat(timespec="seconds"),
                "run_directory": str(run_dir),
            }
            with open(run_dir / "data" / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            self.log(f"Results saved successfully: {run_dir}")
            QMessageBox.information(self, "Saved", f"Results saved successfully:\n{run_dir}")

        except Exception as e:
            self.show_error(f"Failed to save results:\n{e}")
            self.log(f"Save failed: {e}")

    def create_pdf_report(self, pdf_path: Path):
        c = pdf_canvas.Canvas(str(pdf_path), pagesize=A4)
        _, height = A4
        y = height - 2 * cm
        left = 2 * cm

        c.setFont("Helvetica-Bold", 15)
        c.drawString(left, y, APP_NAME)
        y -= 1.0 * cm

        c.setFont("Helvetica", 10)
        c.drawString(left, y, f"Generated: {self.last_result['timestamp']}")
        y -= 0.5 * cm
        c.drawString(left, y, f"Software version: {APP_VERSION}")
        y -= 0.5 * cm

        c.setFillColor("#1F4E79")
        c.drawString(left, y, ARTICLE_CITATION)
        c.linkURL(
            ARTICLE_URL,
            (left, y - 4, left + c.stringWidth(ARTICLE_CITATION, "Helvetica", 10), y + 11),
            thickness=0,
        )
        c.setFillColor("black")
        y -= 0.8 * cm

        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Inputs")
        y -= 0.5 * cm

        c.setFont("Helvetica", 9)
        for k, v in self.last_result["inputs"].items():
            c.drawString(left, y, f"{k}: {v}")
            y -= 0.4 * cm
            if y < 3 * cm:
                c.showPage()
                y = height - 2 * cm

        y -= 0.2 * cm
        c.setFont("Helvetica-Bold", 11)
        c.drawString(left, y, "Computed Results")
        y -= 0.5 * cm

        c.setFont("Helvetica", 9)
        for k, v in self.last_result["computed"].items():
            c.drawString(left, y, f"{k}: {v}")
            y -= 0.4 * cm
            if y < 3 * cm:
                c.showPage()
                y = height - 2 * cm

        c.showPage()
        c.save()

    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)


def main():
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "researcher.analysis.toolkit.tauc.macrologic.v1"
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Researcher Analysis Toolkit")

    for name in ["Researcherlogo.ico", "ResearcherLogo.ico"]:
        path = resource_path(name)
        if os.path.exists(path):
            app.setWindowIcon(QIcon(path))
            break

    try:
        window = TaucPlotWindow()
        window.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()
        QMessageBox.critical(None, "Fatal Error", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
