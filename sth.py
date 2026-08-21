import sys
import os
import json
import traceback
import ctypes
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    QFrame,
)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.pdfgen import canvas as pdf_canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

APP_NAME = "Researcher Analysis Toolkit - STH"
OUTPUT_TECHNIQUE = "STH Analysis"
DELTA_G_H2_J_PER_MOL = 237000.0  # Gibbs free energy for H2 formation


def resource_path(filename: str) -> str:
    if getattr(sys, "_MEIPASS", None):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


@dataclass
class STHInputs:
    sample_name: str
    mode: str
    calculation_type: str
    # PEC current-based manual
    current_density_value: float
    current_density_unit: str
    fe_percent: float
    pin_value: float
    pin_unit: str
    # Hydrogen-rate-based manual
    h2_rate_value: float
    h2_rate_unit: str
    illuminated_area_cm2: float
    # File-based
    file_path: str
    x_column: str
    y_column: str
    fe_column: str
    pin_column: str
    area_column: str
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
            return pd.read_excel(file_path)

        raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def clean_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        for col in columns:
            if col and col not in df.columns:
                raise ValueError(f"Column not found: {col}")

        selected = [c for c in columns if c]
        out = df[selected].copy()

        for col in selected:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna()
        if out.empty:
            raise ValueError("No valid numeric rows found.")
        return out.reset_index(drop=True)

    @staticmethod
    def current_density_to_A_cm2(values: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "a/cm2":
            return values
        if unit == "ma/cm2":
            return values / 1000.0
        if unit == "ua/cm2":
            return values / 1e6
        raise ValueError(f"Unsupported current density unit: {unit}")

    @staticmethod
    def pin_to_W_cm2(values: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "w/cm2":
            return values
        if unit == "mw/cm2":
            return values / 1000.0
        if unit == "uw/cm2":
            return values / 1e6
        raise ValueError(f"Unsupported light intensity unit: {unit}")

    @staticmethod
    def h2_rate_to_mol_s(values: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "mol/s":
            return values
        if unit == "mmol/s":
            return values / 1000.0
        if unit == "umol/s":
            return values / 1e6
        if unit == "nmol/s":
            return values / 1e9
        if unit == "mol/min":
            return values / 60.0
        if unit == "mmol/min":
            return values / (1000.0 * 60.0)
        if unit == "umol/min":
            return values / (1e6 * 60.0)
        if unit == "nmol/min":
            return values / (1e9 * 60.0)
        if unit == "mol/h":
            return values / 3600.0
        if unit == "mmol/h":
            return values / (1000.0 * 3600.0)
        if unit == "umol/h":
            return values / (1e6 * 3600.0)
        if unit == "nmol/h":
            return values / (1e9 * 3600.0)
        raise ValueError(f"Unsupported H2 rate unit: {unit}")


class STHCalculator:
    @staticmethod
    def current_based_sth(current_density_A_cm2, fe_fraction, pin_W_cm2):
        current_density_A_cm2 = np.asarray(current_density_A_cm2, dtype=float)
        fe_fraction = np.asarray(fe_fraction, dtype=float)
        pin_W_cm2 = np.asarray(pin_W_cm2, dtype=float)

        if np.any(pin_W_cm2 <= 0):
            raise ValueError("Pin must be greater than zero.")
        if np.any(fe_fraction < 0):
            raise ValueError("FE cannot be negative.")

        return (current_density_A_cm2 * 1.23 * fe_fraction / pin_W_cm2) * 100.0

    @staticmethod
    def rate_based_sth(h2_rate_mol_s, pin_W_cm2, area_cm2):
        h2_rate_mol_s = np.asarray(h2_rate_mol_s, dtype=float)
        pin_W_cm2 = np.asarray(pin_W_cm2, dtype=float)
        area_cm2 = np.asarray(area_cm2, dtype=float)

        if np.any(pin_W_cm2 <= 0):
            raise ValueError("Pin must be greater than zero.")
        if np.any(area_cm2 <= 0):
            raise ValueError("Illuminated area must be greater than zero.")
        if np.any(h2_rate_mol_s < 0):
            raise ValueError("H2 rate cannot be negative.")

        incident_power_W = pin_W_cm2 * area_cm2
        chemical_power_W = h2_rate_mol_s * DELTA_G_H2_J_PER_MOL
        return (chemical_power_W / incident_power_W) * 100.0


class PlotCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = plt.Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self.canvas = FigureCanvasQTAgg(self.figure)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.canvas.draw_idle()


class STHWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.last_result = None
        self.plot_x_label = "X"
        self.plot_title = "STH"

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1500, 920)
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
        title = QLabel("STH Module")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Solar-to-Hydrogen Efficiency Analysis")
        title.setObjectName("subTitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        left_layout.addWidget(title_frame)

        left_layout.addWidget(self.build_general_group())
        left_layout.addWidget(self.build_current_based_group())
        left_layout.addWidget(self.build_rate_based_group())
        left_layout.addWidget(self.build_file_group())
        left_layout.addWidget(self.build_action_group())
        left_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_container)
        scroll.setMinimumWidth(540)
        scroll.setMaximumWidth(700)

        self.tabs = QTabWidget()

        self.raw_plot = PlotCanvas()
        self.sth_plot = PlotCanvas()

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_tabs = QTabWidget()
        preview_tabs.addTab(self.raw_plot, "Raw Data")
        preview_tabs.addTab(self.sth_plot, "STH Plot")
        preview_layout.addWidget(preview_tabs)

        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)

        self.result_summary = QPlainTextEdit()
        self.result_summary.setReadOnly(True)
        self.result_summary.setMinimumHeight(250)

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

    def build_general_group(self) -> QGroupBox:
        box = QGroupBox("General Settings")
        layout = QGridLayout(box)

        self.sample_edit = QLineEdit("Sample_01")

        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Manual",
            "File-based",
        ])
        self.mode_combo.currentTextChanged.connect(self.update_ui_states)

        self.calculation_type_combo = QComboBox()
        self.calculation_type_combo.addItems([
            "PEC Current-Based",
            "Hydrogen Rate-Based",
        ])
        self.calculation_type_combo.currentTextChanged.connect(self.update_ui_states)

        self.file_path_edit = QLineEdit()
        self.file_path_edit.setReadOnly(True)

        btn_file = QPushButton("Browse Data File")
        btn_file.clicked.connect(self.load_data_file)

        btn_file_clear = QPushButton("Refresh")
        btn_file_clear.clicked.connect(self.clear_file)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes...")
        self.notes_edit.setFixedHeight(90)

        layout.addWidget(QLabel("Sample Name:"), 0, 0)
        layout.addWidget(self.sample_edit, 0, 1, 1, 3)

        layout.addWidget(QLabel("Mode:"), 1, 0)
        layout.addWidget(self.mode_combo, 1, 1)

        layout.addWidget(QLabel("Calculation Type:"), 1, 2)
        layout.addWidget(self.calculation_type_combo, 1, 3)

        layout.addWidget(QLabel("Data File:"), 2, 0)
        layout.addWidget(self.file_path_edit, 2, 1)
        layout.addWidget(btn_file, 2, 2)
        layout.addWidget(btn_file_clear, 2, 3)

        layout.addWidget(QLabel("Notes:"), 3, 0)
        layout.addWidget(self.notes_edit, 3, 1, 1, 3)

        return box

    def build_current_based_group(self) -> QGroupBox:
        box = QGroupBox("PEC Current-Based STH")
        layout = QGridLayout(box)

        self.current_density_spin = QDoubleSpinBox()
        self.current_density_spin.setDecimals(10)
        self.current_density_spin.setMaximum(1e12)
        self.current_density_spin.setValue(10.0)

        self.current_density_unit_combo = QComboBox()
        self.current_density_unit_combo.addItems(["mA/cm2", "A/cm2", "uA/cm2"])
        self.current_density_unit_combo.setCurrentText("mA/cm2")

        self.fe_spin = QDoubleSpinBox()
        self.fe_spin.setDecimals(6)
        self.fe_spin.setMaximum(1000.0)
        self.fe_spin.setValue(100.0)
        self.fe_spin.setSuffix(" %")

        self.pin_spin = QDoubleSpinBox()
        self.pin_spin.setDecimals(10)
        self.pin_spin.setMaximum(1e12)
        self.pin_spin.setValue(100.0)

        self.pin_unit_combo = QComboBox()
        self.pin_unit_combo.addItems(["mW/cm2", "W/cm2", "uW/cm2"])
        self.pin_unit_combo.setCurrentText("mW/cm2")

        layout.addWidget(QLabel("Current Density:"), 0, 0)
        layout.addWidget(self.current_density_spin, 0, 1)
        layout.addWidget(self.current_density_unit_combo, 0, 2)

        layout.addWidget(QLabel("Faradaic Efficiency:"), 1, 0)
        layout.addWidget(self.fe_spin, 1, 1)

        layout.addWidget(QLabel("Incident Light Intensity:"), 2, 0)
        layout.addWidget(self.pin_spin, 2, 1)
        layout.addWidget(self.pin_unit_combo, 2, 2)

        return box

    def build_rate_based_group(self) -> QGroupBox:
        box = QGroupBox("Hydrogen-Rate-Based STH")
        layout = QGridLayout(box)

        self.h2_rate_spin = QDoubleSpinBox()
        self.h2_rate_spin.setDecimals(12)
        self.h2_rate_spin.setMaximum(1e12)
        self.h2_rate_spin.setValue(1.0)

        self.h2_rate_unit_combo = QComboBox()
        self.h2_rate_unit_combo.addItems([
            "mol/s", "mmol/s", "umol/s", "nmol/s",
            "mol/min", "mmol/min", "umol/min", "nmol/min",
            "mol/h", "mmol/h", "umol/h", "nmol/h",
        ])
        self.h2_rate_unit_combo.setCurrentText("umol/h")

        self.area_spin = QDoubleSpinBox()
        self.area_spin.setDecimals(8)
        self.area_spin.setMaximum(1e12)
        self.area_spin.setValue(1.0)
        self.area_spin.setSuffix(" cm2")

        layout.addWidget(QLabel("H2 Production Rate:"), 0, 0)
        layout.addWidget(self.h2_rate_spin, 0, 1)
        layout.addWidget(self.h2_rate_unit_combo, 0, 2)

        layout.addWidget(QLabel("Illuminated Area:"), 1, 0)
        layout.addWidget(self.area_spin, 1, 1)

        return box

    def build_file_group(self) -> QGroupBox:
        box = QGroupBox("File-Based STH")
        layout = QGridLayout(box)

        self.x_col_combo = QComboBox()
        self.y_col_combo = QComboBox()
        self.fe_col_combo = QComboBox()
        self.pin_col_combo = QComboBox()
        self.area_col_combo = QComboBox()

        self.file_current_density_unit_combo = QComboBox()
        self.file_current_density_unit_combo.addItems(["mA/cm2", "A/cm2", "uA/cm2"])
        self.file_current_density_unit_combo.setCurrentText("mA/cm2")

        self.file_pin_unit_combo = QComboBox()
        self.file_pin_unit_combo.addItems(["mW/cm2", "W/cm2", "uW/cm2"])
        self.file_pin_unit_combo.setCurrentText("mW/cm2")

        self.file_h2_rate_unit_combo = QComboBox()
        self.file_h2_rate_unit_combo.addItems([
            "mol/s", "mmol/s", "umol/s", "nmol/s",
            "mol/min", "mmol/min", "umol/min", "nmol/min",
            "mol/h", "mmol/h", "umol/h", "nmol/h",
        ])
        self.file_h2_rate_unit_combo.setCurrentText("umol/h")

        layout.addWidget(QLabel("X Column (Potential / Wavelength / Time):"), 0, 0)
        layout.addWidget(self.x_col_combo, 0, 1)

        layout.addWidget(QLabel("Y Column:"), 1, 0)
        layout.addWidget(self.y_col_combo, 1, 1)

        layout.addWidget(self.file_current_density_unit_combo, 1, 2)
        layout.addWidget(self.file_h2_rate_unit_combo, 1, 3)

        layout.addWidget(QLabel("FE Column (%) [Current-Based Only]:"), 2, 0)
        layout.addWidget(self.fe_col_combo, 2, 1)

        layout.addWidget(QLabel("Pin Column:"), 3, 0)
        layout.addWidget(self.pin_col_combo, 3, 1)
        layout.addWidget(self.file_pin_unit_combo, 3, 2)

        layout.addWidget(QLabel("Area Column (cm2) [Rate-Based Only]:"), 4, 0)
        layout.addWidget(self.area_col_combo, 4, 1)

        return box

    def build_action_group(self) -> QGroupBox:
        box = QGroupBox("Actions")
        layout = QHBoxLayout(box)

        self.calc_btn = QPushButton("Calculate STH")
        self.calc_btn.clicked.connect(self.calculate_sth)

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

    def log(self, text: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{stamp}] {text}")
        self.statusBar().showMessage(text, 5000)

    def update_ui_states(self):
        manual_mode = self.mode_combo.currentText() == "Manual"
        file_mode = self.mode_combo.currentText() == "File-based"
        current_based = self.calculation_type_combo.currentText() == "PEC Current-Based"
        rate_based = self.calculation_type_combo.currentText() == "Hydrogen Rate-Based"

        # manual groups
        for w in [
            self.current_density_spin,
            self.current_density_unit_combo,
            self.fe_spin,
            self.pin_spin,
            self.pin_unit_combo,
        ]:
            w.setEnabled(manual_mode and current_based)

        for w in [
            self.h2_rate_spin,
            self.h2_rate_unit_combo,
            self.area_spin,
        ]:
            w.setEnabled(manual_mode and rate_based)

        # file groups
        self.file_path_edit.setEnabled(file_mode)
        self.x_col_combo.setEnabled(file_mode)
        self.y_col_combo.setEnabled(file_mode)
        self.pin_col_combo.setEnabled(file_mode)

        self.fe_col_combo.setEnabled(file_mode and current_based)
        self.area_col_combo.setEnabled(file_mode and rate_based)

        self.file_current_density_unit_combo.setEnabled(file_mode and current_based)
        self.file_h2_rate_unit_combo.setEnabled(file_mode and rate_based)
        self.file_pin_unit_combo.setEnabled(file_mode)

    def clear_file(self):
        self.raw_df = None
        self.processed_df = None
        self.file_path_edit.clear()
        self.x_col_combo.clear()
        self.y_col_combo.clear()
        self.fe_col_combo.clear()
        self.pin_col_combo.clear()
        self.area_col_combo.clear()
        self.raw_plot.clear()
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
            for combo in [self.x_col_combo, self.y_col_combo, self.fe_col_combo, self.pin_col_combo, self.area_col_combo]:
                combo.clear()
                combo.addItems(cols)

            if len(cols) >= 2:
                self.x_col_combo.setCurrentIndex(0)
                self.y_col_combo.setCurrentIndex(1)
            if len(cols) >= 3:
                self.pin_col_combo.setCurrentIndex(2)

            self.preview_raw_data()
            self.log(f"Loaded data file: {Path(path).name}")

        except Exception as e:
            self.show_error(f"Failed to load data file:\n{e}")

    def preview_raw_data(self):
        if self.raw_df is None:
            return
        try:
            x_col = self.x_col_combo.currentText() or self.raw_df.columns[0]
            y_col = self.y_col_combo.currentText() or self.raw_df.columns[1]
            df = DataUtils.clean_numeric_columns(self.raw_df, [x_col, y_col])

            self.raw_plot.ax.clear()
            self.raw_plot.ax.plot(df[x_col], df[y_col], linewidth=1.8)
            self.raw_plot.ax.set_title("Raw Data Preview")
            self.raw_plot.ax.set_xlabel(x_col)
            self.raw_plot.ax.set_ylabel(y_col)
            self.raw_plot.ax.grid(True, alpha=0.3)
            self.raw_plot.figure.tight_layout()
            self.raw_plot.canvas.draw_idle()
        except Exception as e:
            self.log(f"Raw preview warning: {e}")

    def collect_inputs(self) -> STHInputs:
        return STHInputs(
            sample_name=self.sample_edit.text().strip() or "Sample_01",
            mode=self.mode_combo.currentText(),
            calculation_type=self.calculation_type_combo.currentText(),
            current_density_value=float(self.current_density_spin.value()),
            current_density_unit=self.current_density_unit_combo.currentText(),
            fe_percent=float(self.fe_spin.value()),
            pin_value=float(self.pin_spin.value()),
            pin_unit=self.pin_unit_combo.currentText(),
            h2_rate_value=float(self.h2_rate_spin.value()),
            h2_rate_unit=self.h2_rate_unit_combo.currentText(),
            illuminated_area_cm2=float(self.area_spin.value()),
            file_path=self.file_path_edit.text().strip(),
            x_column=self.x_col_combo.currentText(),
            y_column=self.y_col_combo.currentText(),
            fe_column=self.fe_col_combo.currentText(),
            pin_column=self.pin_col_combo.currentText(),
            area_column=self.area_col_combo.currentText(),
            notes=self.notes_edit.toPlainText().strip(),
        )

    def calculate_sth(self):
        try:
            inp = self.collect_inputs()

            if inp.mode == "Manual":
                if inp.calculation_type == "PEC Current-Based":
                    self.calculate_manual_current_based(inp)
                else:
                    self.calculate_manual_rate_based(inp)
            else:
                if self.raw_df is None:
                    raise ValueError("Please load a data file first.")
                if inp.calculation_type == "PEC Current-Based":
                    self.calculate_file_current_based(inp)
                else:
                    self.calculate_file_rate_based(inp)

            self.tabs.setCurrentIndex(1)

        except Exception as e:
            self.show_error(f"Calculation failed:\n{e}")
            self.log(f"Calculation failed: {e}")

    def calculate_manual_current_based(self, inp: STHInputs):
        j_A_cm2 = float(DataUtils.current_density_to_A_cm2(
            np.array([inp.current_density_value]), inp.current_density_unit
        )[0])
        fe_fraction = inp.fe_percent / 100.0
        pin_W_cm2 = float(DataUtils.pin_to_W_cm2(
            np.array([inp.pin_value]), inp.pin_unit
        )[0])

        sth = float(STHCalculator.current_based_sth(
            np.array([j_A_cm2]),
            np.array([fe_fraction]),
            np.array([pin_W_cm2])
        )[0])

        self.processed_df = pd.DataFrame({
            "CurrentDensity_A_cm2": [j_A_cm2],
            "FE_fraction": [fe_fraction],
            "Pin_W_cm2": [pin_W_cm2],
            "STH_percent": [sth],
        })

        self.plot_x_label = "Single Point"
        self.plot_title = "Manual PEC Current-Based STH"

        self.last_result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inputs": asdict(inp),
            "computed": {
                "mode": inp.mode,
                "calculation_type": inp.calculation_type,
                "current_density_A_cm2": j_A_cm2,
                "fe_percent": inp.fe_percent,
                "pin_W_cm2": pin_W_cm2,
                "sth_percent": sth,
            },
            "warnings": self.build_warnings(np.array([sth])),
        }

        self.update_result_views()
        self.update_sth_plot(single_point=True)
        self.log(f"Manual current-based STH calculated: {sth:.6f}%")

    def calculate_manual_rate_based(self, inp: STHInputs):
        rate_mol_s = float(DataUtils.h2_rate_to_mol_s(
            np.array([inp.h2_rate_value]), inp.h2_rate_unit
        )[0])
        pin_W_cm2 = float(DataUtils.pin_to_W_cm2(
            np.array([inp.pin_value]), inp.pin_unit
        )[0])
        area_cm2 = inp.illuminated_area_cm2

        sth = float(STHCalculator.rate_based_sth(
            np.array([rate_mol_s]),
            np.array([pin_W_cm2]),
            np.array([area_cm2])
        )[0])

        self.processed_df = pd.DataFrame({
            "H2Rate_mol_s": [rate_mol_s],
            "Pin_W_cm2": [pin_W_cm2],
            "Area_cm2": [area_cm2],
            "STH_percent": [sth],
        })

        self.plot_x_label = "Single Point"
        self.plot_title = "Manual Hydrogen-Rate-Based STH"

        self.last_result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inputs": asdict(inp),
            "computed": {
                "mode": inp.mode,
                "calculation_type": inp.calculation_type,
                "h2_rate_mol_s": rate_mol_s,
                "pin_W_cm2": pin_W_cm2,
                "area_cm2": area_cm2,
                "sth_percent": sth,
            },
            "warnings": self.build_warnings(np.array([sth])),
        }

        self.update_result_views()
        self.update_sth_plot(single_point=True)
        self.log(f"Manual rate-based STH calculated: {sth:.6f}%")

    def calculate_file_current_based(self, inp: STHInputs):
        df = DataUtils.clean_numeric_columns(
            self.raw_df,
            [inp.x_column, inp.y_column, inp.fe_column, inp.pin_column]
        )

        x = df[inp.x_column].to_numpy(dtype=float)
        j_A_cm2 = DataUtils.current_density_to_A_cm2(
            df[inp.y_column].to_numpy(dtype=float),
            self.file_current_density_unit_combo.currentText()
        )
        fe_fraction = df[inp.fe_column].to_numpy(dtype=float) / 100.0
        pin_W_cm2 = DataUtils.pin_to_W_cm2(
            df[inp.pin_column].to_numpy(dtype=float),
            self.file_pin_unit_combo.currentText()
        )

        sth = STHCalculator.current_based_sth(j_A_cm2, fe_fraction, pin_W_cm2)

        self.processed_df = pd.DataFrame({
            "X": x,
            "CurrentDensity_A_cm2": j_A_cm2,
            "FE_fraction": fe_fraction,
            "Pin_W_cm2": pin_W_cm2,
            "STH_percent": sth,
        }).sort_values("X").reset_index(drop=True)

        self.plot_x_label = inp.x_column
        self.plot_title = "File-Based PEC Current-Based STH"

        self.last_result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inputs": asdict(inp),
            "computed": {
                "mode": inp.mode,
                "calculation_type": inp.calculation_type,
                "points": int(len(self.processed_df)),
                "x_min": float(self.processed_df["X"].min()),
                "x_max": float(self.processed_df["X"].max()),
                "sth_min_percent": float(self.processed_df["STH_percent"].min()),
                "sth_max_percent": float(self.processed_df["STH_percent"].max()),
                "sth_mean_percent": float(self.processed_df["STH_percent"].mean()),
            },
            "warnings": self.build_warnings(self.processed_df["STH_percent"].to_numpy(dtype=float)),
        }

        self.update_result_views()
        self.update_sth_plot(single_point=False)
        self.log("File-based current-based STH calculated successfully.")

    def calculate_file_rate_based(self, inp: STHInputs):
        df = DataUtils.clean_numeric_columns(
            self.raw_df,
            [inp.x_column, inp.y_column, inp.pin_column, inp.area_column]
        )

        x = df[inp.x_column].to_numpy(dtype=float)
        rate_mol_s = DataUtils.h2_rate_to_mol_s(
            df[inp.y_column].to_numpy(dtype=float),
            self.file_h2_rate_unit_combo.currentText()
        )
        pin_W_cm2 = DataUtils.pin_to_W_cm2(
            df[inp.pin_column].to_numpy(dtype=float),
            self.file_pin_unit_combo.currentText()
        )
        area_cm2 = df[inp.area_column].to_numpy(dtype=float)

        sth = STHCalculator.rate_based_sth(rate_mol_s, pin_W_cm2, area_cm2)

        self.processed_df = pd.DataFrame({
            "X": x,
            "H2Rate_mol_s": rate_mol_s,
            "Pin_W_cm2": pin_W_cm2,
            "Area_cm2": area_cm2,
            "STH_percent": sth,
        }).sort_values("X").reset_index(drop=True)

        self.plot_x_label = inp.x_column
        self.plot_title = "File-Based Hydrogen-Rate-Based STH"

        self.last_result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inputs": asdict(inp),
            "computed": {
                "mode": inp.mode,
                "calculation_type": inp.calculation_type,
                "points": int(len(self.processed_df)),
                "x_min": float(self.processed_df["X"].min()),
                "x_max": float(self.processed_df["X"].max()),
                "sth_min_percent": float(self.processed_df["STH_percent"].min()),
                "sth_max_percent": float(self.processed_df["STH_percent"].max()),
                "sth_mean_percent": float(self.processed_df["STH_percent"].mean()),
            },
            "warnings": self.build_warnings(self.processed_df["STH_percent"].to_numpy(dtype=float)),
        }

        self.update_result_views()
        self.update_sth_plot(single_point=False)
        self.log("File-based rate-based STH calculated successfully.")

    def build_warnings(self, sth_values: np.ndarray):
        warnings = []
        if np.any(sth_values > 100):
            warnings.append("Some STH values are greater than 100%. Check current/rate, FE, Pin, area, and unit conversions.")
        if np.any(sth_values < 0):
            warnings.append("Some STH values are negative. Check sign conventions and input data.")
        return warnings

    def update_result_views(self):
        if not self.last_result:
            return

        comp = self.last_result["computed"]
        inp = self.last_result["inputs"]
        warnings = self.last_result["warnings"]

        lines = [
            f"Sample: {inp['sample_name']}",
            f"Mode: {inp['mode']}",
            f"Calculation Type: {inp['calculation_type']}",
        ]

        for k, v in comp.items():
            lines.append(f"{k}: {v}")

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

    def update_sth_plot(self, single_point: bool):
        if self.processed_df is None or self.processed_df.empty:
            return

        self.sth_plot.ax.clear()

        if single_point:
            self.sth_plot.ax.bar(["STH (%)"], [self.processed_df["STH_percent"].iloc[0]])
            self.sth_plot.ax.set_xlabel(self.plot_x_label)
        else:
            self.sth_plot.ax.plot(
                self.processed_df["X"],
                self.processed_df["STH_percent"],
                marker="o",
                linewidth=1.8
            )
            self.sth_plot.ax.set_xlabel(self.plot_x_label)

        self.sth_plot.ax.set_title(self.plot_title)
        self.sth_plot.ax.set_ylabel("STH (%)")
        self.sth_plot.ax.grid(True, alpha=0.3)
        self.sth_plot.figure.tight_layout()
        self.sth_plot.canvas.draw_idle()

    def clear_results(self):
        self.processed_df = None
        self.last_result = None
        self.result_summary.clear()
        self.result_table.setRowCount(0)
        self.sth_plot.clear()
        self.raw_plot.clear()
        self.log("Results cleared.")

    def save_results(self):
        if not self.last_result:
            self.show_error("No results to save. Please calculate STH first.")
            return

        try:
            sample_name = self.last_result["inputs"]["sample_name"]
            run_dir = OutputManager.create_run_folder(sample_name)

            with open(run_dir / "json" / "sth_summary.json", "w", encoding="utf-8") as f:
                json.dump(self.last_result, f, indent=2, ensure_ascii=False)

            if self.raw_df is not None:
                self.raw_df.to_csv(run_dir / "csv" / "raw_data.csv", index=False)

            if self.processed_df is not None:
                self.processed_df.to_csv(run_dir / "csv" / "processed_sth.csv", index=False)

            self.raw_plot.figure.savefig(run_dir / "png" / "raw_plot.png", dpi=300, bbox_inches="tight")
            self.sth_plot.figure.savefig(run_dir / "png" / "sth_plot.png", dpi=300, bbox_inches="tight")

            txt_path = run_dir / "report" / "sth_report.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(APP_NAME + "\n")
                f.write("=" * len(APP_NAME) + "\n\n")
                f.write(f"Generated: {self.last_result['timestamp']}\n\n")
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
                self.create_pdf_report(run_dir / "report" / "sth_report.pdf")

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
        width, height = A4
        y = height - 2 * cm
        left = 2 * cm

        c.setFont("Helvetica-Bold", 15)
        c.drawString(left, y, APP_NAME)
        y -= 1.0 * cm

        c.setFont("Helvetica", 10)
        c.drawString(left, y, f"Generated: {self.last_result['timestamp']}")
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

        if self.last_result["warnings"]:
            y -= 0.2 * cm
            c.setFont("Helvetica-Bold", 11)
            c.drawString(left, y, "Warnings")
            y -= 0.5 * cm
            c.setFont("Helvetica", 9)
            for w in self.last_result["warnings"]:
                c.drawString(left, y, f"- {w}")
                y -= 0.4 * cm
                if y < 3 * cm:
                    c.showPage()
                    y = height - 2 * cm

        c.showPage()
        c.save()

    def show_error(self, message: str):
        QMessageBox.critical(self, "Error", message)


def main():
    import ctypes

    myappid = "researcher.analysis.toolkit.sth.v1"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)

    icon_path = resource_path("Researcherlogo.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = STHWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()