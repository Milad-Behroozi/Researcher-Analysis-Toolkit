import sys
import os
import json
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt6.QtCore import Qt
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

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas as pdf_canvas

FARADAY_CONSTANT = 96485.33212
APP_NAME = "Researcher Analysis Toolkit - Faradaic Efficiency"
OUTPUT_TECHNIQUE = "Faradaic Efficiency Analysis"


def resource_path(filename: str) -> str:
    if getattr(sys, "_MEIPASS", None):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


@dataclass
class FEInputs:
    sample_name: str
    product_name: str
    electrons_n: float
    gas_mode: str
    charge_source: str
    manual_charge_c: float
    manual_current_a: float
    manual_time_s: float
    current_file_path: str
    gas_file_path: str
    current_time_column: str
    current_value_column: str
    gas_time_column: str
    gas_value_column: str
    current_unit: str
    current_time_unit: str
    gas_value_unit: str
    gas_time_unit: str
    integration_mode: str
    manual_moles_product: float
    peak_area_to_mol_factor: float
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
    def clean_xy(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        if x_col not in df.columns or y_col not in df.columns:
            raise ValueError("Selected columns were not found in the file.")

        out = df[[x_col, y_col]].copy()
        out.columns = ["x", "y"]
        out["x"] = pd.to_numeric(out["x"], errors="coerce")
        out["y"] = pd.to_numeric(out["y"], errors="coerce")
        out = out.dropna().sort_values("x").drop_duplicates(subset="x")

        if len(out) < 2:
            raise ValueError("Not enough valid numeric points.")
        return out.reset_index(drop=True)

    @staticmethod
    def time_to_seconds(arr: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "s":
            return arr
        if unit == "min":
            return arr * 60.0
        if unit == "h":
            return arr * 3600.0
        raise ValueError(f"Unsupported time unit: {unit}")

    @staticmethod
    def current_to_ampere(arr: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "a":
            return arr
        if unit == "ma":
            return arr / 1000.0
        if unit == "ua":
            return arr / 1e6
        raise ValueError(f"Unsupported current unit: {unit}")

    @staticmethod
    def cumulative_amount_to_mol(arr: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "mol":
            return arr
        if unit == "mmol":
            return arr / 1000.0
        if unit == "umol":
            return arr / 1e6
        if unit == "nmol":
            return arr / 1e9
        raise ValueError(f"Unsupported cumulative amount unit: {unit}")

    @staticmethod
    def rate_to_mol_per_s(arr: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "mol/s":
            return arr
        if unit == "mmol/s":
            return arr / 1000.0
        if unit == "umol/s":
            return arr / 1e6
        if unit == "nmol/s":
            return arr / 1e9
        if unit == "mol/min":
            return arr / 60.0
        if unit == "mmol/min":
            return arr / (1000.0 * 60.0)
        if unit == "umol/min":
            return arr / (1e6 * 60.0)
        if unit == "nmol/min":
            return arr / (1e9 * 60.0)
        if unit == "mol/h":
            return arr / 3600.0
        if unit == "mmol/h":
            return arr / (1000.0 * 3600.0)
        if unit == "umol/h":
            return arr / (1e6 * 3600.0)
        if unit == "nmol/h":
            return arr / (1e9 * 3600.0)
        raise ValueError(f"Unsupported rate unit: {unit}")

    @staticmethod
    def integrate_charge(time_s: np.ndarray, current_a: np.ndarray, mode: str) -> float:
        mode = mode.lower()
        curr = current_a.copy()
        if mode == "absolute":
            curr = np.abs(curr)
        elif mode == "anodic_only":
            curr = np.where(curr > 0, curr, 0.0)
        elif mode == "cathodic_only":
            curr = np.where(curr < 0, np.abs(curr), 0.0)
        elif mode == "signed":
            curr = curr
        else:
            raise ValueError("Invalid integration mode.")
        return float(np.trapz(curr, time_s))

    @staticmethod
    def load_processed_current(df: pd.DataFrame, time_col: str, current_col: str, time_unit: str, current_unit: str) -> pd.DataFrame:
        out = DataUtils.clean_xy(df, time_col, current_col)
        out["time_s"] = DataUtils.time_to_seconds(out["x"].to_numpy(dtype=float), time_unit)
        out["current_a"] = DataUtils.current_to_ampere(out["y"].to_numpy(dtype=float), current_unit)
        return out[["time_s", "current_a"]].copy()

    @staticmethod
    def load_processed_gas_generic(df: pd.DataFrame, time_col: str, value_col: str, time_unit: str) -> pd.DataFrame:
        out = DataUtils.clean_xy(df, time_col, value_col)
        out["time_s"] = DataUtils.time_to_seconds(out["x"].to_numpy(dtype=float), time_unit)
        out["value"] = out["y"].to_numpy(dtype=float)
        return out[["time_s", "value"]].copy()


class FECalculator:
    @staticmethod
    def fe_percent(n: float, mol_product: float, charge_c: float) -> float:
        if n <= 0:
            raise ValueError("n must be greater than zero.")
        if charge_c <= 0:
            raise ValueError("Charge must be greater than zero.")
        if mol_product < 0:
            raise ValueError("Product amount cannot be negative.")
        return (n * FARADAY_CONSTANT * mol_product / charge_c) * 100.0


class PlotCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = plt.Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    def clear(self):
        self.ax.clear()
        self.canvas.draw_idle()


class FEWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_df: Optional[pd.DataFrame] = None
        self.gas_df: Optional[pd.DataFrame] = None
        self.last_result = None
        self.last_current_processed: Optional[pd.DataFrame] = None
        self.last_gas_processed: Optional[pd.DataFrame] = None
        self.last_gas_plot_label = ("Signal", "Value")

        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(1450, 920)
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
        open_current = QAction("Open Current File", self)
        open_current.triggered.connect(self.load_current_file)
        menu.addAction(open_current)
        open_gas = QAction("Open Gas File", self)
        open_gas.triggered.connect(self.load_gas_file)
        menu.addAction(open_gas)
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
        title = QLabel("Faradaic Efficiency Module")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Supports Manual, Cumulative, Instantaneous Rate, and Instantaneous Peak-Area modes")
        subtitle.setObjectName("subTitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        left_layout.addWidget(title_frame)

        left_layout.addWidget(self.build_general_group())
        left_layout.addWidget(self.build_charge_group())
        left_layout.addWidget(self.build_product_group())
        left_layout.addWidget(self.build_action_group())
        left_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_container)
        scroll.setMinimumWidth(520)
        scroll.setMaximumWidth(660)

        self.tabs = QTabWidget()

        self.current_plot = PlotCanvas()
        self.gas_plot = PlotCanvas()
        self.fe_plot = PlotCanvas()

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_tabs = QTabWidget()
        preview_tabs.addTab(self.current_plot, "Current Plot")
        preview_tabs.addTab(self.gas_plot, "Gas Plot")
        preview_tabs.addTab(self.fe_plot, "FE Plot")
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
        self.current_path_edit = QLineEdit()
        self.current_path_edit.setReadOnly(True)
        self.gas_path_edit = QLineEdit()
        self.gas_path_edit.setReadOnly(True)
        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText("Optional notes...")
        self.notes_edit.setFixedHeight(90)

        btn_current = QPushButton("Browse Current File")
        btn_current.clicked.connect(self.load_current_file)
        btn_current_clear = QPushButton("Refresh")
        btn_current_clear.clicked.connect(self.clear_current_file)

        btn_gas = QPushButton("Browse Gas File")
        btn_gas.clicked.connect(self.load_gas_file)
        btn_gas_clear = QPushButton("Refresh")
        btn_gas_clear.clicked.connect(self.clear_gas_file)

        layout.addWidget(QLabel("Sample Name:"), 0, 0)
        layout.addWidget(self.sample_edit, 0, 1, 1, 3)
        layout.addWidget(QLabel("Current File:"), 1, 0)
        layout.addWidget(self.current_path_edit, 1, 1)
        layout.addWidget(btn_current, 1, 2)
        layout.addWidget(btn_current_clear, 1, 3)
        layout.addWidget(QLabel("Gas File / GC File:"), 2, 0)
        layout.addWidget(self.gas_path_edit, 2, 1)
        layout.addWidget(btn_gas, 2, 2)
        layout.addWidget(btn_gas_clear, 2, 3)
        layout.addWidget(QLabel("Notes:"), 3, 0)
        layout.addWidget(self.notes_edit, 3, 1, 1, 3)
        return box

    def build_charge_group(self) -> QGroupBox:
        box = QGroupBox("Charge / Current Settings")
        layout = QGridLayout(box)

        self.charge_source_combo = QComboBox()
        self.charge_source_combo.addItems([
            "Integrate current file",
            "Manual charge (Q)",
            "Manual current + time",
        ])
        self.charge_source_combo.currentTextChanged.connect(self.update_ui_states)

        self.current_time_col_combo = QComboBox()
        self.current_value_col_combo = QComboBox()
        self.current_time_unit_combo = QComboBox()
        self.current_time_unit_combo.addItems(["s", "min", "h"])
        self.current_unit_combo = QComboBox()
        self.current_unit_combo.addItems(["A", "mA", "uA"])
        self.current_unit_combo.setCurrentText("A")
        self.integration_mode_combo = QComboBox()
        self.integration_mode_combo.addItems(["absolute", "cathodic_only", "anodic_only", "signed"])
        self.integration_mode_combo.setCurrentText("absolute")

        self.manual_charge_spin = QDoubleSpinBox()
        self.manual_charge_spin.setDecimals(10)
        self.manual_charge_spin.setMaximum(1e12)
        self.manual_charge_spin.setValue(100.0)
        self.manual_charge_spin.setSuffix(" C")

        self.manual_current_spin = QDoubleSpinBox()
        self.manual_current_spin.setDecimals(10)
        self.manual_current_spin.setMaximum(1e12)
        self.manual_current_spin.setValue(0.01)
        self.manual_current_spin.setSuffix(" A")

        self.manual_time_spin = QDoubleSpinBox()
        self.manual_time_spin.setDecimals(6)
        self.manual_time_spin.setMaximum(1e12)
        self.manual_time_spin.setValue(3600.0)
        self.manual_time_spin.setSuffix(" s")

        layout.addWidget(QLabel("Charge Source:"), 0, 0)
        layout.addWidget(self.charge_source_combo, 0, 1, 1, 3)
        layout.addWidget(QLabel("Current Time Column:"), 1, 0)
        layout.addWidget(self.current_time_col_combo, 1, 1)
        layout.addWidget(QLabel("Time Unit:"), 1, 2)
        layout.addWidget(self.current_time_unit_combo, 1, 3)
        layout.addWidget(QLabel("Current Column:"), 2, 0)
        layout.addWidget(self.current_value_col_combo, 2, 1)
        layout.addWidget(QLabel("Current Unit:"), 2, 2)
        layout.addWidget(self.current_unit_combo, 2, 3)
        layout.addWidget(QLabel("Integration Mode:"), 3, 0)
        layout.addWidget(self.integration_mode_combo, 3, 1, 1, 3)
        layout.addWidget(QLabel("Manual Charge:"), 4, 0)
        layout.addWidget(self.manual_charge_spin, 4, 1)
        layout.addWidget(QLabel("Manual Current:"), 5, 0)
        layout.addWidget(self.manual_current_spin, 5, 1)
        layout.addWidget(QLabel("Manual Time:"), 5, 2)
        layout.addWidget(self.manual_time_spin, 5, 3)
        return box

    def build_product_group(self) -> QGroupBox:
        box = QGroupBox("Product / Gas Settings")
        layout = QGridLayout(box)

        self.product_combo = QComboBox()
        self.product_combo.addItems(["H2", "O2", "Custom"])
        self.product_combo.currentTextChanged.connect(self.on_product_changed)

        self.n_spin = QDoubleSpinBox()
        self.n_spin.setDecimals(4)
        self.n_spin.setMaximum(1000)
        self.n_spin.setValue(2.0)

        self.gas_mode_combo = QComboBox()
        self.gas_mode_combo.addItems([
            "Manual product amount",
            "Cumulative gas file",
            "Instantaneous rate gas file",
            "Instantaneous GC peak-area file",
        ])
        self.gas_mode_combo.currentTextChanged.connect(self.update_ui_states)

        self.manual_moles_spin = QDoubleSpinBox()
        self.manual_moles_spin.setDecimals(14)
        self.manual_moles_spin.setMaximum(1e12)
        self.manual_moles_spin.setValue(1e-6)
        self.manual_moles_spin.setSuffix(" mol")

        self.gas_time_col_combo = QComboBox()
        self.gas_value_col_combo = QComboBox()
        self.gas_time_unit_combo = QComboBox()
        self.gas_time_unit_combo.addItems(["s", "min", "h"])

        self.gas_value_unit_combo = QComboBox()
        self.gas_value_unit_combo.addItems([
            "mol", "mmol", "umol", "nmol",
            "mol/s", "mmol/s", "umol/s", "nmol/s",
            "mol/min", "mmol/min", "umol/min", "nmol/min",
            "mol/h", "mmol/h", "umol/h", "nmol/h",
            "peak_area"
        ])
        self.gas_value_unit_combo.setCurrentText("umol")

        self.peak_factor_spin = QDoubleSpinBox()
        self.peak_factor_spin.setDecimals(16)
        self.peak_factor_spin.setMaximum(1e12)
        self.peak_factor_spin.setValue(1e-12)
        self.peak_factor_spin.setSuffix(" mol/area")

        layout.addWidget(QLabel("Product:"), 0, 0)
        layout.addWidget(self.product_combo, 0, 1)
        layout.addWidget(QLabel("n (electrons):"), 0, 2)
        layout.addWidget(self.n_spin, 0, 3)

        layout.addWidget(QLabel("Gas Data Mode:"), 1, 0)
        layout.addWidget(self.gas_mode_combo, 1, 1, 1, 3)

        layout.addWidget(QLabel("Manual Product Amount:"), 2, 0)
        layout.addWidget(self.manual_moles_spin, 2, 1)

        layout.addWidget(QLabel("Gas Time Column:"), 3, 0)
        layout.addWidget(self.gas_time_col_combo, 3, 1)
        layout.addWidget(QLabel("Time Unit:"), 3, 2)
        layout.addWidget(self.gas_time_unit_combo, 3, 3)

        layout.addWidget(QLabel("Gas Value Column:"), 4, 0)
        layout.addWidget(self.gas_value_col_combo, 4, 1)
        layout.addWidget(QLabel("Value Unit / Type:"), 4, 2)
        layout.addWidget(self.gas_value_unit_combo, 4, 3)

        layout.addWidget(QLabel("Peak Area → Mol Factor:"), 5, 0)
        layout.addWidget(self.peak_factor_spin, 5, 1)
        return box

    def build_action_group(self) -> QGroupBox:
        box = QGroupBox("Actions")
        layout = QHBoxLayout(box)
        self.calc_btn = QPushButton("Calculate FE")
        self.calc_btn.clicked.connect(self.calculate_fe)
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

    def on_product_changed(self):
        product = self.product_combo.currentText()
        if product == "H2":
            self.n_spin.setValue(2.0)
        elif product == "O2":
            self.n_spin.setValue(4.0)

    def update_ui_states(self):
        charge_source = self.charge_source_combo.currentText()
        use_current_file = charge_source == "Integrate current file"
        use_manual_q = charge_source == "Manual charge (Q)"
        use_manual_it = charge_source == "Manual current + time"

        for w in [
            self.current_time_col_combo, self.current_value_col_combo,
            self.current_time_unit_combo, self.current_unit_combo,
            self.integration_mode_combo,
        ]:
            w.setEnabled(use_current_file)

        self.manual_charge_spin.setEnabled(use_manual_q)
        self.manual_current_spin.setEnabled(use_manual_it)
        self.manual_time_spin.setEnabled(use_manual_it)

        gas_mode = self.gas_mode_combo.currentText()
        use_manual_amount = gas_mode == "Manual product amount"
        use_file_modes = gas_mode != "Manual product amount"
        use_peak_mode = gas_mode == "Instantaneous GC peak-area file"

        self.manual_moles_spin.setEnabled(use_manual_amount)
        for w in [self.gas_time_col_combo, self.gas_value_col_combo, self.gas_time_unit_combo, self.gas_value_unit_combo]:
            w.setEnabled(use_file_modes)
        self.peak_factor_spin.setEnabled(use_peak_mode)

        if gas_mode == "Manual product amount":
            self.gas_value_unit_combo.setCurrentText("umol")
        elif gas_mode == "Cumulative gas file":
            if self.gas_value_unit_combo.currentText() not in ["mol", "mmol", "umol", "nmol"]:
                self.gas_value_unit_combo.setCurrentText("umol")
        elif gas_mode == "Instantaneous rate gas file":
            if self.gas_value_unit_combo.currentText() not in [
                "mol/s", "mmol/s", "umol/s", "nmol/s",
                "mol/min", "mmol/min", "umol/min", "nmol/min",
                "mol/h", "mmol/h", "umol/h", "nmol/h"
            ]:
                self.gas_value_unit_combo.setCurrentText("umol/min")
        elif gas_mode == "Instantaneous GC peak-area file":
            self.gas_value_unit_combo.setCurrentText("peak_area")

    def clear_current_file(self):
        self.current_df = None
        self.current_path_edit.clear()
        self.current_time_col_combo.clear()
        self.current_value_col_combo.clear()
        self.current_plot.clear()
        self.log("Current file cleared.")

    def clear_gas_file(self):
        self.gas_df = None
        self.gas_path_edit.clear()
        self.gas_time_col_combo.clear()
        self.gas_value_col_combo.clear()
        self.gas_plot.clear()
        self.log("Gas file cleared.")

    def load_current_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Current File", "", "Data Files (*.csv *.txt *.dat *.xlsx *.xls)")
        if not path:
            return
        try:
            self.current_df = DataUtils.read_table(path)
            self.current_path_edit.setText(path)
            cols = list(self.current_df.columns)
            self.current_time_col_combo.clear()
            self.current_value_col_combo.clear()
            self.current_time_col_combo.addItems(cols)
            self.current_value_col_combo.addItems(cols)
            if len(cols) >= 2:
                self.current_time_col_combo.setCurrentIndex(0)
                self.current_value_col_combo.setCurrentIndex(1)
            self.preview_current()
            self.log(f"Loaded current file: {Path(path).name}")
        except Exception as e:
            self.show_error(f"Failed to load current file:\n{e}")

    def load_gas_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Gas File", "", "Data Files (*.csv *.txt *.dat *.xlsx *.xls)")
        if not path:
            return
        try:
            self.gas_df = DataUtils.read_table(path)
            self.gas_path_edit.setText(path)
            cols = list(self.gas_df.columns)
            self.gas_time_col_combo.clear()
            self.gas_value_col_combo.clear()
            self.gas_time_col_combo.addItems(cols)
            self.gas_value_col_combo.addItems(cols)
            if len(cols) >= 2:
                self.gas_time_col_combo.setCurrentIndex(0)
                self.gas_value_col_combo.setCurrentIndex(1)
            self.preview_gas()
            self.log(f"Loaded gas file: {Path(path).name}")
        except Exception as e:
            self.show_error(f"Failed to load gas file:\n{e}")

    def preview_current(self):
        if self.current_df is None:
            return
        try:
            x_col = self.current_time_col_combo.currentText()
            y_col = self.current_value_col_combo.currentText()
            df = DataUtils.clean_xy(self.current_df, x_col, y_col)
            self.current_plot.ax.clear()
            self.current_plot.ax.plot(df["x"], df["y"], linewidth=1.8)
            self.current_plot.ax.set_title("Current Data Preview")
            self.current_plot.ax.set_xlabel(x_col)
            self.current_plot.ax.set_ylabel(y_col)
            self.current_plot.ax.grid(True, alpha=0.3)
            self.current_plot.figure.tight_layout()
            self.current_plot.canvas.draw_idle()
        except Exception as e:
            self.log(f"Current preview warning: {e}")

    def preview_gas(self):
        if self.gas_df is None:
            return
        try:
            x_col = self.gas_time_col_combo.currentText()
            y_col = self.gas_value_col_combo.currentText()
            df = DataUtils.clean_xy(self.gas_df, x_col, y_col)
            self.gas_plot.ax.clear()
            self.gas_plot.ax.plot(df["x"], df["y"], linewidth=1.8)
            self.gas_plot.ax.set_title("Gas / GC Data Preview")
            self.gas_plot.ax.set_xlabel(x_col)
            self.gas_plot.ax.set_ylabel(y_col)
            self.gas_plot.ax.grid(True, alpha=0.3)
            self.gas_plot.figure.tight_layout()
            self.gas_plot.canvas.draw_idle()
        except Exception as e:
            self.log(f"Gas preview warning: {e}")

    def collect_inputs(self) -> FEInputs:
        return FEInputs(
            sample_name=self.sample_edit.text().strip() or "Sample_01",
            product_name=self.product_combo.currentText(),
            electrons_n=float(self.n_spin.value()),
            gas_mode=self.gas_mode_combo.currentText(),
            charge_source=self.charge_source_combo.currentText(),
            manual_charge_c=float(self.manual_charge_spin.value()),
            manual_current_a=float(self.manual_current_spin.value()),
            manual_time_s=float(self.manual_time_spin.value()),
            current_file_path=self.current_path_edit.text().strip(),
            gas_file_path=self.gas_path_edit.text().strip(),
            current_time_column=self.current_time_col_combo.currentText(),
            current_value_column=self.current_value_col_combo.currentText(),
            gas_time_column=self.gas_time_col_combo.currentText(),
            gas_value_column=self.gas_value_col_combo.currentText(),
            current_unit=self.current_unit_combo.currentText(),
            current_time_unit=self.current_time_unit_combo.currentText(),
            gas_value_unit=self.gas_value_unit_combo.currentText(),
            gas_time_unit=self.gas_time_unit_combo.currentText(),
            integration_mode=self.integration_mode_combo.currentText(),
            manual_moles_product=float(self.manual_moles_spin.value()),
            peak_area_to_mol_factor=float(self.peak_factor_spin.value()),
            notes=self.notes_edit.toPlainText().strip(),
        )

    def calculate_fe(self):
        try:
            inp = self.collect_inputs()
            self.last_current_processed = None
            self.last_gas_processed = None
            self.last_gas_plot_label = ("Signal", "Value")

            # Charge part
            if inp.charge_source == "Manual charge (Q)":
                charge_c = inp.manual_charge_c
                total_time_s = None
                self.log("Using manual charge.")
            elif inp.charge_source == "Manual current + time":
                charge_c = inp.manual_current_a * inp.manual_time_s
                total_time_s = inp.manual_time_s
                self.log("Using manual current × time.")
            else:
                if self.current_df is None:
                    raise ValueError("Please load a current file.")
                self.last_current_processed = DataUtils.load_processed_current(
                    self.current_df,
                    inp.current_time_column,
                    inp.current_value_column,
                    inp.current_time_unit,
                    inp.current_unit,
                )
                charge_c = DataUtils.integrate_charge(
                    self.last_current_processed["time_s"].to_numpy(dtype=float),
                    self.last_current_processed["current_a"].to_numpy(dtype=float),
                    inp.integration_mode,
                )
                total_time_s = float(self.last_current_processed["time_s"].iloc[-1] - self.last_current_processed["time_s"].iloc[0])
                self.log("Charge integrated from current file.")

            # Gas/product part
            mol_product, gas_details = self.compute_product_moles(inp)
            fe = FECalculator.fe_percent(inp.electrons_n, mol_product, charge_c)
            theo_mol = charge_c / (inp.electrons_n * FARADAY_CONSTANT)

            computed = {
                "charge_c": float(charge_c),
                "moles_product": float(mol_product),
                "faradaic_efficiency_percent": float(fe),
                "theoretical_moles_from_charge": float(theo_mol),
                "total_time_s": None if total_time_s is None else float(total_time_s),
            }
            computed.update(gas_details)

            self.last_result = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "inputs": asdict(inp),
                "computed": computed,
                "warnings": self.build_warnings(fe),
            }

            self.update_result_views()
            self.update_processed_plots()
            self.log(f"Calculation completed successfully. FE = {fe:.6f}%")
            self.tabs.setCurrentIndex(1)

        except Exception as e:
            self.show_error(f"Calculation failed:\n{e}")
            self.log(f"Calculation failed: {e}")

    def compute_product_moles(self, inp: FEInputs):
        gas_mode = inp.gas_mode

        if gas_mode == "Manual product amount":
            self.log("Using manual product amount.")
            return float(inp.manual_moles_product), {"gas_mode": gas_mode}

        if self.gas_df is None:
            raise ValueError("Please load a gas/GC file.")

        raw = DataUtils.load_processed_gas_generic(
            self.gas_df,
            inp.gas_time_column,
            inp.gas_value_column,
            inp.gas_time_unit,
        )

        if gas_mode == "Cumulative gas file":
            mol_arr = DataUtils.cumulative_amount_to_mol(raw["value"].to_numpy(dtype=float), inp.gas_value_unit)
            processed = pd.DataFrame({
                "time_s": raw["time_s"],
                "cumulative_mol": mol_arr,
            })
            mol_product = float(processed["cumulative_mol"].iloc[-1])
            self.last_gas_processed = processed
            self.last_gas_plot_label = ("Cumulative Product", "Moles (mol)")
            self.log("Using cumulative gas file. Final cumulative product was used.")
            return mol_product, {
                "gas_mode": gas_mode,
                "gas_points": int(len(processed)),
                "gas_final_time_s": float(processed["time_s"].iloc[-1]),
            }

        if gas_mode == "Instantaneous rate gas file":
            rate_mol_s = DataUtils.rate_to_mol_per_s(raw["value"].to_numpy(dtype=float), inp.gas_value_unit)
            if len(rate_mol_s) > 1:
                t_arr = raw["time_s"].to_numpy(dtype=float)
                dt = np.diff(t_arr)
                trap_segments = 0.5 * (rate_mol_s[:-1] + rate_mol_s[1:]) * dt
                cumulative_mol = np.concatenate(([0.0], np.cumsum(trap_segments)))
            else:
                cumulative_mol = np.array([0.0])
            processed = pd.DataFrame({
                "time_s": raw["time_s"],
                "rate_mol_s": rate_mol_s,
                "cumulative_mol": cumulative_mol,
            })
            mol_product = float(np.trapz(processed["rate_mol_s"], processed["time_s"]))
            self.last_gas_processed = processed
            self.last_gas_plot_label = ("Instantaneous Production Rate", "Rate (mol/s)")
            self.log("Using instantaneous rate gas file. Total product obtained by integrating rate over time.")
            return mol_product, {
                "gas_mode": gas_mode,
                "gas_points": int(len(processed)),
                "gas_final_time_s": float(processed["time_s"].iloc[-1]),
                "integrated_from_rate": True,
            }

        if gas_mode == "Instantaneous GC peak-area file":
            if inp.gas_value_unit != "peak_area":
                raise ValueError("For peak-area mode, Value Unit / Type must be 'peak_area'.")
            if inp.peak_area_to_mol_factor <= 0:
                raise ValueError("Peak area to mol factor must be greater than zero.")

            mol_per_event = raw["value"].to_numpy(dtype=float) * inp.peak_area_to_mol_factor
            cumulative_mol = np.cumsum(mol_per_event)
            processed = pd.DataFrame({
                "time_s": raw["time_s"],
                "peak_area": raw["value"],
                "mol_per_event": mol_per_event,
                "cumulative_mol": cumulative_mol,
            })
            mol_product = float(np.sum(processed["mol_per_event"]))
            self.last_gas_processed = processed
            self.last_gas_plot_label = ("GC Instantaneous Peaks", "Moles per event (mol)")
            self.log("Using instantaneous GC peak-area file. Each peak area was converted to mol and summed.")
            return mol_product, {
                "gas_mode": gas_mode,
                "gas_points": int(len(processed)),
                "gas_final_time_s": float(processed["time_s"].iloc[-1]),
                "peak_area_to_mol_factor": float(inp.peak_area_to_mol_factor),
                "summed_peak_events": True,
            }

        raise ValueError("Unsupported gas mode.")

    def build_warnings(self, fe: float):
        warnings = []
        if fe > 100:
            warnings.append("FE > 100%. Check calibration, leaks, gas quantification, or current integration.")
        elif fe > 95:
            warnings.append("Very high FE. Usually good, but verify calibration and baseline.")
        elif fe < 50:
            warnings.append("Low FE. Side reactions or incomplete product quantification may exist.")
        return warnings

    def update_result_views(self):
        if not self.last_result:
            return
        comp = self.last_result["computed"]
        inp = self.last_result["inputs"]
        warnings = self.last_result["warnings"]

        lines = [
            f"Sample: {inp['sample_name']}",
            f"Product: {inp['product_name']}",
            f"Gas Mode: {inp['gas_mode']}",
            f"n (electrons): {inp['electrons_n']}",
            f"Charge (C): {comp['charge_c']:.12g}",
            f"Product amount (mol): {comp['moles_product']:.12g}",
            f"Theoretical moles from charge: {comp['theoretical_moles_from_charge']:.12g}",
            f"Faradaic Efficiency (%): {comp['faradaic_efficiency_percent']:.6f}",
        ]
        if comp.get("total_time_s") is not None:
            lines.append(f"Total time (s): {comp['total_time_s']:.6f}")
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

    def update_processed_plots(self):
        if self.last_current_processed is not None:
            self.current_plot.ax.clear()
            self.current_plot.ax.plot(self.last_current_processed["time_s"], self.last_current_processed["current_a"], linewidth=1.8)
            self.current_plot.ax.set_title("Processed Current Data")
            self.current_plot.ax.set_xlabel("Time (s)")
            self.current_plot.ax.set_ylabel("Current (A)")
            self.current_plot.ax.grid(True, alpha=0.3)
            self.current_plot.figure.tight_layout()
            self.current_plot.canvas.draw_idle()

        if self.last_gas_processed is not None:
            self.gas_plot.ax.clear()
            title, ylabel = self.last_gas_plot_label
            if "rate_mol_s" in self.last_gas_processed.columns:
                self.gas_plot.ax.plot(self.last_gas_processed["time_s"], self.last_gas_processed["rate_mol_s"], linewidth=1.8)
            elif "mol_per_event" in self.last_gas_processed.columns:
                self.gas_plot.ax.plot(self.last_gas_processed["time_s"], self.last_gas_processed["mol_per_event"], marker='o', linewidth=1.2)
            elif "cumulative_mol" in self.last_gas_processed.columns:
                self.gas_plot.ax.plot(self.last_gas_processed["time_s"], self.last_gas_processed["cumulative_mol"], linewidth=1.8)
            else:
                self.gas_plot.ax.plot(self.last_gas_processed["time_s"], self.last_gas_processed.iloc[:, 1], linewidth=1.8)
            self.gas_plot.ax.set_title(title)
            self.gas_plot.ax.set_xlabel("Time (s)")
            self.gas_plot.ax.set_ylabel(ylabel)
            self.gas_plot.ax.grid(True, alpha=0.3)
            self.gas_plot.figure.tight_layout()
            self.gas_plot.canvas.draw_idle()

        if self.last_result is not None:
            fe = self.last_result["computed"]["faradaic_efficiency_percent"]
            self.fe_plot.ax.clear()
            self.fe_plot.ax.bar(["FE (%)"], [fe])
            self.fe_plot.ax.axhline(100, linestyle="--", linewidth=1)
            self.fe_plot.ax.set_title("Faradaic Efficiency")
            self.fe_plot.ax.set_ylabel("Efficiency (%)")
            self.fe_plot.ax.grid(True, axis="y", alpha=0.3)
            self.fe_plot.figure.tight_layout()
            self.fe_plot.canvas.draw_idle()

    def clear_results(self):
        self.last_result = None
        self.last_current_processed = None
        self.last_gas_processed = None
        self.result_summary.clear()
        self.result_table.setRowCount(0)
        self.current_plot.clear()
        self.gas_plot.clear()
        self.fe_plot.clear()
        self.log("Results cleared.")

    def save_results(self):
        if not self.last_result:
            self.show_error("No results to save. Please calculate FE first.")
            return
        try:
            sample_name = self.last_result["inputs"]["sample_name"]
            run_dir = OutputManager.create_run_folder(sample_name)

            with open(run_dir / "json" / "fe_summary.json", "w", encoding="utf-8") as f:
                json.dump(self.last_result, f, indent=2, ensure_ascii=False)

            comp_df = pd.DataFrame(list(self.last_result["computed"].items()), columns=["Parameter", "Value"])
            comp_df.to_csv(run_dir / "csv" / "fe_results.csv", index=False)

            if self.current_df is not None:
                self.current_df.to_csv(run_dir / "csv" / "raw_current.csv", index=False)
            if self.gas_df is not None:
                self.gas_df.to_csv(run_dir / "csv" / "raw_gas.csv", index=False)
            if self.last_current_processed is not None:
                self.last_current_processed.to_csv(run_dir / "csv" / "processed_current.csv", index=False)
            if self.last_gas_processed is not None:
                self.last_gas_processed.to_csv(run_dir / "csv" / "processed_gas.csv", index=False)

            self.current_plot.figure.savefig(run_dir / "png" / "current_plot.png", dpi=300, bbox_inches="tight")
            self.gas_plot.figure.savefig(run_dir / "png" / "gas_plot.png", dpi=300, bbox_inches="tight")
            self.fe_plot.figure.savefig(run_dir / "png" / "fe_plot.png", dpi=300, bbox_inches="tight")

            txt_path = run_dir / "report" / "fe_report.txt"
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

            self.create_pdf_report(run_dir / "report" / "fe_report.pdf")

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

    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "researcher.analysis.toolkit.fe"
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("Researcher Analysis Toolkit")

    for name in ["Researcherlogo.ico", "Researcherlogo.ico"]:
        path = resource_path(name)
        if os.path.exists(path):
            app.setWindowIcon(QIcon(path))
            break

    try:
        window = FEWindow()
        window.show()
        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()
        QMessageBox.critical(None, "Fatal Error", traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
