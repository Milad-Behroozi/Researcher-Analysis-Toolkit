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


APP_NAME = "Researcher Analysis Toolkit - IPCE"
OUTPUT_TECHNIQUE = "IPCE Analysis"
APP_VERSION = "1.0.0"
ARTICLE_DOI = "10.1039/D6RA08618G"
ARTICLE_URL = f"https://doi.org/{ARTICLE_DOI}"
ARTICLE_CITATION = f"Suggested citation · Behroozi et al., RSC Adv. (2026). {ARTICLE_DOI}"


def resource_path(filename: str) -> str:
    if getattr(sys, "_MEIPASS", None):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


@dataclass
class IPCEInputs:
    sample_name: str
    mode: str
    wavelength_nm: float
    photocurrent_value: float
    light_power_value: float
    photocurrent_unit: str
    light_power_unit: str
    file_path: str
    wavelength_column: str
    photocurrent_column: str
    power_column: str
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
            if col not in df.columns:
                raise ValueError(f"Column not found: {col}")

        out = df[columns].copy()
        for col in columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        out = out.dropna()
        if out.empty:
            raise ValueError("No valid numeric rows found.")
        return out.reset_index(drop=True)

    @staticmethod
    def convert_photocurrent_to_mA_cm2(values: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "ma/cm2":
            return values
        if unit == "a/cm2":
            return values * 1000.0
        if unit == "ua/cm2":
            return values / 1000.0
        raise ValueError(f"Unsupported photocurrent unit: {unit}")

    @staticmethod
    def convert_power_to_mW_cm2(values: np.ndarray, unit: str) -> np.ndarray:
        unit = unit.lower()
        if unit == "mw/cm2":
            return values
        if unit == "w/cm2":
            return values * 1000.0
        if unit == "uw/cm2":
            return values / 1000.0
        raise ValueError(f"Unsupported light power unit: {unit}")


class IPCECalculator:
    @staticmethod
    def calculate_ipce(wavelength_nm, photocurrent_mA_cm2, power_mW_cm2):
        wavelength_nm = np.asarray(wavelength_nm, dtype=float)
        photocurrent_mA_cm2 = np.asarray(photocurrent_mA_cm2, dtype=float)
        power_mW_cm2 = np.asarray(power_mW_cm2, dtype=float)

        if np.any(wavelength_nm <= 0):
            raise ValueError("Wavelength must be greater than zero.")
        if np.any(power_mW_cm2 <= 0):
            raise ValueError("Light power must be greater than zero.")

        return (1240.0 * photocurrent_mA_cm2 / (wavelength_nm * power_mW_cm2)) * 100.0


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


class IPCEWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.last_result = None

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
        title = QLabel("IPCE Module")
        title.setObjectName("mainTitle")
        subtitle = QLabel("Incident Photon-to-Current Efficiency Analysis")
        subtitle.setObjectName("subTitle")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        left_layout.addWidget(title_frame)

        left_layout.addWidget(self.build_general_group())
        left_layout.addWidget(self.build_manual_group())
        left_layout.addWidget(self.build_file_group())
        left_layout.addWidget(self.build_action_group())
        left_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_container)
        scroll.setMinimumWidth(520)
        scroll.setMaximumWidth(660)

        self.tabs = QTabWidget()

        self.raw_plot = PlotCanvas()
        self.ipce_plot = PlotCanvas()

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_tabs = QTabWidget()
        preview_tabs.addTab(self.raw_plot, "Raw Data")
        preview_tabs.addTab(self.ipce_plot, "IPCE Plot")
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
            "Single-point manual",
            "File-based spectrum",
        ])
        self.mode_combo.currentTextChanged.connect(self.update_ui_states)

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
        layout.addWidget(self.mode_combo, 1, 1, 1, 3)

        layout.addWidget(QLabel("Data File:"), 2, 0)
        layout.addWidget(self.file_path_edit, 2, 1)
        layout.addWidget(btn_file, 2, 2)
        layout.addWidget(btn_file_clear, 2, 3)

        layout.addWidget(QLabel("Notes:"), 3, 0)
        layout.addWidget(self.notes_edit, 3, 1, 1, 3)

        return box

    def build_manual_group(self) -> QGroupBox:
        box = QGroupBox("Single-Point Manual Input")
        layout = QGridLayout(box)

        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setDecimals(4)
        self.wavelength_spin.setMaximum(1e6)
        self.wavelength_spin.setValue(420.0)
        self.wavelength_spin.setSuffix(" nm")

        self.photocurrent_spin = QDoubleSpinBox()
        self.photocurrent_spin.setDecimals(10)
        self.photocurrent_spin.setMaximum(1e12)
        self.photocurrent_spin.setValue(0.10)

        self.photocurrent_unit_combo = QComboBox()
        self.photocurrent_unit_combo.addItems(["mA/cm2", "A/cm2", "uA/cm2"])
        self.photocurrent_unit_combo.setCurrentText("mA/cm2")

        self.power_spin = QDoubleSpinBox()
        self.power_spin.setDecimals(10)
        self.power_spin.setMaximum(1e12)
        self.power_spin.setValue(100.0)

        self.power_unit_combo = QComboBox()
        self.power_unit_combo.addItems(["mW/cm2", "W/cm2", "uW/cm2"])
        self.power_unit_combo.setCurrentText("mW/cm2")

        layout.addWidget(QLabel("Wavelength:"), 0, 0)
        layout.addWidget(self.wavelength_spin, 0, 1)

        layout.addWidget(QLabel("Photocurrent Density:"), 1, 0)
        layout.addWidget(self.photocurrent_spin, 1, 1)
        layout.addWidget(self.photocurrent_unit_combo, 1, 2)

        layout.addWidget(QLabel("Light Power Density:"), 2, 0)
        layout.addWidget(self.power_spin, 2, 1)
        layout.addWidget(self.power_unit_combo, 2, 2)

        return box

    def build_file_group(self) -> QGroupBox:
        box = QGroupBox("File-Based IPCE Spectrum")
        layout = QGridLayout(box)

        self.wavelength_col_combo = QComboBox()
        self.photocurrent_col_combo = QComboBox()
        self.power_col_combo = QComboBox()

        self.file_photocurrent_unit_combo = QComboBox()
        self.file_photocurrent_unit_combo.addItems(["mA/cm2", "A/cm2", "uA/cm2"])
        self.file_photocurrent_unit_combo.setCurrentText("mA/cm2")

        self.file_power_unit_combo = QComboBox()
        self.file_power_unit_combo.addItems(["mW/cm2", "W/cm2", "uW/cm2"])
        self.file_power_unit_combo.setCurrentText("mW/cm2")

        layout.addWidget(QLabel("Wavelength Column:"), 0, 0)
        layout.addWidget(self.wavelength_col_combo, 0, 1)

        layout.addWidget(QLabel("Photocurrent Column:"), 1, 0)
        layout.addWidget(self.photocurrent_col_combo, 1, 1)
        layout.addWidget(self.file_photocurrent_unit_combo, 1, 2)

        layout.addWidget(QLabel("Power Column:"), 2, 0)
        layout.addWidget(self.power_col_combo, 2, 1)
        layout.addWidget(self.file_power_unit_combo, 2, 2)

        return box

    def build_action_group(self) -> QGroupBox:
        box = QGroupBox("Actions")
        layout = QHBoxLayout(box)

        self.calc_btn = QPushButton("Calculate IPCE")
        self.calc_btn.clicked.connect(self.calculate_ipce)

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
        mode = self.mode_combo.currentText()
        manual_mode = mode == "Single-point manual"
        file_mode = mode == "File-based spectrum"

        for w in [
            self.wavelength_spin,
            self.photocurrent_spin,
            self.power_spin,
            self.photocurrent_unit_combo,
            self.power_unit_combo,
        ]:
            w.setEnabled(manual_mode)

        self.file_path_edit.setEnabled(file_mode)
        self.wavelength_col_combo.setEnabled(file_mode)
        self.photocurrent_col_combo.setEnabled(file_mode)
        self.power_col_combo.setEnabled(file_mode)
        self.file_photocurrent_unit_combo.setEnabled(file_mode)
        self.file_power_unit_combo.setEnabled(file_mode)

    def clear_file(self):
        self.raw_df = None
        self.processed_df = None
        self.file_path_edit.clear()
        self.wavelength_col_combo.clear()
        self.photocurrent_col_combo.clear()
        self.power_col_combo.clear()
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
            self.wavelength_col_combo.clear()
            self.photocurrent_col_combo.clear()
            self.power_col_combo.clear()

            self.wavelength_col_combo.addItems(cols)
            self.photocurrent_col_combo.addItems(cols)
            self.power_col_combo.addItems(cols)

            if len(cols) >= 3:
                self.wavelength_col_combo.setCurrentIndex(0)
                self.photocurrent_col_combo.setCurrentIndex(1)
                self.power_col_combo.setCurrentIndex(2)

            self.preview_raw_data()
            self.log(f"Loaded data file: {Path(path).name}")

        except Exception as e:
            self.show_error(f"Failed to load data file:\n{e}")

    def preview_raw_data(self):
        if self.raw_df is None:
            return

        try:
            x_col = self.wavelength_col_combo.currentText() or self.raw_df.columns[0]
            y_col = self.photocurrent_col_combo.currentText() or self.raw_df.columns[1]

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

    def collect_inputs(self) -> IPCEInputs:
        return IPCEInputs(
            sample_name=self.sample_edit.text().strip() or "Sample_01",
            mode=self.mode_combo.currentText(),
            wavelength_nm=float(self.wavelength_spin.value()),
            photocurrent_value=float(self.photocurrent_spin.value()),
            light_power_value=float(self.power_spin.value()),
            photocurrent_unit=self.photocurrent_unit_combo.currentText(),
            light_power_unit=self.power_unit_combo.currentText(),
            file_path=self.file_path_edit.text().strip(),
            wavelength_column=self.wavelength_col_combo.currentText(),
            photocurrent_column=self.photocurrent_col_combo.currentText(),
            power_column=self.power_col_combo.currentText(),
            notes=self.notes_edit.toPlainText().strip(),
        )

    def calculate_ipce(self):
        try:
            inp = self.collect_inputs()

            if inp.mode == "Single-point manual":
                self.calculate_manual_ipce(inp)
            else:
                self.calculate_file_ipce(inp)

            self.tabs.setCurrentIndex(1)

        except Exception as e:
            self.show_error(f"Calculation failed:\n{e}")
            self.log(f"Calculation failed: {e}")

    def calculate_manual_ipce(self, inp: IPCEInputs):
        wavelength = inp.wavelength_nm
        current_mA_cm2 = float(DataUtils.convert_photocurrent_to_mA_cm2(
            np.array([inp.photocurrent_value]), inp.photocurrent_unit
        )[0])
        power_mW_cm2 = float(DataUtils.convert_power_to_mW_cm2(
            np.array([inp.light_power_value]), inp.light_power_unit
        )[0])

        ipce = float(IPCECalculator.calculate_ipce(
            np.array([wavelength]),
            np.array([current_mA_cm2]),
            np.array([power_mW_cm2])
        )[0])

        self.processed_df = pd.DataFrame({
            "Wavelength_nm": [wavelength],
            "Photocurrent_mA_cm2": [current_mA_cm2],
            "Power_mW_cm2": [power_mW_cm2],
            "IPCE_percent": [ipce],
        })

        self.last_result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inputs": asdict(inp),
            "computed": {
                "mode": inp.mode,
                "wavelength_nm": wavelength,
                "photocurrent_mA_cm2": current_mA_cm2,
                "power_mW_cm2": power_mW_cm2,
                "ipce_percent": ipce,
            },
            "warnings": self.build_warnings(np.array([ipce])),
        }

        self.update_result_views()
        self.update_ipce_plot(single_point=True)
        self.log(f"Single-point IPCE calculated: {ipce:.6f}%")

    def calculate_file_ipce(self, inp: IPCEInputs):
        if self.raw_df is None:
            raise ValueError("Please load a data file.")

        df = DataUtils.clean_numeric_columns(
            self.raw_df,
            [inp.wavelength_column, inp.photocurrent_column, inp.power_column]
        )

        wavelength = df[inp.wavelength_column].to_numpy(dtype=float)
        photocurrent = DataUtils.convert_photocurrent_to_mA_cm2(
            df[inp.photocurrent_column].to_numpy(dtype=float),
            self.file_photocurrent_unit_combo.currentText()
        )
        power = DataUtils.convert_power_to_mW_cm2(
            df[inp.power_column].to_numpy(dtype=float),
            self.file_power_unit_combo.currentText()
        )

        ipce = IPCECalculator.calculate_ipce(wavelength, photocurrent, power)

        self.processed_df = pd.DataFrame({
            "Wavelength_nm": wavelength,
            "Photocurrent_mA_cm2": photocurrent,
            "Power_mW_cm2": power,
            "IPCE_percent": ipce,
        }).sort_values("Wavelength_nm").reset_index(drop=True)

        self.last_result = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "inputs": asdict(inp),
            "computed": {
                "mode": inp.mode,
                "points": int(len(self.processed_df)),
                "wavelength_min_nm": float(self.processed_df["Wavelength_nm"].min()),
                "wavelength_max_nm": float(self.processed_df["Wavelength_nm"].max()),
                "ipce_min_percent": float(self.processed_df["IPCE_percent"].min()),
                "ipce_max_percent": float(self.processed_df["IPCE_percent"].max()),
                "ipce_mean_percent": float(self.processed_df["IPCE_percent"].mean()),
            },
            "warnings": self.build_warnings(self.processed_df["IPCE_percent"].to_numpy(dtype=float)),
        }

        self.update_result_views()
        self.update_ipce_plot(single_point=False)
        self.log("File-based IPCE spectrum calculated successfully.")

    def build_warnings(self, ipce_values: np.ndarray):
        warnings = []
        if np.any(ipce_values > 100):
            warnings.append("Some IPCE values are greater than 100%. Check current, power, wavelength, and unit conversions.")
        if np.any(ipce_values < 0):
            warnings.append("Some IPCE values are negative. Check sign convention for photocurrent.")
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

    def update_ipce_plot(self, single_point: bool):
        if self.processed_df is None or self.processed_df.empty:
            return

        self.ipce_plot.ax.clear()

        if single_point:
            self.ipce_plot.ax.bar(
                [self.processed_df["Wavelength_nm"].iloc[0]],
                [self.processed_df["IPCE_percent"].iloc[0]],
                width=10
            )
        else:
            self.ipce_plot.ax.plot(
                self.processed_df["Wavelength_nm"],
                self.processed_df["IPCE_percent"],
                marker="o",
                linewidth=1.8
            )

        self.ipce_plot.ax.set_title("IPCE")
        self.ipce_plot.ax.set_xlabel("Wavelength (nm)")
        self.ipce_plot.ax.set_ylabel("IPCE (%)")
        self.ipce_plot.ax.grid(True, alpha=0.3)
        self.ipce_plot.figure.tight_layout()
        self.ipce_plot.canvas.draw_idle()

    def clear_results(self):
        self.processed_df = None
        self.last_result = None
        self.result_summary.clear()
        self.result_table.setRowCount(0)
        self.ipce_plot.clear()
        self.raw_plot.clear()
        self.log("Results cleared.")

    def save_results(self):
        if not self.last_result:
            self.show_error("No results to save. Please calculate IPCE first.")
            return

        try:
            sample_name = self.last_result["inputs"]["sample_name"]
            run_dir = OutputManager.create_run_folder(sample_name)

            with open(run_dir / "json" / "ipce_summary.json", "w", encoding="utf-8") as f:
                json.dump(self.last_result, f, indent=2, ensure_ascii=False)

            if self.raw_df is not None:
                self.raw_df.to_csv(run_dir / "csv" / "raw_data.csv", index=False)

            if self.processed_df is not None:
                self.processed_df.to_csv(run_dir / "csv" / "processed_ipce.csv", index=False)

            self.raw_plot.figure.savefig(run_dir / "png" / "raw_plot.png", dpi=300, bbox_inches="tight")
            self.ipce_plot.figure.savefig(run_dir / "png" / "ipce_plot.png", dpi=300, bbox_inches="tight")

            txt_path = run_dir / "report" / "ipce_report.txt"
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
                self.create_pdf_report(run_dir / "report" / "ipce_report.pdf")

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

    myappid = "researcher.analysis.toolkit.ipce.v1"
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    app = QApplication(sys.argv)

    icon_path = resource_path("Researcherlogo.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))

    window = IPCEWindow()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
