
import sys
from dataclasses import dataclass
from typing import Optional, List

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDesktopServices, QFont
from PyQt6.QtCore import QUrl
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


@dataclass
class ReferenceScale:
    key: str
    label: str
    base_vs_nhe: Optional[float] = None
    is_rhe: bool = False
    notes: str = ""

    def vs_nhe(self, ph: float) -> float:
        if self.is_rhe:
            return -0.05916 * ph
        return float(self.base_vs_nhe if self.base_vs_nhe is not None else 0.0)


REFERENCE_SCALES: List[ReferenceScale] = [
    ReferenceScale("NHE", "NHE / SHE", 0.000, notes="Normal or standard hydrogen electrode."),
    ReferenceScale("RHE", "RHE", is_rhe=True, notes="Depends on pH at 25 °C."),
    ReferenceScale("AgAgCl_sat", "Ag/AgCl (sat. KCl)", 0.197, notes="Typical value at 25 °C."),
    ReferenceScale("AgAgCl_3M", "Ag/AgCl (3 M KCl)", 0.210, notes="Typical value at 25 °C."),
    ReferenceScale("AgAgCl_1M", "Ag/AgCl (1 M KCl)", 0.235, notes="Typical value at 25 °C."),
    ReferenceScale("SCE", "SCE", 0.244, notes="Saturated calomel electrode."),
    ReferenceScale("HgHgO_1M", "Hg/HgO (1 M NaOH)", 0.098, notes="Alkaline reference."),
    ReferenceScale("HgHgO_0_1M", "Hg/HgO (0.1 M NaOH)", 0.165, notes="Typical practical value."),
    ReferenceScale("HgHg2SO4_sat", "Hg/Hg2SO4 (sat. K2SO4)", 0.640, notes="Mercurous sulfate electrode."),
    ReferenceScale("HgHg2SO4_1M", "Hg/Hg2SO4 (1 M H2SO4)", 0.680, notes="Typical practical value."),
    ReferenceScale("MSE", "MSE (Mercurous sulfate)", 0.650, notes="Approximate practical value."),
    ReferenceScale("CuCuSO4_sat", "Cu/CuSO4 (sat.)", 0.316, notes="Often used in corrosion studies."),
    ReferenceScale("LiLi+", "Li/Li+", -3.040, notes="Typical battery reference value vs NHE."),
    ReferenceScale("NaNa+", "Na/Na+", -2.710, notes="Typical battery reference value vs NHE."),
    ReferenceScale("KK+", "K/K+", -2.931, notes="Typical battery reference value vs NHE."),
    ReferenceScale("FcFc+", "Fc/Fc+", 0.400, notes="Common approximate value vs NHE."),
]

AUTHOR_NAME = "Milad Behroozi"
GOOGLE_SCHOLAR_URL = "https://scholar.google.com/citations?user=ZNF36DQAAAAJ&hl=en"
GITHUB_URL = "https://github.com/Milad-Behroozi"


class Card(QFrame):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title_label = QLabel(title)
        title_label.setObjectName("cardTitle")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        self.body = layout


class ScrollColumn(QWidget):
    def __init__(self):
        super().__init__()
        self.layout_main = QVBoxLayout(self)
        self.layout_main.setContentsMargins(0, 0, 0, 0)
        self.layout_main.setSpacing(0)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.layout_main.addWidget(self.scroll)

        self.content = QWidget()
        self.scroll.setWidget(self.content)

        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(16)


class PotentialConverterPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Electrochemical Potential Converter Pro")
        self.resize(1400, 920)
        self.setMinimumSize(1050, 700)

        self.references = {ref.key: ref for ref in REFERENCE_SCALES}
        self.df: Optional[pd.DataFrame] = None
        self.current_file: Optional[str] = None

        root = QWidget()
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(20, 16, 20, 20)
        outer.setSpacing(16)

        title = QLabel("Electrochemical Potential Converter Pro")
        title.setObjectName("mainTitle")
        subtitle = QLabel(
            "Convert between electrochemical reference scales, calculate the exact correction for raw data, "
            "show the conversion formula, and batch-convert CSV or Excel files."
        )
        subtitle.setObjectName("subTitle")
        subtitle.setWordWrap(True)

        outer.addWidget(title)
        outer.addWidget(subtitle)
        #outer.addWidget(self.build_footer_bar())

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter, 1)

        self.left_panel = ScrollColumn()
        self.right_panel = ScrollColumn()
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([560, 760])

        self.build_left_panel()
        self.build_right_panel()
        self.apply_styles()
        self.update_results()

    def build_footer_bar(self) -> QWidget:
        footer = QWidget()
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(10)

        self.author_label = QLabel(f"Developer: {AUTHOR_NAME}")
        self.author_label.setObjectName("footerLabel")

        self.scholar_link = QLabel(f'<a href="{GOOGLE_SCHOLAR_URL}">Google Scholar</a>')
        self.scholar_link.setObjectName("footerLink")
        self.scholar_link.setTextFormat(Qt.TextFormat.RichText)
        self.scholar_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.scholar_link.setOpenExternalLinks(True)

        self.github_link = QLabel(f'<a href="{GITHUB_URL}">GitHub</a>')
        self.github_link.setObjectName("footerLink")
        self.github_link.setTextFormat(Qt.TextFormat.RichText)
        self.github_link.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        self.github_link.setOpenExternalLinks(True)

        footer_layout.addWidget(self.author_label, 0, Qt.AlignmentFlag.AlignLeft)
        footer_layout.addSpacing(8)
        footer_layout.addWidget(self.scholar_link, 0, Qt.AlignmentFlag.AlignLeft)
        footer_layout.addWidget(self.github_link, 0, Qt.AlignmentFlag.AlignLeft)
        footer_layout.addStretch(1)
        return footer

    def build_left_panel(self):
        left = self.left_panel.content_layout

        setup_card = Card("Single Conversion")
        left.addWidget(setup_card)

        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        grid.setColumnStretch(1, 1)
        setup_card.body.addLayout(grid)

        self.source_combo = QComboBox()
        self.target_combo = QComboBox()
        for ref in REFERENCE_SCALES:
            self.source_combo.addItem(ref.label, ref.key)
            self.target_combo.addItem(ref.label, ref.key)

        self.source_combo.setCurrentText("Ag/AgCl (sat. KCl)")
        self.target_combo.setCurrentText("RHE")


        self.potential_input = QDoubleSpinBox()
        self.potential_input.setRange(-10000.0, 10000.0)
        self.potential_input.setDecimals(6)
        self.potential_input.setValue(0.0)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["V", "mV"])

        self.ph_input = QDoubleSpinBox()
        self.ph_input.setRange(0.0, 14.0)
        self.ph_input.setDecimals(2)
        self.ph_input.setValue(7.00)

        self.temp_input = QDoubleSpinBox()
        self.temp_input.setRange(-20.0, 200.0)
        self.temp_input.setDecimals(1)
        self.temp_input.setValue(25.0)
        self.temp_input.setSuffix(" °C")

        self.include_temp_note = QCheckBox("Show temperature note in formula panel")
        self.include_temp_note.setChecked(True)

        self.note_input = QLineEdit()
        self.note_input.setPlaceholderText("Example: 1 M KOH, dark, Pt counter, scan rate 10 mV/s")

        grid.addWidget(QLabel("Source reference"), 0, 0)
        grid.addWidget(self.source_combo, 0, 1)

        grid.addWidget(QLabel("Target reference"), 1, 0)
        grid.addWidget(self.target_combo, 1, 1)

        grid.addWidget(QLabel("Measured potential"), 2, 0)
        grid.addWidget(self.potential_input, 2, 1)

        grid.addWidget(QLabel("Potential unit"), 3, 0)
        grid.addWidget(self.unit_combo, 3, 1)

        grid.addWidget(QLabel("pH"), 4, 0)
        grid.addWidget(self.ph_input, 4, 1)

        grid.addWidget(QLabel("Temperature"), 5, 0)
        grid.addWidget(self.temp_input, 5, 1)

        grid.addWidget(QLabel("Optional note"), 6, 0)
        grid.addWidget(self.note_input, 6, 1)

        setup_card.body.addWidget(self.include_temp_note)

        btn_row = QHBoxLayout()
        self.convert_btn = QPushButton("Convert")
        self.swap_btn = QPushButton("Swap")
        self.reset_btn = QPushButton("Reset")
        btn_row.addWidget(self.convert_btn)
        btn_row.addWidget(self.swap_btn)
        btn_row.addWidget(self.reset_btn)
        setup_card.body.addLayout(btn_row)

        notes_card = Card("Reference Electrode Notes")
        left.addWidget(notes_card)

        self.reference_note_box = QTextEdit()
        self.reference_note_box.setReadOnly(True)
        self.reference_note_box.setMinimumHeight(240)
        notes_card.body.addWidget(self.reference_note_box)

        left.addStretch(1)

        self.convert_btn.clicked.connect(self.update_results)
        self.swap_btn.clicked.connect(self.swap_references)
        self.reset_btn.clicked.connect(self.reset_fields)

        for w in [
            self.source_combo,
            self.target_combo,
            self.unit_combo,
            self.ph_input,
            self.temp_input,
            self.potential_input,
        ]:
            if isinstance(w, QComboBox):
                w.currentIndexChanged.connect(self.update_results)
            else:
                w.valueChanged.connect(self.update_results)
        self.note_input.textChanged.connect(self.update_results)
        self.include_temp_note.stateChanged.connect(self.update_results)

    def build_right_panel(self):
        right = self.right_panel.content_layout

        result_card = Card("Result, Formula, and Data Correction")
        right.addWidget(result_card)

        self.result_value = QLabel("0.000000 V")
        self.result_value.setObjectName("resultValue")
        self.result_value.setWordWrap(True)
        result_card.body.addWidget(self.result_value)

        self.result_summary = QLabel("")
        self.result_summary.setObjectName("resultSummary")
        self.result_summary.setWordWrap(True)
        result_card.body.addWidget(self.result_summary)

        self.correction_box = QTextEdit()
        self.correction_box.setReadOnly(True)
        self.correction_box.setMinimumHeight(140)
        result_card.body.addWidget(self.correction_box)

        self.formula_box = QTextEdit()
        self.formula_box.setReadOnly(True)
        self.formula_box.setMinimumHeight(260)
        result_card.body.addWidget(self.formula_box)

        file_card = Card("Batch Conversion for Excel / CSV")
        right.addWidget(file_card)

        top_buttons = QHBoxLayout()
        top_buttons.setSpacing(10)
        self.load_btn = QPushButton("Load File")
        self.preview_btn = QPushButton("Preview Conversion")
        self.export_btn = QPushButton("Export Converted File")
        top_buttons.addWidget(self.load_btn)
        top_buttons.addWidget(self.preview_btn)
        top_buttons.addWidget(self.export_btn)
        file_card.body.addLayout(top_buttons)

        file_grid = QGridLayout()
        file_grid.setHorizontalSpacing(14)
        file_grid.setVerticalSpacing(12)
        file_grid.setColumnStretch(1, 1)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)

        self.column_combo = QComboBox()
        self.output_name_input = QLineEdit()
        self.output_name_input.setPlaceholderText("Example: converted_potentials.xlsx")
        self.new_column_input = QLineEdit()
        self.new_column_input.setText("Converted_Potential")
        self.keep_original_check = QCheckBox("Keep original potential column")
        self.keep_original_check.setChecked(True)
        self.overwrite_check = QCheckBox("Overwrite selected column instead of creating a new one")
        self.overwrite_check.setChecked(False)

        file_grid.addWidget(QLabel("Loaded file"), 0, 0)
        file_grid.addWidget(self.file_label, 0, 1)
        file_grid.addWidget(QLabel("Potential column"), 1, 0)
        file_grid.addWidget(self.column_combo, 1, 1)
        file_grid.addWidget(QLabel("New column name"), 2, 0)
        file_grid.addWidget(self.new_column_input, 2, 1)
        file_grid.addWidget(QLabel("Output file name"), 3, 0)
        file_grid.addWidget(self.output_name_input, 3, 1)

        file_card.body.addLayout(file_grid)
        file_card.body.addWidget(self.keep_original_check)
        file_card.body.addWidget(self.overwrite_check)

        self.table = QTableWidget()
        self.table.setObjectName("previewTable")
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setMinimumHeight(320)
        self.table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        file_card.body.addWidget(self.table)

        right.addStretch(1)

        self.load_btn.clicked.connect(self.load_file)
        self.preview_btn.clicked.connect(self.preview_conversion)
        self.export_btn.clicked.connect(self.export_conversion)

    def get_reference(self, combo: QComboBox) -> ReferenceScale:
        key = combo.currentData()
        return self.references[key]

    def calculate_offset(self, ph: float) -> float:
        source = self.get_reference(self.source_combo)
        target = self.get_reference(self.target_combo)
        return source.vs_nhe(ph) - target.vs_nhe(ph)

    def get_input_potential_volts(self) -> float:
        value = self.potential_input.value()
        return value / 1000.0 if self.unit_combo.currentText() == "mV" else value

    def format_potential(self, value_v: float) -> str:
        if self.unit_combo.currentText() == "mV":
            return f"{value_v * 1000:.3f} mV"
        return f"{value_v:.6f} V"

    def swap_references(self):
        source_idx = self.source_combo.currentIndex()
        target_idx = self.target_combo.currentIndex()
        self.source_combo.setCurrentIndex(target_idx)
        self.target_combo.setCurrentIndex(source_idx)

        self.update_results()

    def reset_fields(self):
        self.source_combo.setCurrentText("Ag/AgCl (sat. KCl)")
        self.target_combo.setCurrentText("RHE")
        self.potential_input.setValue(0.0)
        self.unit_combo.setCurrentText("V")
        self.ph_input.setValue(7.0)
        self.temp_input.setValue(25.0)
        self.note_input.clear()
        self.include_temp_note.setChecked(True)
        self.update_results()

    def update_results(self):
        try:
            source = self.get_reference(self.source_combo)
            target = self.get_reference(self.target_combo)
            ph = self.ph_input.value()
            temp = self.temp_input.value()
            input_v = self.get_input_potential_volts()
            offset = self.calculate_offset(ph)
            output_v = input_v + offset

            self.result_value.setText(self.format_potential(output_v))
            self.result_summary.setText(
                f"{source.label} → {target.label} | pH = {ph:.2f} | correction = {offset:+.6f} V"
            )

            self.correction_box.setPlainText(self.build_correction_text(input_v, output_v, offset))
            self.formula_box.setPlainText(self.build_formula_text(source, target, ph, temp, input_v, output_v, offset))
            self.reference_note_box.setPlainText(self.build_reference_notes(source, target))
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{exc}")

    def build_correction_text(self, input_v: float, output_v: float, offset: float) -> str:
        lines = ["What should be done to the raw potential data?\n\n"]
        if abs(offset) < 1e-12:
            lines.append("No correction is needed.\n")
            lines.append("E_target = E_source\n\n")
        elif offset > 0:
            lines.append(f"Add {offset:.6f} V to every value in the raw potential data.\n")
            lines.append(f"If your source file uses mV, add {offset * 1000:.3f} mV to each value.\n\n")
        else:
            lines.append(f"Subtract {abs(offset):.6f} V from every value in the raw potential data.\n")
            lines.append(f"If your source file uses mV, subtract {abs(offset) * 1000:.3f} mV from each value.\n\n")

        lines.append("Worked example\n")
        lines.append(f"Input value = {input_v:.6f} V\n")
        lines.append(f"Converted value = {output_v:.6f} V\n")
        return "".join(lines)

    def build_formula_text(self, source, target, ph, temp, input_v, output_v, offset) -> str:
        source_val = source.vs_nhe(ph)
        target_val = target.vs_nhe(ph)

        lines = []
        lines.append("General conversion formula\n")
        lines.append("E_target = E_source + E°(source vs NHE) - E°(target vs NHE)\n\n")
        lines.append("For RHE at 25 °C\n")
        lines.append("E°(RHE vs NHE) = -0.05916 × pH\n\n")

        if target.is_rhe and not source.is_rhe:
            lines.append(f"Special case\nE_RHE = E_{source.label} + E°({source.label} vs NHE) + 0.05916 × pH\n\n")
        elif source.is_rhe and not target.is_rhe:
            lines.append(f"Special case\nE_{target.label} = E_RHE + (-0.05916 × pH) - E°({target.label} vs NHE)\n\n")

        lines.append("Selected values\n")
        lines.append(f"Source = {source.label}\n")
        lines.append(f"Target = {target.label}\n")
        lines.append(f"pH = {ph:.2f}\n")
        if self.include_temp_note.isChecked():
            lines.append(f"Temperature note = {temp:.1f} °C (coefficient currently fixed to the 25 °C convention)\n")
        lines.append(f"E°(source vs NHE) = {source_val:.6f} V\n")
        lines.append(f"E°(target vs NHE) = {target_val:.6f} V\n")
        lines.append(f"Correction = {source_val:.6f} - {target_val:.6f} = {offset:.6f} V\n\n")

        lines.append("Worked example\n")
        lines.append(f"E_source = {input_v:.6f} V\n")
        lines.append(f"E_target = {input_v:.6f} + ({offset:.6f})\n")
        lines.append(f"E_target = {output_v:.6f} V\n\n")

        note = self.note_input.text().strip()
        if note:
            lines.append(f"User note: {note}\n")
        return "".join(lines)

    def build_reference_notes(self, source, target) -> str:
        return (
            "Reference electrode notes\n\n"
            f"Source: {source.label}\n"
            f"Note: {source.notes or 'No additional note.'}\n\n"
            f"Target: {target.label}\n"
            f"Note: {target.notes or 'No additional note.'}\n\n"
            "Important caution:\n"
            "Built-in values are practical standard values commonly used in electrochemistry. "
            "Real offsets can vary slightly with electrolyte concentration, liquid junction potential, solvent, and temperature. "
            "For publication-quality reporting, always confirm the exact reference conditions used experimentally.\n"
        )

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "Data files (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls)"
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith(".csv"):
                self.df = pd.read_csv(file_path)
            else:
                self.df = pd.read_excel(file_path)

            self.current_file = file_path
            self.file_label.setText(file_path)
            self.column_combo.clear()
            self.column_combo.addItems([str(c) for c in self.df.columns])

            filename = file_path.split("/")[-1].split("\\")[-1]
            self.output_name_input.setText("converted_" + filename)

            self.show_dataframe_preview(self.df.head(40))
        except Exception as exc:
            QMessageBox.critical(self, "File Error", f"Could not load file:\n{exc}")

    def convert_dataframe(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No file loaded.")

        col = self.column_combo.currentText().strip()
        if not col:
            raise ValueError("Please select a potential column.")
        if col not in self.df.columns:
            raise ValueError("Selected column was not found in the file.")

        data = self.df.copy()
        offset = self.calculate_offset(self.ph_input.value())

        source_unit = self.unit_combo.currentText()
        values = pd.to_numeric(data[col], errors="coerce")

        if source_unit == "mV":
            converted = values + (offset * 1000.0)
        else:
            converted = values + offset

        overwrite = self.overwrite_check.isChecked()
        keep_original = self.keep_original_check.isChecked()
        new_col = self.new_column_input.text().strip() or "Converted_Potential"

        if overwrite:
            data[col] = converted
            applied_col = col
        else:
            data[new_col] = converted
            applied_col = new_col
            if not keep_original:
                data = data.drop(columns=[col])

        source = self.get_reference(self.source_combo)
        target = self.get_reference(self.target_combo)

        meta_lines = [
            f"Source reference: {source.label}",
            f"Target reference: {target.label}",
            f"pH: {self.ph_input.value():.2f}",
            f"Temperature note: {self.temp_input.value():.1f} °C",
            f"Correction applied: {offset * 1000.0:.6f} mV" if source_unit == "mV" else f"Correction applied: {offset:.6f} V",
            f"Applied column: {applied_col}",
        ]
        data.attrs["conversion_summary"] = " | ".join(meta_lines)
        return data

    def preview_conversion(self):
        if self.df is None:
            QMessageBox.warning(self, "No file", "Please load a CSV or Excel file first.")
            return
        try:
            converted_df = self.convert_dataframe()
            self.show_dataframe_preview(converted_df.head(50))
        except Exception as exc:
            QMessageBox.critical(self, "Preview Error", str(exc))

    def export_conversion(self):
        if self.df is None:
            QMessageBox.warning(self, "No file", "Please load a CSV or Excel file first.")
            return
        try:
            converted_df = self.convert_dataframe()

            suggested = self.output_name_input.text().strip() or "converted_output.xlsx"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save converted file",
                suggested,
                "Excel (*.xlsx);;CSV (*.csv)"
            )
            if not file_path:
                return

            if file_path.lower().endswith(".csv"):
                converted_df.to_csv(file_path, index=False)
            else:
                with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                    converted_df.to_excel(writer, index=False, sheet_name="converted_data")
                    doi_url = "https://doi.org/10.1039/D6RA08618G"
                    pd.DataFrame({
                        "Conversion Summary": [converted_df.attrs.get("conversion_summary", "")],
                        "Software": ["Researcher Analysis Toolkit v1.0.0"],
                        "Suggested citation": ["Behroozi et al., RSC Adv. (2026)."],
                        "Article DOI": [doi_url],
                    }).to_excel(writer, index=False, sheet_name="metadata")

                    doi_cell = writer.sheets["metadata"]["D2"]
                    doi_cell.hyperlink = doi_url
                    doi_cell.style = "Hyperlink"

            QMessageBox.information(self, "Success", f"Converted file saved successfully:\n{file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", str(exc))

    def show_dataframe_preview(self, df: pd.DataFrame):
        self.table.clear()
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in df.columns])

        for i in range(len(df)):
            for j, col in enumerate(df.columns):
                value = df.iloc[i, j]
                item = QTableWidgetItem("" if pd.isna(value) else str(value))
                self.table.setItem(i, j, item)

        self.table.resizeColumnsToContents()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #0b1020;
                color: #e8edf8;
                font-family: Segoe UI, Arial, sans-serif;
                font-size: 14px;
            }
            QLabel#mainTitle {
                font-size: 28px;
                font-weight: 800;
                color: #ffffff;
            }
            QLabel#subTitle {
                color: #a9b8d8;
                font-size: 14px;
                padding-bottom: 4px;
            }
            QLabel#footerLabel {
                color: #9fb0d4;
                font-size: 12px;
                padding: 2px 0 4px 0;
            }
            QLabel#footerLink {
                color: #8cb4ff;
                font-size: 12px;
                padding: 2px 0 4px 0;
            }
            QLabel#footerLink a {
                color: #8cb4ff;
                text-decoration: none;
            }
            QLabel#footerLink a:hover {
                text-decoration: underline;
            }
            QFrame#card {
                background-color: #121a2f;
                border: 1px solid #253252;
                border-radius: 22px;
            }
            QLabel#cardTitle {
                font-size: 17px;
                font-weight: 700;
                color: #f5f8ff;
                padding-bottom: 4px;
            }
            QLabel#resultValue {
                font-size: 33px;
                font-weight: 800;
                color: #ffffff;
                padding: 4px 0;
            }
            QLabel#resultSummary {
                color: #aebee0;
                padding-bottom: 6px;
            }
            QComboBox, QDoubleSpinBox, QLineEdit, QTextEdit, QTableWidget {
                background-color: #0e1528;
                border: 1px solid #304469;
                border-radius: 14px;
                padding: 9px 11px;
                selection-background-color: #35518e;
            }
            QComboBox:hover, QDoubleSpinBox:hover, QLineEdit:hover, QTextEdit:hover, QTableWidget:hover {
                border: 1px solid #4a6db8;
            }
            QHeaderView::section {
                background-color: #16203a;
                color: #e8edf8;
                border: 1px solid #304469;
                padding: 6px;
            }
            QPushButton {
                background-color: #2d4b88;
                border: none;
                border-radius: 14px;
                padding: 11px 15px;
                font-weight: 700;
                min-height: 22px;
            }
            QPushButton:hover {
                background-color: #3b60aa;
            }
            QPushButton:pressed {
                background-color: #28457a;
            }
            QCheckBox {
                padding: 2px;
            }
            QScrollArea {
                border: none;
                background: transparent;
            }
        """)


def main():
    import sys
    import ctypes
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication

    # This line is very important for the Windows Taskbar
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "researcher.analysis.toolkit"
    )

    app = QApplication(sys.argv)

    app.setApplicationName("Research Analysis Toolkit — Electrochemical Potential Converter")

    app.setWindowIcon(QIcon("Researcherlogo.ico"))

    app.setFont(QFont("Segoe UI", 10))

    window = PotentialConverterPro()

    window.setWindowIcon(QIcon("Researcherlogo.ico"))

    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
