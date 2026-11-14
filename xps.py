"""Publication-oriented, reference-driven XPS peak fitting application.

The optimizer is element-agnostic: any element, core-level region, oxidation
state, satellite, or multiplet can be described in the editable Model Builder
or imported through the versioned JSON schema ``xps-chemical-model/v2``.
Chemical assignments are deliberately never guessed from binding energy alone;
the model records the reference used for each assignment.

Version 2 center controls are explicit and auditable: Free (global safety
interval), Fixed, Bounded (per-component limits), and Tied (fixed or bounded
offset to another independent component). Legacy v1 model JSON remains
importable and is upgraded during validation.

Version 2.3 adds a simple, element-agnostic Free Fit workflow. Any number of
independent peaks can be added, duplicated, removed, dragged and assigned an
individual line shape in the Interactive Fit Studio. Physics-constrained models
remain available as an optional advanced mode.

Version 2.4 adds an auditable manual background made from draggable anchor
points, keeps the workspace geometry stable while fitting, and strengthens
background/model quality checks. The numerical engine is broadly applicable,
but a chemically meaningful fit still requires a sample-appropriate component
model, acquisition-aware constraints, and reference spectra.

Version 2.5 reorganizes the desktop interface for narrow and high-DPI screens:
sidebar forms use full-width stacked controls, primary actions remain visible
in a fixed footer, hidden rows no longer leave blank labels, and both plot
workspaces use compact titled toolbars and deliberate empty states.

Version 2.6 fixes the interactive-workspace action state after the first fit,
adds an explicitly provisional export for operator-adjusted previews, and makes
both sidebars resizable with a persistent width and a one-click layout reset.

Version 2.6.3 adds a compact Origin-ready CSV beside the complete audit data.
Its component curves are already placed on the final background and its
binding-energy rows are ordered high-to-low to match the conventional XPS view.

Core dependencies: numpy, pandas, scipy/lmfit, matplotlib, PyQt6, openpyxl.
Optional dependency: xraydb (advisory atomic-edge lookup only; never treated as
a chemical-state XPS binding-energy database).
"""

import sys
import json
import copy
import math
import platform
import traceback
import ctypes
import datetime
import base64
import html as html_lib
from io import BytesIO
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import pandas as pd
import lmfit
from scipy.optimize import nnls

# ---------------------------------------------------------
# HYBRID X-RAY & COMPOUND DATABASE INTEGRATION
# ---------------------------------------------------------
try:
    import xraydb
    HAS_XRAYDB = True
except ImportError:
    HAS_XRAYDB = False

SHELL_MAP = {'1': 'K', '2': 'L', '3': 'M', '4': 'N', '5': 'O', '6': 'P'}
ORBITAL_MAP = {
    's': {'edges': ['1'], 'ratio': 1.0, 'doublet': False},
    'p': {'edges': ['2', '3'], 'ratio': 0.5000, 'doublet': True},   
    'd': {'edges': ['4', '5'], 'ratio': 0.6667, 'doublet': True},   
    'f': {'edges': ['6', '7'], 'ratio': 0.7500, 'doublet': True},   
}

# Reference centers are physically informed starting points, not immutable assignments.
# Every assignment must still be checked against calibration, acquisition conditions,
# the residuals, and appropriate reference spectra.
#
# The Ce 3d template uses the conventional 10-component mixed Ce3+/Ce4+ model.
# "primary" denotes the lower-binding-energy 3d5/2 member and "partner" the
# 3d3/2 member.  Equal line shapes and the statistical 3:2 area ratio are
# enforced within each v/u multiplet pair when physics constraints are enabled.
CE3D_MIXED_TEMPLATE = [
    {"label": "Ce3+ v0",   "state": "Ce3+", "center": 880.8, "role": "primary", "pair": "ce3_0", "delta_e": 18.1, "ratio": 2/3},
    {"label": "Ce4+ v",    "state": "Ce4+", "center": 882.4, "role": "primary", "pair": "ce4_0", "delta_e": 18.5, "ratio": 2/3},
    {"label": "Ce3+ v'",   "state": "Ce3+", "center": 885.1, "role": "primary", "pair": "ce3_1", "delta_e": 18.3, "ratio": 2/3},
    {"label": "Ce4+ v''",  "state": "Ce4+", "center": 888.6, "role": "primary", "pair": "ce4_1", "delta_e": 18.5, "ratio": 2/3},
    {"label": "Ce4+ v'''", "state": "Ce4+", "center": 898.2, "role": "primary", "pair": "ce4_2", "delta_e": 18.3, "ratio": 2/3},
    {"label": "Ce3+ u0",   "state": "Ce3+", "center": 898.9, "role": "partner", "pair": "ce3_0", "delta_e": 18.1, "ratio": 2/3},
    {"label": "Ce4+ u",    "state": "Ce4+", "center": 900.9, "role": "partner", "pair": "ce4_0", "delta_e": 18.5, "ratio": 2/3},
    {"label": "Ce3+ u'",   "state": "Ce3+", "center": 903.4, "role": "partner", "pair": "ce3_1", "delta_e": 18.3, "ratio": 2/3},
    {"label": "Ce4+ u''",  "state": "Ce4+", "center": 907.1, "role": "partner", "pair": "ce4_1", "delta_e": 18.5, "ratio": 2/3},
    {"label": "Ce4+ u'''", "state": "Ce4+", "center": 916.5, "role": "partner", "pair": "ce4_2", "delta_e": 18.3, "ratio": 2/3},
]

# Express the conventional Ce model through the same generic group constraints
# available to user-authored models. This keeps the solver element-agnostic.
for _ce_spec in CE3D_MIXED_TEMPLATE:
    _ce_group = "ce3" if _ce_spec["state"] == "Ce3+" else "ce4"
    _ce_spec["shift_group"] = _ce_group
    _ce_spec["fwhm_group"] = _ce_group
    _ce_spec["shape_group"] = _ce_group
    _ce_spec["fwhm_init"] = 1.45
    _ce_spec["lineshape"] = "pseudo_voigt"
    _ce_spec["source"] = "Literature/reference template; verify against a sample-appropriate Ce 3d reference."

CE3D_CE3_TEMPLATE = [p for p in CE3D_MIXED_TEMPLATE if p["state"] == "Ce3+"]

# Oxide W 4f lines are symmetric doublets.  The broad W 5p3/2 contribution is
# included because it commonly enters a sufficiently wide W 4f acquisition.
# It is automatically omitted from the model if it lies outside the data range.
W6_W4F_TEMPLATE = [
    {"label": "W6+ 4f7/2", "state": "W6+", "center": 35.7, "center_mode": "free", "role": "primary", "pair": "w6", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W6+ 4f5/2", "state": "W6+", "center": 37.87, "center_mode": "tied", "role": "partner", "pair": "w6", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W 5p3/2", "state": "W 5p3/2", "center": 41.5, "center_mode": "bounded", "center_min": 40.5, "center_max": 42.5, "role": "singlet", "quantify": False, "independent_shape": True, "fwhm_init": 2.4, "fwhm_min": 0.9, "fwhm_max": 5.0},
]

W4_W4F_TEMPLATE = [
    {"label": "W4+ 4f7/2", "state": "W4+", "center": 32.8, "center_mode": "free", "role": "primary", "pair": "w4", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W4+ 4f5/2", "state": "W4+", "center": 34.97, "center_mode": "tied", "role": "partner", "pair": "w4", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W 5p3/2", "state": "W 5p3/2", "center": 41.0, "center_mode": "bounded", "center_min": 40.0, "center_max": 42.2, "role": "singlet", "quantify": False, "independent_shape": True, "fwhm_init": 2.4, "fwhm_min": 0.9, "fwhm_max": 5.0},
]

WO3X_W4F_TEMPLATE = [
    {"label": "W6+ 4f7/2", "state": "W6+", "center": 35.7, "center_mode": "free", "role": "primary", "pair": "w6", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W6+ 4f5/2", "state": "W6+", "center": 37.87, "center_mode": "tied", "role": "partner", "pair": "w6", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W5+ 4f7/2", "state": "W5+", "center": 34.6, "center_mode": "tied", "tie_target": "W6+ 4f7/2", "offset_init": -1.10, "offset_min": -1.40, "offset_max": -0.80, "role": "primary", "pair": "w5", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W5+ 4f5/2", "state": "W5+", "center": 36.77, "center_mode": "tied", "role": "partner", "pair": "w5", "delta_e": 2.17, "ratio": 0.75},
    {"label": "W 5p3/2", "state": "W 5p3/2", "center": 41.5, "center_mode": "bounded", "center_min": 40.5, "center_max": 42.5, "role": "singlet", "quantify": False, "independent_shape": True, "fwhm_init": 2.4, "fwhm_min": 0.9, "fwhm_max": 5.0},
]

for _w_template in (W6_W4F_TEMPLATE, W4_W4F_TEMPLATE, WO3X_W4F_TEMPLATE):
    for _w_spec in _w_template:
        _w_spec.setdefault("lineshape", "pseudo_voigt")
        if "5p3/2" not in _w_spec["label"]:
            _w_spec.setdefault("fwhm_init", 1.25)
            _w_spec.setdefault("fwhm_min", 0.65)
            _w_spec.setdefault("fwhm_max", 2.20)
        _w_spec.setdefault("source", "Built-in literature starting template; verify with a sample-appropriate XPS reference.")

# ZrO2 contributes a broad Zr 4p doublet in the low-binding-energy side of a
# wide W 4f acquisition.  It must not be represented with W 4f splitting or a
# W-like branching ratio.  The values below reproduce the reference starting
# spectrum (4p3/2 30.61 eV, 4p1/2 32.29 eV, approximately 2:1 area and 2.2 eV
# FWHM); calibration and the sample-specific Zr chemical state must still be
# verified independently, preferably in the Zr 3d region.
ZR4P_ZRO2_TEMPLATE = [
    {
        "label": "Zr4+ 4p3/2", "state": "Zr4+", "center": 30.61,
        "center_mode": "bounded", "center_min": 30.10, "center_max": 31.15,
        "role": "primary", "pair": "zr4p", "delta_e": 1.68, "ratio": 0.50,
        "quantify": False, "fwhm_init": 2.20, "fwhm_min": 1.40,
        "fwhm_max": 3.20, "lineshape": "pseudo_voigt",
        "source": (
            "ZrO2 Zr 4p reference starting values: 30.61/32.29 eV, "
            "2:1 area, FWHM about 2.2 eV; verify against Zr 3d and a "
            "sample-appropriate reference."
        ),
    },
    {
        "label": "Zr4+ 4p1/2", "state": "Zr4+", "center": 32.29,
        "center_mode": "tied", "role": "partner", "pair": "zr4p",
        "delta_e": 1.68, "ratio": 0.50, "quantify": False,
        "fwhm_init": 2.20, "fwhm_min": 1.40, "fwhm_max": 3.20,
        "lineshape": "pseudo_voigt",
        "source": (
            "ZrO2 Zr 4p reference starting values: 30.61/32.29 eV, "
            "2:1 area, FWHM about 2.2 eV; verify against Zr 3d and a "
            "sample-appropriate reference."
        ),
    },
]

WO3X_ZR4P_TEMPLATE = (
    copy.deepcopy(WO3X_W4F_TEMPLATE) + copy.deepcopy(ZR4P_ZRO2_TEMPLATE)
)

WZR_WIDE_REGION_DATABASE_ENTRY = {
    "doublet": True,
    "delta_e": 2.17,
    "ratio": 0.750,
    "primary_centers": [35.7, 34.6, 30.61],
    "state": "W6+/W5+ with Zr4+ 4p overlap",
    "template": "wo3x_zr4p",
    "template_specs": WO3X_ZR4P_TEMPLATE,
}

COMPOUND_DATABASE = {
    "tio2 2p": {"doublet": True, "delta_e": 5.70, "ratio": 0.500, "primary_centers": [458.5], "state": "Ti4+"},
    "ti2o3 2p": {"doublet": True, "delta_e": 5.70, "ratio": 0.500, "primary_centers": [456.0], "state": "Ti3+"},
    "tin 2p": {"doublet": True, "delta_e": 5.70, "ratio": 0.500, "primary_centers": [455.5], "state": "TiN"},
    "wo3 4f": {"doublet": True, "delta_e": 2.17, "ratio": 0.750, "primary_centers": [35.7], "state": "W6+", "template": "w6_w4f", "template_specs": W6_W4F_TEMPLATE},
    "wo2 4f": {"doublet": True, "delta_e": 2.17, "ratio": 0.750, "primary_centers": [32.8], "state": "W4+", "template": "w4_w4f", "template_specs": W4_W4F_TEMPLATE},
    "wo3-x 4f": {"doublet": True, "delta_e": 2.17, "ratio": 0.750, "primary_centers": [35.7, 34.6], "state": "W6+/W5+", "template": "wo3x_w4f", "template_specs": WO3X_W4F_TEMPLATE},
    "wo2.9 4f": {"doublet": True, "delta_e": 2.17, "ratio": 0.750, "primary_centers": [35.7, 34.6], "state": "W6+/W5+", "template": "wo3x_w4f", "template_specs": WO3X_W4F_TEMPLATE},
    "wo2.9/zro2 4f": copy.deepcopy(WZR_WIDE_REGION_DATABASE_ENTRY),
    "wo2.9-zro2 4f": copy.deepcopy(WZR_WIDE_REGION_DATABASE_ENTRY),
    "zro2-wo2.9 4f": copy.deepcopy(WZR_WIDE_REGION_DATABASE_ENTRY),
    "wo3-x/zro2 4f": copy.deepcopy(WZR_WIDE_REGION_DATABASE_ENTRY),
    "wo3-x-zro2 4f": copy.deepcopy(WZR_WIDE_REGION_DATABASE_ENTRY),
    "zro2-wo3-x 4f": copy.deepcopy(WZR_WIDE_REGION_DATABASE_ENTRY),
    "w-zr-oxide 4f": copy.deepcopy(WZR_WIDE_REGION_DATABASE_ENTRY),
    "zro2 4p": {"doublet": True, "delta_e": 1.68, "ratio": 0.500, "primary_centers": [30.61], "state": "Zr4+", "template": "zr4p_zro2", "template_specs": ZR4P_ZRO2_TEMPLATE},
    "zro2 3d": {"doublet": True, "delta_e": 2.40, "ratio": 0.667, "primary_centers": [182.2], "state": "Zr4+"},
    "ceo2 3d": {"doublet": False, "template": "ce3d_mixed", "primary_centers": [p["center"] for p in CE3D_MIXED_TEMPLATE]},
    "ce2o3 3d": {"doublet": False, "template": "ce3d_ce3", "primary_centers": [p["center"] for p in CE3D_CE3_TEMPLATE]},
    "sio2 2p": {"doublet": False, "primary_centers": [103.3], "state": "Si4+"},
    "al2o3 2p": {"doublet": False, "primary_centers": [74.5], "state": "Al3+"},
}

def get_xps_parameters(target: str, orbital: str) -> Tuple[Optional[Dict], str]:
    target = target.strip()
    orbital = orbital.strip().lower()
    formula_translation = str.maketrans({
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        '₋': '-', 'ₓ': 'x', '−': '-', '–': '-'
    })
    normalized_target = target.lower().translate(formula_translation).replace(' ', '')
    query_key = f"{normalized_target} {orbital.replace(' ', '')}"

    if query_key in COMPOUND_DATABASE:
        db_info = copy.deepcopy(COMPOUND_DATABASE[query_key])
        c_str = ", ".join([f"{c:.1f}" for c in db_info.get('primary_centers', [])])
        if db_info.get('template') == 'ce3d_mixed':
            msg = f"Ce 3d mixed-valence template: 10 components (Ce3+ + Ce4+) | Reference centers~{c_str} eV"
        elif db_info.get('template') == 'ce3d_ce3':
            msg = f"Ce 3d Ce3+ template: 4 components | Reference centers~{c_str} eV"
        elif db_info.get('template') in {'w6_w4f', 'w4_w4f', 'wo3x_w4f', 'wo3x_zr4p'}:
            states = "/".join(dict.fromkeys(p['state'] for p in db_info['template_specs'] if p['state'].startswith('W') and '5p' not in p['state']))
            overlap = " and constrained Zr 4p overlap" if db_info.get('template') == 'wo3x_zr4p' else ""
            msg = (f"Constrained W 4f template ({states}) with W 5p3/2{overlap} | "
                   f"W Delta E={db_info['delta_e']:.2f} eV | W partner/primary area={db_info['ratio']:.3f}")
        elif db_info.get('template') == 'zr4p_zro2':
            msg = (
                "Constrained ZrO2 Zr 4p starting template | Delta E=1.68 eV | "
                "partner/primary area=0.500 | broad FWHM about 2.2 eV"
            )
        elif db_info.get('doublet', False):
            delta_e = db_info['delta_e']
            ratio = db_info['ratio']
            msg = f"Chemical-state starting model ({target}) | Delta E={delta_e:.2f} eV | partner/primary area={ratio:.3f} | Primary centers~{c_str} eV"
        else:
            msg = f"Chemical-state starting model ({target}) | Singlet centers~{c_str} eV"
        return db_info, msg

    if not HAS_XRAYDB:
        return None, (
            "Target is not in the built-in chemical-state starting database. "
            "Enter centers and constraints from an appropriate XPS reference or standard."
        )
    
    element = target.capitalize()
    if len(orbital) < 2:
        return None, "Invalid orbital format (e.g., use '2p', '4f')"
        
    n = orbital[0]
    l = orbital[1]
    
    if n not in SHELL_MAP or l not in ORBITAL_MAP:
        return None, "Unknown shell or orbital sub-level"
        
    shell_prefix = SHELL_MAP[n]
    orb_info = ORBITAL_MAP[l]
    
    try:
        edges = xraydb.xray_edges(element)
    except Exception:
        return None, f"Unknown compound or element '{target}'. Check spelling."
    
    if not orb_info['doublet']:
        edge_name = shell_prefix + orb_info['edges'][0]
        if edge_name in edges:
            center_1 = edges[edge_name].energy
            return {
                'doublet': False, 'primary_centers': [center_1], 'reference_only': True
            }, (f"xraydb returned an X-ray absorption edge (~{center_1:.1f} eV), not a chemical-state XPS reference. "
                "It will not be used automatically for publication fitting.")
        return {'doublet': False}, f"{element} {orbital} is a Singlet (No splitting)"
        
    try:
        edge1 = shell_prefix + orb_info['edges'][0]
        edge2 = shell_prefix + orb_info['edges'][1]
        
        if edge1 in edges and edge2 in edges:
            e1 = edges[edge1].energy  
            e2 = edges[edge2].energy  
            
            delta_e = abs(e1 - e2)
            ratio = orb_info['ratio']
            primary_center = min(e1, e2) 
            
            return {
                'doublet': True, 'delta_e': delta_e, 'ratio': ratio,
                'primary_centers': [primary_center], 'reference_only': True
            }, (f"xraydb absorption-edge estimate for {element} {orbital}: Delta E~{delta_e:.2f} eV, "
                f"primary~{primary_center:.1f} eV. This is not a chemical-state XPS binding-energy reference "
                "and will not be applied automatically.")
        else:
            return None, f"Binding energies for {element} {edge1}/{edge2} not found."
    except Exception as e:
        return None, f"Database error: {str(e)}"
# ---------------------------------------------------------

from PyQt6.QtCore import Qt, pyqtSignal, QUrl, QThread, QTimer, QSize, QSettings
from PyQt6.QtGui import QFont, QDesktopServices, QIcon, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QToolButton, QLineEdit, QComboBox, QDoubleSpinBox, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QGroupBox,
    QFormLayout, QCheckBox, QScrollArea,
    QTabWidget, QStackedWidget, QFrame, QStatusBar, QProgressBar, QSpinBox,
    QSlider, QAbstractItemView, QInputDialog, QSizePolicy
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

APP_NAME = "Researcher Analysis Toolkit — XPS Fitting"
APP_SUBTITLE = "X-ray photoelectron spectroscopy (XPS) fitting system"
APP_VERSION = "2.6.3"
FIT_RANDOM_SEED = 20260723
MODEL_SCHEMA = "xps-chemical-model/v2"


class ResponsiveScrollArea(QScrollArea):
    """Keep a vertical control panel fitted to its visible viewport width.

    Qt normally honours a child layout's minimum-size hint.  A long status
    label can therefore make the hidden scroll-area canvas wider than the
    viewport even when the horizontal scrollbar is disabled, which clips the
    right side of every row.  Capping the canvas at the live viewport width
    makes the panel genuinely responsive while labels wrap normally.
    """

    def resizeEvent(self, event):
        super().resizeEvent(event)
        content = self.widget()
        viewport_width = self.viewport().width()
        if content is not None and viewport_width > 0:
            content.setMaximumWidth(viewport_width)
SUPPORTED_MODEL_SCHEMAS = {"xps-chemical-model/v1", MODEL_SCHEMA}
VALID_LINE_SHAPES = {
    "pseudo_voigt", "asymmetric_pseudo_voigt", "gaussian", "lorentzian", "doniach_sunjic"
}
VALID_COMPONENT_ROLES = {"singlet", "primary", "partner", "satellite", "multiplet"}
VALID_CENTER_MODES = {"free", "fixed", "bounded", "tied"}
MODEL_TABLE_FIELDS = [
    ("Use", "enabled"), ("Label", "label"), ("State", "state"),
    ("Center", "center"), ("Center mode", "center_mode"),
    ("Center min", "center_min"), ("Center max", "center_max"),
    ("Tie target", "tie_target"), ("Offset init", "offset_init"),
    ("Offset min", "offset_min"), ("Offset max", "offset_max"),
    ("Role", "role"), ("Pair", "pair"), ("ΔE", "delta_e"), ("Area ratio", "ratio"),
    ("FWHM init", "fwhm_init"), ("FWHM min", "fwhm_min"), ("FWHM max", "fwhm_max"),
    ("Line shape", "lineshape"), ("Shift group", "shift_group"),
    ("FWHM group", "fwhm_group"), ("Shape group", "shape_group"),
    ("Independent shape", "independent_shape"), ("Quantify", "quantify"), ("Source / DOI", "source"),
]
MODEL_FIELD_INDEX = {field: index for index, (_, field) in enumerate(MODEL_TABLE_FIELDS)}
MODEL_BOOL_FIELDS = {"enabled", "independent_shape", "quantify"}


@dataclass
class PeakResult:
    label: str
    center_input: float
    center_fitted: float
    fwhm_fitted: float
    height_fitted: float
    area_fitted: float
    eta_fitted: float
    asym_fitted: float
    state: str = ""
    role: str = "singlet"
    pair: str = ""
    lineshape: str = "pseudo_voigt"
    source: str = ""
    quantify: bool = True
    center_mode: str = "free"
    center_min: Optional[float] = None
    center_max: Optional[float] = None
    tie_target: str = ""
    offset_fitted: Optional[float] = None
    offset_stderr: Optional[float] = None
    offset_min: Optional[float] = None
    offset_max: Optional[float] = None
    center_stderr: Optional[float] = None
    fwhm_stderr: Optional[float] = None
    height_stderr: Optional[float] = None
    area_stderr: Optional[float] = None
    area_fraction: Optional[float] = None
    area_fraction_stderr: Optional[float] = None
    bound_warning: str = ""


@dataclass
class FitMetrics:
    rmse: float
    nrmse_percent: float
    r2: float
    rss: float
    reduced_chi2: float  # reduced objective; chi-square only when justified Poisson weights are selected
    aic: float
    aicc: float
    bic: float
    nvarys: int
    max_abs_corr: float
    residual_lag1_corr: float
    durbin_watson: float
    objective_value: float
    weighting: str
    fit_quality: str
    warning_count: int


def safe_float(text: str) -> Optional[float]:
    try: return float(text)
    except Exception: return None

def parse_centers(text: str) -> List[float]:
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    centers: List[float] = []
    for p in parts:
        if not p: continue
        val = safe_float(p)
        if val is None: raise ValueError(f"Invalid center value: {p}")
        centers.append(val)
    return centers

def format_centers(centers: List[float]) -> str:
    return ", ".join(f"{v:.4f}" for v in sorted(centers))


def format_fraction_percent(
    fraction: Optional[float], decimals: int = 2, include_symbol: bool = True
) -> str:
    """Format fractional area without disguising a small non-zero value as zero."""
    if fraction is None:
        return "Excluded"
    percent = 100.0 * float(fraction)
    symbol = "%" if include_symbol else ""
    threshold = 10.0 ** (-decimals)
    if 0.0 < percent < threshold:
        return f"<{threshold:.{decimals}f}{symbol}"
    return f"{percent:.{decimals}f}{symbol}"


def _parse_bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"{field} must be true or false, not '{value}'.")


def canonical_lineshape(value: str) -> str:
    text = str(value or "pseudo_voigt").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "pseudovoigt": "pseudo_voigt", "pseudo_voigt_(symmetric)": "pseudo_voigt",
        "gl": "pseudo_voigt", "gauss": "gaussian", "lorentz": "lorentzian",
        "asymmetric_pseudovoigt": "asymmetric_pseudo_voigt",
        "asymmetric_pseudo_voigt_(integrable;_metallic)": "asymmetric_pseudo_voigt",
        "asymmetric_pseudo_voigt_(integrable_metallic)": "asymmetric_pseudo_voigt",
        "apv": "asymmetric_pseudo_voigt",
        "doniach_sunjic_(be_tail;_metallic_only)": "doniach_sunjic",
        "doniach_sunjic_(be_tail_metallic_only)": "doniach_sunjic", "ds": "doniach_sunjic",
    }
    text = aliases.get(text, text)
    if text not in VALID_LINE_SHAPES:
        raise ValueError(
            f"Unknown line shape '{value}'. Use pseudo_voigt, asymmetric_pseudo_voigt, "
            "gaussian, lorentzian, or doniach_sunjic."
        )
    return text


def canonical_center_mode(value: Any, role: str = "singlet", has_bounds: bool = False) -> str:
    """Normalize the user-facing peak-center control to the model schema."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "tied" if role == "partner" else ("bounded" if has_bounds else "free"),
        "free_(global_bound)": "free", "free_global_bound": "free", "vary": "free",
        "constant": "fixed", "lock": "fixed", "locked": "fixed",
        "bound": "bounded", "range": "bounded",
        "linked": "tied", "tie": "tied",
    }
    text = aliases.get(text, text)
    if text not in VALID_CENTER_MODES:
        raise ValueError(
            f"Unknown center mode '{value}'. Use free, fixed, bounded, or tied."
        )
    return text


def default_doublet_ratio(orbital: str) -> Optional[float]:
    """Return the statistical partner/primary area ratio; splitting is compound-specific."""
    text = str(orbital or "").strip().lower().replace(" ", "")
    if len(text) < 2:
        return None
    info = ORBITAL_MAP.get(text[1])
    return float(info["ratio"]) if info and info.get("doublet") else None


def normalize_model_specs(
    components: List[Dict[str, Any]],
    default_shape: str = "pseudo_voigt",
    default_fwhm_min: float = 0.35,
    default_fwhm_max: float = 4.0,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Validate and normalize an arbitrary, reference-driven XPS component model.

    The numerical engine is element-agnostic. Chemical meaning enters only through
    these explicit components and their documented constraints.
    """
    if not isinstance(components, list) or not components:
        raise ValueError("The chemical model must contain at least one component.")
    default_shape = canonical_lineshape(default_shape)
    if not np.isfinite(default_fwhm_min) or not np.isfinite(default_fwhm_max) or default_fwhm_max <= default_fwhm_min:
        raise ValueError("Invalid global FWHM limits.")

    normalized: List[Dict[str, Any]] = []
    warnings_out: List[str] = []
    labels = set()
    for row_no, raw in enumerate(components, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Model component {row_no} is not an object.")
        spec = dict(raw)
        enabled = _parse_bool(spec.get("enabled", True), f"component {row_no} enabled")
        if not enabled:
            continue
        label = str(spec.get("label", f"Peak {row_no}")).strip() or f"Peak {row_no}"
        if label in labels:
            raise ValueError(f"Duplicate component label: '{label}'. Labels must be unique.")
        labels.add(label)
        try:
            center = float(spec["center"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"Component '{label}' needs a numeric center in eV.")
        if not np.isfinite(center):
            raise ValueError(f"Component '{label}' has a non-finite center.")

        role = str(spec.get("role", "singlet")).strip().lower()
        if role not in VALID_COMPONENT_ROLES:
            raise ValueError(f"Component '{label}' has unknown role '{role}'.")
        has_explicit_center_bounds = (
            str(spec.get("center_min", "")).strip() != ""
            and str(spec.get("center_max", "")).strip() != ""
        )
        center_mode = canonical_center_mode(
            spec.get("center_mode", ""), role, has_explicit_center_bounds
        )
        if role == "partner" and center_mode != "tied":
            warnings_out.append(
                f"Center mode of doublet partner '{label}' was changed to tied; "
                "partner centers are fixed by the primary + spin-orbit splitting."
            )
            center_mode = "tied"
        shape = canonical_lineshape(spec.get("lineshape", default_shape))
        fmin = float(spec.get("fwhm_min", default_fwhm_min))
        fmax = float(spec.get("fwhm_max", default_fwhm_max))
        finit = float(spec.get("fwhm_init", min(max(1.20, fmin), fmax)))
        if not all(np.isfinite(v) for v in (fmin, fmax, finit)) or fmin <= 0 or fmax <= fmin:
            raise ValueError(f"Component '{label}' has invalid FWHM settings.")
        if not (fmin <= finit <= fmax):
            raise ValueError(f"Component '{label}' has FWHM init outside its min/max interval.")

        clean = {
            "enabled": True,
            "label": label,
            "state": str(spec.get("state", "")).strip(),
            "center": center,
            "center_mode": center_mode,
            "tie_target": str(spec.get("tie_target", "")).strip(),
            "role": role,
            "pair": str(spec.get("pair", "")).strip(),
            "lineshape": shape,
            "fwhm_init": finit,
            "fwhm_min": fmin,
            "fwhm_max": fmax,
            "shift_group": str(spec.get("shift_group", "")).strip(),
            "fwhm_group": str(spec.get("fwhm_group", "")).strip(),
            "shape_group": str(spec.get("shape_group", "")).strip(),
            "independent_shape": _parse_bool(spec.get("independent_shape", False), f"{label} independent_shape"),
            "quantify": _parse_bool(spec.get("quantify", True), f"{label} quantify"),
            "source": str(spec.get("source", "")).strip(),
        }
        for key in (
            "center_min", "center_max", "offset_init", "offset_min", "offset_max",
            "height_init", "eta_init", "eta_min", "eta_max", "asym_init", "asym_min", "asym_max"
        ):
            if key in spec and str(spec[key]).strip() != "":
                value = float(spec[key])
                if not np.isfinite(value):
                    raise ValueError(f"Component '{label}' has non-finite {key}.")
                clean[key] = value

        if "height_init" in clean and clean["height_init"] <= 0:
            raise ValueError(f"Component '{label}' requires height_init > 0.")

        has_cmin = "center_min" in clean
        has_cmax = "center_max" in clean
        if has_cmin != has_cmax:
            raise ValueError(f"Component '{label}' must provide both center_min and center_max.")
        if has_cmin and clean["center_max"] <= clean["center_min"]:
            raise ValueError(f"Component '{label}' has center_max <= center_min.")
        if center_mode == "bounded" and not (has_cmin and has_cmax):
            raise ValueError(f"Bounded component '{label}' needs center_min and center_max.")
        if center_mode == "bounded" and not clean["center_min"] <= center <= clean["center_max"]:
            raise ValueError(f"Component '{label}' has its reference center outside center_min/center_max.")
        if center_mode in {"free", "fixed", "tied"} and (has_cmin or has_cmax):
            warnings_out.append(
                f"Center min/max of '{label}' are ignored in {center_mode} mode."
            )
            clean.pop("center_min", None)
            clean.pop("center_max", None)
        if center_mode != "tied":
            stale_tie_fields = bool(clean.get("tie_target")) or any(
                key in clean for key in ("offset_init", "offset_min", "offset_max")
            )
            if stale_tie_fields:
                warnings_out.append(
                    f"Tie target/offset fields of '{label}' are ignored in {center_mode} mode."
                )
            clean["tie_target"] = ""
            clean.pop("offset_init", None)
            clean.pop("offset_min", None)
            clean.pop("offset_max", None)
        if center_mode == "fixed" and clean["shift_group"]:
            raise ValueError(f"Fixed component '{label}' cannot also belong to a shift_group.")
        if center_mode == "tied" and role != "partner" and clean["shift_group"]:
            raise ValueError(f"Tied component '{label}' cannot also belong to a shift_group.")
        eta_values = [clean[key] for key in ("eta_init", "eta_min", "eta_max") if key in clean]
        if any(value < 0.0 or value > 1.0 for value in eta_values):
            raise ValueError(f"Component '{label}' requires pseudo-Voigt eta values within [0, 1].")
        if "eta_min" in clean and "eta_max" in clean and clean["eta_max"] <= clean["eta_min"]:
            raise ValueError(f"Component '{label}' has eta_max <= eta_min.")
        if "eta_init" in clean:
            if clean["eta_init"] < clean.get("eta_min", 0.0) or clean["eta_init"] > clean.get("eta_max", 1.0):
                raise ValueError(f"Component '{label}' has eta_init outside its eta bounds.")
        asym_values = [clean[key] for key in ("asym_init", "asym_min", "asym_max") if key in clean]
        if any(value < 0.0 or value >= 0.99 for value in asym_values):
            raise ValueError(f"Component '{label}' requires Doniach-Sunjic asymmetry in [0, 0.99).")
        if "asym_min" in clean and "asym_max" in clean and clean["asym_max"] <= clean["asym_min"]:
            raise ValueError(f"Component '{label}' has asym_max <= asym_min.")
        if "asym_init" in clean:
            default_asym_max = 0.80 if shape == "asymmetric_pseudo_voigt" else 0.25
            if clean["asym_init"] < clean.get("asym_min", 0.0) or clean["asym_init"] > clean.get("asym_max", default_asym_max):
                raise ValueError(f"Component '{label}' has asym_init outside its asymmetry bounds.")
        if role in {"primary", "partner"} and not clean["pair"]:
            raise ValueError(f"Component '{label}' is a {role} but has no pair identifier.")
        if role == "partner":
            try:
                clean["delta_e"] = float(spec["delta_e"])
                clean["ratio"] = float(spec["ratio"])
            except (KeyError, TypeError, ValueError):
                raise ValueError(f"Partner '{label}' needs numeric delta_e and ratio values.")
            if not np.isfinite(clean["delta_e"]) or clean["delta_e"] <= 0:
                raise ValueError(f"Partner '{label}' needs delta_e > 0.")
            if not np.isfinite(clean["ratio"]) or clean["ratio"] <= 0:
                raise ValueError(f"Partner '{label}' needs area ratio > 0.")
        normalized.append(clean)

    if not normalized:
        raise ValueError("All model components are disabled.")

    primaries: Dict[str, Tuple[int, Dict[str, Any]]] = {}
    for idx, spec in enumerate(normalized):
        if spec["role"] == "primary":
            if spec["pair"] in primaries:
                raise ValueError(f"Pair '{spec['pair']}' has more than one primary component.")
            primaries[spec["pair"]] = (idx, spec)
    for spec in normalized:
        if spec["role"] != "partner":
            continue
        if spec["pair"] not in primaries:
            raise ValueError(f"Partner '{spec['label']}' has no matching primary for pair '{spec['pair']}'.")
        _, primary = primaries[spec["pair"]]
        if spec["lineshape"] != primary["lineshape"]:
            raise ValueError(f"Doublet pair '{spec['pair']}' must use one line shape for both members.")
        predicted = primary["center"] + spec["delta_e"]
        if abs(spec["center"] - predicted) > 0.15:
            warnings_out.append(
                f"Reference center of '{spec['label']}' differs from primary + splitting by "
                f"{spec['center']-predicted:+.3f} eV; the constrained value ({predicted:.3f} eV) will be used."
            )
        # Enforced equal width/shape makes a height ratio equal to the area ratio.
        spec["center"] = predicted
        spec["center_mode"] = "tied"
        spec["tie_target"] = primary["label"]
        spec["offset_init"] = float(spec["delta_e"])
        spec.pop("offset_min", None)
        spec.pop("offset_max", None)

    # Validate cross-component center ties after all labels and doublet primaries
    # are known. A tied offset may be exact (no min/max) or independently bounded.
    label_map = {spec["label"]: spec for spec in normalized}
    tied_edges: Dict[str, str] = {}
    for spec in normalized:
        if spec["role"] == "partner" or spec["center_mode"] != "tied":
            continue
        target_label = str(spec.get("tie_target", "")).strip()
        if not target_label:
            raise ValueError(f"Tied component '{spec['label']}' needs a tie_target label.")
        if target_label not in label_map:
            raise ValueError(f"Tied component '{spec['label']}' targets unknown component '{target_label}'.")
        if target_label == spec["label"]:
            raise ValueError(f"Component '{spec['label']}' cannot be tied to itself.")
        if label_map[target_label]["role"] == "partner":
            raise ValueError(
                f"Tied component '{spec['label']}' must target an independent component, not partner '{target_label}'."
            )
        reference_offset = float(spec["center"] - label_map[target_label]["center"])
        offset_init = float(spec.get("offset_init", reference_offset))
        has_omin = "offset_min" in spec
        has_omax = "offset_max" in spec
        if has_omin != has_omax:
            raise ValueError(f"Tied component '{spec['label']}' must provide both offset_min and offset_max.")
        if has_omin:
            omin, omax = float(spec["offset_min"]), float(spec["offset_max"])
            if omax < omin:
                raise ValueError(f"Tied component '{spec['label']}' has offset_max < offset_min.")
            if not omin <= offset_init <= omax:
                raise ValueError(f"Tied component '{spec['label']}' has offset_init outside its offset bounds.")
        predicted_center = float(label_map[target_label]["center"] + offset_init)
        if abs(float(spec["center"]) - predicted_center) > 0.15:
            warnings_out.append(
                f"Reference center of tied component '{spec['label']}' differs from target + offset by "
                f"{float(spec['center'])-predicted_center:+.3f} eV; {predicted_center:.3f} eV will be used."
            )
        spec["center"] = predicted_center
        spec["offset_init"] = offset_init
        tied_edges[spec["label"]] = target_label

    visiting = set()
    visited = set()
    def visit_tie(label: str):
        if label in visited:
            return
        if label in visiting:
            raise ValueError(f"Center ties contain a dependency cycle involving '{label}'.")
        visiting.add(label)
        target = tied_edges.get(label)
        if target in tied_edges:
            visit_tie(target)
        visiting.remove(label)
        visited.add(label)
    for tied_label in tied_edges:
        visit_tie(tied_label)

    for group_field in ("shift_group", "fwhm_group", "shape_group"):
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for spec in normalized:
            if spec["role"] != "partner" and spec[group_field]:
                groups.setdefault(spec[group_field], []).append(spec)
        for group_name, members in groups.items():
            if len(members) < 2:
                warnings_out.append(f"{group_field} '{group_name}' contains only one independent component.")
            if group_field == "shape_group" and len({m["lineshape"] for m in members}) > 1:
                raise ValueError(f"shape_group '{group_name}' mixes incompatible line shapes.")

    return normalized, warnings_out


def validate_model_document(document: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    if not isinstance(document, dict):
        raise ValueError("Model JSON root must be an object.")
    schema = str(document.get("schema", MODEL_SCHEMA))
    if schema not in SUPPORTED_MODEL_SCHEMAS:
        raise ValueError(
            f"Unsupported model schema '{schema}'; supported schemas are: "
            + ", ".join(sorted(SUPPORTED_MODEL_SCHEMAS))
        )
    specs, warnings_out = normalize_model_specs(
        document.get("components", []),
        document.get("default_lineshape", "pseudo_voigt"),
        float(document.get("default_fwhm_min", 0.35)),
        float(document.get("default_fwhm_max", 4.0)),
    )
    clean = {
        "schema": MODEL_SCHEMA,
        "name": str(document.get("name", "Untitled XPS model")).strip(),
        "target": str(document.get("target", "")).strip(),
        "orbital": str(document.get("orbital", "")).strip(),
        "reference": str(document.get("reference", "")).strip(),
        "notes": str(document.get("notes", "")).strip(),
        "default_lineshape": canonical_lineshape(document.get("default_lineshape", "pseudo_voigt")),
        "default_fwhm_min": float(document.get("default_fwhm_min", 0.35)),
        "default_fwhm_max": float(document.get("default_fwhm_max", 4.0)),
        "components": specs,
    }
    if schema != MODEL_SCHEMA:
        warnings_out.insert(0, f"Imported legacy schema '{schema}' and upgraded it to '{MODEL_SCHEMA}'.")
    return clean, warnings_out

def load_xps_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".dat"}:
        # header=None preserves the first spectrum point in headerless exports;
        # textual header rows are removed by numeric coercion below.
        try:
            df = pd.read_csv(
                file_path, sep=None, engine="python", comment="#", header=None,
                on_bad_lines="skip"
            )
        except Exception:
            df = pd.read_csv(
                file_path, sep=r"\s+", engine="python", comment="#", header=None,
                on_bad_lines="skip"
            )
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path, header=None)
    else:
        raise ValueError("Unsupported file type.")

    if df.shape[1] < 2:
        raise ValueError("Data must have at least 2 columns (binding energy and intensity).")
    numeric = df.apply(pd.to_numeric, errors="coerce")
    valid_counts = numeric.notna().sum()
    usable_columns = [c for c in numeric.columns if int(valid_counts[c]) >= 10]
    if len(usable_columns) < 2:
        raise ValueError("Could not identify two numeric columns in the file.")

    # XPS exports conventionally place binding energy and intensity in the first
    # two numeric columns.  Retaining this deterministic rule is safer than
    # silently guessing among auxiliary instrument columns.
    energy_col, intensity_col = usable_columns[:2]
    energy = numeric[energy_col].to_numpy(dtype=float)
    intensity = numeric[intensity_col].to_numpy(dtype=float)
    mask = np.isfinite(energy) & np.isfinite(intensity)
    energy, intensity = energy[mask], intensity[mask]
    if len(energy) < 10:
        raise ValueError("Not enough valid data points.")

    # Repeated binding-energy bins occur in averaged or concatenated scans.
    # Averaging them is statistically and physically preferable to keeping the
    # first arbitrary occurrence.
    grouped = (pd.DataFrame({"energy": energy, "intensity": intensity})
               .groupby("energy", as_index=False, sort=True)["intensity"].mean())
    energy = grouped["energy"].to_numpy(dtype=float)
    intensity = grouped["intensity"].to_numpy(dtype=float)
    if len(energy) < 10 or float(np.ptp(energy)) <= 0:
        raise ValueError("The numeric columns do not contain a valid XPS energy axis.")
    return energy, intensity


def extract_fit_region(energy: np.ndarray, intensity: np.ndarray, centers: List[float], margin: float) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
    if not centers: raise ValueError("Select peak centers.")
    lo, hi = min(centers) - margin, max(centers) + margin
    mask = (energy >= lo) & (energy <= hi)
    if mask.sum() < 20: raise ValueError("Region too small.")
    return energy[mask], intensity[mask], (lo, hi)


# --- Background Models ---
def _edge_level(y: np.ndarray, left: bool = True) -> float:
    """Robust endpoint intensity using a small median window instead of one noisy point."""
    n = len(y)
    k = max(3, min(15, max(3, n // 20)))
    part = y[:k] if left else y[-k:]
    return float(np.median(part))


def linear_background(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x0, x1 = float(x[0]), float(x[-1])
    y0, y1 = _edge_level(y, True), _edge_level(y, False)
    if x1 == x0:
        return np.full_like(y, y0)
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def shirley_background(x: np.ndarray, y: np.ndarray, max_iter: int = 250, tol: float = 1e-7) -> np.ndarray:
    """
    Iterative Shirley background with robust endpoint levels and O(N) cumulative integration.
    The spectrum is expected to be sorted in ascending binding energy.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Invalid spectrum for Shirley background.")

    left_level = _edge_level(y, True)
    right_level = _edge_level(y, False)
    bkg = linear_background(x, y)
    scale = max(float(np.ptp(y)), 1.0)

    for _ in range(max_iter):
        old = bkg.copy()
        # Negative background-subtracted noise is not clipped in the final spectrum;
        # clipping here is only part of the classical Shirley integral construction.
        diff = np.maximum(y - bkg, 0.0)
        dx = np.diff(x)
        trap = 0.5 * (diff[:-1] + diff[1:]) * dx
        cumulative = np.concatenate((np.cumsum(trap[::-1])[::-1], [0.0]))
        total = float(cumulative[0])
        if not np.isfinite(total) or abs(total) < 1e-20:
            break
        candidate = right_level + (left_level - right_level) * cumulative / total
        # Mild damping stabilizes difficult spectra without changing the converged solution.
        bkg = 0.5 * old + 0.5 * candidate
        if np.max(np.abs(bkg - old)) <= tol * scale:
            break
    return bkg


def smart_background(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Kept for UI compatibility. Unlike the old implementation, it does NOT clip
    # the background to the measured signal, which biased peak areas.
    return shirley_background(x, y)


def seed_manual_background_points(
    x: np.ndarray, y: np.ndarray
) -> List[Tuple[float, float]]:
    """Create robust left/right anchors for starting a manual background.

    The anchors are placed at the actual fit-window limits while their
    intensities use the same median edge estimate as the linear/Shirley
    backgrounds.  They are only a transparent starting point and are never
    presented as a physically derived inelastic-scattering background.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y) or len(x) < 3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid spectrum for a manual background.")
    return [
        (float(x[0]), _edge_level(y, True)),
        (float(x[-1]), _edge_level(y, False)),
    ]


def normalize_manual_background_points(
    points: Any, x: np.ndarray, require_edge_coverage: bool = True
) -> List[Tuple[float, float]]:
    """Validate, sort and de-duplicate manual background anchors.

    A manual background must span the fitted interval.  Constant extrapolation
    from interior anchors can look innocuous while materially changing peak
    areas, so publication fits require an anchor close to each fit boundary.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Invalid energy axis for a manual background.")
    parsed: List[Tuple[float, float]] = []
    for item in points or []:
        if isinstance(item, dict):
            px = item.get("binding_energy_eV", item.get("x"))
            py = item.get("intensity", item.get("y"))
        else:
            try:
                px, py = item
            except Exception as exc:
                raise ValueError("Each manual-background anchor must contain energy and intensity.") from exc
        try:
            px, py = float(px), float(py)
        except (TypeError, ValueError) as exc:
            raise ValueError("Manual-background anchors must be numeric.") from exc
        if not np.isfinite(px) or not np.isfinite(py):
            raise ValueError("Manual-background anchors must be finite.")
        parsed.append((px, py))
    if len(parsed) < 2:
        raise ValueError("Manual background requires at least two anchor points.")

    parsed.sort(key=lambda point: point[0])
    xmin, xmax = float(np.min(x)), float(np.max(x))
    span = max(xmax - xmin, 1e-12)
    x_tolerance = max(1e-9, 1e-7 * span)
    outside = [point for point in parsed if point[0] < xmin-x_tolerance or point[0] > xmax+x_tolerance]
    if outside:
        raise ValueError(
            "Manual-background anchors must lie inside the current fit interval. "
            "Reset the anchors after changing peak centers or the fit margin."
        )

    # Merge anchors at effectively identical energies. Their median intensity
    # avoids an arbitrary last-click-wins rule and keeps the definition stable.
    merged: List[Tuple[float, float]] = []
    group_x: List[float] = []
    group_y: List[float] = []
    for px, py in parsed:
        if group_x and abs(px - group_x[-1]) > x_tolerance:
            merged.append((float(np.mean(group_x)), float(np.median(group_y))))
            group_x, group_y = [], []
        group_x.append(px); group_y.append(py)
    if group_x:
        merged.append((float(np.mean(group_x)), float(np.median(group_y))))
    if len(merged) < 2:
        raise ValueError("Manual background requires anchors at two distinct binding energies.")

    if require_edge_coverage:
        steps = np.diff(np.unique(x))
        median_step = float(np.median(steps)) if len(steps) else span
        edge_allowance = max(2.5 * median_step, 0.035 * span)
        if merged[0][0] > xmin + edge_allowance or merged[-1][0] < xmax - edge_allowance:
            raise ValueError(
                "Manual background must include one anchor near each fit-window boundary "
                "(within 3.5% of the interval)."
            )
    return merged


def manual_background(
    x: np.ndarray, y: np.ndarray, points: Any
) -> np.ndarray:
    """Piecewise-linear background through explicit user-defined anchors."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    anchors = normalize_manual_background_points(points, x, require_edge_coverage=True)
    anchor_x = np.asarray([point[0] for point in anchors], dtype=float)
    anchor_y = np.asarray([point[1] for point in anchors], dtype=float)
    return np.interp(x, anchor_x, anchor_y)


def compute_background(
    x: np.ndarray, y: np.ndarray, method: str, manual_points: Any = None
) -> np.ndarray:
    method = method.lower().strip()
    if method == "linear":
        return linear_background(x, y)
    if method in {"shirley", "shirley (iterative)", "smart (shirley)"}:
        return shirley_background(x, y)
    if method in {"manual", "manual (anchor points)", "manual anchor points"}:
        return manual_background(x, y, manual_points)
    if method == "none":
        return np.zeros_like(y, dtype=float)
    raise ValueError(f"Unknown background: {method}")


def build_origin_ready_plot_frame(
    binding_energy: Any,
    experimental_data: Any,
    final_background: Any,
    total_fit_raw_scale: Any,
    residual_data_minus_fit: Any,
    components: Any,
    peak_results: Optional[List[Any]] = None,
) -> pd.DataFrame:
    """Return one unambiguous table that reproduces the software plot.

    The detailed numerical export intentionally stores each fitted component
    relative to zero.  The GUI, however, draws ``final background + component``.
    Origin users should not have to reconstruct that convention manually, so
    this compact table exports the exact displayed curves on the raw-intensity
    scale.  Binding energies are written in descending order, matching the
    conventional high-to-low XPS presentation.
    """
    arrays = {
        "binding energy": np.asarray(binding_energy, dtype=float),
        "experimental data": np.asarray(experimental_data, dtype=float),
        "final background": np.asarray(final_background, dtype=float),
        "total fit": np.asarray(total_fit_raw_scale, dtype=float),
        "residual": np.asarray(residual_data_minus_fit, dtype=float),
    }
    lengths = {name: len(values) for name, values in arrays.items() if values.ndim == 1}
    if len(lengths) != len(arrays) or len(set(lengths.values())) != 1:
        raise ValueError("Origin-ready export requires aligned one-dimensional fit arrays.")
    n_points = next(iter(lengths.values()), 0)
    if n_points < 2 or any(not np.all(np.isfinite(values)) for values in arrays.values()):
        raise ValueError("Origin-ready export contains too few points or non-finite values.")

    component_source = [] if components is None else list(components)
    component_arrays = [np.asarray(component, dtype=float) for component in component_source]
    if any(
        component.ndim != 1
        or len(component) != n_points
        or not np.all(np.isfinite(component))
        for component in component_arrays
    ):
        raise ValueError("Origin-ready component arrays are not aligned with the fit window.")

    # A stable numerical sort makes the CSV read high-to-low without changing
    # any measured or fitted values.
    order = np.argsort(-arrays["binding energy"], kind="mergesort")
    origin_data: Dict[str, Any] = {
        "Binding_Energy_eV": arrays["binding energy"][order],
        "Experimental_Data": arrays["experimental data"][order],
        "Total_Fit": arrays["total fit"][order],
        "Final_Background": arrays["final background"][order],
    }
    peak_results = [] if peak_results is None else list(peak_results)
    for index, component in enumerate(component_arrays, start=1):
        center = None
        assignment = ""
        if index <= len(peak_results):
            peak_result = peak_results[index - 1]
            try:
                center = float(getattr(peak_result, "center_fitted"))
            except (AttributeError, TypeError, ValueError):
                center = None
            label = str(getattr(peak_result, "label", "") or "").strip()
            state = str(getattr(peak_result, "state", "") or "").strip()
            generic_labels = {f"peak {index}".casefold(), f"peak_{index}".casefold()}
            if label and label.casefold() not in generic_labels:
                assignment = label
            elif state and state.casefold() != "unassigned":
                assignment = state

        # Keep column names readable in Origin while retaining chemically useful
        # labels such as W6+ or W5+ when the fit carries an assignment.
        assignment_token = ""
        if assignment:
            cleaned = "".join(
                character if character.isalnum() or character in "+-" else "_"
                for character in assignment
            )
            cleaned = "_".join(part for part in cleaned.split("_") if part)
            if cleaned:
                assignment_token = f"_{cleaned}"
        center_token = f"_{center:.3f}eV" if center is not None and np.isfinite(center) else ""
        column_name = f"Peak_{index}{assignment_token}{center_token}_On_Background"
        origin_data[column_name] = (arrays["final background"] + component)[order]

    origin_data["Residual_Data_Minus_Fit"] = arrays["residual"][order]
    origin_data["Residual_Zero_Line"] = np.zeros(n_points, dtype=float)
    return pd.DataFrame(origin_data)


# --- Lineshapes ---
def pseudo_voigt(x: np.ndarray, center: float, fwhm: float, height: float, eta: float) -> np.ndarray:
    fwhm = max(float(fwhm), 1e-9)
    eta = float(np.clip(eta, 0.0, 1.0))
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma**2))
    lorentz = 1.0 / (1.0 + 4.0 * ((x - center) / fwhm) ** 2)
    return height * (eta * lorentz + (1.0 - eta) * gaussian)


def asymmetric_pseudo_voigt(
    x: np.ndarray, center: float, fwhm: float, height: float, eta: float, asym: float
) -> np.ndarray:
    """Integrable split pseudo-Voigt with a tail toward higher binding energy.

    ``fwhm`` is the mean of the left/right FWHM values.  The two widths are
    fwhm*(1-asym) and fwhm*(1+asym), so the analytic full area is identical to
    that of a symmetric pseudo-Voigt with the same fwhm/height/eta.  This makes
    spin-orbit area-ratio constraints well-defined for asymmetric metallic lines.
    """
    fwhm = max(float(fwhm), 1e-9)
    asym = float(np.clip(asym, 0.0, 0.98))
    local_fwhm = np.where(np.asarray(x) < center, fwhm * (1.0 - asym), fwhm * (1.0 + asym))
    eta = float(np.clip(eta, 0.0, 1.0))
    sigma = local_fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-((x - center) ** 2) / (2 * sigma**2))
    lorentz = 1.0 / (1.0 + 4.0 * ((x - center) / local_fwhm) ** 2)
    return height * (eta * lorentz + (1.0 - eta) * gaussian)

def doniach_sunjic(x: np.ndarray, center: float, fwhm: float, height: float, asym: float) -> np.ndarray:
    """Doniach-Sunjic profile oriented for a binding-energy axis.

    The conventional analytic form is often written for kinetic energy and has
    its long tail toward lower x.  XPS data here use binding energy, so the sign
    is reversed to place the loss tail toward higher binding energy.
    """
    asym = np.clip(asym, 0.0, 0.99)
    fwhm = max(float(fwhm), 1e-9)
    gamma = fwhm / 2.0
    diff = center - x
    phase = np.pi * asym / 2.0 + (1.0 - asym) * np.arctan(diff / gamma)
    denominator = (diff**2 + gamma**2) ** ((1.0 - asym) / 2.0)
    norm = (gamma**2)**((1.0 - asym) / 2.0) / np.cos(np.pi * asym / 2.0)
    return height * norm * np.cos(phase) / denominator

def _baseline_correction(params, x):
    if 'baseline_offset' not in params:
        return np.zeros_like(x, dtype=float)
    span = max(float(np.ptp(x)), 1e-12)
    scaled_x = (x - float(np.mean(x))) / span
    return (params['baseline_offset'].value
            + params['baseline_slope'].value * scaled_x)


def _shape_for_peak(shape, peak_index: int) -> str:
    if isinstance(shape, (list, tuple, np.ndarray)):
        if len(shape) < peak_index:
            raise ValueError("The line-shape list is shorter than the component list.")
        return canonical_lineshape(shape[peak_index - 1])
    return canonical_lineshape(shape)


def xps_model_eval(params, x, n_peaks, shape='pseudo_voigt'):
    model_y = np.zeros_like(x, dtype=float)
    comps = []
    for i in range(1, n_peaks + 1):
        c = params[f'p{i}_center'].value
        w = params[f'p{i}_fwhm'].value
        h = params[f'p{i}_height'].value
        peak_shape = _shape_for_peak(shape, i)
        if peak_shape == 'doniach_sunjic':
            a = params[f'p{i}_asym'].value
            comp = doniach_sunjic(x, c, w, h, a)
        elif peak_shape == 'asymmetric_pseudo_voigt':
            e = params[f'p{i}_eta'].value
            a = params[f'p{i}_asym'].value
            comp = asymmetric_pseudo_voigt(x, c, w, h, e, a)
        else:
            e = params[f'p{i}_eta'].value
            comp = pseudo_voigt(x, c, w, h, e)
        comps.append(comp)
        model_y += comp
    model_y += _baseline_correction(params, x)
    return model_y, comps


def xps_residual(params, x, y, n_peaks, shape='pseudo_voigt', residual_scale=1.0, weights=None):
    model_y, _ = xps_model_eval(params, x, n_peaks, shape)
    scale = max(float(residual_scale), 1e-12)
    residual = (model_y - y) / scale
    if weights is not None:
        residual = residual * np.asarray(weights, dtype=float)
    return residual


def _pseudo_voigt_full_area(params, peak_index: int) -> float:
    """Analytic full-line area, independent of the selected fit window."""
    height = float(params[f'p{peak_index}_height'].value)
    fwhm = float(params[f'p{peak_index}_fwhm'].value)
    eta = float(params[f'p{peak_index}_eta'].value)
    gaussian_factor = math.sqrt(math.pi) / (2.0 * math.sqrt(math.log(2.0)))
    lorentzian_factor = math.pi / 2.0
    return height * fwhm * ((1.0 - eta) * gaussian_factor + eta * lorentzian_factor)


def _component_area(params, peak_index: int, shape: str, x: np.ndarray) -> float:
    peak_shape = _shape_for_peak(shape, peak_index)
    if peak_shape in {'pseudo_voigt', 'asymmetric_pseudo_voigt', 'gaussian', 'lorentzian'}:
        return float(_pseudo_voigt_full_area(params, peak_index))
    # The ideal Doniach-Sunjic line has a non-integrable asymptotic tail.
    # Its reported area is therefore explicitly the area inside the fit window.
    comp = doniach_sunjic(
        x,
        params[f'p{peak_index}_center'].value,
        params[f'p{peak_index}_fwhm'].value,
        params[f'p{peak_index}_height'].value,
        params[f'p{peak_index}_asym'].value,
    )
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(abs(trapezoid(comp, x)))


def _propagated_stderr(result, evaluator) -> Optional[float]:
    """First-order covariance propagation for tied and derived quantities."""
    covar = getattr(result, 'covar', None)
    names = list(getattr(result, 'var_names', []) or [])
    if covar is None or not names:
        return None
    covar = np.asarray(covar, dtype=float)
    if covar.shape != (len(names), len(names)) or not np.all(np.isfinite(covar)):
        return None

    gradients = np.zeros(len(names), dtype=float)
    for j, name in enumerate(names):
        par = result.params[name]
        value = float(par.value)
        lo = float(par.min) if np.isfinite(par.min) else -np.inf
        hi = float(par.max) if np.isfinite(par.max) else np.inf
        stderr_hint = float(par.stderr) if par.stderr is not None and np.isfinite(par.stderr) else 0.0
        step = max(abs(value) * 1e-6, stderr_hint * 1e-3, 1e-7)
        plus_value = min(value + step, hi)
        minus_value = max(value - step, lo)
        if plus_value <= minus_value:
            continue
        p_plus = copy.deepcopy(result.params)
        p_minus = copy.deepcopy(result.params)
        p_plus[name].value = plus_value
        p_minus[name].value = minus_value
        p_plus.update_constraints()
        p_minus.update_constraints()
        try:
            f_plus = float(evaluator(p_plus))
            f_minus = float(evaluator(p_minus))
            gradients[j] = (f_plus - f_minus) / (plus_value - minus_value)
        except Exception:
            return None

    variance = float(gradients @ covar @ gradients)
    if not np.isfinite(variance) or variance < -1e-12:
        return None
    return float(math.sqrt(max(variance, 0.0)))


def _clip_to_bounds(par, value: float) -> float:
    lo = par.min if np.isfinite(par.min) else -np.inf
    hi = par.max if np.isfinite(par.max) else np.inf
    if np.isfinite(lo):
        value = max(value, lo + 1e-12 * max(abs(lo), 1.0))
    if np.isfinite(hi):
        value = min(value, hi - 1e-12 * max(abs(hi), 1.0))
    return float(value)


def _initial_height(x, y, center, refs, floor):
    idx = int(np.argmin(np.abs(x - center)))
    local = max(float(y[idx]), floor)
    overlap = max(1, sum(abs(center - r) <= 1.5 for r in refs))
    return max(local / overlap, floor)


def _build_peak_specs(raw_centers, db_info, auto_doublet):
    raw_centers = [float(v) for v in raw_centers]
    warnings_out = []

    if auto_doublet and db_info and db_info.get('template') == 'ce3d_mixed':
        return [dict(p) for p in CE3D_MIXED_TEMPLATE], warnings_out, 'ce3d_mixed'
    if auto_doublet and db_info and db_info.get('template') == 'ce3d_ce3':
        return [dict(p) for p in CE3D_CE3_TEMPLATE], warnings_out, 'ce3d_ce3'
    if auto_doublet and db_info and db_info.get('template_specs'):
        template_specs = [dict(p) for p in db_info['template_specs']]
        template_name = str(db_info.get('template', 'template'))
        if template_name in {
            'w6_w4f', 'w4_w4f', 'wo3x_w4f', 'wo3x_zr4p', 'zr4p_zro2'
        } and raw_centers:
            grouped = {}
            for spec in template_specs:
                group_key = spec.get('pair', f"single:{spec['label']}")
                grouped.setdefault(group_key, []).append(spec)
            selected_groups = set()
            for center in raw_centers:
                distances = [
                    (min(abs(center - float(spec['center'])) for spec in group), key)
                    for key, group in grouped.items()
                ]
                best_distance, best_key = min(distances, key=lambda item: item[0])
                if best_distance <= 0.70:
                    selected_groups.add(best_key)
            selected = [
                spec for spec in template_specs
                if spec.get('pair', f"single:{spec['label']}") in selected_groups
            ]
            if selected:
                template_specs = selected
        return template_specs, warnings_out, template_name

    if auto_doublet and db_info and db_info.get('doublet', False):
        delta = float(db_info['delta_e'])
        ratio = float(db_info['ratio'])
        state = str(db_info.get('state', ''))
        reference_primaries = [float(v) for v in db_info.get('primary_centers', [])]
        centers = sorted(raw_centers)
        used = set()
        specs = []
        pair_no = 1
        tolerance = max(0.35, min(1.25, 0.12 * delta + 0.25))

        for i, c1 in enumerate(centers):
            if i in used:
                continue
            target = c1 + delta
            candidates = [(abs(c2 - target), j, c2) for j, c2 in enumerate(centers)
                          if j != i and j not in used and c2 > c1]
            candidates.sort(key=lambda z: z[0])
            if candidates and candidates[0][0] <= tolerance:
                _, j, c2 = candidates[0]
                used.update({i, j})
                specs.append({"label": f"{state or 'State'} {pair_no} primary", "state": state,
                              "center": c1, "role": "primary", "pair": pair_no,
                              "delta_e": delta, "ratio": ratio})
                specs.append({"label": f"{state or 'State'} {pair_no} SO partner", "state": state,
                              "center": c2, "role": "partner", "pair": pair_no,
                              "delta_e": delta, "ratio": ratio})
                pair_no += 1
            else:
                used.add(i)
                distance_to_primary = min((abs(c1-r) for r in reference_primaries), default=0.0)
                distance_to_partner = min((abs(c1-(r+delta)) for r in reference_primaries), default=float('inf'))
                core_tolerance = max(1.25, min(3.0, 1.25*delta))
                if min(distance_to_primary, distance_to_partner) <= core_tolerance:
                    if distance_to_partner + 0.15 < distance_to_primary:
                        primary_center = c1 - delta
                        partner_center = c1
                    else:
                        primary_center = c1
                        partner_center = c1 + delta
                    specs.append({"label": f"{state or 'State'} {pair_no} primary", "state": state,
                                  "center": primary_center, "role": "primary", "pair": pair_no,
                                  "delta_e": delta, "ratio": ratio})
                    specs.append({"label": f"{state or 'State'} {pair_no} SO partner", "state": state,
                                  "center": partner_center, "role": "partner", "pair": pair_no,
                                  "delta_e": delta, "ratio": ratio})
                    warnings_out.append(
                        f"Generated the missing spin-orbit partner for the component near {c1:.2f} eV."
                    )
                    pair_no += 1
                else:
                    specs.append({"label": f"Peak {len(specs)+1}", "state": "", "center": c1, "role": "singlet"})
                    warnings_out.append(
                        f"The unpaired component at {c1:.2f} eV is outside the expected core-level doublet and was kept as an unassigned singlet."
                    )
        return specs, warnings_out, 'doublet'

    generic_state = "" if (db_info or {}).get('template_specs') else str((db_info or {}).get('state', ''))
    specs = [{"label": f"Peak {i}", "state": generic_state,
              "center": float(c), "role": "singlet"}
             for i, c in enumerate(raw_centers, start=1)]
    return specs, warnings_out, 'generic'


def _expanded_centers(raw_centers, db_info, auto_doublet):
    specs, _, _ = _build_peak_specs(raw_centers, db_info, auto_doublet)
    return sorted({float(spec['center']) for spec in specs})


def _filter_specs_to_data_range(specs, x, shift_bound):
    """Remove wholly unobserved optional/template components, but keep doublets intact."""
    xmin, xmax = float(np.min(x)), float(np.max(x))
    pad = max(float(shift_bound), 0.25)
    grouped = {}
    singles = []
    for spec in specs:
        pair = spec.get('pair')
        if not pair or spec.get('role') not in {'primary', 'partner'}:
            singles.append(spec)
        else:
            grouped.setdefault(pair, []).append(spec)

    kept = []
    removed = []
    for group in grouped.values():
        if any(xmin-pad <= float(s['center']) <= xmax+pad for s in group):
            kept.extend(group)
        else:
            removed.extend(group)
    for spec in singles:
        if xmin-pad <= float(spec['center']) <= xmax+pad:
            kept.append(spec)
        else:
            removed.append(spec)

    # Preserve the original display/template order.
    kept_ids = {id(s) for s in kept}
    ordered = [s for s in specs if id(s) in kept_ids]
    warning = None
    if removed:
        warning = "Components outside the acquired energy range were omitted: " + ", ".join(str(s.get('label', 'peak')) for s in removed)
    return ordered, warning


def _add_baseline_parameters(params, x, y, enabled):
    if not enabled:
        return
    y_span = max(float(np.ptp(y)), 1.0)
    edge_points = max(3, min(15, len(y)//15))
    edge_data = np.concatenate((y[:edge_points], y[-edge_points:]))
    mad = float(np.median(np.abs(edge_data - np.median(edge_data))))
    robust_noise = 1.4826 * mad
    correction_limit = max(4.0 * robust_noise, 0.04 * y_span, 1e-8)
    params.add('baseline_offset', value=0.0, min=-correction_limit, max=correction_limit)
    # baseline_slope is the intensity change across one full x-span.
    params.add('baseline_slope', value=0.0, min=-2.0*correction_limit, max=2.0*correction_limit)


def _build_parameters(x, y, specs, mode, shift_bound, shape,
                      fwhm_min=0.35, fwhm_max=4.0,
                      shared_shape=True, baseline_refine=True):
    """Build lmfit parameters from the generic component schema.

    Pair constraints enforce spin-orbit splitting, area ratio, FWHM, and shape.
    Named shift/FWHM/shape groups support multiplets and chemically related peaks
    without hard-coding any particular element.
    """
    if fwhm_max <= fwhm_min:
        raise ValueError("Maximum FWHM must be greater than minimum FWHM.")
    params = lmfit.Parameters()
    y_span = max(float(np.ptp(y)), 1.0)
    y_max = max(float(np.max(y)), y_span, 1.0)
    floor = max(y_span * 0.002, 1e-8)
    refs = [float(s['center']) for s in specs]

    independent = [(i, spec) for i, spec in enumerate(specs, start=1)
                   if spec.get('role', 'singlet') != 'partner']
    pair_primary_idx = {
        str(spec.get('pair', '')): i for i, spec in independent
        if spec.get('role') == 'primary'
    }
    independent_label_idx = {str(spec.get('label', '')): i for i, spec in independent}

    def center_limits(spec):
        center = float(spec['center'])
        center_mode = canonical_center_mode(
            spec.get('center_mode', ''), spec.get('role', 'singlet'),
            'center_min' in spec and 'center_max' in spec
        )
        if center_mode == 'bounded':
            lo = float(spec['center_min'])
            hi = float(spec['center_max'])
        else:
            # Free centers remain inside a global safety interval. This avoids
            # numerically valid but chemically meaningless peak migration.
            lo = float(spec.get('center_min', center - shift_bound))
            hi = float(spec.get('center_max', center + shift_bound))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError(f"Invalid center bounds for '{spec.get('label', 'peak')}'.")
        if not lo <= center <= hi:
            raise ValueError(f"Reference center of '{spec.get('label', 'peak')}' lies outside its center bounds.")
        return lo, hi

    def fwhm_limits(spec):
        lo = float(spec.get('fwhm_min', fwhm_min))
        hi = float(spec.get('fwhm_max', fwhm_max))
        start = float(spec.get('fwhm_init', 1.20))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo or lo <= 0:
            raise ValueError(f"Invalid FWHM bounds for '{spec.get('label', 'peak')}'.")
        if not lo <= start <= hi:
            raise ValueError(f"Initial FWHM of '{spec.get('label', 'peak')}' lies outside its bounds.")
        return lo, hi, start

    # Map user-defined group labels to safe lmfit parameter names.
    shift_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    width_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    shape_groups: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for item in independent:
        _, spec = item
        if spec.get('shift_group'):
            shift_groups.setdefault(str(spec['shift_group']), []).append(item)
        if spec.get('fwhm_group'):
            width_groups.setdefault(str(spec['fwhm_group']), []).append(item)
        peak_shape = canonical_lineshape(spec.get('lineshape', shape))
        explicit_shape_group = str(spec.get('shape_group', '')).strip()
        if explicit_shape_group:
            group_key = f"named:{explicit_shape_group}"
        elif shared_shape and not bool(spec.get('independent_shape', False)):
            group_key = f"shared:{peak_shape}"
        else:
            group_key = ""
        if group_key and peak_shape in {'pseudo_voigt', 'asymmetric_pseudo_voigt', 'doniach_sunjic'}:
            shape_groups.setdefault(group_key, []).append(item)

    shift_param: Dict[str, str] = {}
    for group_no, (group, members) in enumerate(shift_groups.items(), start=1):
        lower_offsets = []
        upper_offsets = []
        for _, spec in members:
            lo, hi = center_limits(spec)
            lower_offsets.append(lo - float(spec['center']))
            upper_offsets.append(hi - float(spec['center']))
        lo, hi = max(lower_offsets), min(upper_offsets)
        if hi <= lo:
            raise ValueError(f"shift_group '{group}' has incompatible center bounds.")
        name = f'group_shift_{group_no}'
        params.add(name, value=float(np.clip(0.0, lo, hi)), min=lo, max=hi)
        shift_param[group] = name

    width_param: Dict[str, str] = {}
    for group_no, (group, members) in enumerate(width_groups.items(), start=1):
        bounds = [fwhm_limits(spec) for _, spec in members]
        lo, hi = max(v[0] for v in bounds), min(v[1] for v in bounds)
        if hi <= lo:
            raise ValueError(f"fwhm_group '{group}' has incompatible width bounds.")
        start = float(np.clip(np.median([v[2] for v in bounds]), lo, hi))
        name = f'group_fwhm_{group_no}'
        params.add(name, value=start, min=lo, max=hi)
        width_param[group] = name

    shape_param: Dict[str, Dict[str, str]] = {}
    for group_no, (group, members) in enumerate(shape_groups.items(), start=1):
        shapes = {canonical_lineshape(spec.get('lineshape', shape)) for _, spec in members}
        if len(shapes) != 1:
            raise ValueError(f"Shape group '{group}' contains incompatible line shapes.")
        peak_shape = next(iter(shapes))
        group_parameters: Dict[str, str] = {}
        if peak_shape in {'pseudo_voigt', 'asymmetric_pseudo_voigt'}:
            lows = [float(spec.get('eta_min', 0.0)) for _, spec in members]
            highs = [float(spec.get('eta_max', 1.0)) for _, spec in members]
            starts = [float(spec.get('eta_init', 0.35)) for _, spec in members]
            lo, hi = max(lows), min(highs)
            if hi <= lo:
                raise ValueError(f"Shape group '{group}' has incompatible eta bounds.")
            eta_name = f'group_eta_{group_no}'
            params.add(eta_name, value=float(np.clip(np.median(starts), lo, hi)), min=lo, max=hi)
            group_parameters['eta'] = eta_name
        if peak_shape in {'doniach_sunjic', 'asymmetric_pseudo_voigt'}:
            lows = [float(spec.get('asym_min', 0.0)) for _, spec in members]
            default_asym_max = 0.80 if peak_shape == 'asymmetric_pseudo_voigt' else 0.25
            default_asym_init = 0.10 if peak_shape == 'asymmetric_pseudo_voigt' else 0.06
            highs = [float(spec.get('asym_max', default_asym_max)) for _, spec in members]
            starts = [float(spec.get('asym_init', default_asym_init)) for _, spec in members]
            lo, hi = max(lows), min(highs)
            if hi <= lo:
                raise ValueError(f"Shape group '{group}' has incompatible asymmetry bounds.")
            asym_name = f'group_asym_{group_no}'
            params.add(asym_name, value=float(np.clip(np.median(starts), lo, hi)), min=lo, max=hi)
            group_parameters['asym'] = asym_name
        shape_param[group] = group_parameters

    # Create all independent center parameters first. Cross-component ties are
    # attached only after every possible target exists, so table/JSON row order
    # cannot change the meaning of a model.
    for i, spec in independent:
        center = float(spec['center'])
        center_mode = canonical_center_mode(
            spec.get('center_mode', ''), spec.get('role', 'singlet'),
            'center_min' in spec and 'center_max' in spec
        )
        center_group = str(spec.get('shift_group', '')).strip()
        if center_group:
            params.add(f'p{i}_center', expr=f'{center:.10f}+{shift_param[center_group]}')
        elif center_mode == 'fixed':
            params.add(f'p{i}_center', value=center, vary=False)
        elif center_mode == 'tied':
            params.add(f'p{i}_center', value=center, vary=False)
        else:
            lo, hi = center_limits(spec)
            params.add(f'p{i}_center', value=center, min=lo, max=hi)

    for i, spec in independent:
        center_mode = canonical_center_mode(
            spec.get('center_mode', ''), spec.get('role', 'singlet'),
            'center_min' in spec and 'center_max' in spec
        )
        if center_mode != 'tied':
            continue
        target_label = str(spec.get('tie_target', '')).strip()
        if target_label not in independent_label_idx:
            raise ValueError(
                f"Tied component '{spec.get('label', i)}' targets unavailable independent component '{target_label}'."
            )
        target_idx = independent_label_idx[target_label]
        offset_init = float(spec.get('offset_init', float(spec['center']) - float(specs[target_idx-1]['center'])))
        offset_name = f'p{i}_center_offset'
        if 'offset_min' in spec and 'offset_max' in spec:
            omin, omax = float(spec['offset_min']), float(spec['offset_max'])
            if abs(omax - omin) <= 1e-12:
                params.add(offset_name, value=offset_init, vary=False)
            else:
                params.add(offset_name, value=offset_init, min=omin, max=omax)
        else:
            params.add(offset_name, value=offset_init, vary=False)
        params[f'p{i}_center'].set(expr=f'p{target_idx}_center+{offset_name}')

    # Add remaining parameters for independent components. Partner expressions
    # are created afterward from their primary doublet member.
    for i, spec in independent:
        center = float(spec['center'])
        estimated_h0 = _initial_height(x, y, center, refs, floor)
        h0 = float(spec.get('height_init', estimated_h0))
        if not np.isfinite(h0) or h0 <= 0:
            raise ValueError(f"Initial height of '{spec.get('label', 'peak')}' must be positive and finite.")
        h0 = max(h0, floor)
        params.add(f'p{i}_height', value=h0, min=0.0, max=max(3.0*y_max, h0*8.0))

        width_group = str(spec.get('fwhm_group', '')).strip()
        if width_group:
            params.add(f'p{i}_fwhm', expr=width_param[width_group])
        else:
            lo, hi, start = fwhm_limits(spec)
            params.add(f'p{i}_fwhm', value=start, min=lo, max=hi)

        peak_shape = canonical_lineshape(spec.get('lineshape', shape))
        explicit_shape_group = str(spec.get('shape_group', '')).strip()
        if explicit_shape_group:
            group_key = f"named:{explicit_shape_group}"
        elif shared_shape and not bool(spec.get('independent_shape', False)):
            group_key = f"shared:{peak_shape}"
        else:
            group_key = ""

        if peak_shape == 'doniach_sunjic':
            params.add(f'p{i}_eta', value=0.0, vary=False)
            if group_key:
                params.add(f'p{i}_asym', expr=shape_param[group_key]['asym'])
            else:
                amin = float(spec.get('asym_min', 0.0)); amax = float(spec.get('asym_max', 0.25))
                ainit = float(spec.get('asym_init', 0.06))
                if amax <= amin:
                    raise ValueError(f"Invalid asymmetry bounds for '{spec.get('label', 'peak')}'.")
                params.add(f'p{i}_asym', value=float(np.clip(ainit, amin, amax)), min=amin, max=amax)
        elif peak_shape == 'gaussian':
            params.add(f'p{i}_eta', value=0.0, vary=False)
            params.add(f'p{i}_asym', value=0.0, vary=False)
        elif peak_shape == 'lorentzian':
            params.add(f'p{i}_eta', value=1.0, vary=False)
            params.add(f'p{i}_asym', value=0.0, vary=False)
        elif peak_shape == 'asymmetric_pseudo_voigt':
            if group_key:
                params.add(f'p{i}_eta', expr=shape_param[group_key]['eta'])
                params.add(f'p{i}_asym', expr=shape_param[group_key]['asym'])
            else:
                emin = float(spec.get('eta_min', 0.0)); emax = float(spec.get('eta_max', 1.0))
                einit = float(spec.get('eta_init', 0.35))
                amin = float(spec.get('asym_min', 0.0)); amax = float(spec.get('asym_max', 0.80))
                ainit = float(spec.get('asym_init', 0.10))
                if emax <= emin or amax <= amin:
                    raise ValueError(f"Invalid asymmetric pseudo-Voigt bounds for '{spec.get('label', 'peak')}'.")
                params.add(f'p{i}_eta', value=float(np.clip(einit, emin, emax)), min=emin, max=emax)
                params.add(f'p{i}_asym', value=float(np.clip(ainit, amin, amax)), min=amin, max=amax)
        else:
            if group_key:
                params.add(f'p{i}_eta', expr=shape_param[group_key]['eta'])
            else:
                emin = float(spec.get('eta_min', 0.0)); emax = float(spec.get('eta_max', 1.0))
                einit = float(spec.get('eta_init', 0.35))
                if emax <= emin:
                    raise ValueError(f"Invalid pseudo-Voigt eta bounds for '{spec.get('label', 'peak')}'.")
                params.add(f'p{i}_eta', value=float(np.clip(einit, emin, emax)), min=emin, max=emax)
            params.add(f'p{i}_asym', value=0.0, vary=False)

    for i, spec in enumerate(specs, start=1):
        if spec.get('role', 'singlet') != 'partner':
            continue
        pair = str(spec.get('pair', ''))
        if pair not in pair_primary_idx:
            raise ValueError(f"Partner '{spec.get('label', i)}' has no matching primary.")
        lead = pair_primary_idx[pair]
        delta = float(spec['delta_e'])
        ratio = float(spec['ratio'])
        params.add(f'p{i}_center', expr=f'p{lead}_center+{delta:.10f}')
        params.add(f'p{i}_fwhm', expr=f'p{lead}_fwhm')
        params.add(f'p{i}_height', expr=f'p{lead}_height*{ratio:.10f}')
        params.add(f'p{i}_eta', expr=f'p{lead}_eta')
        params.add(f'p{i}_asym', expr=f'p{lead}_asym')

    _add_baseline_parameters(params, x, y, baseline_refine)
    return params


def _apply_manual_constraints(params, text):
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        target, expr = line.split('=', 1)
        target, expr = target.strip(), expr.strip()
        if target not in params:
            raise ValueError(f"Manual constraint target '{target}' does not exist.")
        params[target].set(expr=expr)


def _perturb_parameters(base_params, rng, strength=1.0):
    params = copy.deepcopy(base_params)
    for name, par in params.items():
        if not par.vary or par.expr is not None:
            continue
        v = float(par.value)
        lname = name.lower()
        if 'height' in lname:
            factor = math.exp(rng.normal(0.0, 0.55 * strength))
            nv = max(v * factor, 1e-12)
        elif 'fwhm' in lname:
            factor = math.exp(rng.normal(0.0, 0.28 * strength))
            nv = max(v * factor, 0.05)
        elif 'eta' in lname:
            nv = rng.uniform(0.15, 0.85)
        elif 'asym' in lname:
            nv = rng.uniform(0.01, 0.20)
        elif 'center' in lname or 'shift' in lname:
            span = (par.max - par.min) if np.isfinite(par.max) and np.isfinite(par.min) else 1.0
            nv = v + rng.normal(0.0, 0.18 * max(span, 0.1) * strength)
        else:
            nv = v + rng.normal(0.0, 0.15 * max(abs(v), 1.0) * strength)
        par.value = _clip_to_bounds(par, nv)
    params.update_constraints()
    return params


def _raw_rss(params, x, y, n_peaks, shape, weights=None):
    yhat, _ = xps_model_eval(params, x, n_peaks, shape)
    residual = y - yhat
    if weights is not None:
        residual = residual * np.asarray(weights, dtype=float)
    return float(np.sum(residual**2))


def _near_bound(par, frac=0.0075):
    if not par.vary or par.expr is not None:
        return False
    lo, hi, val = par.min, par.max, par.value
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        return ((val-lo)/(hi-lo) < frac) or ((hi-val)/(hi-lo) < frac)
    if np.isfinite(lo):
        return abs(val-lo) <= frac * max(abs(val), 1.0)
    if np.isfinite(hi):
        return abs(hi-val) <= frac * max(abs(val), 1.0)
    return False


def _max_parameter_correlation(params):
    max_corr = 0.0
    pair = None
    for name, par in params.items():
        corr = getattr(par, 'correl', None) or {}
        for other, val in corr.items():
            av = abs(float(val))
            if av > max_corr:
                max_corr = av
                pair = (name, other, float(val))
    return max_corr, pair


def _stderr_value(result, name):
    par = result.params[name]
    if par.expr is None and par.stderr is not None and np.isfinite(par.stderr):
        return float(par.stderr)
    return _propagated_stderr(result, lambda p: p[name].value)


# --- Multi-threaded Fit Worker ---
class FitWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, ctx, settings):
        super().__init__()
        self.ctx = ctx
        self.settings = settings

    def run(self):
        try:
            x = np.asarray(self.ctx["x_fit"], dtype=float)
            y = np.asarray(self.ctx["y_corr"], dtype=float)
            y_raw = np.asarray(self.ctx["y_raw"], dtype=float)
            raw_centers = list(self.ctx["centers"])
            shift_bound = float(self.ctx["shift_bound"])
            global_shape = canonical_lineshape(self.settings['shape'])
            db_info = self.settings.get('db_info')
            auto_doublet = bool(self.settings.get('auto_doublet'))

            custom_specs = self.settings.get('model_specs')
            if custom_specs is not None:
                specs = copy.deepcopy(custom_specs)
                build_warnings = []
                mode = 'custom_model'
            else:
                specs, build_warnings, mode = _build_peak_specs(raw_centers, db_info, auto_doublet)

            specs, model_warnings = normalize_model_specs(
                specs, global_shape,
                float(self.settings.get('fwhm_min', 0.35)),
                float(self.settings.get('fwhm_max', 4.0)),
            )
            build_warnings.extend(model_warnings)
            cross_ties = [
                spec for spec in specs
                if spec.get('center_mode') == 'tied' and spec.get('role') != 'partner'
            ]
            if cross_ties:
                tie_summary = "; ".join(
                    f"{spec['label']} -> {spec.get('tie_target')} "
                    + (
                        f"[{spec['offset_min']:+.3f}, {spec['offset_max']:+.3f}] eV"
                        if 'offset_min' in spec and 'offset_max' in spec
                        else f"{spec.get('offset_init', 0.0):+.3f} eV (fixed)"
                    )
                    for spec in cross_ties
                )
                self.progress.emit("Cross-component center constraints: " + tie_summary)
            specs, range_warning = _filter_specs_to_data_range(specs, x, shift_bound)
            if range_warning:
                build_warnings.append(range_warning)
            n_peaks = len(specs)
            if n_peaks < 1:
                raise ValueError("No peak components were generated.")
            peak_shapes = [canonical_lineshape(spec.get('lineshape', global_shape)) for spec in specs]

            if 'doniach_sunjic' in peak_shapes:
                self.progress.emit("Warning: Doniach-Sunjic is intended mainly for genuinely asymmetric metallic lines; oxide peaks should normally remain symmetric.")
                if any(
                    spec.get('role') == 'partner' and peak_shapes[i] == 'doniach_sunjic'
                    for i, spec in enumerate(specs)
                ):
                    build_warnings.append(
                        "A Doniach-Sunjic doublet uses the requested ratio as a height ratio, while reported DS areas are fit-window areas. "
                        "For quantitative branching ratios, prefer the integrable asymmetric pseudo-Voigt model."
                    )
            model_reference = str(self.settings.get('model_reference', '')).strip()
            has_state_assignments = any(
                str(spec.get('state', '')).strip()
                for spec in specs if spec.get('quantify', True)
            )
            unverified_source = lambda spec: (
                not str(spec.get('source', '')).strip()
                or 'verify' in str(spec.get('source', '')).lower()
                or 'built-in' in str(spec.get('source', '')).lower()
            )
            if (
                has_state_assignments and not model_reference
                and any(unverified_source(spec) for spec in specs)
            ):
                build_warnings.append(
                    "One or more chemical assignments have no recorded source. Add a DOI, standard, or reference-spectrum identifier before publication."
                )
            if not has_state_assignments:
                build_warnings.append(
                    "Exploratory free fit: no chemical-state labels were supplied. The envelope can be fitted, "
                    "but peak count alone is not evidence for oxidation states and state fractions cannot be reported."
                )

            background_method = str(
                self.ctx.get("background_method", "")
            ).strip()
            background_method_lower = background_method.lower()
            if "manual" in background_method_lower:
                anchor_count = len(self.ctx.get("manual_baseline_points", []) or [])
                build_warnings.append(
                    f"Operator-defined piecewise-linear background ({anchor_count} recorded anchors). "
                    "This is auditable but not a physical inelastic-scattering model; justify it and test "
                    "background sensitivity before reporting component areas."
                )
            elif background_method_lower == "linear":
                build_warnings.append(
                    "A linear background is normally defensible only when the fitted interval has no "
                    "appreciable inelastic-loss step; compare against a sample-appropriate alternative."
                )
            elif background_method_lower == "none":
                build_warnings.append(
                    "No background was subtracted. This is appropriate only for data already background-corrected "
                    "or for a region with a justified zero/constant baseline."
                )

            x_steps = np.diff(x)
            if len(x_steps) and np.all(x_steps > 0):
                median_step = max(float(np.median(x_steps)), 1e-30)
                if float(np.max(x_steps) / median_step) > 1.15 or float(median_step / np.min(x_steps)) > 1.15:
                    build_warnings.append(
                        "Binding-energy spacing is non-uniform by more than 15%; ordinary pointwise least squares "
                        "gives densely sampled intervals more influence. Resample or use acquisition-derived uncertainties if needed."
                    )

            weighting = str(self.settings.get('weighting', 'uniform')).strip().lower()
            weights = None
            if weighting == 'poisson':
                if np.any(~np.isfinite(y_raw)) or np.any(y_raw < 0):
                    raise ValueError("Poisson weighting requires finite, non-negative raw count data.")
                weights = 1.0 / np.sqrt(np.maximum(y_raw, 1.0))
                build_warnings.append(
                    "Poisson weighting assumes intensity is proportional to raw counts with consistent dwell time/scans; "
                    "do not use it for arbitrarily normalized spectra."
                )
                weighting_label = "Poisson (raw-count variance)"
            elif weighting == 'uniform':
                weighting_label = "Uniform"
            else:
                raise ValueError(f"Unknown residual weighting method: {weighting}")

            if mode == 'ce3d_mixed':
                self.progress.emit("Ce 3d physics template: 10 components (4 Ce3+ + 6 Ce4+) with state-coupled shifts/FWHM and paired u/v areas.")
            elif mode == 'ce3d_ce3':
                self.progress.emit("Ce 3d Ce3+ template: 4 components with coupled shift/FWHM and paired u/v areas.")
            elif mode == 'doublet':
                self.progress.emit("Spin-orbit constraints active: fixed splitting, fixed area ratio, equal FWHM and shared line shape within each doublet.")
            elif mode == 'wo3x_zr4p':
                self.progress.emit(
                    "Combined W/Zr template active: constrained W6+/W5+ 4f doublets, "
                    "broad Zr 4p doublet, and optional W 5p3/2. Zr 4p and W 5p are "
                    "excluded from W oxidation-state fractions."
                )
            elif mode in {'w6_w4f', 'w4_w4f', 'wo3x_w4f'}:
                self.progress.emit("W 4f template active: constrained oxide doublet(s) with an optional broad W 5p3/2 contribution.")
            elif mode == 'zr4p_zro2':
                self.progress.emit(
                    "ZrO2 Zr 4p template active: broad 4p3/2/4p1/2 doublet with "
                    "1.68 eV splitting and 0.50 partner/primary area ratio."
                )
            elif mode == 'custom_model':
                self.progress.emit("Reference-driven Model Builder active: arbitrary singlets, doublets, satellites, multiplets, and named parameter groups.")
            else:
                self.progress.emit("Independent-component fitting with physical bounds.")

            base_params = _build_parameters(
                x, y, specs, mode, shift_bound, global_shape,
                fwhm_min=float(self.settings.get('fwhm_min', 0.35)),
                fwhm_max=float(self.settings.get('fwhm_max', 4.0)),
                shared_shape=bool(self.settings.get('shared_shape', True)),
                baseline_refine=bool(self.settings.get('baseline_refine', True)),
            )
            _apply_manual_constraints(base_params, self.settings.get('constraints', ''))

            varying_count = sum(bool(p.vary and p.expr is None) for p in base_params.values())
            if len(x) <= varying_count + 5:
                raise ValueError(
                    f"The fit has {varying_count} free parameters but only {len(x)} data points. "
                    "Use a wider region, fewer components, or stronger physical constraints."
                )

            # A constant robust scale only normalizes the unweighted residual magnitude;
            # it cannot alter the optimum. Poisson residuals retain their statistical scale.
            residual_scale = (
                1.0 if weights is not None else
                max(float(np.percentile(np.abs(y - np.median(y)), 90)), float(np.std(y)), 1.0)
            )
            rng = np.random.default_rng(FIT_RANDOM_SEED)
            candidates = []
            starts_done = 0
            stage_sizes = [8, 8, 8]
            if mode.startswith('ce3d'):
                stage_sizes = [10, 10, 8]

            coarse_ftol = max(float(self.settings['ftol']), 1e-7)
            coarse_xtol = max(float(self.settings['xtol']), 1e-7)
            coarse_nfev = min(int(self.settings['max_iter']), 3000)

            for stage_idx, stage_size in enumerate(stage_sizes, start=1):
                for j in range(stage_size):
                    starts_done += 1
                    p0 = copy.deepcopy(base_params) if starts_done == 1 else _perturb_parameters(base_params, rng, strength=1.0 + 0.15*stage_idx)
                    try:
                        out = lmfit.minimize(
                            xps_residual, p0,
                            args=(x, y, n_peaks, peak_shapes, residual_scale, weights),
                            method='least_squares',
                            max_nfev=coarse_nfev,
                            ftol=coarse_ftol, xtol=coarse_xtol, gtol=coarse_ftol,
                            loss='soft_l1', f_scale=1.0, x_scale='jac'
                        )
                        objective = _raw_rss(out.params, x, y, n_peaks, peak_shapes, weights)
                        if np.isfinite(objective):
                            candidates.append((objective, out))
                    except Exception:
                        continue

                candidates.sort(key=lambda z: z[0])
                self.progress.emit(f"Adaptive multi-start: {starts_done} starts evaluated; {len(candidates)} converged.")
                if len(candidates) >= 3:
                    top = np.array([z[0] for z in candidates[:3]], dtype=float)
                    spread = float((top.max() - top.min()) / max(top.min(), 1e-30))
                    if spread < 0.005:
                        self.progress.emit("Independent starts converged to the same basin; stopping multi-start early.")
                        break

            if not candidates:
                raise RuntimeError("Fitting failed for all multi-start trials.")

            self.progress.emit("Running high-precision refinement of the best independent basins...")
            final_ftol = min(float(self.settings['ftol']), 1e-9)
            final_xtol = min(float(self.settings['xtol']), 1e-9)
            refined = []
            # Refining several robust-loss solutions avoids assuming that the
            # best coarse basin is also the best ordinary least-squares basin.
            for _, coarse_result in candidates[:min(5, len(candidates))]:
                try:
                    candidate_result = lmfit.minimize(
                        xps_residual, copy.deepcopy(coarse_result.params),
                        args=(x, y, n_peaks, peak_shapes, residual_scale, weights),
                        method='least_squares',
                        max_nfev=max(int(self.settings['max_iter']), 7000),
                        ftol=final_ftol, xtol=final_xtol, gtol=final_ftol,
                        loss='linear', x_scale='jac'
                    )
                    candidate_objective = _raw_rss(candidate_result.params, x, y, n_peaks, peak_shapes, weights)
                    if np.isfinite(candidate_objective):
                        refined.append((candidate_objective, candidate_result))
                except Exception:
                    continue
            if not refined:
                raise RuntimeError("High-precision refinement failed for all candidate basins.")
            refined.sort(key=lambda item: item[0])
            final_result = refined[0][1]
            self.progress.emit(f"High-precision refinement completed for {len(refined)} candidate basins.")

            y_fit, comps = xps_model_eval(final_result.params, x, n_peaks, peak_shapes)
            resid = y - y_fit
            rss = float(np.sum(resid**2))
            objective_value = float(np.sum((resid * weights)**2)) if weights is not None else rss
            n = len(y)
            k = max(int(getattr(final_result, 'nvarys', len(getattr(final_result, 'var_names', [])))), 1)
            dof = max(n - k, 1)
            ss_tot = float(np.sum((y - np.mean(y))**2))
            r2 = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(rss / max(n, 1)))
            nrmse_percent = float(100.0 * rmse / max(float(np.ptp(y)), 1e-30))
            reduced_sse = objective_value / dof
            if weights is not None:
                # Gaussian likelihood with known pointwise Poisson variance; the
                # model-independent normalization constant is omitted.
                aic = float(objective_value + 2*k)
                bic = float(objective_value + k*np.log(max(n, 2)))
            else:
                rss_per_n = max(rss / max(n, 1), 1e-300)
                aic = float(n * np.log(rss_per_n) + 2*k)
                bic = float(n * np.log(rss_per_n) + k*np.log(max(n, 2)))
            aicc = float(aic + (2*k*(k+1))/(n-k-1)) if n > k + 1 else float('inf')

            max_corr, corr_pair = _max_parameter_correlation(final_result.params)
            lag1 = 0.0
            if len(resid) > 3 and np.std(resid[:-1]) > 0 and np.std(resid[1:]) > 0:
                lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
            durbin_watson = float(np.sum(np.diff(resid)**2) / max(np.sum(resid**2), 1e-30))

            warnings_list = list(build_warnings)
            if not final_result.success:
                warnings_list.append("Optimizer did not report successful convergence: " + str(final_result.message))
            bound_names = [name for name, par in final_result.params.items() if _near_bound(par)]
            if bound_names:
                warnings_list.append("Parameters at/near bounds: " + ", ".join(bound_names))

            high_unc = []
            for name in getattr(final_result, 'var_names', []):
                par = final_result.params[name]
                if par.stderr is None or not np.isfinite(par.stderr):
                    high_unc.append(f"{name} (uncertainty unavailable)")
                elif abs(par.value) > 1e-12 and abs(par.stderr/par.value) > 0.50:
                    high_unc.append(f"{name} ({abs(par.stderr/par.value):.0%})")
            if high_unc:
                warnings_list.append("Poorly determined parameters: " + ", ".join(high_unc[:8]))

            if max_corr > 0.95 and corr_pair:
                warnings_list.append(f"Strong parameter correlation: {corr_pair[0]} vs {corr_pair[1]} = {corr_pair[2]:+.3f}")
            if abs(lag1) > 0.60:
                warnings_list.append(f"Structured residuals detected (lag-1 correlation {lag1:+.2f}); consider background/peak model review.")
            if nrmse_percent > 5.0:
                warnings_list.append(f"The normalized RMSE is {nrmse_percent:.2f}% of the corrected signal range.")
            if r2 < 0.95:
                warnings_list.append(f"Low envelope agreement (R²={r2:.4f}).")
            if n < 15 * max(k, 1):
                warnings_list.append(
                    f"Only {n/max(k, 1):.1f} data points per varying parameter are available; "
                    "component areas may be sensitive to model complexity."
                )
            background_array = np.asarray(
                self.ctx.get("background", np.zeros_like(y_raw)), dtype=float
            )
            background_scale = max(float(np.ptp(y_raw)), 1e-30)
            background_above = float(np.mean(background_array > y_raw + 0.02 * background_scale))
            if background_above > 0.05:
                warnings_list.append(
                    f"The selected static background lies appreciably above {100*background_above:.1f}% "
                    "of measured points; review endpoints/anchors and the fit interval."
                )
            corrected_span = max(float(np.ptp(y)), 1e-30)
            negative_fraction = float(np.mean(y < -0.02 * corrected_span))
            if negative_fraction > 0.10:
                warnings_list.append(
                    f"{100*negative_fraction:.1f}% of background-subtracted points are appreciably negative; "
                    "review the fit window and background endpoints."
                )
            if rmse > 0:
                outlier_count = int(np.sum(np.abs(resid) > 5.0 * rmse))
                if outlier_count:
                    warnings_list.append(
                        f"{outlier_count} point(s) exceed five RMSE from the fitted envelope; inspect for spikes or acquisition artefacts."
                    )
            if 'doniach_sunjic' in peak_shapes:
                warnings_list.append(
                    "Doniach-Sunjic areas are finite-window areas and must not be compared across different fit windows; "
                    "the FWHM column reports the model width parameter."
                )

            baseline = _baseline_correction(final_result.params, x)
            if np.max(np.abs(baseline)) > 0.08 * max(float(np.ptp(y)), 1e-30):
                warnings_list.append("The residual baseline correction exceeds 8% of the corrected signal range; review the background endpoints/model.")

            center_bound_names = [name for name in bound_names if 'center' in name]
            if center_bound_names:
                warnings_list.append(
                    "Peak-center constraint(s) reached a bound: " + ", ".join(center_bound_names)
                    + ". Do not report chemical-state fractions until the assignment and bounds are reviewed."
                )
            severe = (
                (r2 < 0.90)
                or (max_corr > 0.995)
                or (len(bound_names) >= max(2, k//2))
                or (abs(lag1) > 0.85)
                or (durbin_watson < 0.35)
                or (bool(center_bound_names) and max_corr > 0.95)
                or (not final_result.success)
            )

            peak_results = []
            areas = []
            for i, spec in enumerate(specs, start=1):
                p = final_result.params
                area = _component_area(p, i, peak_shapes, x)
                areas.append(area)
                bw = []
                for pname in (f'p{i}_center', f'p{i}_fwhm', f'p{i}_height'):
                    if _near_bound(p[pname]):
                        bw.append(pname)
                offset_name = f'p{i}_center_offset'
                if offset_name in p and _near_bound(p[offset_name]):
                    bw.append(offset_name)
                center_mode = canonical_center_mode(
                    spec.get('center_mode', ''), spec.get('role', 'singlet'),
                    'center_min' in spec and 'center_max' in spec
                )
                peak_results.append(PeakResult(
                    label=str(spec.get('label', f'Peak {i}')),
                    center_input=float(spec['center']),
                    center_fitted=float(p[f'p{i}_center'].value),
                    fwhm_fitted=float(p[f'p{i}_fwhm'].value),
                    height_fitted=float(p[f'p{i}_height'].value),
                    area_fitted=area,
                    eta_fitted=float(p[f'p{i}_eta'].value),
                    asym_fitted=float(p[f'p{i}_asym'].value),
                    state=str(spec.get('state', '')),
                    role=str(spec.get('role', 'singlet')),
                    pair=str(spec.get('pair', '')),
                    lineshape=peak_shapes[i-1],
                    source=str(spec.get('source', '')),
                    quantify=bool(spec.get('quantify', True)),
                    center_mode=center_mode,
                    center_min=float(spec['center_min']) if 'center_min' in spec else None,
                    center_max=float(spec['center_max']) if 'center_max' in spec else None,
                    tie_target=str(spec.get('tie_target', '')),
                    offset_fitted=float(p[offset_name].value) if offset_name in p else (
                        float(spec['delta_e']) if spec.get('role') == 'partner' else None
                    ),
                    offset_stderr=_stderr_value(final_result, offset_name) if offset_name in p else None,
                    offset_min=float(spec['offset_min']) if 'offset_min' in spec else None,
                    offset_max=float(spec['offset_max']) if 'offset_max' in spec else None,
                    center_stderr=_stderr_value(final_result, f'p{i}_center'),
                    fwhm_stderr=_stderr_value(final_result, f'p{i}_fwhm'),
                    height_stderr=_stderr_value(final_result, f'p{i}_height'),
                    area_stderr=_propagated_stderr(
                        final_result,
                        lambda pars, idx=i: _component_area(pars, idx, peak_shapes, x)
                    ),
                    bound_warning=", ".join(bw)
                ))

            quant_indices = [
                i for i, spec in enumerate(specs, start=1)
                if bool(spec.get('quantify', True))
            ]
            total_area = max(float(sum(areas[i-1] for i in quant_indices)), 1e-30)
            for i, pr in enumerate(peak_results, start=1):
                if pr.quantify:
                    pr.area_fraction = float(pr.area_fitted / total_area)
                    pr.area_fraction_stderr = _propagated_stderr(
                        final_result,
                        lambda pars, idx=i, qidx=tuple(quant_indices): (
                            _component_area(pars, idx, peak_shapes, x)
                            / max(sum(_component_area(pars, j, peak_shapes, x) for j in qidx), 1e-30)
                        )
                    )
                else:
                    pr.area_fraction = None
                    pr.area_fraction_stderr = None

            state_areas = {}
            for pr in peak_results:
                if pr.state and pr.quantify:
                    state_areas[pr.state] = state_areas.get(pr.state, 0.0) + pr.area_fitted
            state_fractions = {state: area / max(sum(state_areas.values()), 1e-30) for state, area in state_areas.items()}
            state_fraction_stderr = {}
            state_to_indices = {
                state: [i for i, pr in enumerate(peak_results, start=1) if pr.state == state and pr.quantify]
                for state in state_areas
            }
            quant_indices = [i for i, pr in enumerate(peak_results, start=1) if pr.quantify]
            for state, indices in state_to_indices.items():
                state_fraction_stderr[state] = _propagated_stderr(
                    final_result,
                    lambda pars, idxs=tuple(indices): (
                        sum(_component_area(pars, j, peak_shapes, x) for j in idxs)
                        / max(sum(_component_area(pars, j, peak_shapes, x) for j in quant_indices), 1e-30)
                    )
                )

            effectively_zero_states = [
                state for state, fraction in state_fractions.items()
                if fraction < 1e-4
            ]
            if effectively_zero_states:
                warnings_list.append(
                    "State fraction converged to effectively zero (<0.01%): "
                    + ", ".join(effectively_zero_states)
                    + ". Adding a component to the model does not by itself demonstrate that state; "
                    "review its binding-energy constraints, background and residual before forcing a non-zero value."
                )

            unresolved = [
                pr.label for pr in peak_results
                if pr.quantify and pr.area_stderr is not None and pr.area_stderr > 0
                and pr.area_fitted / pr.area_stderr < 2.0
            ]
            if unresolved:
                warnings_list.append(
                    "Components not resolved above approximately 2 standard errors: "
                    + ", ".join(unresolved)
                )
            edge_limited = [
                pr.label for pr in peak_results
                if (pr.center_fitted - float(np.min(x)) < pr.fwhm_fitted)
                or (float(np.max(x)) - pr.center_fitted < pr.fwhm_fitted)
            ]
            if edge_limited:
                warnings_list.append(
                    "Components within one FWHM of a fit boundary: " + ", ".join(edge_limited)
                    + ". Expand the acquired/fit range before quantitative reporting."
                )
            heavy_overlap = []
            for left_index, left_peak in enumerate(peak_results):
                for right_peak in peak_results[left_index+1:]:
                    separation = abs(left_peak.center_fitted - right_peak.center_fitted)
                    width_scale = 0.50 * min(left_peak.fwhm_fitted, right_peak.fwhm_fitted)
                    if separation < width_scale:
                        heavy_overlap.append(f"{left_peak.label} / {right_peak.label}")
            if heavy_overlap:
                warnings_list.append(
                    "Strongly overlapping components (center separation < 0.5 × narrower FWHM): "
                    + ", ".join(heavy_overlap[:6])
                    + ". Their individual areas are model-dependent and require independent chemical justification."
                )
            # Include post-fit component diagnostics in the overall assessment.
            fit_quality = "POOR" if severe else ("REVIEW" if warnings_list else "GOOD")

            fit_m = FitMetrics(
                rmse=rmse, nrmse_percent=nrmse_percent, r2=float(r2), rss=rss, reduced_chi2=float(reduced_sse),
                aic=aic, aicc=aicc, bic=bic, nvarys=k,
                max_abs_corr=float(max_corr), residual_lag1_corr=float(lag1),
                durbin_watson=durbin_watson,
                objective_value=objective_value, weighting=weighting_label,
                fit_quality=fit_quality, warning_count=len(warnings_list)
            )

            self.finished.emit({
                "success": bool(final_result.success),
                "message": str(final_result.message),
                "y_fit": y_fit,
                "components": comps,
                "metrics": fit_m,
                "peak_results": peak_results,
                "warnings": warnings_list,
                "state_fractions": state_fractions,
                "state_fraction_stderr": state_fraction_stderr,
                "baseline_correction": baseline,
                "n_starts": starts_done,
                "n_refined": len(refined),
                "mode": mode,
                "model_specs": specs,
                "peak_shapes": peak_shapes,
            })
        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


# --- Plot Canvas ---
class XPSPlotCanvas(FigureCanvas):
    peak_selected = pyqtSignal(float)
    cursor_moved = pyqtSignal(float, float)
    cursor_left_axes = pyqtSignal()
    centers_updated = pyqtSignal(list) # Signal for drag & drop
    fit_component_selected = pyqtSignal(int)
    fit_component_edited = pyqtSignal(int, object)
    manual_baseline_changed = pyqtSignal(list)

    def __init__(self):
        self.figure = Figure(figsize=(9.2, 7.0), dpi=110, facecolor="white")
        super().__init__(self.figure)
        grid = self.figure.add_gridspec(2, 1, height_ratios=[3.25, 1.0], hspace=0.16)
        self.ax_main = self.figure.add_subplot(grid[0])
        self.ax_resid = self.figure.add_subplot(grid[1], sharex=self.ax_main)
        self.figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.84)
        self._selection_mode, self._reverse_x = False, True
        self._crosshair_v, self._crosshair_h, self._coord_text = None, None, None
        
        self._dragging_line = None
        self._dragging_idx = None
        self._lines, self._points, self._texts = [], [], []
        self.x_data, self.y_data = None, None
        self._fit_editing = False
        self._fit_selected_idx = -1
        self._fit_component_lines = []
        self._fit_apex_handle = None
        self._fit_width_handles = []
        self._fit_drag_kind = None
        self._fit_drag_state = None
        self._fit_background = None
        self._manual_baseline_editing = False
        self._manual_baseline_points: List[Tuple[float, float]] = []
        self._manual_baseline_artist = None
        self._manual_baseline_line = None
        self._manual_dragging_idx: Optional[int] = None
        self._empty_message_artist = None

        self.mpl_connect("button_press_event", self._on_click)
        self.mpl_connect("motion_notify_event", self._on_mouse_move)
        self.mpl_connect("button_release_event", self._on_release)
        self.mpl_connect("axes_leave_event", self._on_axes_leave)
        self.mpl_connect("scroll_event", self._on_scroll)
        self.plot_empty_state()

    def set_reverse_x(self, enabled): self._reverse_x = enabled
    def enable_peak_selection(self, enabled): self._selection_mode = enabled
    def set_fit_editing(self, enabled: bool): self._fit_editing = bool(enabled)
    def set_fit_selected_component(self, index: int): self._fit_selected_idx = int(index)

    def _set_figure_title(self, text: str = ""):
        """Keep fit titles inside the canvas instead of clipping above the axes."""
        title_artist = getattr(self.figure, "_suptitle", None)
        if title_artist is None:
            self.figure.suptitle(
                text, x=0.105, y=0.975, ha="left", va="top",
                fontsize=11.5, fontweight="semibold", color="#0F172A"
            )
        else:
            title_artist.set_text(text)

    def set_manual_baseline_editing(self, enabled: bool):
        self._manual_baseline_editing = bool(enabled)
        if enabled:
            # Background editing and peak/component dragging are intentionally
            # mutually exclusive so one click can never change two models.
            self._selection_mode = False
            self._fit_editing = False

    def set_manual_baseline_points(self, points: Any):
        self._manual_baseline_points = [
            (float(point[0]), float(point[1])) for point in (points or [])
        ]
        self._update_manual_baseline_artists()

    def plot_empty_state(self, message: str = "Open an XPS data file to begin"):
        """Show a calm workspace before data are loaded instead of empty axes."""
        self._set_figure_title("")
        self._reset_crosshair()
        self._reset_fit_artists()
        self._reset_manual_baseline_artists(keep_points=False)
        self.ax_main.clear(); self.ax_resid.clear()
        self.ax_main.axis("off"); self.ax_resid.axis("off")
        if self._empty_message_artist is not None:
            try:
                self._empty_message_artist.remove()
            except (AttributeError, ValueError, NotImplementedError):
                pass
        self._empty_message_artist = self.figure.text(
            0.50, 0.53, message,
            ha="center", va="center", fontsize=13, color="#64748B",
            bbox=dict(boxstyle="round,pad=0.75", fc="#F8FAFC", ec="#CBD5E1")
        )
        self.draw_idle()

    def _leave_empty_state(self):
        if self._empty_message_artist is not None:
            try:
                self._empty_message_artist.remove()
            except (AttributeError, ValueError, NotImplementedError):
                pass
            self._empty_message_artist = None
        self.ax_main.axis("on")

    @staticmethod
    def _interp_on_axis(value: float, x_values, y_values) -> float:
        """Interpolate correctly for both ascending and conventional descending XPS axes."""
        x_array = np.asarray(x_values, dtype=float)
        y_array = np.asarray(y_values, dtype=float)
        order = np.argsort(x_array)
        return float(np.interp(float(value), x_array[order], y_array[order]))

    def _apply_style(self):
        if self.x_data is not None and len(self.x_data):
            xmin, xmax = float(np.min(self.x_data)), float(np.max(self.x_data))
            limits = (xmax, xmin) if self._reverse_x else (xmin, xmax)
            self.ax_main.set_xlim(*limits)
            if self.ax_resid.axison:
                self.ax_resid.set_xlim(*limits)
        for ax in (self.ax_main, self.ax_resid):
            if not ax.axison:
                continue
            ax.set_facecolor("white")
            ax.tick_params(axis="both", which="major", labelsize=9, width=0.8, length=4, color="#475569")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#94A3B8")
            ax.spines["bottom"].set_color("#94A3B8")
            ax.grid(axis="y", color="#CBD5E1", alpha=0.45, linewidth=0.7)
        self.figure.subplots_adjust(left=0.105, right=0.985, bottom=0.105, top=0.84, hspace=0.16)
        self.draw_idle()

    def _hide_crosshair(self, redraw=True):
        """Hide the interactive cursor without detaching Matplotlib artists.

        Removing an artist after ``Axes.clear()`` raises ``NotImplementedError``
        because the artist has already been detached from its axes. Visibility
        toggling is safer and cheaper for frequent enter/leave events.
        """
        for artist in (self._crosshair_v, self._crosshair_h, self._coord_text):
            if artist is not None:
                try:
                    artist.set_visible(False)
                except (AttributeError, RuntimeError, ValueError, NotImplementedError):
                    pass
        if redraw:
            self.draw_idle()

    def _reset_crosshair(self):
        """Forget cursor artists before the axes are rebuilt."""
        self._hide_crosshair(redraw=False)
        self._crosshair_v, self._crosshair_h, self._coord_text = None, None, None

    def _reset_fit_artists(self):
        self._fit_component_lines = []
        self._fit_apex_handle = None
        self._fit_width_handles = []
        self._fit_drag_kind = None
        self._fit_drag_state = None
        self._fit_background = None

    def _reset_manual_baseline_artists(self, keep_points: bool = True):
        self._manual_baseline_artist = None
        self._manual_baseline_line = None
        self._manual_dragging_idx = None
        if not keep_points:
            self._manual_baseline_points = []

    @staticmethod
    def _preview_manual_curve(x_values, points):
        """Interpolation used only while dragging; strict QA runs before fitting."""
        if not points:
            return np.zeros_like(np.asarray(x_values, dtype=float))
        ordered = sorted((float(px), float(py)) for px, py in points)
        compact: List[Tuple[float, float]] = []
        for px, py in ordered:
            if compact and abs(px - compact[-1][0]) <= 1e-12:
                compact[-1] = (px, 0.5 * (compact[-1][1] + py))
            else:
                compact.append((px, py))
        if len(compact) == 1:
            return np.full_like(np.asarray(x_values, dtype=float), compact[0][1])
        xp = np.asarray([point[0] for point in compact], dtype=float)
        yp = np.asarray([point[1] for point in compact], dtype=float)
        return np.interp(np.asarray(x_values, dtype=float), xp, yp)

    def _update_manual_baseline_artists(self):
        if self._manual_baseline_artist is not None:
            offsets = (
                np.asarray(self._manual_baseline_points, dtype=float).reshape((-1, 2))
                if self._manual_baseline_points else np.empty((0, 2), dtype=float)
            )
            self._manual_baseline_artist.set_offsets(offsets)
        if (
            self._manual_baseline_line is not None
            and self.x_data is not None
            and self._manual_baseline_points
        ):
            self._manual_baseline_line.set_ydata(
                self._preview_manual_curve(self.x_data, self._manual_baseline_points)
            )
        self.draw_idle()

    def _on_click(self, event):
        if event.inaxes != self.ax_main: return

        if self._manual_baseline_editing:
            hit_index = None
            if self._manual_baseline_artist is not None:
                contains, details = self._manual_baseline_artist.contains(event)
                if contains and details.get("ind") is not None and len(details["ind"]):
                    hit_index = int(details["ind"][0])
            if event.button == 3 and hit_index is not None:
                del self._manual_baseline_points[hit_index]
                self._manual_dragging_idx = None
                self._update_manual_baseline_artists()
                self.manual_baseline_changed.emit(list(self._manual_baseline_points))
                return
            if event.button == 1:
                if hit_index is not None:
                    self._manual_dragging_idx = hit_index
                elif event.xdata is not None and event.ydata is not None:
                    self._manual_baseline_points.append(
                        (float(event.xdata), float(event.ydata))
                    )
                    self._manual_baseline_points.sort(key=lambda point: point[0])
                    self._update_manual_baseline_artists()
                    self.manual_baseline_changed.emit(list(self._manual_baseline_points))
                return

        if self._fit_editing and event.button == 1:
            if self._fit_apex_handle is not None:
                contains, _ = self._fit_apex_handle.contains(event)
                if contains:
                    x0 = float(self._fit_apex_handle.get_xdata()[0])
                    y0 = float(self._fit_apex_handle.get_ydata()[0])
                    self._fit_drag_kind = "apex"
                    self._fit_drag_state = {
                        "index": self._fit_selected_idx,
                        "center": x0,
                        "apex_y": y0,
                    }
                    return
            for handle in self._fit_width_handles:
                contains, _ = handle.contains(event)
                if contains:
                    self._fit_drag_kind = "width"
                    self._fit_drag_state = {
                        "index": self._fit_selected_idx,
                        "center": float(handle.get_gid().split(":")[1]),
                    }
                    return
            for idx, line in enumerate(self._fit_component_lines):
                contains, _ = line.contains(event)
                if contains:
                    self.fit_component_selected.emit(idx)
                    return
        
        if self._lines:
            for idx, line in enumerate(self._lines):
                cont, ind = line.contains(event)
                if cont:
                    self._dragging_line = line
                    self._dragging_idx = idx
                    line.set_color("#ef4444")
                    self.draw_idle()
                    return

        if self._selection_mode and event.xdata is not None:
            self.peak_selected.emit(float(event.xdata))

    def _on_mouse_move(self, event):
        if event.inaxes == self.ax_main and event.xdata is not None and event.ydata is not None:
            if self._manual_dragging_idx is not None:
                self._manual_baseline_points[self._manual_dragging_idx] = (
                    float(event.xdata), float(event.ydata)
                )
                self._update_manual_baseline_artists()
                return
            if self._fit_drag_kind and self._fit_drag_state is not None:
                if self._fit_drag_kind == "apex":
                    new_center = float(event.xdata)
                    new_apex_y = float(event.ydata)
                    self._fit_drag_state["center"] = new_center
                    self._fit_drag_state["apex_y"] = new_apex_y
                    self._fit_apex_handle.set_data([new_center], [new_apex_y])
                elif self._fit_drag_kind == "width":
                    center = float(self._fit_drag_state["center"])
                    fwhm = max(0.02, 2.0 * abs(float(event.xdata) - center))
                    self._fit_drag_state["fwhm"] = fwhm
                    if len(self._fit_width_handles) == 2:
                        half = 0.5 * fwhm
                        for handle, xpos in zip(self._fit_width_handles, (center-half, center+half)):
                            handle.set_xdata([xpos])
                self.draw_idle()
                return
            if self._dragging_line is not None:
                new_x = float(event.xdata)
                self._dragging_line.set_xdata([new_x, new_x])
                if self.x_data is not None and self.y_data is not None:
                    new_y = self._interp_on_axis(new_x, self.x_data, self.y_data)
                    self._points[self._dragging_idx].set_data([new_x], [new_y])
                    self._texts[self._dragging_idx].set_x(new_x)
                self.draw_idle()
                return

            if self._crosshair_v is None:
                self._crosshair_v = self.ax_main.axvline(0, color="#7dd3fc", linestyle=":", lw=1, alpha=0.9)
                self._crosshair_h = self.ax_main.axhline(0, color="#7dd3fc", linestyle=":", lw=1, alpha=0.9)
                self._coord_text = self.ax_main.text(0.02, 0.97, "", transform=self.ax_main.transAxes, va="top", 
                                                     color="white", bbox=dict(boxstyle="round", fc="#0F172A", ec="#35507C", alpha=0.9))
            self._crosshair_v.set_xdata([event.xdata, event.xdata]); self._crosshair_v.set_visible(True)
            self._crosshair_h.set_ydata([event.ydata, event.ydata]); self._crosshair_h.set_visible(True)
            self._coord_text.set_text(f"X = {event.xdata:.4f} eV\nY = {event.ydata:.4f}"); self._coord_text.set_visible(True)
            self.cursor_moved.emit(float(event.xdata), float(event.ydata))
            self.draw_idle()

    def _on_release(self, event):
        if self._manual_dragging_idx is not None:
            self._manual_dragging_idx = None
            self._manual_baseline_points.sort(key=lambda point: point[0])
            self._update_manual_baseline_artists()
            self.manual_baseline_changed.emit(list(self._manual_baseline_points))
            return
        if self._fit_drag_kind and self._fit_drag_state is not None:
            state = dict(self._fit_drag_state)
            index = int(state.pop("index"))
            if self._fit_drag_kind == "apex":
                center = float(state["center"])
                background_y = 0.0
                if self.x_data is not None and self._fit_background is not None:
                    background_y = self._interp_on_axis(
                        center, self.x_data, self._fit_background
                    )
                state["height"] = max(float(state.pop("apex_y")) - background_y, 0.0)
            self._fit_drag_kind = None
            self._fit_drag_state = None
            self.fit_component_edited.emit(index, state)
            return
        if self._dragging_line is not None:
            self._dragging_line.set_color("#f59e0b") 
            self._dragging_line = None
            
            new_centers = [float(l.get_xdata()[0]) for l in self._lines]
            self.centers_updated.emit(new_centers)
            self._dragging_idx = None

    def _on_axes_leave(self, event):
        self._hide_crosshair(); self.cursor_left_axes.emit()

    def _on_scroll(self, event):
        if event.inaxes not in (self.ax_main, self.ax_resid) or event.xdata is None:
            return
        ax = event.inaxes
        scale = 0.80 if event.button == "up" else 1.25
        x0, x1 = ax.get_xlim()
        anchor_x = float(event.xdata)
        ax.set_xlim(anchor_x + (x0-anchor_x)*scale, anchor_x + (x1-anchor_x)*scale)
        if event.ydata is not None:
            y0, y1 = ax.get_ylim()
            anchor_y = float(event.ydata)
            ax.set_ylim(anchor_y + (y0-anchor_y)*scale, anchor_y + (y1-anchor_y)*scale)
        self.draw_idle()

    def plot_loaded_data(self, x, y, centers=None):
        self._leave_empty_state()
        self._set_figure_title("")
        self.x_data, self.y_data = x, y
        self._reset_crosshair(); self._reset_fit_artists(); self._reset_manual_baseline_artists()
        self.set_manual_baseline_editing(False)
        self.ax_main.clear(); self.ax_resid.clear()
        self._lines, self._points, self._texts = [], [], [] 
        
        self.ax_main.plot(x, y, label="Loaded data", lw=1.35, color="#334155")
        if centers:
            ymax = float(np.max(y)) if len(y) else 1.0
            for i, c in enumerate(centers, start=1):
                line = self.ax_main.axvline(c, ls="--", lw=1.5, color="#f59e0b", picker=5)
                pt, = self.ax_main.plot(
                    [c], [self._interp_on_axis(c, x, y)],
                    marker="o", ms=6, color="#ef4444"
                )
                txt = self.ax_main.text(c, ymax*0.95, f"P{i}", rotation=90, va="top", ha="right", color="#fbbf24")
                
                self._lines.append(line)
                self._points.append(pt)
                self._texts.append(txt)
                
        self.ax_main.set_title("Loaded XPS spectrum", loc="left", fontsize=12, fontweight="semibold", pad=12)
        self.ax_main.set_xlabel("Binding energy (eV)", fontsize=10.5)
        self.ax_main.set_ylabel("Intensity (a.u.)", fontsize=10.5)
        self.ax_main.legend(frameon=False, fontsize=8.5); self.ax_resid.axis("off")
        self._apply_style()

    def plot_raw_and_background(
        self, x, y, bkg, manual_points=None, manual_editing: bool = False
    ):
        self._leave_empty_state()
        self._set_figure_title("")
        self.x_data, self.y_data = x, y
        self._reset_crosshair(); self._reset_fit_artists(); self._reset_manual_baseline_artists()
        self.ax_main.clear(); self.ax_resid.clear()
        self._lines, self._points, self._texts = [], [], []
        self._dragging_line, self._dragging_idx = None, None
        self.ax_main.plot(x, y, label="Raw data", lw=1.35, color="#334155")
        self._manual_baseline_line, = self.ax_main.plot(
            x, bkg, label="Background", lw=1.8, color="#0F172A", ls="--"
        )
        self._manual_baseline_points = [
            (float(point[0]), float(point[1])) for point in (manual_points or [])
        ]
        if self._manual_baseline_points:
            point_array = np.asarray(self._manual_baseline_points, dtype=float)
            self._manual_baseline_artist = self.ax_main.scatter(
                point_array[:, 0], point_array[:, 1], s=58,
                facecolors="white", edgecolors="#DC2626", linewidths=1.7,
                label="Manual anchors", zorder=8, picker=8
            )
        self.set_manual_baseline_editing(bool(manual_editing))
        self.ax_main.set_title("Background preview", loc="left", fontsize=12, fontweight="semibold", pad=12)
        self.ax_main.set_xlabel("Binding energy (eV)", fontsize=10.5)
        self.ax_main.set_ylabel("Intensity (a.u.)", fontsize=10.5)
        self.ax_main.legend(frameon=False, fontsize=8.5); self.ax_resid.axis("off")
        self._apply_style()

    def plot_fit(self, x, y_raw, initial_background, baseline_correction,
                 y_fit_corrected, comps, residual, peak_results=None,
                 metrics=None, state_fractions=None, title="XPS peak fit",
                 preserve_view=False):
        self._leave_empty_state()
        previous_view = None
        if preserve_view and self.ax_main.has_data():
            previous_view = (self.ax_main.get_xlim(), self.ax_main.get_ylim(), self.ax_resid.get_xlim(), self.ax_resid.get_ylim())
        self.x_data, self.y_data = x, y_raw
        self._reset_crosshair(); self._reset_fit_artists(); self._reset_manual_baseline_artists()
        self.set_manual_baseline_editing(False)
        self.ax_main.clear(); self.ax_resid.clear()
        self._lines, self._points, self._texts = [], [], []
        self._dragging_line, self._dragging_idx = None, None
        final_background = initial_background + baseline_correction
        total_raw_fit = initial_background + y_fit_corrected
        self.ax_main.scatter(
            x, y_raw, s=13, facecolors="white", edgecolors="#64748B",
            linewidths=0.65, label="Experimental data", zorder=4
        )
        self.ax_main.plot(x, total_raw_fit, color="#DC2626", label="Total fit", lw=2.25, zorder=5)
        self.ax_main.plot(x, final_background, color="#111827", ls=":", lw=1.55, label="Background", zorder=3)
        self._fit_background = np.asarray(final_background, dtype=float)

        palette = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DB2777", "#0891B2", "#4F46E5"]
        state_colors: Dict[str, str] = {}
        for i, comp in enumerate(comps):
            label = peak_results[i].label if peak_results and i < len(peak_results) else f"Peak {i+1}"
            state = peak_results[i].state if peak_results and i < len(peak_results) else label
            state_key = state or label
            if state_key not in state_colors:
                state_colors[state_key] = palette[len(state_colors) % len(palette)]
            color = state_colors[state_key]
            component_curve = final_background + comp
            legend_label = label
            if peak_results and i < len(peak_results):
                legend_label += f"  {peak_results[i].center_fitted:.2f} eV"
            selected = i == self._fit_selected_idx
            line, = self.ax_main.plot(
                x, component_curve, color=color, ls="-" if selected else "--",
                lw=2.15 if selected else 1.25, label=legend_label,
                zorder=4 if selected else 2, picker=5
            )
            self._fit_component_lines.append(line)
            self.ax_main.fill_between(x, final_background, component_curve, color=color, alpha=0.09, zorder=1)

        if (
            self._fit_editing and peak_results
            and 0 <= self._fit_selected_idx < len(peak_results)
        ):
            selected_peak = peak_results[self._fit_selected_idx]
            center = float(selected_peak.center_fitted)
            height = float(selected_peak.height_fitted)
            fwhm = float(selected_peak.fwhm_fitted)
            apex_y = self._interp_on_axis(center, x, final_background) + height
            self._fit_apex_handle, = self.ax_main.plot(
                [center], [apex_y], marker="D", ms=7.0, mfc="white", mec="#DC2626",
                mew=1.6, linestyle="none", picker=8, zorder=9
            )
            half_y = []
            for xpos in (center - 0.5*fwhm, center + 0.5*fwhm):
                half_y.append(
                    self._interp_on_axis(xpos, x, final_background) + 0.5*height
                )
            for xpos, ypos in zip((center-0.5*fwhm, center+0.5*fwhm), half_y):
                handle, = self.ax_main.plot(
                    [xpos], [ypos], marker="s", ms=6.0, mfc="#FEF2F2", mec="#DC2626",
                    mew=1.3, linestyle="none", picker=8, zorder=9
                )
                handle.set_gid(f"width:{center}")
                self._fit_width_handles.append(handle)

        self.ax_main.set_title("")
        self._set_figure_title(title)
        self.ax_main.set_ylabel("Intensity (a.u.)", fontsize=10.5)
        self.ax_main.tick_params(labelbottom=False)
        n_legend = 3 + len(comps)
        self.ax_main.legend(
            loc="upper center", bbox_to_anchor=(0.5, 1.17),
            ncol=min(4, max(2, n_legend)), fontsize=7.5,
            frameon=False, handlelength=2.4, columnspacing=1.1
        )

        if state_fractions:
            fraction_text = "   ".join(
                f"{state}: {format_fraction_percent(fraction, decimals=1)}"
                for state, fraction in state_fractions.items()
            )
            self.ax_main.text(
                0.99, 0.97, fraction_text, transform=self.ax_main.transAxes,
                ha="right", va="top", fontsize=8.2, color="#0F172A",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#CBD5E1", alpha=0.92)
            )
        if metrics is not None:
            quality_color = {"GOOD": "#047857", "REVIEW": "#B45309", "POOR": "#B91C1C"}.get(metrics.fit_quality, "#475569")
            self.ax_main.text(
                0.01, 0.97, f"Quality: {metrics.fit_quality}", transform=self.ax_main.transAxes,
                ha="left", va="top", fontsize=8.2, color=quality_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=quality_color, alpha=0.92)
            )
        
        self.ax_resid.axis("on")
        self.ax_resid.plot(x, residual, color="#475569", lw=1.05, zorder=2)
        self.ax_resid.axhline(0, ls="--", lw=0.9, color="#0F172A", zorder=3)
        rmse = float(np.sqrt(np.mean(np.asarray(residual, dtype=float)**2))) if len(residual) else 0.0
        if rmse > 0:
            self.ax_resid.axhspan(-rmse, rmse, color="#94A3B8", alpha=0.13, zorder=0, label="±RMSE")
            self.ax_resid.text(
                0.99, 0.88, f"RMSE = {rmse:.3g}", transform=self.ax_resid.transAxes,
                ha="right", va="top", fontsize=8, color="#475569"
            )
        max_abs = float(np.max(np.abs(residual))) if len(residual) else 1.0
        self.ax_resid.set_ylim(-1.15 * max(max_abs, 1e-12), 1.15 * max(max_abs, 1e-12))
        self.ax_resid.set_xlabel("Binding energy (eV)", fontsize=10.5)
        self.ax_resid.set_ylabel("Residual", fontsize=9.5)
        self._apply_style()
        if previous_view is not None:
            self.ax_main.set_xlim(*previous_view[0]); self.ax_main.set_ylim(*previous_view[1])
            self.ax_resid.set_xlim(*previous_view[2]); self.ax_resid.set_ylim(*previous_view[3])
            self.draw_idle()


# --- Main UI ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(QIcon("ResearcherLogo.ico"))
        self.resize(1600, 950)
        self.setMinimumSize(1180, 760)
        self.ui_settings = QSettings("ResearcherAnalysisToolkit", "XPSFitting")
        self.file_path, self.energy, self.intensity = None, None, None
        self.energy_original = None
        self.applied_charge_shift = 0.0
        self.fit_result, self.fit_context = None, {}
        self.db_fetched_info = None
        self._fit_active = False
        self._fit_origin_page = 0
        self._pre_refit_snapshot = None
        self._use_studio_model_once = False
        self.manual_baseline_points: List[Tuple[float, float]] = []
        self._manual_baseline_region: Optional[Tuple[float, float]] = None
        self._manual_baseline_ui_update = False
        self.studio_state: List[Dict[str, Any]] = []
        self.studio_original_state: List[Dict[str, Any]] = []
        self.studio_history: List[List[Dict[str, Any]]] = []
        self.studio_redo: List[List[Dict[str, Any]]] = []
        self.studio_selected_index = -1
        self.studio_dirty = False
        self._studio_updating_controls = False
        self._studio_slider_active_field: Optional[str] = None
        self._build_ui()
        self._build_status_bar()
        self.apply_light_theme()
        QTimer.singleShot(0, self._restore_workspace_layout)

    def open_external_link(self, url: str):
        QDesktopServices.openUrl(QUrl(url))

    def _build_status_bar(self):
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self.progress_bar = QProgressBar(); self.progress_bar.setMaximumWidth(250); self.progress_bar.setVisible(False)
        self.status.addPermanentWidget(self.progress_bar)
        self.cursor_status_label = QLabel("X: -, Y: -")
        self.cursor_status_label.setStyleSheet("color: #2563EB; font-weight: bold; padding: 2px 10px;")
        self.status.addPermanentWidget(self.cursor_status_label)
        self.status.showMessage("Ready")

    @staticmethod
    def _configure_sidebar_form(form: QFormLayout):
        """Use a single-column form that remains readable at high DPI."""
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(0)
        form.setVerticalSpacing(8)
        form.setContentsMargins(10, 22, 10, 10)

    @staticmethod
    def _make_combo_responsive(combo: QComboBox, minimum_characters: int = 18):
        """Let long choices elide instead of forcing a sidebar wider than its viewport."""
        combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        combo.setMinimumContentsLength(int(minimum_characters))
        combo.setMinimumWidth(0)
        combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    @staticmethod
    def _make_label_responsive(label: QLabel):
        """Allow informative sidebar text to wrap without widening the panel."""
        label.setWordWrap(True)
        label.setMinimumWidth(0)
        label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def _build_plot_header(
        self, canvas: XPSPlotCanvas, parent: QWidget, title: str,
        reset_slot=None
    ) -> QFrame:
        """Create a compact, consistent title/toolbar strip for plot panels."""
        header = QFrame(); header.setObjectName("plotToolbar")
        layout = QHBoxLayout(header); layout.setContentsMargins(10, 4, 8, 4)
        layout.setSpacing(6)
        label = QLabel(title); label.setObjectName("panelTitle")
        layout.addWidget(label)
        layout.addStretch(1)
        toolbar = NavigationToolbar(canvas, parent)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        layout.addWidget(toolbar, 0)
        if reset_slot is not None:
            reset_view = QPushButton("Reset view", clicked=reset_slot)
            reset_view.setObjectName("compactButton")
            layout.addWidget(reset_view)
        return header
        
    def _build_ui(self):
        self.setup_canvas = XPSPlotCanvas()
        self.canvas = XPSPlotCanvas()
        
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central); root.setContentsMargins(8, 8, 8, 8); root.setSpacing(6)

        # Header
        hero = QFrame(); hero.setObjectName("hero")
        hero_layout = QHBoxLayout(hero); hero_layout.setContentsMargins(14, 6, 14, 6)
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_box.addWidget(QLabel(APP_NAME, styleSheet="font-size: 20px; font-weight: bold; color: #1E3A8A; border: none; background: transparent;"))
        title_box.addWidget(QLabel(APP_SUBTITLE, styleSheet="color: #3B82F6; border: none; background: transparent;"))
        
        links_row = QHBoxLayout(); links_row.setSpacing(8)
        btn_scholar = QToolButton(); btn_scholar.setText("Google Scholar"); btn_scholar.setObjectName("linkButton")
        btn_scholar.clicked.connect(lambda: self.open_external_link("https://scholar.google.com/citations?user=ZNF36DQAAAAJ&hl=en"))
        btn_github = QToolButton(); btn_github.setText("GitHub"); btn_github.setObjectName("linkButton")
        btn_github.clicked.connect(lambda: self.open_external_link("https://github.com/Milad-Behroozi"))
        links_row.addWidget(btn_scholar); links_row.addWidget(btn_github)

        hero_layout.addLayout(title_box); hero_layout.addStretch(); hero_layout.addLayout(links_row)
        root.addWidget(hero)

        workflow_nav = QFrame(); workflow_nav.setObjectName("workflowNav")
        workflow_nav_layout = QHBoxLayout(workflow_nav)
        workflow_nav_layout.setContentsMargins(8, 3, 8, 3)
        self.setup_step_button = QPushButton("1   Quick Fit")
        self.setup_step_button.setObjectName("stepButton")
        self.setup_step_button.clicked.connect(lambda: self._show_workflow_page(0))
        self.studio_step_button = QPushButton("2   Edit && Review")
        self.studio_step_button.setObjectName("stepButton")
        self.studio_step_button.setEnabled(False)
        self.studio_step_button.clicked.connect(lambda: self._show_workflow_page(1))
        workflow_nav_layout.addWidget(self.setup_step_button)
        workflow_nav_layout.addWidget(QLabel("›", styleSheet="font-size:22px; color:#94A3B8;"))
        workflow_nav_layout.addWidget(self.studio_step_button)
        workflow_nav_layout.addStretch(1)
        self.btn_reset_layout = QPushButton("Reset layout", clicked=self._reset_workspace_layout)
        self.btn_reset_layout.setObjectName("compactButton")
        self.btn_reset_layout.setToolTip(
            "Restore safe sidebar widths, plot proportions and scroll both control panels to the top."
        )
        workflow_nav_layout.addWidget(self.btn_reset_layout)
        root.addWidget(workflow_nav)

        self.workflow_stack = QStackedWidget()
        root.addWidget(self.workflow_stack, 1)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        self.content_splitter.setHandleWidth(7)

        # Left Panel
        left_panel = QFrame(); left_panel.setObjectName("sidePanel")
        left_panel.setMinimumWidth(320)
        left_panel.setMaximumWidth(560)
        left_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        left_panel_layout = QVBoxLayout(left_panel); left_panel_layout.setContentsMargins(8, 8, 8, 8)
        left_panel_layout.setSpacing(8)
        
        left_scroll = ResponsiveScrollArea(); left_scroll.setWidgetResizable(True); left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        left_scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.setup_controls_scroll = left_scroll
        left_widget = QWidget()
        left_widget.setMinimumWidth(0)
        left_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 8)
        left_layout.setSpacing(10)
        left_scroll.setWidget(left_widget)
        
        left_panel_layout.addWidget(left_scroll, 1)

        # 1. Setup
        setup_box = QGroupBox("Spectrum and peaks"); setup_form = QFormLayout(setup_box)
        self._configure_sidebar_form(setup_form)
        self.file_edit = QLineEdit(); self.file_edit.setReadOnly(True)
        self.file_edit.setMinimumWidth(0); self.file_edit.setPlaceholderText("No file selected")
        btn_open = QPushButton("Open", clicked=self.open_file)
        btn_open.setObjectName("secondaryButton"); btn_open.setMinimumWidth(74)
        file_row = QHBoxLayout(); file_row.addWidget(self.file_edit, 1); file_row.addWidget(btn_open, 0)
        setup_form.addRow("Data file", self._wrap_layout(file_row))
        self.fit_mode_combo = QComboBox()
        self.fit_mode_combo.addItem("Free fit — any independent peaks", "free")
        self.fit_mode_combo.addItem("Physics-constrained fit", "physics")
        self._make_combo_responsive(self.fit_mode_combo, 20)
        self.fit_mode_combo.currentIndexChanged.connect(self._on_fit_mode_changed)
        setup_form.addRow("Fit mode", self.fit_mode_combo)
        self.fit_mode_help_label = QLabel(
            "Pick any peak centers. In Free fit, every peak is independent and no chemical assignment is imposed."
        )
        self._make_label_responsive(self.fit_mode_help_label)
        self.fit_mode_help_label.setObjectName("simpleModeHelp")
        setup_form.addRow(self.fit_mode_help_label)
        self.centers_edit = QLineEdit(); self.centers_edit.editingFinished.connect(self.refresh_loaded_plot)
        self.centers_edit.setPlaceholderText("e.g. 34.6, 35.7, 37.9 — or pick directly on the plot")
        setup_form.addRow("Peak centers", self.centers_edit)
        center_btn_row = QHBoxLayout()
        self.btn_pick_peaks = QPushButton("Pick on Plot", checkable=True); self.btn_pick_peaks.toggled.connect(self.setup_canvas.enable_peak_selection)
        self.btn_clear_centers = QPushButton("Clear", clicked=lambda: (self.centers_edit.clear(), self.refresh_loaded_plot()))
        self.btn_clear_centers.setObjectName("secondaryButton")
        center_btn_row.addWidget(self.btn_pick_peaks, 1); center_btn_row.addWidget(self.btn_clear_centers, 1)
        setup_form.addRow("Peak tools", self._wrap_layout(center_btn_row))
        left_layout.addWidget(setup_box)

        self.btn_show_advanced = QPushButton("Show advanced settings", checkable=True)
        self.btn_show_advanced.setObjectName("secondaryButton")
        self.btn_show_advanced.setMinimumHeight(34)
        self.btn_show_advanced.toggled.connect(self._toggle_setup_advanced)
        left_layout.addWidget(self.btn_show_advanced)

        # 2. ENERGY CALIBRATION / CHARGE REFERENCING
        self.calib_box = QGroupBox("Energy calibration"); calib_form = QFormLayout(self.calib_box)
        self._configure_sidebar_form(calib_form)
        self.calibration_reference_edit = QLineEdit("C 1s (user-justified)")
        self.measured_c1s = QDoubleSpinBox(); self.measured_c1s.setRange(-1000.0, 5000.0); self.measured_c1s.setValue(284.8); self.measured_c1s.setDecimals(3)
        self.target_c1s = QDoubleSpinBox(); self.target_c1s.setRange(-1000.0, 5000.0); self.target_c1s.setValue(284.8); self.target_c1s.setDecimals(3)
        self.target_c1s.setStyleSheet("background: #065F46; color: white;") 
        
        self.btn_apply_shift = QPushButton("Apply Energy Shift", clicked=self.apply_charge_shift)
        self.btn_apply_shift.setObjectName("secondaryButton")
        self.btn_apply_shift.setFixedHeight(30)
        
        calib_form.addRow("Reference:", self.calibration_reference_edit)
        calib_form.addRow("Measured BE:", self.measured_c1s)
        calib_form.addRow("Target BE:", self.target_c1s)
        calib_form.addRow(self.btn_apply_shift)
        
        left_layout.addWidget(self.calib_box)

        # 3. REFERENCE MODEL
        self.db_box = QGroupBox("Physics / reference model"); db_form = QFormLayout(self.db_box)
        self._configure_sidebar_form(db_form)
        self.element_edit = QLineEdit(placeholderText="e.g. CeO2, TiO2, W")
        self.orbital_edit = QLineEdit(placeholderText="e.g. 3d, 2p, 4f")
        self.element_edit.textChanged.connect(self._invalidate_database)
        self.orbital_edit.textChanged.connect(self._invalidate_database)
        inp_row = QGridLayout()
        inp_row.setColumnStretch(0, 2); inp_row.setColumnStretch(1, 1)
        inp_row.addWidget(self.element_edit, 0, 0)
        inp_row.addWidget(self.orbital_edit, 0, 1)

        self.btn_fetch_db = QPushButton("Fetch from DB", clicked=self.fetch_database)
        self.btn_fetch_db.setObjectName("secondaryButton")
        inp_row.addWidget(self.btn_fetch_db, 1, 0, 1, 2)
        db_form.addRow("Target", self._wrap_layout(inp_row))
        
        self.db_status_label = QLabel("Not fetched yet.")
        self._make_label_responsive(self.db_status_label)
        self.db_status_label.setStyleSheet("color: #F59E0B; font-weight: bold; padding: 5px;")
        db_form.addRow("DB Status", self.db_status_label)

        self.auto_doublet_check = QCheckBox("Tie splitting and area ratio")
        self.auto_doublet_check.setToolTip(
            "Enable physics constraints for spin-orbit splitting and doublet area ratios."
        )
        self.auto_doublet_check.setChecked(True)
        self.auto_doublet_check.setStyleSheet("color: #22C55E; font-weight: bold;")
        db_form.addRow("Constraint", self.auto_doublet_check)
        self.use_model_check = QCheckBox("Use editable Model Builder table")
        self.use_model_check.setToolTip(
            "Fit with the components and parameter ties currently shown in Model Builder."
        )
        self.use_model_check.setChecked(False)
        self.use_model_check.stateChanged.connect(self._invalidate_fit_after_model_edit)
        db_form.addRow("Universal model", self.use_model_check)
        ref_links = QHBoxLayout()
        btn_nist = QPushButton("NIST XPS")
        btn_nist.setObjectName("secondaryButton")
        btn_nist.setToolTip("Open the NIST XPS Database in your web browser.")
        btn_nist.clicked.connect(lambda: self.open_external_link("https://srdata.nist.gov/xps/"))
        btn_model_tab = QPushButton("Model Builder")
        btn_model_tab.setObjectName("secondaryButton")
        btn_model_tab.clicked.connect(lambda: self.analysis_tabs.setCurrentWidget(self.model_tab))
        ref_links.addWidget(btn_nist); ref_links.addWidget(btn_model_tab)
        db_form.addRow("References", self._wrap_layout(ref_links))
        left_layout.addWidget(self.db_box)

        # 4. Fit Engine
        self.engine_box = QGroupBox("Fitting model"); engine_form = QFormLayout(self.engine_box)
        self.engine_form = engine_form
        self._configure_sidebar_form(engine_form)
        self.background_combo = QComboBox(); self.background_combo.addItems([
            "Shirley (iterative)", "Linear", "Manual (anchor points)", "None"
        ])
        self._make_combo_responsive(self.background_combo, 18)
        self.background_combo.setToolTip(
            "Choose from the spectrum physics and fit interval. Manual is a recorded piecewise-linear line, "
            "not an automatic inelastic-scattering model."
        )
        self.lineshape_combo = QComboBox(); self.lineshape_combo.addItems([
            "Pseudo-Voigt (Symmetric)", "Gaussian", "Lorentzian",
            "Asymmetric pseudo-Voigt (integrable; metallic)",
            "Doniach-Sunjic (BE-tail; metallic only)"
        ])
        self._make_combo_responsive(self.lineshape_combo, 22)
        engine_form.addRow("Background", self.background_combo)

        self.manual_baseline_widget = QFrame()
        manual_baseline_layout = QVBoxLayout(self.manual_baseline_widget)
        manual_baseline_layout.setContentsMargins(0, 2, 0, 2)
        manual_baseline_layout.setSpacing(6)
        manual_button_row = QHBoxLayout(); manual_button_row.setSpacing(6)
        self.btn_edit_manual_baseline = QPushButton("Edit on graph", checkable=True)
        self.btn_edit_manual_baseline.setObjectName("secondaryButton")
        self.btn_edit_manual_baseline.setToolTip(
            "Left-click to add an anchor, drag a circle to move it, and right-click a circle to remove it."
        )
        self.btn_reset_manual_baseline = QPushButton("Reset anchors")
        self.btn_reset_manual_baseline.setObjectName("secondaryButton")
        self.btn_clear_manual_baseline = QPushButton("Clear")
        self.btn_clear_manual_baseline.setObjectName("secondaryButton")
        manual_button_row.addWidget(self.btn_edit_manual_baseline, 1)
        manual_button_row.addWidget(self.btn_reset_manual_baseline)
        manual_button_row.addWidget(self.btn_clear_manual_baseline)
        manual_baseline_layout.addLayout(manual_button_row)
        self.manual_baseline_status = QLabel(
            "No anchors. Start the editor to seed robust edge points."
        )
        self.manual_baseline_status.setWordWrap(True)
        self.manual_baseline_status.setObjectName("manualBaselineStatus")
        manual_baseline_layout.addWidget(self.manual_baseline_status)
        engine_form.addRow("Manual baseline", self.manual_baseline_widget)
        engine_form.addRow("Lineshape", self.lineshape_combo)
        self.weighting_combo = QComboBox(); self.weighting_combo.addItems([
            "Uniform", "Poisson (raw counts only)"
        ])
        self._make_combo_responsive(self.weighting_combo, 18)
        engine_form.addRow("Residual weighting", self.weighting_combo)

        self.margin_spin = QDoubleSpinBox(); self.margin_spin.setRange(0.5, 50.0); self.margin_spin.setValue(5.0)
        self.shift_spin = QDoubleSpinBox(); self.shift_spin.setRange(0.02, 3.0); self.shift_spin.setValue(0.40); self.shift_spin.setDecimals(2)
        self.shift_spin.setToolTip(
            "Safety half-width used only by peaks in Free mode. Bounded, Fixed, Tied and doublet-partner centers use their own constraints."
        )
        engine_form.addRow("Fit-region margin (eV)", self.margin_spin)
        engine_form.addRow("Free-center safety ± (eV)", self.shift_spin)

        self.fwhm_min_spin = QDoubleSpinBox(); self.fwhm_min_spin.setRange(0.10, 5.0); self.fwhm_min_spin.setValue(0.35); self.fwhm_min_spin.setDecimals(2)
        self.fwhm_max_spin = QDoubleSpinBox(); self.fwhm_max_spin.setRange(0.20, 10.0); self.fwhm_max_spin.setValue(4.00); self.fwhm_max_spin.setDecimals(2)
        fwhm_row = QHBoxLayout(); fwhm_row.addWidget(QLabel("Min:")); fwhm_row.addWidget(self.fwhm_min_spin)
        fwhm_row.addWidget(QLabel("Max:")); fwhm_row.addWidget(self.fwhm_max_spin)
        self.fwhm_limits_widget = self._wrap_layout(fwhm_row)
        engine_form.addRow("FWHM limits (eV)", self.fwhm_limits_widget)

        self.shared_shape_check = QCheckBox("Share line-shape mixing/asymmetry across components")
        self.shared_shape_check.setText("Share line shape across peaks")
        self.shared_shape_check.setToolTip(
            "Share line-shape mixing/asymmetry across compatible components to improve identifiability."
        )
        self.shared_shape_check.setChecked(True)
        self.baseline_refine_check = QCheckBox("Allow small residual-baseline refinement")
        self.baseline_refine_check.setChecked(True)
        engine_form.addRow("Identifiability", self.shared_shape_check)
        engine_form.addRow("Background QA", self.baseline_refine_check)

        self.ftol_edit, self.xtol_edit, self.iter_spin = QLineEdit("1e-8"), QLineEdit("1e-8"), QSpinBox()
        self.iter_spin.setRange(100, 10000); self.iter_spin.setValue(5000)
        tol_grid = QGridLayout()
        tol_grid.addWidget(QLabel("ftol"), 0, 0); tol_grid.addWidget(self.ftol_edit, 0, 1)
        tol_grid.addWidget(QLabel("xtol"), 0, 2); tol_grid.addWidget(self.xtol_edit, 0, 3)
        tol_grid.addWidget(QLabel("Maximum evaluations"), 1, 0, 1, 2)
        tol_grid.addWidget(self.iter_spin, 1, 2, 1, 2)
        tol_grid.setColumnStretch(1, 1); tol_grid.setColumnStretch(3, 1)
        self.tolerance_widget = self._wrap_layout(tol_grid)
        engine_form.addRow("Tolerances", self.tolerance_widget)

        self.constraints_box = QTextEdit(
            placeholderText="# Advanced lmfit expressions; table constraints are preferred\n# Example: p1_fwhm = p3_fwhm"
        ); self.constraints_box.setFixedHeight(68)
        engine_form.addRow("Advanced expressions", self.constraints_box)
        self.engine_advanced_rows = [
            self.weighting_combo, self.shift_spin, self.shared_shape_check,
            self.baseline_refine_check, self.tolerance_widget, self.constraints_box,
        ]
        self.background_combo.currentTextChanged.connect(self._on_background_method_changed)
        self.btn_edit_manual_baseline.toggled.connect(self._toggle_manual_baseline_editor)
        self.btn_reset_manual_baseline.clicked.connect(self._reset_manual_baseline)
        self.btn_clear_manual_baseline.clicked.connect(self._clear_manual_baseline)
        self.setup_canvas.manual_baseline_changed.connect(self._manual_baseline_points_changed)
        left_layout.addWidget(self.engine_box)

        # Actions stay visible below the scrolling options.
        left_layout.addStretch(1)
        action_bar = QFrame(); action_bar.setObjectName("actionDock")
        action_layout = QGridLayout(action_bar)
        action_layout.setContentsMargins(8, 8, 8, 8); action_layout.setSpacing(7)
        
        self.btn_preview = QPushButton("Preview BG", clicked=self.preview_background)
        self.btn_preview.setObjectName("secondaryButton")
        self.btn_preview.setToolTip("Preview the selected background before fitting.")
        self.btn_fit = QPushButton("Run Free Fit", clicked=self.run_fit)
        self.btn_fit.setObjectName("primaryAction")
        self.btn_export = QPushButton("Export results", clicked=self.export_all)
        self.btn_export.setObjectName("secondaryButton")
        self.btn_export.setToolTip("Export the report, figures, numerical CSV files, model and audit record.")
        self.btn_export.setEnabled(False)
        self.btn_fit.setMinimumHeight(40)
        self.btn_preview.setMinimumHeight(34); self.btn_export.setMinimumHeight(34)
        action_layout.addWidget(self.btn_fit, 0, 0, 1, 2)
        action_layout.addWidget(self.btn_preview, 1, 0)
        action_layout.addWidget(self.btn_export, 1, 1)
        left_panel_layout.addWidget(action_bar, 0)
        # Tabs
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setObjectName("analysisTabs")
        self.analysis_tabs.setDocumentMode(True)
        self.analysis_tabs.tabBar().setExpanding(False)
        data_tab = QWidget(); data_tab_layout = QVBoxLayout(data_tab)
        data_tab_layout.setContentsMargins(6, 6, 6, 6)
        self.data_table = QTableWidget(0, 2)
        self.data_table.setHorizontalHeaderLabels(["Binding Energy (eV)", "Intensity (a.u.)"])
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.data_table.setAlternatingRowColors(True)
        data_tab_layout.addWidget(self.data_table)
        self.analysis_tabs.addTab(data_tab, "Raw Data")

        self.model_tab = QWidget(); model_tab_layout = QVBoxLayout(self.model_tab)
        metadata_row = QHBoxLayout()
        self.model_name_edit = QLineEdit("Untitled XPS chemical-state model")
        self.model_reference_edit = QLineEdit(placeholderText="Required for publication: DOI, NIST record, standard, or reference spectrum")
        self.model_name_edit.textChanged.connect(self._invalidate_fit_after_model_edit)
        self.model_reference_edit.textChanged.connect(self._invalidate_fit_after_model_edit)
        metadata_row.addWidget(QLabel("Model:")); metadata_row.addWidget(self.model_name_edit, 1)
        metadata_row.addWidget(QLabel("Reference:")); metadata_row.addWidget(self.model_reference_edit, 2)
        model_tab_layout.addLayout(metadata_row)
        self.model_notes_edit = QLineEdit(placeholderText="Sample-specific assumptions, calibration, exclusions, or multiplet notes")
        self.model_notes_edit.textChanged.connect(self._invalidate_fit_after_model_edit)
        model_tab_layout.addWidget(self.model_notes_edit)

        model_buttons = QHBoxLayout()
        for text_value, slot in [
            ("From picked centers", self.build_model_from_centers),
            ("Add component", self.add_model_component),
            ("Add doublet", self.add_model_doublet),
            ("Center constraint…", self.configure_center_constraint),
        ]:
            button = QPushButton(text_value, clicked=slot)
            button.setObjectName("secondaryButton")
            model_buttons.addWidget(button)
        model_buttons.addStretch(1)
        model_tab_layout.addLayout(model_buttons)
        model_buttons_2 = QHBoxLayout()
        for text_value, slot in [
            ("Delete row(s)", self.delete_model_rows),
            ("Validate model", self.validate_model_ui),
            ("Import JSON", self.import_model_json),
            ("Export JSON", self.export_model_json),
        ]:
            button = QPushButton(text_value, clicked=slot)
            button.setObjectName("secondaryButton")
            model_buttons_2.addWidget(button)
        model_buttons_2.addStretch(1)
        model_tab_layout.addLayout(model_buttons_2)

        self.constraint_summary_label = QLabel("Center constraints: no active components")
        self.constraint_summary_label.setObjectName("constraintSummary")
        self.constraint_summary_label.setWordWrap(True)
        model_tab_layout.addWidget(self.constraint_summary_label)

        self.model_table = QTableWidget(0, len(MODEL_TABLE_FIELDS))
        self.model_table.setHorizontalHeaderLabels([title for title, _ in MODEL_TABLE_FIELDS])
        header_tips = {
            "center": "Reference/starting binding energy in eV.",
            "center_mode": "Free = global safety range; Fixed = locked; Bounded = Center min/max; Tied = target + offset.",
            "center_min": "Lower center bound used only in Bounded mode.",
            "center_max": "Upper center bound used only in Bounded mode.",
            "tie_target": "Exact component label used as the center reference in Tied mode.",
            "offset_init": "Starting/fixed value of (this center - target center), in eV.",
            "offset_min": "Minimum allowed tied offset; leave both offset limits blank for an exact tie.",
            "offset_max": "Maximum allowed tied offset; leave both offset limits blank for an exact tie.",
            "delta_e": "Spin-orbit splitting for a partner row.",
            "ratio": "Partner/primary integrated-area ratio.",
            "shift_group": "Optional shared energy shift for chemically related independent components.",
        }
        for field, tip in header_tips.items():
            item = self.model_table.horizontalHeaderItem(MODEL_FIELD_INDEX[field])
            if item is not None:
                item.setToolTip(tip)
        self.model_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.model_table.horizontalHeader().setSectionResizeMode(
            len(MODEL_TABLE_FIELDS)-1, QHeaderView.ResizeMode.Stretch
        )
        self.model_table.setAlternatingRowColors(True)
        self.model_table.itemChanged.connect(self._invalidate_fit_after_model_edit)
        self.model_table.itemChanged.connect(self._update_constraint_summary)
        model_tab_layout.addWidget(self.model_table, 1)
        model_help = QLabel(
            "Partner rows are tied to their primary by Pair, ΔE, area ratio, equal FWHM and equal line shape. "
            "Center modes: Free uses the global safety interval; Fixed is locked; Bounded uses Center min/max; "
            "Tied uses Tie target and a fixed or bounded offset. Named groups couple multiplet/state shifts, widths, or shapes. "
            "Use integrable asymmetric pseudo-Voigt when quantitative asymmetric-doublet areas are required."
        )
        model_help.setWordWrap(True)
        model_tab_layout.addWidget(model_help)
        self.analysis_tabs.addTab(self.model_tab, "Model Builder")

        results_tab = QWidget(); results_tab_layout = QVBoxLayout(results_tab)
        results_tab_layout.setContentsMargins(6, 6, 6, 6)
        self.fit_quality_banner = QLabel("Run a fit to see publication-readiness checks.")
        self.fit_quality_banner.setObjectName("qualityBanner")
        self.fit_quality_banner.setWordWrap(True)
        results_tab_layout.addWidget(self.fit_quality_banner)
        self.state_fraction_label = QLabel("Oxidation/chemical-state fractions: —")
        self.state_fraction_label.setObjectName("fractionBanner")
        self.state_fraction_label.setWordWrap(True)
        results_tab_layout.addWidget(self.state_fraction_label)
        self.results_table = QTableWidget(0, 14)
        self.results_table.setHorizontalHeaderLabels([
            "Peak", "State", "Input Center", "Fit Center ± SE", "FWHM ± SE",
            "Height", "Full Area ± SE", "Area %", "Center constraint",
            "Line shape", "Role", "Eta", "Asym", "QA"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        results_tab_layout.addWidget(self.results_table)
        self.analysis_tabs.addTab(results_tab, "Fit Results")

        log_tab = QWidget(); log_tab_layout = QVBoxLayout(log_tab)
        log_tab_layout.setContentsMargins(8, 8, 8, 8)
        self.metrics_label = QLabel("Metrics will appear here"); self.log_box = QTextEdit()
        log_tab_layout.addWidget(self.metrics_label); log_tab_layout.addWidget(self.log_box, 1)
        self.analysis_tabs.addTab(log_tab, "Logs")
        # Right Panel
        right_panel = QFrame(); right_panel.setObjectName("workspacePanel")
        right_panel.setMinimumWidth(560)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(7, 7, 7, 7)
        self.setup_right_splitter = QSplitter(Qt.Orientation.Vertical)
        self.setup_right_splitter.setChildrenCollapsible(False)
        self.setup_right_splitter.setHandleWidth(5)
        setup_plot_panel = QFrame(); setup_plot_panel.setObjectName("plotPanel")
        setup_plot_layout = QVBoxLayout(setup_plot_panel)
        setup_plot_layout.setContentsMargins(0, 0, 0, 0); setup_plot_layout.setSpacing(4)
        setup_plot_panel.setMinimumHeight(350)
        self.setup_canvas.peak_selected.connect(self._add_peak_from_plot)
        self.setup_canvas.centers_updated.connect(self._on_centers_dragged)
        setup_plot_layout.addWidget(
            self._build_plot_header(self.setup_canvas, setup_plot_panel, "Spectrum")
        )
        setup_plot_layout.addWidget(self.setup_canvas, 1)
        self.analysis_tabs.setMinimumHeight(250)
        self.setup_right_splitter.addWidget(setup_plot_panel)
        self.setup_right_splitter.addWidget(self.analysis_tabs)
        self.setup_right_splitter.setStretchFactor(0, 3)
        self.setup_right_splitter.setStretchFactor(1, 2)
        self.setup_right_splitter.setSizes([520, 360])
        right_layout.addWidget(self.setup_right_splitter, 1)

        self.content_splitter.addWidget(left_panel); self.content_splitter.addWidget(right_panel)
        self.content_splitter.setStretchFactor(0, 0)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([400, 1400])

        self.workflow_stack.addWidget(self.content_splitter)
        self.workflow_stack.addWidget(self._build_fit_studio_page())

        for plot_canvas in (self.setup_canvas, self.canvas):
            plot_canvas.cursor_moved.connect(
                lambda x, y: self.cursor_status_label.setText(f"X: {x:.4f} eV   |   Y: {y:.4f}")
            )
            plot_canvas.cursor_left_axes.connect(
                lambda: self.cursor_status_label.setText("X: -, Y: -")
            )
        self._toggle_setup_advanced(False)
        self._on_fit_mode_changed()
        self._on_background_method_changed(self.background_combo.currentText())
        self._show_workflow_page(0)

    def _build_fit_studio_page(self) -> QWidget:
        """Create a roomy second workspace for constrained, interactive refinement."""
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)

        help_bar = QFrame(); help_bar.setObjectName("studioHelp")
        help_layout = QHBoxLayout(help_bar); help_layout.setContentsMargins(12, 8, 12, 8)
        help_text = QLabel(
            "Select a peak, then drag it or use Center / FWHM / Height. Use the mouse wheel to zoom."
        )
        help_text.setWordWrap(True)
        help_layout.addWidget(help_text, 1)
        self.studio_fit_status = QLabel("No fitted model yet")
        self.studio_fit_status.setObjectName("studioStatusPill")
        help_layout.addWidget(self.studio_fit_status)
        page_layout.addWidget(help_bar)

        summary_frame = QFrame(); summary_frame.setObjectName("studioSummary")
        summary_layout = QVBoxLayout(summary_frame); summary_layout.setContentsMargins(10, 6, 10, 6)
        summary_grid = QGridLayout()
        summary_grid.setContentsMargins(0, 0, 0, 0)
        summary_grid.setHorizontalSpacing(10)
        summary_grid.setVerticalSpacing(6)
        summary_grid.setColumnStretch(1, 1)
        self.studio_quality_label = QLabel("Run the first fit on page 1.")
        self.studio_quality_label.setObjectName("studioQuality")
        self.studio_metrics_label = QLabel("RMSE —   |   R² —   |   DW —")
        self.studio_metrics_label.setWordWrap(True)
        self.studio_fraction_label = QLabel("Fractions: —")
        self.studio_fraction_label.setObjectName("fractionBanner")
        self.studio_fraction_label.setWordWrap(True)
        summary_grid.addWidget(self.studio_quality_label, 0, 0)
        summary_grid.addWidget(self.studio_metrics_label, 0, 1)
        summary_grid.addWidget(self.studio_fraction_label, 1, 0, 1, 2)
        summary_layout.addLayout(summary_grid)
        self.studio_warning_label = QLabel("")
        self.studio_warning_label.setObjectName("studioWarning")
        self.studio_warning_label.setWordWrap(True)
        summary_layout.addWidget(self.studio_warning_label)
        page_layout.addWidget(summary_frame)

        self.studio_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.studio_splitter.setChildrenCollapsible(False)
        self.studio_splitter.setHandleWidth(7)
        self.studio_splitter.splitterMoved.connect(self._remember_studio_sidebar_width)
        page_layout.addWidget(self.studio_splitter, 1)

        controls_panel = QFrame(); controls_panel.setObjectName("sidePanel")
        controls_panel.setMinimumWidth(320)
        controls_panel.setMaximumWidth(560)
        controls_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        controls_outer = QVBoxLayout(controls_panel); controls_outer.setContentsMargins(8, 8, 8, 8)
        controls_outer.setSpacing(8)
        controls_scroll = ResponsiveScrollArea(); controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        controls_scroll.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.studio_controls_scroll = controls_scroll
        controls_widget = QWidget()
        controls_widget.setMinimumWidth(0)
        controls_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        controls = QVBoxLayout(controls_widget)
        controls.setContentsMargins(0, 0, 0, 8); controls.setSpacing(10)
        controls_scroll.setWidget(controls_widget); controls_outer.addWidget(controls_scroll, 1)

        components_box = QGroupBox("Peaks")
        components_layout = QVBoxLayout(components_box)
        self.studio_peak_table = QTableWidget(0, 5)
        self.studio_peak_table.setHorizontalHeaderLabels([
            "Peak", "Center", "FWHM", "Height", "Area %"
        ])
        self.studio_peak_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.studio_peak_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.studio_peak_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.studio_peak_table.setAlternatingRowColors(True)
        self.studio_peak_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.studio_peak_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.studio_peak_table.setMinimumHeight(175)
        self.studio_peak_table.setMaximumHeight(245)
        self.studio_peak_table.itemSelectionChanged.connect(self._studio_table_selection_changed)
        components_layout.addWidget(self.studio_peak_table)
        peak_buttons = QHBoxLayout()
        self.btn_studio_add_peak = QPushButton("+ Add", clicked=self._studio_add_free_peak)
        self.btn_studio_duplicate_peak = QPushButton("Duplicate", clicked=self._studio_duplicate_free_peak)
        self.btn_studio_remove_peak = QPushButton("Remove", clicked=self._studio_remove_free_peak)
        for button in (self.btn_studio_add_peak, self.btn_studio_duplicate_peak, self.btn_studio_remove_peak):
            button.setObjectName("secondaryButton"); peak_buttons.addWidget(button, 1)
        components_layout.addLayout(peak_buttons)
        controls.addWidget(components_box)

        parameter_box = QGroupBox("Selected peak")
        parameter_grid = QGridLayout(parameter_box)
        parameter_grid.setColumnStretch(2, 1)
        self.studio_parameter_spins: Dict[str, QDoubleSpinBox] = {}
        self.studio_parameter_sliders: Dict[str, QSlider] = {}
        self.studio_advanced_parameter_widgets: List[QWidget] = []
        parameter_rows = [
            ("center", "Center (eV)", -1000.0, 5000.0, 4, 0.01),
            ("fwhm", "FWHM (eV)", 0.01, 20.0, 4, 0.01),
            ("height", "Height", 0.0, 1.0e12, 2, 1.0),
            ("eta", "Lorentzian η", 0.0, 1.0, 3, 0.01),
            ("asym", "Asymmetry", 0.0, 0.98, 3, 0.01),
        ]
        for row, (field, label, minimum, maximum, decimals, step) in enumerate(parameter_rows):
            spin = QDoubleSpinBox(); spin.setRange(minimum, maximum)
            spin.setDecimals(decimals); spin.setSingleStep(step); spin.setKeyboardTracking(False)
            slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(0, 1000)
            spin.valueChanged.connect(
                lambda value, parameter=field: self._studio_spin_changed(parameter, value)
            )
            slider.valueChanged.connect(
                lambda value, parameter=field: self._studio_slider_changed(parameter, value)
            )
            slider.sliderPressed.connect(
                lambda parameter=field: self._studio_slider_pressed(parameter)
            )
            slider.sliderReleased.connect(self._studio_slider_released)
            self.studio_parameter_spins[field] = spin
            self.studio_parameter_sliders[field] = slider
            field_label = QLabel(label)
            parameter_grid.addWidget(field_label, row, 0)
            parameter_grid.addWidget(spin, row, 1)
            parameter_grid.addWidget(slider, row, 2)
            if field in {"eta", "asym"}:
                self.studio_advanced_parameter_widgets.extend([field_label, spin, slider])

        shape_row = len(parameter_rows)
        self.studio_shape_label = QLabel("Line shape")
        self.studio_shape_combo = QComboBox()
        for label, value in [
            ("Pseudo-Voigt", "pseudo_voigt"), ("Gaussian", "gaussian"),
            ("Lorentzian", "lorentzian"),
            ("Asymmetric Pseudo-Voigt", "asymmetric_pseudo_voigt"),
            ("Doniach-Sunjic", "doniach_sunjic"),
        ]:
            self.studio_shape_combo.addItem(label, value)
        self._make_combo_responsive(self.studio_shape_combo, 18)
        self.studio_shape_combo.currentIndexChanged.connect(self._studio_shape_changed)
        parameter_grid.addWidget(self.studio_shape_label, shape_row, 0)
        parameter_grid.addWidget(self.studio_shape_combo, shape_row, 1, 1, 2)
        self.studio_advanced_parameter_widgets.extend([
            self.studio_shape_label, self.studio_shape_combo
        ])
        self.btn_studio_more = QPushButton("More peak controls", checkable=True)
        self.btn_studio_more.setObjectName("secondaryButton")
        self.btn_studio_more.toggled.connect(self._studio_toggle_more_controls)
        parameter_grid.addWidget(self.btn_studio_more, shape_row + 1, 0, 1, 3)
        controls.addWidget(parameter_box)

        interaction_box = QGroupBox("Actions")
        interaction_layout = QVBoxLayout(interaction_box)
        self.studio_manual_check = QCheckBox("Enable direct editing on graph")
        self.studio_manual_check.setChecked(True)
        self.studio_manual_check.toggled.connect(self._studio_manual_toggled)
        self.studio_live_refit_check = QCheckBox("Auto-refine peak heights (optional)")
        self.studio_live_refit_check.setChecked(False)
        self.studio_live_refit_check.setToolTip(
            "Live quality metrics update even when this is off. If enabled, non-negative least-squares may "
            "override a manual height and drive an unsupported component (for example W5+) to zero."
        )
        interaction_layout.addWidget(self.studio_manual_check)
        interaction_layout.addWidget(self.studio_live_refit_check)
        history_row = QHBoxLayout()
        self.btn_studio_undo = QPushButton("Undo", clicked=self._studio_undo)
        self.btn_studio_redo = QPushButton("Redo", clicked=self._studio_redo)
        self.btn_studio_reset = QPushButton("Reset fit", clicked=self._studio_reset)
        for button in (self.btn_studio_undo, self.btn_studio_redo, self.btn_studio_reset):
            button.setObjectName("secondaryButton"); history_row.addWidget(button, 1)
        interaction_layout.addLayout(history_row)
        controls.addWidget(interaction_box)
        controls.addStretch(1)

        # Keep the final actions pinned below the scrolling controls.  They
        # remain available on short and high-DPI screens.
        finish_box = QFrame(); finish_box.setObjectName("actionDock")
        finish_layout = QGridLayout(finish_box)
        finish_layout.setContentsMargins(8, 8, 8, 8)
        finish_layout.setHorizontalSpacing(6)
        finish_layout.setVerticalSpacing(6)
        self.btn_studio_final_fit = QPushButton("Run Final Fit", clicked=self._studio_run_final_fit)
        self.btn_studio_final_fit.setObjectName("finalFitButton")
        self.btn_studio_final_fit.setEnabled(False)
        self.btn_studio_export_preview = QPushButton(
            "Export Preview", clicked=self.export_current_preview
        )
        self.btn_studio_export_preview.setObjectName("secondaryButton")
        self.btn_studio_export_preview.setEnabled(False)
        self.btn_studio_export_preview.setToolTip(
            "Export the current operator-adjusted values with a PROVISIONAL label and no parameter uncertainties."
        )
        self.btn_studio_export = QPushButton("Export Final", clicked=self.export_all)
        self.btn_studio_export.setEnabled(False)
        self.btn_studio_export.setToolTip(
            "Export the latest completed optimizer result, including calculated uncertainties."
        )
        self.btn_studio_final_fit.setMinimumHeight(40)
        self.btn_studio_export_preview.setMinimumHeight(34)
        self.btn_studio_export.setMinimumHeight(34)
        finish_layout.addWidget(self.btn_studio_final_fit, 0, 0, 1, 2)
        finish_layout.addWidget(self.btn_studio_export_preview, 1, 0)
        finish_layout.addWidget(self.btn_studio_export, 1, 1)
        controls_outer.addWidget(finish_box, 0)

        plot_panel = QFrame(); plot_panel.setObjectName("plotPanel")
        plot_panel.setMinimumWidth(560)
        plot_layout = QVBoxLayout(plot_panel)
        plot_layout.setContentsMargins(4, 4, 4, 4); plot_layout.setSpacing(4)
        plot_layout.addWidget(
            self._build_plot_header(
                self.canvas, plot_panel, "Interactive fit", self._studio_reset_view
            )
        )
        plot_layout.addWidget(self.canvas, 1)

        self.studio_splitter.addWidget(controls_panel); self.studio_splitter.addWidget(plot_panel)
        self.studio_splitter.setStretchFactor(0, 0)
        self.studio_splitter.setStretchFactor(1, 1)
        self.studio_splitter.setSizes([400, 1200])

        self.canvas.fit_component_selected.connect(self._studio_select_component)
        self.canvas.fit_component_edited.connect(self._studio_canvas_edited)
        self.canvas.set_fit_editing(True)
        self.studio_refit_timer = QTimer(self)
        self.studio_refit_timer.setSingleShot(True)
        self.studio_refit_timer.setInterval(320)
        self.studio_refit_timer.timeout.connect(self._studio_refine_heights)
        self._studio_toggle_more_controls(False)
        self._studio_update_history_buttons()
        return page

    def _show_workflow_page(self, index: int):
        if index == 1 and not self.studio_step_button.isEnabled():
            self.status.showMessage("Run a fit before opening Edit & Review.", 5000)
            return
        self.workflow_stack.setCurrentIndex(index)
        if index == 1:
            QTimer.singleShot(0, self._apply_studio_sidebar_width)
        active = "background:#1D4ED8; color:white; border-radius:7px; padding:7px 18px;"
        inactive = "background:#E2E8F0; color:#475569; border-radius:7px; padding:7px 18px;"
        self.setup_step_button.setStyleSheet(active if index == 0 else inactive)
        self.studio_step_button.setStyleSheet(active if index == 1 else inactive)

    @staticmethod
    def _safe_sidebar_width(value: Any, default: int = 400) -> int:
        try:
            return int(np.clip(int(value), 320, 560))
        except (TypeError, ValueError):
            return int(default)

    def _restore_workspace_layout(self):
        """Restore only bounded panel dimensions; never restore an off-screen window."""
        setup_width = self._safe_sidebar_width(
            self.ui_settings.value("setup_sidebar_width", 400)
        )
        studio_width = self._safe_sidebar_width(
            self.ui_settings.value("studio_sidebar_width", 400)
        )
        self._preferred_studio_sidebar_width = studio_width
        self.content_splitter.setSizes([
            setup_width, max(self.content_splitter.width() - setup_width, 560)
        ])
        self.studio_splitter.setSizes([
            studio_width, max(self.studio_splitter.width() - studio_width, 560)
        ])
        setup_vertical = self.ui_settings.value("setup_vertical_splitter_state")
        if setup_vertical is not None:
            self.setup_right_splitter.restoreState(setup_vertical)

    def _apply_studio_sidebar_width(self):
        if not hasattr(self, "studio_splitter"):
            return
        preferred = self._safe_sidebar_width(
            getattr(self, "_preferred_studio_sidebar_width", 400)
        )
        self.studio_splitter.setSizes([
            preferred, max(self.studio_splitter.width() - preferred, 560)
        ])

    def _remember_studio_sidebar_width(self, *_):
        sizes = self.studio_splitter.sizes() if hasattr(self, "studio_splitter") else []
        if sizes and sizes[0] > 0:
            self._preferred_studio_sidebar_width = self._safe_sidebar_width(sizes[0])

    def _reset_workspace_layout(self):
        """Recover a readable layout after DPI, monitor or manual size changes."""
        self.content_splitter.setSizes([400, max(self.content_splitter.width() - 400, 560)])
        self.studio_splitter.setSizes([400, max(self.studio_splitter.width() - 400, 560)])
        self._preferred_studio_sidebar_width = 400
        self.setup_right_splitter.setSizes([520, 360])
        for scroll in (
            getattr(self, "setup_controls_scroll", None),
            getattr(self, "studio_controls_scroll", None),
        ):
            if scroll is not None:
                scroll.verticalScrollBar().setValue(0)
                scroll.horizontalScrollBar().setValue(0)
        self.ui_settings.setValue("setup_sidebar_width", 400)
        self.ui_settings.setValue("studio_sidebar_width", 400)
        self.ui_settings.remove("setup_vertical_splitter_state")
        self.status.showMessage("Workspace layout restored.", 4000)

    def closeEvent(self, event):
        if hasattr(self, "content_splitter") and self.content_splitter.sizes():
            self.ui_settings.setValue(
                "setup_sidebar_width", self.content_splitter.sizes()[0]
            )
        if hasattr(self, "studio_splitter") and self.studio_splitter.sizes():
            self.ui_settings.setValue(
                "studio_sidebar_width",
                self._safe_sidebar_width(
                    getattr(
                        self, "_preferred_studio_sidebar_width",
                        self.studio_splitter.sizes()[0],
                    )
                ),
            )
        if hasattr(self, "setup_right_splitter"):
            self.ui_settings.setValue(
                "setup_vertical_splitter_state", self.setup_right_splitter.saveState()
            )
        self.ui_settings.sync()
        super().closeEvent(event)

    def _fit_mode(self) -> str:
        return str(self.fit_mode_combo.currentData() or "free")

    def _toggle_setup_advanced(self, visible: bool):
        visible = bool(visible)
        self.btn_show_advanced.blockSignals(True)
        self.btn_show_advanced.setChecked(visible)
        self.btn_show_advanced.blockSignals(False)
        self.btn_show_advanced.setText(
            "Hide advanced settings" if visible else "Show advanced settings"
        )
        self.calib_box.setVisible(visible)
        physics_mode = self._fit_mode() == "physics"
        self.db_box.setVisible(physics_mode or visible)
        for field_widget in self.engine_advanced_rows:
            self.engine_form.setRowVisible(field_widget, visible)
        model_index = self.analysis_tabs.indexOf(self.model_tab)
        if model_index >= 0:
            self.analysis_tabs.setTabVisible(model_index, physics_mode or visible)

    def _on_fit_mode_changed(self, *_):
        if not hasattr(self, "auto_doublet_check"):
            return
        free_mode = self._fit_mode() == "free"
        previous_mode = str(self.fit_context.get("ui_fit_mode", ""))
        if self.fit_result is not None and previous_mode and previous_mode != self._fit_mode():
            self.fit_result = None
            self.fit_context = {}
            self.studio_state = []
            self.studio_original_state = []
            self.studio_step_button.setEnabled(False)
            self._set_export_enabled(False)
            self.status.showMessage("Fit mode changed — run a new fit.", 5000)
        self.auto_doublet_check.blockSignals(True)
        self.use_model_check.blockSignals(True)
        self.shared_shape_check.blockSignals(True)
        if free_mode:
            self.auto_doublet_check.setChecked(False)
            self.use_model_check.setChecked(False)
            self.shared_shape_check.setChecked(False)
            self.btn_fit.setText("Run Free Fit")
            self.fit_mode_help_label.setText(
                "Pick any number of peaks. Center, width, height and line shape are fitted independently; "
                "no element, oxidation state or doublet is assumed."
            )
        else:
            self.auto_doublet_check.setChecked(True)
            if self.model_table.rowCount() > 0:
                self.use_model_check.setChecked(True)
            self.shared_shape_check.setChecked(True)
            self.btn_fit.setText("Run Physics-Constrained Fit")
            self.fit_mode_help_label.setText(
                "Use a reference model for doublets, chemical states and auditable parameter ties."
            )
        self.auto_doublet_check.blockSignals(False)
        self.use_model_check.blockSignals(False)
        self.shared_shape_check.blockSignals(False)
        self._toggle_setup_advanced(self.btn_show_advanced.isChecked())
        if hasattr(self, "btn_studio_final_fit"):
            self.btn_studio_final_fit.setText(
                "Run Final Free Fit" if free_mode else "Run Final Constrained Fit"
            )

    def _wrap_layout(self, layout):
        layout.setContentsMargins(0, 0, 0, 0)
        if hasattr(layout, "setSpacing"):
            layout.setSpacing(6)
        widget = QWidget()
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        widget.setMinimumWidth(0)
        widget.setLayout(layout)
        return widget

    def apply_light_theme(self):
        style = """
            QMainWindow, QWidget {
                background-color: #F8FAFC;
                color: #0F172A;
                font-size: 13px;
            }
            QLabel {
                background: transparent;
                border: none;
            }
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QFrame#hero {
                background: #EFF6FF;
                border: 1px solid #BFDBFE;
                border-radius: 12px;
            }
            QFrame#sidePanel {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
            QFrame#workspacePanel {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 12px;
            }
            QFrame#plotPanel {
                background: #FFFFFF;
                border: none;
            }
            QFrame#actionDock {
                background: #FFFFFF;
                border: 1px solid #DCE4EE;
                border-radius: 10px;
            }
            QFrame#plotToolbar {
                background: #F8FAFC;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            QLabel#panelTitle {
                color: #1E3A8A;
                font-size: 14px;
                font-weight: 700;
            }
            QGroupBox {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 10px;
                margin-top: 13px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 11px;
                color: #1D4ED8;
                background: #FFFFFF;
                padding: 0 5px;
                font-weight: bold;
            }
            QPushButton {
                background: #2563EB;
                color: white;
                border: none;
                border-radius: 7px;
                min-height: 28px;
                font-weight: 600;
                padding: 4px 10px;
            }
            QPushButton:hover { background: #1D4ED8; }
            QPushButton:pressed { background: #1E40AF; }
            QPushButton:disabled {
                background: #E2E8F0;
                color: #94A3B8;
                border: 1px solid #CBD5E1;
            }
            QPushButton#secondaryButton {
                background: #F1F5F9;
                color: #334155;
                border: 1px solid #CBD5E1;
            }
            QPushButton#secondaryButton:hover { background: #E2E8F0; }
            QPushButton#primaryAction,
            QPushButton#finalFitButton {
                background: #059669;
                color: white;
                border: none;
                min-height: 36px;
                font-size: 14px;
            }
            QPushButton#primaryAction:hover,
            QPushButton#finalFitButton:hover { background: #047857; }
            QPushButton#primaryAction:disabled,
            QPushButton#finalFitButton:disabled,
            QPushButton#secondaryButton:disabled {
                background: #E2E8F0;
                color: #94A3B8;
                border: 1px solid #CBD5E1;
            }
            QPushButton#compactButton {
                background: #FFFFFF;
                color: #334155;
                border: 1px solid #CBD5E1;
                min-height: 24px;
                padding: 2px 9px;
                font-size: 12px;
            }
            QPushButton#compactButton:hover { background: #E2E8F0; }
            QPushButton#compactButton:disabled {
                background: #E2E8F0;
                color: #94A3B8;
                border: 1px solid #CBD5E1;
            }
            QToolButton#linkButton {
                background: #F8FAFC;
                border: 1px solid #D8E3F0;
                border-radius: 7px;
                padding: 6px 12px;
                color: #334155;
            }
            QToolButton#linkButton:hover { background: #F1F5F9; }
            QToolBar {
                background: transparent;
                border: none;
                spacing: 2px;
            }
            QToolBar QToolButton {
                background: transparent;
                border: none;
                border-radius: 5px;
                padding: 3px;
            }
            QToolBar QToolButton:hover { background: #E2E8F0; }

            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
                background: #FFFFFF;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 5px 8px;
                min-height: 27px;
            }
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {
                border: 1px solid #3B82F6;
            }
            QComboBox::drop-down {
                border: none;
                width: 24px;
            }
            QCheckBox { spacing: 7px; }
            QTableWidget {
                background: #FFFFFF;
                alternate-background-color: #F8FAFC;
                gridline-color: #E2E8F0;
                border: 1px solid #CBD5E1;
                border-radius: 8px;
                selection-background-color: #DBEAFE;
                selection-color: #0F172A;
            }
            QHeaderView::section {
                background: #F1F5F9;
                padding: 7px;
                border: none;
                border-right: 1px solid #E2E8F0;
                font-weight: bold;
                color: #475569;
            }
            QTabWidget::pane {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
                top: -1px;
            }
            QTabBar::tab {
                background: #F1F5F9;
                padding: 8px 14px;
                border: 1px solid #D8E1EC;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #2563EB;
                font-weight: bold;
                border-top: 3px solid #2563EB;
            }
            QSplitter::handle { background: #DBEAFE; }
            QSplitter::handle:horizontal {
                width: 7px;
                margin: 10px 1px;
                border-radius: 3px;
            }
            QSplitter::handle:vertical {
                height: 7px;
                margin: 1px 10px;
                border-radius: 3px;
            }
            QSplitter::handle:hover { background: #60A5FA; }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #CBD5E1;
                border-radius: 4px;
                min-height: 28px;
            }
            QScrollBar::handle:vertical:hover { background: #94A3B8; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
            }
            QLabel#constraintSummary {
                background: #EFF6FF;
                color: #1E3A8A;
                border: 1px solid #BFDBFE;
                border-radius: 6px;
                padding: 8px 10px;
                font-weight: 600;
            }
            QLabel#qualityBanner {
                background: #F8FAFC;
                color: #334155;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 9px 11px;
                font-weight: bold;
            }
            QLabel#fractionBanner {
                background: #F8FAFC;
                color: #334155;
                border-left: 4px solid #3B82F6;
                padding: 7px 10px;
            }
            QLabel#simpleModeHelp {
                background: #EFF6FF;
                color: #1E3A8A;
                border-left: 3px solid #3B82F6;
                padding: 6px 8px;
            }
            QLabel#manualBaselineStatus {
                background: #FFF7ED;
                color: #9A3412;
                border-left: 3px solid #F97316;
                padding: 6px 8px;
            }
            QFrame#workflowNav {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 9px;
            }
            QFrame#studioHelp {
                background: #EFF6FF;
                border: 1px solid #BFDBFE;
                border-radius: 8px;
            }
            QFrame#studioSummary {
                background: #FFFFFF;
                border: 1px solid #E2E8F0;
                border-radius: 8px;
            }
            QLabel#studioStatusPill {
                background: #E2E8F0;
                color: #334155;
                border-radius: 10px;
                padding: 5px 10px;
                font-weight: bold;
            }
            QLabel#studioQuality {
                background: #F8FAFC;
                color: #334155;
                border: 1px solid #CBD5E1;
                border-radius: 7px;
                padding: 6px 8px;
                font-size: 13px;
                font-weight: bold;
            }
            QLabel#studioWarning {
                background: #FFFBEB;
                color: #92400E;
                border-left: 4px solid #F59E0B;
                padding: 7px 10px;
            }
            QSlider::groove:horizontal {
                height: 5px;
                background: #CBD5E1;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #2563EB;
                border: 1px solid #1D4ED8;
                width: 15px;
                margin: -5px 0;
                border-radius: 7px;
            }
        """
        self.setStyleSheet(style)
        
    def log(self, t): self.log_box.append(t); self.status.showMessage(t, 3000)
    def warn(self, t): QMessageBox.warning(self, APP_NAME, t)
    def info(self, t): QMessageBox.information(self, "Success", t)

    def _selected_lineshape(self) -> str:
        text = self.lineshape_combo.currentText().lower()
        if "doniach" in text:
            return "doniach_sunjic"
        if "gaussian" in text:
            return "gaussian"
        if "lorentzian" in text:
            return "lorentzian"
        if "asymmetric" in text:
            return "asymmetric_pseudo_voigt"
        return "pseudo_voigt"

    # ------------------------------------------------------------------
    # Manual background editor
    # ------------------------------------------------------------------
    def _manual_background_selected(self) -> bool:
        return "manual" in self.background_combo.currentText().strip().lower()

    def _manual_fit_region(self) -> Tuple[np.ndarray, np.ndarray, Tuple[float, float]]:
        if self.energy is None:
            raise ValueError("Load an XPS data file first.")
        free_mode = self._fit_mode() == "free"
        if self.use_model_check.isChecked() and not free_mode:
            model_doc, _ = self._validated_model_from_ui()
            region_centers = [float(spec["center"]) for spec in model_doc["components"]]
        else:
            centers = parse_centers(self.centers_edit.text())
            region_centers = _expanded_centers(
                centers,
                None if free_mode else self.db_fetched_info,
                False if free_mode else self.auto_doublet_check.isChecked(),
            )
        return extract_fit_region(
            self.energy, self.intensity, region_centers, self.margin_spin.value()
        )

    def _discard_fit_for_background_edit(self, message: str) -> bool:
        if self.fit_result is None:
            return False
        self.fit_result = None
        self.fit_context = {}
        self.studio_state = []
        self.studio_original_state = []
        self.studio_history = []
        self.studio_redo = []
        self.studio_dirty = False
        self.studio_step_button.setEnabled(False)
        self._set_export_enabled(False)
        self.status.showMessage(message, 5000)
        return True

    def _on_background_method_changed(self, text: str):
        if not hasattr(self, "manual_baseline_widget"):
            return
        manual = "manual" in str(text).lower()
        none_selected = str(text).strip().lower() == "none"
        # Hide the complete form row (including its label) to avoid the blank
        # "Manual baseline" row visible in the old sidebar.
        self.engine_form.setRowVisible(self.manual_baseline_widget, manual)
        self.btn_preview.setText("Edit Baseline" if manual else "Preview BG")
        self.btn_preview.setToolTip(
            "Preview or edit the manual baseline anchors before fitting."
            if manual else
            "Preview the selected background before fitting."
        )
        self.baseline_refine_check.setEnabled(not manual and not none_selected)
        self.baseline_refine_check.setToolTip(
            "Disabled for a manual background so the exported line passes exactly through the recorded anchors."
            if manual else
            "Fits a small bounded offset/slope after the selected static background."
        )
        if not manual:
            self.btn_edit_manual_baseline.blockSignals(True)
            self.btn_edit_manual_baseline.setChecked(False)
            self.btn_edit_manual_baseline.blockSignals(False)
            self.btn_edit_manual_baseline.setText("Edit on graph")
            self.setup_canvas.set_manual_baseline_editing(False)
        if not self._manual_baseline_ui_update:
            discarded = self._discard_fit_for_background_edit(
                "Background method changed — run a new fit before export."
            )
            if discarded and self.energy is not None:
                self.refresh_loaded_plot()

    def _set_manual_editor_checked(self, checked: bool):
        self.btn_edit_manual_baseline.blockSignals(True)
        self.btn_edit_manual_baseline.setChecked(bool(checked))
        self.btn_edit_manual_baseline.setText(
            "Finish editing" if checked else "Edit on graph"
        )
        self.btn_edit_manual_baseline.blockSignals(False)
        self.setup_canvas.set_manual_baseline_editing(bool(checked))

    def _render_manual_baseline_preview(self, editing: bool):
        x, y, fit_region = self._manual_fit_region()
        self._manual_baseline_region = tuple(float(v) for v in fit_region)
        try:
            anchors = normalize_manual_background_points(
                self.manual_baseline_points, x, require_edge_coverage=True
            )
            self.manual_baseline_points = anchors
            background = manual_background(x, y, anchors)
            self.manual_baseline_status.setText(
                f"{len(anchors)} anchors define the full fit interval. "
                "Left-click adds, drag moves, right-click removes."
            )
        except ValueError as exc:
            background = XPSPlotCanvas._preview_manual_curve(
                x, self.manual_baseline_points
            )
            self.manual_baseline_status.setText(
                f"Incomplete manual baseline: {exc}"
            )
        self.setup_canvas.plot_raw_and_background(
            x, y, background,
            manual_points=self.manual_baseline_points,
            manual_editing=editing,
        )

    def _toggle_manual_baseline_editor(self, enabled: bool):
        if self._manual_baseline_ui_update:
            return
        if not enabled:
            self._set_manual_editor_checked(False)
            return
        try:
            if not self._manual_background_selected():
                self._manual_baseline_ui_update = True
                self.background_combo.setCurrentText("Manual (anchor points)")
                self._manual_baseline_ui_update = False
                self._on_background_method_changed(self.background_combo.currentText())
            self.btn_pick_peaks.setChecked(False)
            x, y, fit_region = self._manual_fit_region()
            try:
                normalize_manual_background_points(
                    self.manual_baseline_points, x, require_edge_coverage=True
                )
            except ValueError:
                self.manual_baseline_points = seed_manual_background_points(x, y)
            self._manual_baseline_region = tuple(float(v) for v in fit_region)
            self._discard_fit_for_background_edit(
                "Manual background changed — run a new fit before export."
            )
            self._set_manual_editor_checked(True)
            self._render_manual_baseline_preview(editing=True)
            self.status.showMessage(
                "Manual baseline: left-click to add, drag to move, right-click an anchor to remove.",
                7000,
            )
        except Exception as exc:
            self._set_manual_editor_checked(False)
            self.warn(str(exc))

    def _manual_baseline_points_changed(self, points: List[Any]):
        if self._manual_baseline_ui_update:
            return
        self.manual_baseline_points = [
            (float(point[0]), float(point[1])) for point in (points or [])
        ]
        self._discard_fit_for_background_edit(
            "Manual background changed — run a new fit before export."
        )
        try:
            self._render_manual_baseline_preview(
                editing=self.btn_edit_manual_baseline.isChecked()
            )
        except Exception as exc:
            self.manual_baseline_status.setText(str(exc))

    def _reset_manual_baseline(self):
        try:
            if not self._manual_background_selected():
                self.background_combo.setCurrentText("Manual (anchor points)")
            x, y, fit_region = self._manual_fit_region()
            self.manual_baseline_points = seed_manual_background_points(x, y)
            self._manual_baseline_region = tuple(float(v) for v in fit_region)
            self._discard_fit_for_background_edit(
                "Manual background reset — run a new fit before export."
            )
            self._set_manual_editor_checked(True)
            self._render_manual_baseline_preview(editing=True)
        except Exception as exc:
            self.warn(str(exc))

    def _clear_manual_baseline(self):
        self.manual_baseline_points = []
        self._manual_baseline_region = None
        self._discard_fit_for_background_edit(
            "Manual background cleared — define anchors and run a new fit."
        )
        try:
            self._set_manual_editor_checked(True)
            self._render_manual_baseline_preview(editing=True)
        except Exception as exc:
            self.manual_baseline_status.setText(str(exc))

    # ------------------------------------------------------------------
    # Interactive Fit Studio
    # ------------------------------------------------------------------
    def _initialize_studio_state(self, result: Dict[str, Any]):
        specs_by_label = {
            str(spec.get("label", "")): spec for spec in result.get("model_specs", [])
        }
        shift_bound = float(self.fit_context.get("shift_bound", 0.40))
        free_studio = str(self.fit_context.get("ui_fit_mode", self._fit_mode())) == "free"
        x_fit = np.asarray(self.fit_context.get("x_fit", [0.0, 1.0]), dtype=float)
        fit_center_min, fit_center_max = float(np.min(x_fit)), float(np.max(x_fit))
        y_span = max(float(np.ptp(self.fit_context.get("y_corr", np.array([1.0])))), 1.0)
        height_ceiling = max(4.0 * y_span, 1.0)
        state: List[Dict[str, Any]] = []
        for peak in result.get("peak_results", []):
            spec = copy.deepcopy(specs_by_label.get(peak.label, {}))
            mode = canonical_center_mode(
                spec.get("center_mode", peak.center_mode), peak.role,
                "center_min" in spec and "center_max" in spec
            )
            if mode == "bounded":
                center_min = float(spec.get("center_min", peak.center_fitted - shift_bound))
                center_max = float(spec.get("center_max", peak.center_fitted + shift_bound))
            elif mode == "free":
                reference_center = float(spec.get("center", peak.center_input))
                if free_studio:
                    # Free Fit is intentionally direct-manipulation friendly:
                    # a selected peak can be dragged anywhere in the fit window.
                    center_min, center_max = fit_center_min, fit_center_max
                else:
                    center_min = reference_center - shift_bound
                    center_max = reference_center + shift_bound
            else:
                center_min = center_max = float(peak.center_fitted)
            offset_current = peak.offset_fitted
            if offset_current is None and mode == "tied":
                offset_current = float(spec.get("offset_init", 0.0))
            shape = canonical_lineshape(spec.get("lineshape", peak.lineshape))
            if shape == "doniach_sunjic":
                eta_min = eta_max = 0.0
                asym_min, asym_max = float(spec.get("asym_min", 0.0)), float(spec.get("asym_max", 0.25))
            elif shape == "gaussian":
                eta_min = eta_max = 0.0; asym_min = asym_max = 0.0
            elif shape == "lorentzian":
                eta_min = eta_max = 1.0; asym_min = asym_max = 0.0
            elif shape == "asymmetric_pseudo_voigt":
                eta_min, eta_max = float(spec.get("eta_min", 0.0)), float(spec.get("eta_max", 1.0))
                asym_min, asym_max = float(spec.get("asym_min", 0.0)), float(spec.get("asym_max", 0.80))
            else:
                eta_min, eta_max = float(spec.get("eta_min", 0.0)), float(spec.get("eta_max", 1.0))
                asym_min = asym_max = 0.0
            shape_group = str(spec.get("shape_group", "")).strip()
            if shape_group:
                shape_link = f"named:{shape_group}"
            elif self.fit_context.get("shared_shape_enabled", False) and not bool(spec.get("independent_shape", False)):
                shape_link = f"shared:{shape}"
            else:
                shape_link = ""
            state.append({
                "label": peak.label,
                "state": peak.state,
                "center_input": float(peak.center_input),
                "center": float(peak.center_fitted),
                "center_mode": mode,
                "center_min": center_min,
                "center_max": center_max,
                "tie_target": str(spec.get("tie_target", peak.tie_target)),
                "offset_current": None if offset_current is None else float(offset_current),
                "offset_min": float(spec["offset_min"]) if "offset_min" in spec else None,
                "offset_max": float(spec["offset_max"]) if "offset_max" in spec else None,
                "fwhm": float(peak.fwhm_fitted),
                "fwhm_min": float(spec.get("fwhm_min", self.fwhm_min_spin.value())),
                "fwhm_max": float(spec.get("fwhm_max", self.fwhm_max_spin.value())),
                "height": float(peak.height_fitted),
                "height_max": max(height_ceiling, 5.0 * float(peak.height_fitted), 1.0),
                "eta": float(peak.eta_fitted),
                "eta_min": eta_min,
                "eta_max": eta_max,
                "asym": float(peak.asym_fitted),
                "asym_min": asym_min,
                "asym_max": asym_max,
                "shape": shape,
                "role": peak.role,
                "pair": peak.pair,
                "ratio": float(spec.get("ratio", 1.0)),
                "delta_e": float(spec.get("delta_e", peak.offset_fitted or 0.0)),
                "quantify": bool(peak.quantify),
                "source": peak.source,
                "shift_group": str(spec.get("shift_group", "")).strip(),
                "fwhm_group": str(spec.get("fwhm_group", "")).strip(),
                "shape_link": shape_link,
                "area": float(peak.area_fitted),
                "area_fraction": peak.area_fraction,
            })
        self.studio_state = state
        self._studio_apply_links()
        self.studio_original_state = copy.deepcopy(self.studio_state)
        self.studio_history = []
        self.studio_redo = []
        self.studio_dirty = False
        self.studio_selected_index = 0 if self.studio_state else -1
        self.studio_step_button.setEnabled(bool(self.studio_state))
        self._studio_update_history_buttons()
        self._studio_update_peak_buttons()
        self._refresh_studio_preview(preserve_view=False)
        if self.studio_state:
            self._studio_select_component(self.studio_selected_index, redraw=False)
        if hasattr(self, "studio_controls_scroll"):
            QTimer.singleShot(
                0, lambda: self.studio_controls_scroll.verticalScrollBar().setValue(0)
            )
        self._set_export_enabled(True)

    def _studio_item_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        return next((item for item in self.studio_state if item["label"] == label), None)

    def _studio_primary_for_partner(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if item.get("role") != "partner":
            return None
        return next(
            (candidate for candidate in self.studio_state
             if candidate.get("role") == "primary" and candidate.get("pair") == item.get("pair")),
            None
        )

    def _studio_apply_links(self):
        """Reapply center ties, named groups and doublet expressions after every edit."""
        label_map = {item["label"]: item for item in self.studio_state}
        resolved = set()

        def resolve_center(item: Dict[str, Any], visiting: set):
            label = item["label"]
            if label in resolved or item.get("role") == "partner":
                return
            if label in visiting:
                return
            visiting.add(label)
            if item.get("center_mode") == "tied":
                target = label_map.get(item.get("tie_target", ""))
                if target is not None:
                    resolve_center(target, visiting)
                    offset = float(item.get("offset_current") or 0.0)
                    if item.get("offset_min") is not None and item.get("offset_max") is not None:
                        offset = float(np.clip(offset, item["offset_min"], item["offset_max"]))
                    item["offset_current"] = offset
                    item["center"] = float(target["center"] + offset)
            visiting.remove(label); resolved.add(label)

        for component in self.studio_state:
            resolve_center(component, set())

        for partner in (item for item in self.studio_state if item.get("role") == "partner"):
            primary = self._studio_primary_for_partner(partner)
            if primary is None:
                continue
            partner["center"] = float(primary["center"] + partner.get("delta_e", 0.0))
            partner["offset_current"] = float(partner.get("delta_e", 0.0))
            partner["fwhm"] = float(primary["fwhm"])
            partner["height"] = float(primary["height"] * partner.get("ratio", 1.0))
            partner["eta"] = float(primary["eta"])
            partner["asym"] = float(primary["asym"])
            partner["shape"] = str(
                primary.get("shape", partner.get("shape", "pseudo_voigt"))
            )
            for field in ("eta_min", "eta_max", "asym_min", "asym_max"):
                if field in primary:
                    partner[field] = float(primary[field])

    def _studio_state_is_modified(self) -> bool:
        """Compare only editable model values; areas/fractions are derived preview data."""
        fields = (
            "label", "center", "fwhm", "height", "eta", "asym",
            "shape", "offset_current",
        )
        current = [tuple(item.get(field) for field in fields) for item in self.studio_state]
        original = [tuple(item.get(field) for field in fields) for item in self.studio_original_state]
        return current != original

    def _studio_center_limits(self, item: Dict[str, Any]) -> Tuple[float, float]:
        if item.get("role") == "partner" or item.get("center_mode") == "fixed":
            return float(item["center"]), float(item["center"])
        if item.get("center_mode") == "tied":
            target = self._studio_item_by_label(item.get("tie_target", ""))
            if target is None:
                return float(item["center"]), float(item["center"])
            offset = float(item.get("offset_current") or 0.0)
            omin = item.get("offset_min")
            omax = item.get("offset_max")
            if omin is None or omax is None:
                return float(target["center"] + offset), float(target["center"] + offset)
            return float(target["center"] + omin), float(target["center"] + omax)
        return float(item["center_min"]), float(item["center_max"])

    def _studio_parameter_limits(self, item: Dict[str, Any], field: str) -> Tuple[float, float]:
        if field == "center":
            return self._studio_center_limits(item)
        if field == "fwhm":
            return float(item["fwhm_min"]), float(item["fwhm_max"])
        if field == "height":
            return 0.0, float(item["height_max"])
        if field == "eta":
            return float(item["eta_min"]), float(item["eta_max"])
        if field == "asym":
            return float(item["asym_min"]), float(item["asym_max"])
        return 0.0, 1.0

    def _studio_constraint_text(self, item: Dict[str, Any]) -> str:
        if item.get("role") == "partner":
            primary = self._studio_primary_for_partner(item)
            return f"Doublet → {primary['label'] if primary else '?'}"
        mode = item.get("center_mode", "free")
        if mode == "fixed":
            return "Fixed"
        if mode == "bounded":
            return f"Bounded {item['center_min']:.2f}…{item['center_max']:.2f}"
        if mode == "tied":
            if item.get("offset_min") is None:
                return f"Tied → {item.get('tie_target', '?')} ({item.get('offset_current', 0.0):+.2f})"
            return (
                f"Tied → {item.get('tie_target', '?')} "
                f"[{item['offset_min']:+.2f},{item['offset_max']:+.2f}]"
            )
        return f"Free {item['center_min']:.2f}…{item['center_max']:.2f}"

    def _studio_table_selection_changed(self):
        rows = self.studio_peak_table.selectionModel().selectedRows()
        if rows:
            self._studio_select_component(rows[0].row())

    def _studio_select_component(self, index: int, redraw: bool = True):
        if not (0 <= int(index) < len(self.studio_state)):
            return
        self.studio_selected_index = int(index)
        self.studio_peak_table.blockSignals(True)
        self.studio_peak_table.selectRow(self.studio_selected_index)
        self.studio_peak_table.blockSignals(False)
        self.canvas.set_fit_selected_component(self.studio_selected_index)
        item = self.studio_state[self.studio_selected_index]
        editable_on_canvas = self.studio_manual_check.isChecked() and item.get("role") != "partner"
        self.canvas.set_fit_editing(editable_on_canvas)
        self._studio_update_parameter_controls()
        self._studio_update_peak_buttons()
        if redraw:
            self._refresh_studio_preview(preserve_view=True)

    def _studio_update_parameter_controls(self):
        self._studio_updating_controls = True
        try:
            valid = 0 <= self.studio_selected_index < len(self.studio_state)
            if not valid:
                for widget in list(self.studio_parameter_spins.values()) + list(self.studio_parameter_sliders.values()):
                    widget.setEnabled(False)
                self.studio_shape_combo.setEnabled(False)
                return
            item = self.studio_state[self.studio_selected_index]
            partner = item.get("role") == "partner"
            shape_index = self.studio_shape_combo.findData(item.get("shape", "pseudo_voigt"))
            self.studio_shape_combo.setCurrentIndex(max(shape_index, 0))
            self.studio_shape_combo.setEnabled(not partner)
            self.studio_shape_combo.setToolTip(
                "Edit the primary member of this doublet." if partner
                else "Choose the line shape for this peak."
            )
            for field, spin in self.studio_parameter_spins.items():
                lo, hi = self._studio_parameter_limits(item, field)
                if hi < lo:
                    lo, hi = hi, lo
                spin.setRange(lo, hi)
                spin.setValue(float(item[field]))
                slider = self.studio_parameter_sliders[field]
                if hi > lo:
                    slider.setValue(int(round(1000.0 * (float(item[field]) - lo) / (hi - lo))))
                else:
                    slider.setValue(0)
                enabled = not partner and hi > lo
                if field == "center" and item.get("center_mode") == "fixed":
                    enabled = False
                if field == "eta" and item.get("shape") not in {"pseudo_voigt", "asymmetric_pseudo_voigt"}:
                    enabled = False
                if field == "asym" and item.get("shape") not in {"asymmetric_pseudo_voigt", "doniach_sunjic"}:
                    enabled = False
                spin.setEnabled(enabled); slider.setEnabled(enabled)
                if partner:
                    spin.setToolTip("This value is derived from the doublet primary and cannot be edited independently.")
                elif not enabled:
                    spin.setToolTip("This parameter is fixed by the selected model/line shape.")
                else:
                    spin.setToolTip(f"Allowed interval: {lo:.5g} to {hi:.5g}")
        finally:
            self._studio_updating_controls = False

    def _studio_toggle_more_controls(self, visible: bool):
        """Keep the normal Studio view compact; expose shape/eta tools on demand."""
        visible = bool(visible)
        self.btn_studio_more.blockSignals(True)
        self.btn_studio_more.setChecked(visible)
        self.btn_studio_more.blockSignals(False)
        self.btn_studio_more.setText(
            "Fewer peak controls" if visible else "More peak controls"
        )
        for widget in self.studio_advanced_parameter_widgets:
            widget.setVisible(visible)
        self.studio_live_refit_check.setVisible(visible)

    @staticmethod
    def _studio_set_shape_values(item: Dict[str, Any], shape: str):
        shape = canonical_lineshape(shape)
        item["shape"] = shape
        if shape == "gaussian":
            item["eta"] = 0.0; item["eta_min"] = item["eta_max"] = 0.0
            item["asym"] = 0.0; item["asym_min"] = item["asym_max"] = 0.0
        elif shape == "lorentzian":
            item["eta"] = 1.0; item["eta_min"] = item["eta_max"] = 1.0
            item["asym"] = 0.0; item["asym_min"] = item["asym_max"] = 0.0
        elif shape == "asymmetric_pseudo_voigt":
            item["eta_min"], item["eta_max"] = 0.0, 1.0
            item["eta"] = float(np.clip(item.get("eta", 0.30), 0.0, 1.0))
            item["asym_min"], item["asym_max"] = 0.0, 0.80
            previous = float(item.get("asym", 0.0))
            item["asym"] = float(np.clip(previous if previous > 0 else 0.08, 0.0, 0.80))
        elif shape == "doniach_sunjic":
            item["eta"] = 0.0; item["eta_min"] = item["eta_max"] = 0.0
            item["asym_min"], item["asym_max"] = 0.0, 0.25
            previous = float(item.get("asym", 0.0))
            item["asym"] = float(np.clip(previous if previous > 0 else 0.06, 0.0, 0.25))
        else:
            item["eta_min"], item["eta_max"] = 0.0, 1.0
            item["eta"] = float(np.clip(item.get("eta", 0.30), 0.0, 1.0))
            item["asym"] = 0.0; item["asym_min"] = item["asym_max"] = 0.0

    def _studio_shape_changed(self, *_):
        if self._studio_updating_controls or not (
            0 <= self.studio_selected_index < len(self.studio_state)
        ):
            return
        item = self.studio_state[self.studio_selected_index]
        if item.get("role") == "partner":
            return
        requested = str(self.studio_shape_combo.currentData() or "pseudo_voigt")
        if requested == item.get("shape"):
            return
        self._studio_push_history()
        link = item.get("shape_link", "")
        peers = [
            peer for peer in self.studio_state
            if peer.get("role") != "partner" and (
                peer is item or (link and peer.get("shape_link") == link)
            )
        ]
        for peer in peers:
            self._studio_set_shape_values(peer, requested)
        self._studio_apply_links()
        self.studio_dirty = self._studio_state_is_modified()
        self._set_export_enabled(not self.studio_dirty)
        self._studio_update_parameter_controls()
        self._refresh_studio_preview(preserve_view=True)

    def _studio_free_editing_allowed(self) -> bool:
        return str(self.fit_context.get("ui_fit_mode", self._fit_mode())) == "free"

    def _studio_update_peak_buttons(self):
        if not hasattr(self, "btn_studio_add_peak"):
            return
        free = bool(
            self._studio_free_editing_allowed()
            and self.fit_result is not None
            and not self._fit_is_running()
        )
        valid = 0 <= self.studio_selected_index < len(self.studio_state)
        self.btn_studio_add_peak.setEnabled(free)
        self.btn_studio_duplicate_peak.setEnabled(free and valid)
        self.btn_studio_remove_peak.setEnabled(free and valid and len(self.studio_state) > 1)
        tip = (
            "Free Fit: add, copy or remove independent peaks."
            if free else "Switch to Free Fit on page 1 to change the number of peaks."
        )
        for button in (
            self.btn_studio_add_peak,
            self.btn_studio_duplicate_peak,
            self.btn_studio_remove_peak,
        ):
            button.setToolTip(tip)

    def _studio_unique_label(self, prefix: str = "Peak") -> str:
        labels = {str(item.get("label", "")) for item in self.studio_state}
        number = 1
        while f"{prefix} {number}" in labels:
            number += 1
        return f"{prefix} {number}"

    def _studio_make_free_item(
        self, center: float, template: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        template = copy.deepcopy(template or {})
        x = np.asarray(self.fit_context.get("x_fit", [center]), dtype=float)
        y = np.asarray(self.fit_context.get("y_corr", [1.0]), dtype=float)
        baseline = np.asarray(
            (self.fit_result or {}).get("baseline_correction", np.zeros_like(x)),
            dtype=float,
        )
        xmin, xmax = float(np.min(x)), float(np.max(x))
        center = float(np.clip(center, xmin, xmax))
        order = np.argsort(x)
        corrected = y - baseline
        local_height = float(np.interp(center, x[order], corrected[order]))
        y_span = max(float(np.ptp(corrected)), 1.0)
        height = max(float(template.get("height", local_height)), 0.02 * y_span, 1e-12)
        fmin = float(template.get("fwhm_min", self.fwhm_min_spin.value()))
        fmax = float(template.get("fwhm_max", self.fwhm_max_spin.value()))
        fwhm = float(np.clip(template.get("fwhm", 1.20), fmin, fmax))
        item = {
            "label": self._studio_unique_label(), "state": "",
            "center_input": center, "center": center, "center_mode": "free",
            "center_min": xmin, "center_max": xmax, "tie_target": "",
            "offset_current": None, "offset_min": None, "offset_max": None,
            "fwhm": fwhm, "fwhm_min": fmin, "fwhm_max": fmax,
            "height": height, "height_max": max(4.0 * y_span, 5.0 * height, 1.0),
            "eta": float(template.get("eta", 0.30)), "eta_min": 0.0, "eta_max": 1.0,
            "asym": float(template.get("asym", 0.0)), "asym_min": 0.0, "asym_max": 0.0,
            "shape": canonical_lineshape(template.get("shape", self._selected_lineshape())),
            "role": "singlet", "pair": "", "ratio": 1.0, "delta_e": 0.0,
            "quantify": True, "source": "", "shift_group": "",
            "fwhm_group": "", "shape_link": "", "area": 0.0,
            "area_fraction": None,
        }
        self._studio_set_shape_values(item, item["shape"])
        return item

    def _studio_add_free_peak(self):
        if not self._studio_free_editing_allowed() or self.fit_result is None:
            return
        x = np.asarray(self.fit_context["x_fit"], dtype=float)
        default = (
            float(self.studio_state[self.studio_selected_index]["center"])
            if 0 <= self.studio_selected_index < len(self.studio_state)
            else float(np.mean(x))
        )
        center, ok = QInputDialog.getDouble(
            self, "Add peak", "Peak center (eV):", default,
            float(np.min(x)), float(np.max(x)), 4,
        )
        if not ok:
            return
        self._studio_push_history()
        self.studio_state.append(self._studio_make_free_item(center))
        self.studio_selected_index = len(self.studio_state) - 1
        self.studio_dirty = True
        self._set_export_enabled(False)
        self._studio_update_history_buttons()
        self._studio_select_component(self.studio_selected_index, redraw=False)
        self._refresh_studio_preview(preserve_view=True)

    def _studio_duplicate_free_peak(self):
        if not self._studio_free_editing_allowed() or not (
            0 <= self.studio_selected_index < len(self.studio_state)
        ):
            return
        source = self.studio_state[self.studio_selected_index]
        x = np.asarray(self.fit_context["x_fit"], dtype=float)
        default = float(np.clip(source["center"] + 0.50, np.min(x), np.max(x)))
        center, ok = QInputDialog.getDouble(
            self, "Duplicate peak", "Center for the copy (eV):", default,
            float(np.min(x)), float(np.max(x)), 4,
        )
        if not ok:
            return
        self._studio_push_history()
        self.studio_state.append(self._studio_make_free_item(center, source))
        self.studio_selected_index = len(self.studio_state) - 1
        self.studio_dirty = True
        self._set_export_enabled(False)
        self._studio_update_history_buttons()
        self._studio_select_component(self.studio_selected_index, redraw=False)
        self._refresh_studio_preview(preserve_view=True)

    def _studio_remove_free_peak(self):
        if (
            not self._studio_free_editing_allowed() or len(self.studio_state) <= 1
            or not (0 <= self.studio_selected_index < len(self.studio_state))
        ):
            return
        self._studio_push_history()
        self.studio_state.pop(self.studio_selected_index)
        self.studio_selected_index = min(self.studio_selected_index, len(self.studio_state) - 1)
        self.studio_dirty = True
        self._set_export_enabled(False)
        self._studio_update_history_buttons()
        self._studio_select_component(self.studio_selected_index, redraw=False)
        self._refresh_studio_preview(preserve_view=True)

    def _studio_spin_changed(self, field: str, value: float):
        if self._studio_updating_controls or self.studio_selected_index < 0:
            return
        self._studio_apply_edit(self.studio_selected_index, {field: float(value)})

    def _studio_slider_changed(self, field: str, slider_value: int):
        if self._studio_updating_controls or self.studio_selected_index < 0:
            return
        item = self.studio_state[self.studio_selected_index]
        lo, hi = self._studio_parameter_limits(item, field)
        value = lo if hi <= lo else lo + (hi - lo) * float(slider_value) / 1000.0
        self._studio_apply_edit(
            self.studio_selected_index, {field: value},
            push_history=self._studio_slider_active_field != field
        )

    def _studio_slider_pressed(self, field: str):
        if self._studio_updating_controls or self.studio_selected_index < 0:
            return
        self._studio_push_history()
        self._studio_slider_active_field = field

    def _studio_slider_released(self):
        self._studio_slider_active_field = None

    def _studio_push_history(self):
        self.studio_history.append(copy.deepcopy(self.studio_state))
        if len(self.studio_history) > 50:
            self.studio_history.pop(0)
        self.studio_redo.clear()
        self._studio_update_history_buttons()

    def _studio_update_history_buttons(self):
        if hasattr(self, "btn_studio_undo"):
            self.btn_studio_undo.setEnabled(bool(self.studio_history))
            self.btn_studio_redo.setEnabled(bool(self.studio_redo))
            self.btn_studio_reset.setEnabled(bool(self.studio_state))
        self._studio_update_peak_buttons()

    def _studio_clamp_center(self, item: Dict[str, Any], requested: float) -> float:
        if item.get("center_mode") == "fixed" or item.get("role") == "partner":
            return float(item["center"])
        if item.get("center_mode") == "tied":
            target = self._studio_item_by_label(item.get("tie_target", ""))
            if target is None:
                return float(item["center"])
            offset = float(requested - target["center"])
            if item.get("offset_min") is not None and item.get("offset_max") is not None:
                offset = float(np.clip(offset, item["offset_min"], item["offset_max"]))
            else:
                offset = float(item.get("offset_current") or 0.0)
            item["offset_current"] = offset
            return float(target["center"] + offset)
        lo, hi = self._studio_center_limits(item)
        return float(np.clip(requested, lo, hi))

    def _studio_apply_edit(self, index: int, changes: Dict[str, float], push_history: bool = True):
        if not (0 <= index < len(self.studio_state)):
            return
        item = self.studio_state[index]
        if item.get("role") == "partner":
            self.status.showMessage("Edit the doublet primary; its partner updates automatically.", 5000)
            return
        if push_history:
            self._studio_push_history()
        changed_fields = set()
        for field, requested in changes.items():
            if field not in {"center", "fwhm", "height", "eta", "asym"}:
                continue
            old_value = float(item[field])
            if field == "center":
                group = item.get("shift_group", "")
                if group:
                    peers = [
                        peer for peer in self.studio_state
                        if peer.get("role") != "partner" and peer.get("shift_group") == group
                    ]
                    desired_delta = float(requested) - old_value
                    delta_min = max(
                        self._studio_center_limits(peer)[0] - float(peer["center"])
                        for peer in peers
                    )
                    delta_max = min(
                        self._studio_center_limits(peer)[1] - float(peer["center"])
                        for peer in peers
                    )
                    delta = float(np.clip(desired_delta, delta_min, delta_max))
                    for peer in peers:
                        peer["center"] = float(peer["center"] + delta)
                else:
                    item[field] = self._studio_clamp_center(item, float(requested))
            else:
                lo, hi = self._studio_parameter_limits(item, field)
                group_key = "fwhm_group" if field == "fwhm" else "shape_link"
                group = item.get(group_key, "") if field in {"fwhm", "eta", "asym"} else ""
                if group:
                    peers = [
                        peer for peer in self.studio_state
                        if peer.get("role") != "partner" and peer.get(group_key) == group
                    ]
                    lo = max(self._studio_parameter_limits(peer, field)[0] for peer in peers)
                    hi = min(self._studio_parameter_limits(peer, field)[1] for peer in peers)
                    item[field] = float(np.clip(requested, lo, hi))
                    for peer in peers:
                        peer[field] = item[field]
                else:
                    item[field] = float(np.clip(requested, lo, hi))
            if abs(float(item[field]) - old_value) > 1e-12:
                changed_fields.add(field)
        if not changed_fields:
            if push_history and self.studio_history:
                self.studio_history.pop()
                self._studio_update_history_buttons()
            self._studio_update_parameter_controls()
            return
        self._studio_apply_links()
        self.studio_dirty = self._studio_state_is_modified()
        self._set_export_enabled(not self.studio_dirty)
        self._studio_update_parameter_controls()
        self._refresh_studio_preview(preserve_view=True)
        if self.studio_live_refit_check.isChecked():
            self.studio_refit_timer.start()

    def _studio_canvas_edited(self, index: int, changes: object):
        if not self.studio_manual_check.isChecked() or not isinstance(changes, dict):
            return
        if 0 <= index < len(self.studio_state):
            item = self.studio_state[index]
            if item.get("center_mode") == "fixed":
                changes = {key: value for key, value in changes.items() if key != "center"}
            self._studio_apply_edit(index, changes)

    def _studio_manual_toggled(self, enabled: bool):
        if 0 <= self.studio_selected_index < len(self.studio_state):
            enabled = bool(enabled and self.studio_state[self.studio_selected_index].get("role") != "partner")
        self.canvas.set_fit_editing(enabled)
        if self.studio_state:
            self._refresh_studio_preview(preserve_view=True)

    def _studio_component_curve(self, item: Dict[str, Any], x: np.ndarray, height: Optional[float] = None) -> np.ndarray:
        h = float(item["height"] if height is None else height)
        shape = item["shape"]
        if shape == "doniach_sunjic":
            return doniach_sunjic(x, item["center"], item["fwhm"], h, item["asym"])
        if shape == "asymmetric_pseudo_voigt":
            return asymmetric_pseudo_voigt(x, item["center"], item["fwhm"], h, item["eta"], item["asym"])
        return pseudo_voigt(x, item["center"], item["fwhm"], h, item["eta"])

    def _studio_component_area(self, item: Dict[str, Any], x: np.ndarray) -> float:
        if item["shape"] == "doniach_sunjic":
            curve = self._studio_component_curve(item, x)
            integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
            order = np.argsort(x)
            return float(integrate(curve[order], x[order]))
        gaussian_factor = math.sqrt(math.pi) / (2.0 * math.sqrt(math.log(2.0)))
        lorentzian_factor = math.pi / 2.0
        return float(
            item["height"] * item["fwhm"]
            * ((1.0 - item["eta"]) * gaussian_factor + item["eta"] * lorentzian_factor)
        )

    def _studio_preview_result(self) -> Dict[str, Any]:
        x = np.asarray(self.fit_context["x_fit"], dtype=float)
        y = np.asarray(self.fit_context["y_corr"], dtype=float)
        baseline = np.asarray(
            self.fit_result.get("baseline_correction", np.zeros_like(x)), dtype=float
        )
        components = [self._studio_component_curve(item, x) for item in self.studio_state]
        y_fit = baseline + np.sum(components, axis=0) if components else baseline.copy()
        residual = y - y_fit
        areas = [max(self._studio_component_area(item, x), 0.0) for item in self.studio_state]
        quant_total = max(sum(area for area, item in zip(areas, self.studio_state) if item["quantify"]), 1e-30)
        state_areas: Dict[str, float] = {}
        for area, item in zip(areas, self.studio_state):
            item["area"] = area
            item["area_fraction"] = area / quant_total if item["quantify"] else None
            if item["state"] and item["quantify"]:
                state_areas[item["state"]] = state_areas.get(item["state"], 0.0) + area
        state_total = max(sum(state_areas.values()), 1e-30)
        state_fractions = {name: value/state_total for name, value in state_areas.items()}

        n = len(y); k = max(int(self.fit_result["metrics"].nvarys), 1)
        rss = float(np.sum(residual**2)); rmse = float(np.sqrt(rss/max(n, 1)))
        ss_tot = float(np.sum((y - np.mean(y))**2))
        r2 = float(1.0 - rss/ss_tot) if ss_tot > 0 else 0.0
        nrmse = float(100.0*rmse/max(float(np.ptp(y)), 1e-30))
        lag1 = 0.0
        if n > 3 and np.std(residual[:-1]) > 0 and np.std(residual[1:]) > 0:
            lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
        dw = float(np.sum(np.diff(residual)**2)/max(np.sum(residual**2), 1e-30))
        poisson = "poisson" in self.fit_result["metrics"].weighting.lower()
        if poisson:
            weights = 1.0/np.sqrt(np.maximum(np.asarray(self.fit_context["y_raw"], dtype=float), 1.0))
            objective = float(np.sum((residual*weights)**2))
            aic = float(objective + 2*k)
            bic = float(objective + k*np.log(max(n, 2)))
        else:
            objective = rss
            aic = float(n*np.log(max(rss/max(n, 1), 1e-300)) + 2*k)
            bic = float(n*np.log(max(rss/max(n, 1), 1e-300)) + k*np.log(max(n, 2)))
        aicc = float(aic + (2*k*(k+1))/(n-k-1)) if n > k+1 else float("inf")
        warnings_out = []
        bound_labels = []
        for item in self.studio_state:
            clo, chi = self._studio_center_limits(item)
            if chi > clo and min(abs(item["center"]-clo), abs(chi-item["center"])) < 0.008*(chi-clo):
                bound_labels.append(item["label"] + " center")
            if min(abs(item["fwhm"]-item["fwhm_min"]), abs(item["fwhm_max"]-item["fwhm"])) < 0.008*(item["fwhm_max"]-item["fwhm_min"]):
                bound_labels.append(item["label"] + " FWHM")
        if bound_labels:
            warnings_out.append("At/near bounds: " + ", ".join(bound_labels))
        effectively_zero_states = [
            state for state, fraction in state_fractions.items() if fraction < 1e-4
        ]
        if effectively_zero_states:
            warnings_out.append(
                "Effectively zero state fraction (<0.01%): "
                + ", ".join(effectively_zero_states)
                + ". The current peak parameters do not give this state measurable fitted area."
            )
        if abs(lag1) > 0.60:
            warnings_out.append(f"Structured residuals (lag-1 {lag1:+.2f}).")
        if nrmse > 5.0:
            warnings_out.append(f"NRMSE is {nrmse:.2f}%.")
        if r2 < 0.95:
            warnings_out.append(f"Low envelope agreement (R²={r2:.4f}).")
        severe = r2 < 0.90 or abs(lag1) > 0.85 or dw < 0.35
        quality = "POOR" if severe else ("REVIEW" if warnings_out else "GOOD")
        metrics = FitMetrics(
            rmse=rmse, nrmse_percent=nrmse, r2=r2, rss=rss,
            reduced_chi2=objective/max(n-k, 1), aic=aic, aicc=aicc, bic=bic,
            nvarys=k, max_abs_corr=self.fit_result["metrics"].max_abs_corr,
            residual_lag1_corr=lag1, durbin_watson=dw, objective_value=objective,
            weighting=self.fit_result["metrics"].weighting, fit_quality=quality,
            warning_count=len(warnings_out)
        )
        peak_results = []
        for item in self.studio_state:
            warning = ""
            peak_results.append(PeakResult(
                label=item["label"], state=item["state"], center_input=item["center_input"],
                center_fitted=item["center"], fwhm_fitted=item["fwhm"],
                height_fitted=item["height"], area_fitted=item["area"],
                eta_fitted=item["eta"], asym_fitted=item["asym"], role=item["role"],
                pair=item["pair"], lineshape=item["shape"], source=item["source"],
                quantify=item["quantify"], center_mode=item["center_mode"],
                center_min=item["center_min"] if item["center_mode"] == "bounded" else None,
                center_max=item["center_max"] if item["center_mode"] == "bounded" else None,
                tie_target=item["tie_target"], offset_fitted=item["offset_current"],
                offset_min=item["offset_min"], offset_max=item["offset_max"],
                area_fraction=item["area_fraction"], bound_warning=warning
            ))
        return {
            "components": components, "y_fit": y_fit, "residual": residual,
            "areas": areas, "state_fractions": state_fractions,
            "metrics": metrics, "warnings": warnings_out, "peak_results": peak_results,
        }

    def _refresh_studio_preview(self, preserve_view: bool = True):
        if not self.studio_state or self.fit_result is None:
            return
        preview = self._studio_preview_result()
        self.studio_preview = preview
        metrics = preview["metrics"]
        free_fit = self._studio_free_editing_allowed()
        title_prefix = (
            "Manual preview" if self.studio_dirty
            else ("Validated free fit" if free_fit else "Validated constrained fit")
        )
        self.canvas.plot_fit(
            self.fit_context["x_fit"], self.fit_context["y_raw"], self.fit_context["background"],
            self.fit_result.get("baseline_correction", np.zeros_like(preview["y_fit"])),
            preview["y_fit"], preview["components"], preview["residual"],
            preview["peak_results"], metrics=metrics,
            state_fractions=preview["state_fractions"],
            title=f"{title_prefix} — select or drag a component", preserve_view=preserve_view
        )
        self.studio_peak_table.blockSignals(True)
        self.studio_peak_table.setRowCount(len(self.studio_state))
        for row, item in enumerate(self.studio_state):
            fraction = format_fraction_percent(
                item["area_fraction"], decimals=2, include_symbol=False
            )
            values = [
                item["label"], f"{item['center']:.4f}", f"{item['fwhm']:.4f}",
                f"{item['height']:.2f}", fraction,
            ]
            for column, value in enumerate(values):
                table_item = QTableWidgetItem(value)
                table_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if column == 0:
                    table_item.setToolTip(
                        f"{self._studio_constraint_text(item)} • {item['shape'].replace('_', ' ')}"
                    )
                self.studio_peak_table.setItem(row, column, table_item)
        if 0 <= self.studio_selected_index < len(self.studio_state):
            self.studio_peak_table.selectRow(self.studio_selected_index)
        self.studio_peak_table.blockSignals(False)
        if free_fit:
            fractions = " | ".join(
                f"{item['label']}: {format_fraction_percent(item['area_fraction'])}"
                for item in self.studio_state if item.get("quantify", True)
            ) or "not available"
            self.studio_fraction_label.setText("Peak areas: " + fractions)
        else:
            fractions = " | ".join(
                f"{state}: {format_fraction_percent(fraction)}"
                for state, fraction in preview["state_fractions"].items()
            ) or "not available"
            self.studio_fraction_label.setText("Chemical-state fractions: " + fractions)
        self.studio_metrics_label.setText(
            f"RMSE {metrics.rmse:.4g}   |   NRMSE {metrics.nrmse_percent:.2f}%   |   "
            f"R² {metrics.r2:.5f}   |   DW {metrics.durbin_watson:.2f}   |   AICc {metrics.aicc:.2f}"
        )
        palette = {
            "GOOD": ("#ECFDF5", "#047857", "#6EE7B7"),
            "REVIEW": ("#FFFBEB", "#B45309", "#FCD34D"),
            "POOR": ("#FEF2F2", "#B91C1C", "#FCA5A5"),
        }
        bg, fg, border = palette.get(metrics.fit_quality, ("#F8FAFC", "#334155", "#CBD5E1"))
        prefix = "PROVISIONAL PREVIEW" if self.studio_dirty else "VALIDATED FIT"
        self.studio_quality_label.setText(
            f"{prefix} — {metrics.fit_quality}   •   {len(preview['warnings'])} QA note(s)"
        )
        self.studio_quality_label.setStyleSheet(
            f"background:{bg}; color:{fg}; border:1px solid {border}; border-radius:7px; padding:9px; font-weight:bold;"
        )
        if self.studio_dirty:
            final_name = "Final Free Fit" if free_fit else "Final Constrained Fit"
            warning_text = (
                f"Manual values are provisional. Use Export Preview to keep these exact values, "
                f"or run {final_name} to recalculate uncertainties and enable validated export."
            )
            if preview["warnings"]:
                warning_text += " | QA: " + " | ".join(preview["warnings"])
            self.studio_fit_status.setText("Manual preview")
            self.studio_fit_status.setStyleSheet("background:#FEF3C7; color:#92400E; border-radius:10px; padding:5px 10px; font-weight:bold;")
        else:
            warning_text = " | ".join(preview["warnings"])
            self.studio_fit_status.setText("Validated / exportable")
            self.studio_fit_status.setStyleSheet("background:#D1FAE5; color:#047857; border-radius:10px; padding:5px 10px; font-weight:bold;")
        self.studio_warning_label.setText(warning_text)
        self.studio_warning_label.setVisible(bool(warning_text))

    def _studio_refine_heights(self):
        if (
            self._fit_is_running() or not self.studio_dirty
            or not self.studio_live_refit_check.isChecked() or not self.studio_state
        ):
            return
        x = np.asarray(self.fit_context["x_fit"], dtype=float)
        baseline = np.asarray(self.fit_result.get("baseline_correction", np.zeros_like(x)), dtype=float)
        target = np.asarray(self.fit_context["y_corr"], dtype=float) - baseline
        independent_indices = [
            index for index, item in enumerate(self.studio_state) if item.get("role") != "partner"
        ]
        basis = []
        for index in independent_indices:
            item = self.studio_state[index]
            column = self._studio_component_curve(item, x, height=1.0)
            if item.get("role") == "primary":
                for partner in self.studio_state:
                    if partner.get("role") == "partner" and partner.get("pair") == item.get("pair"):
                        column = column + self._studio_component_curve(
                            partner, x, height=float(partner.get("ratio", 1.0))
                        )
            basis.append(column)
        if not basis:
            return
        try:
            heights, _ = nnls(np.column_stack(basis), target)
        except Exception as exc:
            self.status.showMessage(f"Live height refinement skipped: {exc}", 5000)
            return
        for index, height in zip(independent_indices, heights):
            self.studio_state[index]["height"] = float(np.clip(
                height, 0.0, self.studio_state[index]["height_max"]
            ))
        self._studio_apply_links()
        self.studio_dirty = self._studio_state_is_modified()
        self._set_export_enabled(not self.studio_dirty)
        self._studio_update_parameter_controls()
        self._refresh_studio_preview(preserve_view=True)

    def _studio_undo(self):
        if not self.studio_history:
            return
        self.studio_redo.append(copy.deepcopy(self.studio_state))
        self.studio_state = self.studio_history.pop()
        self._studio_apply_links()
        self.studio_dirty = self._studio_state_is_modified()
        self._set_export_enabled(not self.studio_dirty)
        self._studio_update_history_buttons(); self._studio_update_parameter_controls()
        self._studio_update_peak_buttons()
        self._refresh_studio_preview(preserve_view=True)

    def _studio_redo(self):
        if not self.studio_redo:
            return
        self.studio_history.append(copy.deepcopy(self.studio_state))
        self.studio_state = self.studio_redo.pop()
        self._studio_apply_links()
        self.studio_dirty = self._studio_state_is_modified()
        self._set_export_enabled(not self.studio_dirty)
        self._studio_update_history_buttons(); self._studio_update_parameter_controls()
        self._studio_update_peak_buttons()
        self._refresh_studio_preview(preserve_view=True)

    def _studio_reset(self):
        if not self.studio_original_state:
            return
        self._studio_push_history()
        self.studio_state = copy.deepcopy(self.studio_original_state)
        self._studio_apply_links(); self.studio_dirty = False
        self._set_export_enabled(True)
        self._studio_update_parameter_controls(); self._studio_update_history_buttons()
        self._studio_update_peak_buttons()
        self._refresh_studio_preview(preserve_view=False)

    def _studio_reset_view(self):
        if self.studio_state and self.fit_result is not None:
            self._refresh_studio_preview(preserve_view=False)

    def _studio_specs_from_state(self) -> List[Dict[str, Any]]:
        """Serialize the current Studio, including user-added/removed free peaks."""
        originals = {
            str(spec.get("label", "")): copy.deepcopy(spec)
            for spec in (self.fit_result or {}).get("model_specs", [])
        }
        specs: List[Dict[str, Any]] = []
        for item in self.studio_state:
            spec = originals.get(item["label"], {})
            spec.update({
                "enabled": True,
                "label": str(item["label"]),
                "state": str(item.get("state", "")),
                "center": float(item["center"]),
                "center_mode": str(item.get("center_mode", "free")),
                "role": str(item.get("role", "singlet")),
                "pair": str(item.get("pair", "")),
                "lineshape": canonical_lineshape(item.get("shape", "pseudo_voigt")),
                "fwhm_init": float(item["fwhm"]),
                "fwhm_min": float(item["fwhm_min"]),
                "fwhm_max": float(item["fwhm_max"]),
                "height_init": max(float(item["height"]), 1e-12),
                "quantify": bool(item.get("quantify", True)),
                "source": str(item.get("source", "")),
                "shift_group": str(item.get("shift_group", "")),
                "fwhm_group": str(item.get("fwhm_group", "")),
            })
            for stale in (
                "center_min", "center_max", "tie_target", "offset_init",
                "offset_min", "offset_max", "eta_init", "eta_min", "eta_max",
                "asym_init", "asym_min", "asym_max",
            ):
                spec.pop(stale, None)

            mode = spec["center_mode"]
            if mode == "bounded":
                spec["center_min"] = float(item["center_min"])
                spec["center_max"] = float(item["center_max"])
            elif mode == "tied":
                spec["tie_target"] = str(item.get("tie_target", ""))
                spec["offset_init"] = float(item.get("offset_current") or 0.0)
                if item.get("offset_min") is not None and item.get("offset_max") is not None:
                    spec["offset_min"] = float(item["offset_min"])
                    spec["offset_max"] = float(item["offset_max"])

            if spec["role"] == "partner":
                spec["delta_e"] = float(item.get("delta_e", 0.0))
                spec["ratio"] = float(item.get("ratio", 1.0))

            shape = spec["lineshape"]
            if shape == "gaussian":
                spec["eta_init"] = 0.0; spec["asym_init"] = 0.0
            elif shape == "lorentzian":
                spec["eta_init"] = 1.0; spec["asym_init"] = 0.0
            elif shape == "doniach_sunjic":
                spec["eta_init"] = 0.0
                spec["asym_init"] = float(item["asym"])
                spec["asym_min"] = float(item["asym_min"])
                spec["asym_max"] = float(item["asym_max"])
            elif shape == "asymmetric_pseudo_voigt":
                spec["eta_init"] = float(item["eta"])
                spec["eta_min"] = float(item["eta_min"])
                spec["eta_max"] = float(item["eta_max"])
                spec["asym_init"] = float(item["asym"])
                spec["asym_min"] = float(item["asym_min"])
                spec["asym_max"] = float(item["asym_max"])
            else:
                spec["eta_init"] = float(item["eta"])
                spec["eta_min"] = float(item["eta_min"])
                spec["eta_max"] = float(item["eta_max"])
                spec["asym_init"] = 0.0
            specs.append(spec)
        return specs

    def _studio_run_final_fit(self):
        if self._fit_is_running() or not self.studio_state or self.fit_result is None:
            return
        self.studio_refit_timer.stop()
        self._pre_refit_snapshot = (
            self.fit_result, copy.deepcopy(self.fit_context), bool(self.studio_dirty)
        )
        free_fit = self._studio_free_editing_allowed()
        specs = self._studio_specs_from_state()
        metadata = {
            "name": self.fit_context.get("model_name", "") or (
                "Interactive free-fit model" if free_fit else "Interactive refined XPS model"
            ),
            "reference": self.fit_context.get("model_reference", ""),
            "notes": self.fit_context.get("model_notes", ""),
        }
        self._set_model_specs(specs, metadata)
        self._use_studio_model_once = True
        self.studio_fit_status.setText("Optimizing…")
        self.status.showMessage(
            "Running final free optimization from the interactive values…"
            if free_fit else
            "Running full constrained optimization from the interactive starting values…"
        )
        self.run_fit()

    def _set_export_enabled(self, enabled: bool):
        running = self._fit_is_running()
        has_result = self.fit_result is not None
        has_workspace = bool(has_result and self.studio_state)
        allowed = bool(enabled and has_result and not running and not self.studio_dirty)
        if hasattr(self, "btn_export"):
            self.btn_export.setEnabled(allowed)
        if hasattr(self, "btn_studio_export"):
            self.btn_studio_export.setEnabled(allowed)
        if hasattr(self, "btn_studio_export_preview"):
            self.btn_studio_export_preview.setEnabled(
                bool(has_workspace and self.studio_dirty and not running)
            )
        if hasattr(self, "btn_studio_final_fit"):
            final_allowed = bool(has_workspace and not running)
            self.btn_studio_final_fit.setEnabled(final_allowed)
            if final_allowed:
                self.btn_studio_final_fit.setToolTip(
                    "Optimize from the current interactive values and calculate parameter uncertainties."
                )
            elif running:
                self.btn_studio_final_fit.setToolTip("A fit is currently running.")
            else:
                self.btn_studio_final_fit.setToolTip(
                    "Run the first fit on the Quick Fit page before final refinement."
                )

    def _invalidate_fit_after_model_edit(self, *_):
        # The exported audit always corresponds to the last completed fit; an
        # edited model must be refitted before export.
        if self.fit_result is not None:
            self.fit_result = None
            self.studio_step_button.setEnabled(False)
            self._set_export_enabled(False)
            self.status.showMessage("Model changed — run the fit again before export.", 5000)

    def _add_model_spec_row(self, spec: Dict[str, Any]):
        row = self.model_table.rowCount()
        self.model_table.insertRow(row)
        for col, (_, field) in enumerate(MODEL_TABLE_FIELDS):
            value = spec.get(field, "")
            item = QTableWidgetItem()
            if field in MODEL_BOOL_FIELDS:
                default = True if field in {"enabled", "quantify"} else False
                checked = _parse_bool(value if value != "" else default, field)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                item.setText("")
            elif field == "center_mode":
                role = str(spec.get("role", "singlet")).strip().lower()
                mode = canonical_center_mode(
                    value, role,
                    str(spec.get("center_min", "")).strip() != ""
                    and str(spec.get("center_max", "")).strip() != ""
                )
                item.setText(mode)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            else:
                if isinstance(value, float):
                    item.setText(f"{value:.10g}")
                else:
                    item.setText(str(value))
            if col == 0:
                item.setData(Qt.ItemDataRole.UserRole, copy.deepcopy(spec))
            self.model_table.setItem(row, col, item)
            if field == "center_mode":
                combo = QComboBox()
                for label, data in [
                    ("Free (global)", "free"), ("Fixed", "fixed"),
                    ("Bounded", "bounded"), ("Tied", "tied")
                ]:
                    combo.addItem(label, data)
                combo.setCurrentIndex(max(combo.findData(mode), 0))
                combo.setToolTip(
                    "Free: global ± interval | Fixed: locked | Bounded: explicit min/max | Tied: target + offset"
                )
                combo.currentIndexChanged.connect(lambda _index, widget=combo: self._center_mode_changed(widget))
                self._style_center_mode_combo(combo)
                self.model_table.setCellWidget(row, col, combo)
        self._update_constraint_summary()

    def _style_center_mode_combo(self, combo: QComboBox):
        colors = {
            "free": ("#EFF6FF", "#1D4ED8", "#93C5FD"),
            "fixed": ("#F1F5F9", "#334155", "#CBD5E1"),
            "bounded": ("#FFFBEB", "#B45309", "#FCD34D"),
            "tied": ("#F5F3FF", "#6D28D9", "#C4B5FD"),
        }
        bg, fg, border = colors.get(str(combo.currentData()), colors["free"])
        combo.setStyleSheet(
            f"QComboBox {{ background:{bg}; color:{fg}; border:1px solid {border}; "
            "border-radius:5px; padding:3px 8px; font-weight:600; }"
        )

    def _row_for_cell_widget(self, widget: QWidget) -> int:
        mode_col = MODEL_FIELD_INDEX["center_mode"]
        for row in range(self.model_table.rowCount()):
            if self.model_table.cellWidget(row, mode_col) is widget:
                return row
        return -1

    def _table_text(self, row: int, field: str) -> str:
        item = self.model_table.item(row, MODEL_FIELD_INDEX[field])
        return item.text().strip() if item is not None else ""

    def _set_table_text(self, row: int, field: str, value: Any):
        item = self.model_table.item(row, MODEL_FIELD_INDEX[field])
        if item is None:
            item = QTableWidgetItem()
            self.model_table.setItem(row, MODEL_FIELD_INDEX[field], item)
        if value is None or value == "":
            item.setText("")
        elif isinstance(value, (float, np.floating)):
            item.setText(f"{float(value):.10g}")
        else:
            item.setText(str(value))

    def _center_mode_changed(self, combo: QComboBox):
        self._style_center_mode_combo(combo)
        row = self._row_for_cell_widget(combo)
        if row < 0:
            return
        mode = str(combo.currentData())
        role = self._table_text(row, "role").lower()
        if role == "partner" and mode != "tied":
            combo.blockSignals(True)
            combo.setCurrentIndex(combo.findData("tied"))
            combo.blockSignals(False)
            self._style_center_mode_combo(combo)
            self.status.showMessage("A doublet partner is always tied to its primary through ΔE.", 5000)
            return
        if mode == "bounded":
            center = safe_float(self._table_text(row, "center"))
            if center is not None and not self._table_text(row, "center_min") and not self._table_text(row, "center_max"):
                self._set_table_text(row, "center_min", center - 0.20)
                self._set_table_text(row, "center_max", center + 0.20)
        self._invalidate_fit_after_model_edit()
        self._update_constraint_summary()

    def _update_constraint_summary(self, *_):
        if not hasattr(self, "constraint_summary_label"):
            return
        counts = {mode: 0 for mode in VALID_CENTER_MODES}
        mode_col = MODEL_FIELD_INDEX["center_mode"]
        for row in range(self.model_table.rowCount()):
            enabled_item = self.model_table.item(row, MODEL_FIELD_INDEX["enabled"])
            if enabled_item is not None and enabled_item.checkState() != Qt.CheckState.Checked:
                continue
            combo = self.model_table.cellWidget(row, mode_col)
            mode = str(combo.currentData()) if isinstance(combo, QComboBox) else self._table_text(row, "center_mode")
            if mode in counts:
                counts[mode] += 1
        total = sum(counts.values())
        if total == 0:
            self.constraint_summary_label.setText("Center constraints: no active components")
            return
        self.constraint_summary_label.setText(
            "Center constraints — "
            f"Free: {counts['free']}   |   Fixed: {counts['fixed']}   |   "
            f"Bounded: {counts['bounded']}   |   Tied: {counts['tied']}"
        )

    def _set_model_specs(self, specs: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None):
        self.model_table.setRowCount(0)
        for spec in specs:
            self._add_model_spec_row(spec)
        metadata = metadata or {}
        if "name" in metadata:
            self.model_name_edit.setText(str(metadata.get("name", "")))
        if "reference" in metadata:
            self.model_reference_edit.setText(str(metadata.get("reference", "")))
        if "notes" in metadata:
            self.model_notes_edit.setText(str(metadata.get("notes", "")))
        active_centers = [float(s["center"]) for s in specs if _parse_bool(s.get("enabled", True), "enabled")]
        if active_centers:
            self.centers_edit.setText(format_centers(active_centers))
            self.refresh_loaded_plot()
        self.use_model_check.setChecked(True)
        self._update_constraint_summary()

    def _model_raw_components_from_table(self) -> List[Dict[str, Any]]:
        numeric_fields = {
            "center", "center_min", "center_max", "offset_init", "offset_min", "offset_max",
            "delta_e", "ratio",
            "fwhm_init", "fwhm_min", "fwhm_max"
        }
        components: List[Dict[str, Any]] = []
        for row in range(self.model_table.rowCount()):
            first = self.model_table.item(row, 0)
            stored = first.data(Qt.ItemDataRole.UserRole) if first is not None else None
            spec = copy.deepcopy(stored) if isinstance(stored, dict) else {}
            for col, (_, field) in enumerate(MODEL_TABLE_FIELDS):
                item = self.model_table.item(row, col)
                if field in MODEL_BOOL_FIELDS:
                    spec[field] = bool(item and item.checkState() == Qt.CheckState.Checked)
                    continue
                if field == "center_mode":
                    combo = self.model_table.cellWidget(row, col)
                    spec[field] = str(combo.currentData()) if isinstance(combo, QComboBox) else (
                        item.text().strip() if item is not None else "free"
                    )
                    continue
                text_value = item.text().strip() if item is not None else ""
                if text_value == "":
                    spec.pop(field, None)
                elif field in numeric_fields:
                    try:
                        spec[field] = float(text_value)
                    except ValueError:
                        raise ValueError(f"Model row {row+1}, field '{field}' must be numeric.")
                else:
                    spec[field] = text_value
            components.append(spec)
        return components

    def _model_document_from_ui(self) -> Dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "name": self.model_name_edit.text().strip() or "Untitled XPS model",
            "target": self.element_edit.text().strip(),
            "orbital": self.orbital_edit.text().strip(),
            "reference": self.model_reference_edit.text().strip(),
            "notes": self.model_notes_edit.text().strip(),
            "default_lineshape": self._selected_lineshape(),
            "default_fwhm_min": self.fwhm_min_spin.value(),
            "default_fwhm_max": self.fwhm_max_spin.value(),
            "components": self._model_raw_components_from_table(),
        }

    def _validated_model_from_ui(self) -> Tuple[Dict[str, Any], List[str]]:
        return validate_model_document(self._model_document_from_ui())

    def build_model_from_centers(self):
        try:
            centers = parse_centers(self.centers_edit.text())
            if not centers:
                raise ValueError("Pick or enter at least one center first.")
            shape = self._selected_lineshape()
            specs = [
                {
                    "enabled": True, "label": f"Peak {i}", "state": "",
                    "center": center, "center_mode": "free", "role": "singlet", "lineshape": shape,
                    "fwhm_init": 1.20, "fwhm_min": self.fwhm_min_spin.value(),
                    "fwhm_max": self.fwhm_max_spin.value(), "quantify": True,
                    "source": "",
                }
                for i, center in enumerate(centers, start=1)
            ]
            self._set_model_specs(specs, {
                "name": f"{self.element_edit.text().strip()} {self.orbital_edit.text().strip()} custom model".strip(),
                "reference": "", "notes": "",
            })
            self.analysis_tabs.setCurrentWidget(self.model_tab)
            self.log("Created an editable element-agnostic model from the picked centers.")
        except Exception as exc:
            self.warn(str(exc))

    def add_model_component(self):
        center, ok = QInputDialog.getDouble(self, "Add XPS component", "Reference center (eV):", 100.0, -1000.0, 5000.0, 4)
        if not ok:
            return
        label, ok = QInputDialog.getText(self, "Add XPS component", "Component label:", text=f"Peak {self.model_table.rowCount()+1}")
        if not ok:
            return
        state, ok = QInputDialog.getText(self, "Add XPS component", "Oxidation/chemical state (may be blank):")
        if not ok:
            return
        role, ok = QInputDialog.getItem(
            self, "Add XPS component", "Role:",
            ["singlet", "satellite", "multiplet"], 0, False
        )
        if not ok:
            return
        self._add_model_spec_row({
            "enabled": True, "label": label.strip() or f"Peak {self.model_table.rowCount()+1}",
            "state": state.strip(), "center": center, "center_mode": "free", "role": role,
            "lineshape": self._selected_lineshape(), "fwhm_init": 1.20,
            "fwhm_min": self.fwhm_min_spin.value(), "fwhm_max": self.fwhm_max_spin.value(),
            "quantify": role != "satellite", "source": "",
        })
        self.use_model_check.setChecked(True)

    def add_model_doublet(self):
        primary_center, ok = QInputDialog.getDouble(
            self, "Add spin-orbit doublet", "Lower-binding-energy primary center (eV):",
            100.0, -1000.0, 5000.0, 4
        )
        if not ok:
            return
        splitting, ok = QInputDialog.getDouble(
            self, "Add spin-orbit doublet", "Spin-orbit splitting ΔE (eV; use a validated reference):",
            1.0, 0.0001, 100.0, 4
        )
        if not ok:
            return
        statistical_ratio = default_doublet_ratio(self.orbital_edit.text()) or 0.5
        ratio, ok = QInputDialog.getDouble(
            self, "Add spin-orbit doublet", "Partner/primary AREA ratio:",
            statistical_ratio, 0.0001, 10.0, 4
        )
        if not ok:
            return
        state, ok = QInputDialog.getText(self, "Add spin-orbit doublet", "Oxidation/chemical state:")
        if not ok:
            return
        label, ok = QInputDialog.getText(
            self, "Add spin-orbit doublet", "Label prefix:",
            text=state.strip() or f"State {self.model_table.rowCount()+1}"
        )
        if not ok:
            return
        pair = f"so_{self.model_table.rowCount()+1}_{int(round(primary_center*1000))}"
        common = {
            "enabled": True, "state": state.strip(), "pair": pair,
            "lineshape": self._selected_lineshape(), "fwhm_init": 1.20,
            "fwhm_min": self.fwhm_min_spin.value(), "fwhm_max": self.fwhm_max_spin.value(),
            "quantify": True, "source": "",
        }
        primary_label = f"{label.strip()} primary"
        self._add_model_spec_row({**common, "label": primary_label, "center": primary_center, "center_mode": "free", "role": "primary"})
        self._add_model_spec_row({
            **common, "label": f"{label.strip()} partner", "center": primary_center + splitting,
            "center_mode": "tied", "tie_target": primary_label, "offset_init": splitting,
            "role": "partner", "delta_e": splitting, "ratio": ratio,
        })
        self.use_model_check.setChecked(True)

    def configure_center_constraint(self):
        row = self.model_table.currentRow()
        if row < 0:
            self.warn("Select one component row first.")
            return
        if self._table_text(row, "role").lower() == "partner":
            self.info(
                "This row is a spin-orbit partner. Its center is already tied to the primary by ΔE. "
                "Edit the primary center constraint or the ΔE value instead."
            )
            return
        choices = ["Free (global safety interval)", "Fixed", "Bounded", "Tied to another component"]
        current_combo = self.model_table.cellWidget(row, MODEL_FIELD_INDEX["center_mode"])
        current_mode = str(current_combo.currentData()) if isinstance(current_combo, QComboBox) else "free"
        current_index = {"free": 0, "fixed": 1, "bounded": 2, "tied": 3}.get(current_mode, 0)
        selected, ok = QInputDialog.getItem(
            self, "Peak-center constraint", "Constraint mode:", choices, current_index, False
        )
        if not ok:
            return
        mode = {
            choices[0]: "free", choices[1]: "fixed",
            choices[2]: "bounded", choices[3]: "tied"
        }[selected]
        center = safe_float(self._table_text(row, "center"))
        if center is None:
            self.warn("The selected row needs a numeric reference center first.")
            return

        if mode == "bounded":
            current_min = safe_float(self._table_text(row, "center_min"))
            current_max = safe_float(self._table_text(row, "center_max"))
            lower, ok = QInputDialog.getDouble(
                self, "Bounded center", "Minimum binding energy (eV):",
                current_min if current_min is not None else center - 0.20,
                -1000.0, 5000.0, 4
            )
            if not ok:
                return
            upper, ok = QInputDialog.getDouble(
                self, "Bounded center", "Maximum binding energy (eV):",
                current_max if current_max is not None else center + 0.20,
                -1000.0, 5000.0, 4
            )
            if not ok:
                return
            if upper <= lower or not lower <= center <= upper:
                self.warn("Use minimum < reference center < maximum.")
                return
            self._set_table_text(row, "center_min", lower)
            self._set_table_text(row, "center_max", upper)
            for field in ("tie_target", "offset_init", "offset_min", "offset_max"):
                self._set_table_text(row, field, "")
        elif mode == "tied":
            targets = []
            target_centers = {}
            for other in range(self.model_table.rowCount()):
                if other == row or self._table_text(other, "role").lower() == "partner":
                    continue
                label = self._table_text(other, "label")
                other_center = safe_float(self._table_text(other, "center"))
                if label and other_center is not None:
                    targets.append(label)
                    target_centers[label] = other_center
            if not targets:
                self.warn("Add another independent component before using Tied mode.")
                return
            target, ok = QInputDialog.getItem(
                self, "Tied center", "Reference component:", targets, 0, False
            )
            if not ok:
                return
            default_offset = center - target_centers[target]
            offset, ok = QInputDialog.getDouble(
                self, "Tied center", "Initial offset (this center − target center), eV:",
                default_offset, -100.0, 100.0, 4
            )
            if not ok:
                return
            half_width, ok = QInputDialog.getDouble(
                self, "Tied center", "Allowed offset half-width (0 = exact fixed offset), eV:",
                0.20, 0.0, 20.0, 4
            )
            if not ok:
                return
            self._set_table_text(row, "tie_target", target)
            self._set_table_text(row, "offset_init", offset)
            self._set_table_text(row, "center", target_centers[target] + offset)
            if half_width > 0:
                self._set_table_text(row, "offset_min", offset - half_width)
                self._set_table_text(row, "offset_max", offset + half_width)
            else:
                self._set_table_text(row, "offset_min", "")
                self._set_table_text(row, "offset_max", "")
            self._set_table_text(row, "center_min", "")
            self._set_table_text(row, "center_max", "")
        else:
            for field in (
                "center_min", "center_max", "tie_target",
                "offset_init", "offset_min", "offset_max"
            ):
                self._set_table_text(row, field, "")

        combo = self.model_table.cellWidget(row, MODEL_FIELD_INDEX["center_mode"])
        if isinstance(combo, QComboBox):
            combo.setCurrentIndex(combo.findData(mode))
        self._invalidate_fit_after_model_edit()
        self._update_constraint_summary()
        self.status.showMessage(f"Updated center constraint for '{self._table_text(row, 'label')}'.", 5000)

    def validate_model_ui(self):
        try:
            clean, warnings_out = self._validated_model_from_ui()
            tied_count = sum(
                spec.get("center_mode") == "tied" for spec in clean["components"]
            )
            message = (
                f"Model is valid.\n\nActive components: {len(clean['components'])}\n"
                f"Tied centers: {tied_count}\nSchema: {clean['schema']}"
            )
            if warnings_out:
                message += "\n\nReview notes:\n• " + "\n• ".join(warnings_out)
            self.info(message)
        except Exception as exc:
            self.warn("Model validation failed:\n\n" + str(exc))

    def delete_model_rows(self):
        rows = sorted({index.row() for index in self.model_table.selectedIndexes()}, reverse=True)
        if not rows and self.model_table.currentRow() >= 0:
            rows = [self.model_table.currentRow()]
        for row in rows:
            self.model_table.removeRow(row)
        self._update_constraint_summary()

    def import_model_json(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import XPS chemical model", "", "JSON model (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                document = json.load(handle)
            clean, warnings_out = validate_model_document(document)
            self._set_model_specs(clean["components"], clean)
            if clean.get("target"):
                self.element_edit.setText(clean["target"])
            if clean.get("orbital"):
                self.orbital_edit.setText(clean["orbital"])
            self.analysis_tabs.setCurrentWidget(self.model_tab)
            self.log(f"Imported validated model: {Path(path).name}")
            for warning in warnings_out:
                self.log("MODEL WARNING: " + warning)
        except Exception as exc:
            self.warn(str(exc))

    def export_model_json(self):
        try:
            clean, warnings_out = self._validated_model_from_ui()
            path, _ = QFileDialog.getSaveFileName(
                self, "Export XPS chemical model", "xps_model.json", "JSON model (*.json)"
            )
            if not path:
                return
            if not path.lower().endswith(".json"):
                path += ".json"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(clean, handle, ensure_ascii=False, indent=2)
            self.log(f"Exported validated chemical model: {path}")
            for warning in warnings_out:
                self.log("MODEL WARNING: " + warning)
        except Exception as exc:
            self.warn(str(exc))

    def _invalidate_database(self, *_):
        self.db_fetched_info = None
        if hasattr(self, 'db_status_label'):
            self.db_status_label.setText("Target/orbital changed — fetch the model again.")
            self.db_status_label.setStyleSheet("color: #F59E0B; font-weight: bold; padding: 5px;")

    def _set_fit_running(self, running: bool):
        self._fit_active = bool(running)
        self.progress_bar.setVisible(running)
        self.btn_fit.setEnabled(not running)
        self.btn_preview.setEnabled(not running)
        self.fit_mode_combo.setEnabled(not running)
        if running:
            self.studio_step_button.setEnabled(False)
        if hasattr(self, "btn_studio_final_fit"):
            self.btn_studio_final_fit.setEnabled(not running and bool(self.studio_state))
        self._studio_update_peak_buttons()
        self._set_export_enabled(not running and not self.studio_dirty)

    def _fit_is_running(self) -> bool:
        return bool(getattr(self, "_fit_active", False))

    def _on_fit_error(self, message: str):
        self._use_studio_model_once = False
        self._set_fit_running(False)
        if self._pre_refit_snapshot is not None:
            previous_result, previous_context, previous_dirty = self._pre_refit_snapshot
            self.fit_result = previous_result
            self.fit_context = previous_context
            self._pre_refit_snapshot = None
            self.studio_step_button.setEnabled(bool(self.studio_state))
            self.studio_dirty = bool(previous_dirty)
            self._set_export_enabled(not self.studio_dirty)
            self.studio_fit_status.setText("Fit failed — previous workspace restored")
            self._show_workflow_page(
                1 if int(getattr(self, "_fit_origin_page", 0)) == 1 else 0
            )
        self.warn(message)

    def apply_charge_shift(self):
        if self._fit_is_running():
            self.warn("Wait for the current fit to finish before changing the energy axis.")
            return
        if self.energy is None:
            self.warn("Please load a data file first.")
            return
            
        measured = self.measured_c1s.value()
        target = self.target_c1s.value()
        shift = target - measured
        
        if self.energy_original is None:
            self.energy_original = self.energy.copy()
        # Always reference the untouched loaded axis so repeated clicks cannot
        # accumulate the same charge correction.
        self.energy = self.energy_original + shift
        self.applied_charge_shift = float(shift)
        self.fit_result = None
        self.fit_context = {}
        self.studio_state = []
        self.manual_baseline_points = []
        self._manual_baseline_region = None
        self._set_manual_editor_checked(False)
        self.studio_step_button.setEnabled(False)
        self._set_export_enabled(False)
        
        self.data_table.setRowCount(len(self.energy))
        for r, (e, i) in enumerate(zip(self.energy, self.intensity)):
            self.data_table.setItem(r, 0, QTableWidgetItem(f"{e:.4f}"))
            self.data_table.setItem(r, 1, QTableWidgetItem(f"{i:.4f}"))
            
        if self.use_model_check.isChecked() and self.model_table.rowCount() > 0:
            try:
                model_doc, _ = self._validated_model_from_ui()
                self.centers_edit.setText(format_centers([s['center'] for s in model_doc['components']]))
            except Exception:
                self.centers_edit.clear()
        else:
            self.centers_edit.clear()
        self.refresh_loaded_plot()
        
        if abs(shift) < 1e-12:
            self.log("Energy axis restored to the originally loaded values (no charge shift).")
        else:
            self.log(f"Charge Correction Applied: Data shifted by {shift:+.2f} eV")
        reference_name = self.calibration_reference_edit.text().strip() or "Unspecified reference"
        self.log(
            f"Energy reference: {reference_name}; measured {measured:.3f} eV -> target {target:.3f} eV. "
            "Verify this reference against the sample and report the method."
        )
        self.info(
            f"Energy axis shifted by {shift:+.2f} eV.\n\n"
            "Verify that the chosen energy reference is justified for this sample, then select the peak centers again."
        )

    def fetch_database(self):
        el = self.element_edit.text()
        orb = self.orbital_edit.text()
        if not el or not orb:
            self.warn("Please enter both Target and Orbital.")
            return
            
        db_info, msg = get_xps_parameters(el, orb)
        if db_info is not None and db_info.get('reference_only'):
            self.db_fetched_info = None
            self.db_status_label.setText(msg)
            self.db_status_label.setStyleSheet("color: #D97706; font-weight: bold; padding: 5px;")
            self.log("REFERENCE-ONLY WARNING: " + msg)
            self.warn(msg + "\n\nUse a validated XPS chemical-state reference (for example NIST XPS or a suitable standard/literature spectrum).")
            return

        if db_info is not None:
            self.db_fetched_info = db_info
            self.db_status_label.setText(msg)
            self.db_status_label.setStyleSheet("color: #10B981; font-weight: bold; padding: 5px;")
            
            if db_info.get('template_specs'):
                all_centers = [f"{float(spec['center']):.2f}" for spec in db_info['template_specs']]
                new_center_str = ", ".join(all_centers)
                self.centers_edit.setText(new_center_str)
                self.log(f"Auto-filled template centers: {new_center_str} eV")
                self.refresh_loaded_plot()
            elif 'primary_centers' in db_info:
                all_centers = []
                for c1 in db_info['primary_centers']:
                    all_centers.append(f"{c1:.2f}")
                    if db_info.get('doublet'):
                        c2 = c1 + db_info['delta_e']
                        all_centers.append(f"{c2:.2f}")
                        
                new_center_str = ", ".join(all_centers)
                self.centers_edit.setText(new_center_str)
                self.log(f"Auto-filled peak center(s): {new_center_str} eV")
                self.refresh_loaded_plot()
            self.log(msg)
            try:
                current_centers = parse_centers(self.centers_edit.text())
                model_specs, model_warnings, _ = _build_peak_specs(
                    current_centers, db_info, self.auto_doublet_check.isChecked()
                )
                self._set_model_specs(model_specs, {
                    "name": f"{el.strip()} {orb.strip()} starting model",
                    "reference": "",
                    "notes": "Built-in centers are starting values only. Add the exact sample-appropriate reference before publication.",
                })
                for warning in model_warnings:
                    self.log("MODEL WARNING: " + warning)
                self.log("The starting model was copied to Model Builder for inspection and editing.")
            except Exception as exc:
                self.log("MODEL BUILDER WARNING: " + str(exc))
        else:
            self.db_fetched_info = None
            self.db_status_label.setText(msg)
            self.db_status_label.setStyleSheet("color: #EF4444; font-weight: bold; padding: 5px;")
            self.warn(msg)

    def open_file(self):
        if self._fit_is_running():
            self.warn("Wait for the current fit to finish before loading another spectrum.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open XPS", "", "Data (*.csv *.txt *.dat *.xlsx *.xls)")
        if path:
            try:
                self.energy, self.intensity = load_xps_file(path)
                self.energy_original = self.energy.copy()
                self.applied_charge_shift = 0.0
                self.fit_result = None
                self.fit_context = {}
                self.studio_state = []
                self.studio_original_state = []
                self.studio_history = []
                self.studio_redo = []
                self.manual_baseline_points = []
                self._manual_baseline_region = None
                self._set_manual_editor_checked(False)
                self.studio_step_button.setEnabled(False)
                self._set_export_enabled(False)
                self._show_workflow_page(0)
                self.file_path = path; self.file_edit.setText(path)
                
                self.data_table.setRowCount(len(self.energy))
                for r, (e, i) in enumerate(zip(self.energy, self.intensity)):
                    self.data_table.setItem(r, 0, QTableWidgetItem(f"{e:.4f}"))
                    self.data_table.setItem(r, 1, QTableWidgetItem(f"{i:.4f}"))
                    
                self.refresh_loaded_plot(); self.log(f"Loaded {Path(path).name}")
            except Exception as e: self.warn(str(e))

    def _add_peak_from_plot(self, x):
        idx = int(np.argmin(np.abs(self.energy - x)))
        exact_x = float(self.energy[idx])
        
        current_text = self.centers_edit.text().strip()
        if current_text:
            try: c = parse_centers(current_text) + [exact_x]
            except Exception: c = [exact_x]
        else:
            c = [exact_x]
            
        self.centers_edit.setText(format_centers(c))
        self.refresh_loaded_plot()

    def _on_centers_dragged(self, new_centers):
        if self.energy is not None:
            snapped_centers = []
            for c in new_centers:
                idx = int(np.argmin(np.abs(self.energy - c)))
                snapped_centers.append(float(self.energy[idx]))
            self.centers_edit.setText(format_centers(snapped_centers))
            self.refresh_loaded_plot() 
        else:
            self.centers_edit.setText(format_centers(new_centers))

    def refresh_loaded_plot(self):
        if self.energy is not None:
            if (
                self._manual_background_selected()
                and self.btn_edit_manual_baseline.isChecked()
            ):
                try:
                    self._render_manual_baseline_preview(editing=True)
                    return
                except Exception:
                    self._set_manual_editor_checked(False)
            try: centers = parse_centers(self.centers_edit.text())
            except Exception: centers = []
            self.setup_canvas.plot_loaded_data(self.energy, self.intensity, centers)

    def preview_background(self):
        try:
            if self.energy is None:
                raise ValueError("Load an XPS data file first.")
            if self._manual_background_selected():
                x, y, fit_region = self._manual_fit_region()
                try:
                    normalize_manual_background_points(
                        self.manual_baseline_points, x, require_edge_coverage=True
                    )
                except ValueError:
                    self.manual_baseline_points = seed_manual_background_points(x, y)
                self._manual_baseline_region = tuple(float(v) for v in fit_region)
                self._set_manual_editor_checked(True)
                self._render_manual_baseline_preview(editing=True)
                return
            free_mode = self._fit_mode() == "free"
            if self.use_model_check.isChecked() and not free_mode:
                model_doc, model_warnings = self._validated_model_from_ui()
                region_centers = [float(spec['center']) for spec in model_doc['components']]
                for warning in model_warnings:
                    self.log("MODEL WARNING: " + warning)
            else:
                c = parse_centers(self.centers_edit.text())
                region_centers = _expanded_centers(
                    c, None if free_mode else self.db_fetched_info,
                    False if free_mode else self.auto_doublet_check.isChecked(),
                )
            x, y, _ = extract_fit_region(self.energy, self.intensity, region_centers, self.margin_spin.value())
            self._set_manual_editor_checked(False)
            self.setup_canvas.plot_raw_and_background(
                x, y, compute_background(x, y, self.background_combo.currentText())
            )
        except Exception as e: self.warn(str(e))

    def run_fit(self):
        try:
            self._fit_origin_page = int(self.workflow_stack.currentIndex())
            free_mode = self._fit_mode() == "free"
            use_studio_model = bool(self._use_studio_model_once)
            self._use_studio_model_once = False
            use_model_now = bool(
                self.use_model_check.isChecked()
                and (not free_mode or use_studio_model)
            )
            if self.fit_result is not None and self._pre_refit_snapshot is None:
                self._pre_refit_snapshot = (
                    self.fit_result, copy.deepcopy(self.fit_context), bool(self.studio_dirty)
                )
            if self.energy is None:
                raise ValueError("Load an XPS data file first.")
            if self.fwhm_max_spin.value() <= self.fwhm_min_spin.value():
                raise ValueError("Maximum FWHM must be greater than minimum FWHM.")
            model_doc = None
            model_warnings = []
            if use_model_now:
                model_doc, model_warnings = self._validated_model_from_ui()
                c = [float(spec['center']) for spec in model_doc['components']]
                region_centers = c
            else:
                c = parse_centers(self.centers_edit.text())
                region_centers = _expanded_centers(
                    c, None if free_mode else self.db_fetched_info,
                    False if free_mode else self.auto_doublet_check.isChecked(),
                )
            # A final Studio fit must use the same measured window that was
            # reviewed interactively.  Rebuilding the window from manually
            # moved centers can shift an edge by a fraction of an eV and leave
            # otherwise-valid manual-background anchors just outside it.  It
            # also changes the data being fitted between preview and final
            # optimization.  Preserve the original sampled window for an
            # auditable like-for-like refit; ordinary Quick Fit runs still use
            # the current centers and margin.
            fit_window_source = "current centers and fit-region margin"
            prior_x = np.asarray(
                self.fit_context.get("x_fit", []), dtype=float
            ) if use_studio_model else np.asarray([], dtype=float)
            if (
                use_studio_model
                and len(prior_x) >= 20
                and np.all(np.isfinite(prior_x))
            ):
                sampled_lo = float(np.min(prior_x))
                sampled_hi = float(np.max(prior_x))
                energy_steps = np.diff(np.unique(np.asarray(self.energy, dtype=float)))
                edge_tolerance = max(
                    1e-9,
                    0.25 * float(np.median(energy_steps)) if len(energy_steps) else 1e-9,
                )
                preserved_mask = (
                    (self.energy >= sampled_lo - edge_tolerance)
                    & (self.energy <= sampled_hi + edge_tolerance)
                )
                if int(np.count_nonzero(preserved_mask)) < 20:
                    raise ValueError(
                        "The reviewed fit window is no longer available on the current energy axis. "
                        "Return to Quick Fit and start a new fit."
                    )
                x = self.energy[preserved_mask]
                y = self.intensity[preserved_mask]
                if any(
                    float(center) < float(np.min(x))
                    or float(center) > float(np.max(x))
                    for center in region_centers
                ):
                    raise ValueError(
                        "An edited peak center lies outside the reviewed fit window. "
                        "Move it inside the plotted interval or start a new Quick Fit with a wider margin."
                    )
                prior_region = self.fit_context.get("fit_region", (sampled_lo, sampled_hi))
                try:
                    fr = (float(prior_region[0]), float(prior_region[1]))
                except (TypeError, ValueError, IndexError):
                    fr = (sampled_lo, sampled_hi)
                fit_window_source = "preserved from interactive workspace"
            else:
                x, y, fr = extract_fit_region(
                    self.energy, self.intensity, region_centers,
                    self.margin_spin.value()
                )
            background_method = self.background_combo.currentText()
            manual_points = (
                copy.deepcopy(self.manual_baseline_points)
                if self._manual_background_selected() else None
            )
            bkg = compute_background(
                x, y, background_method, manual_points=manual_points
            )
            use_baseline_refine = (
                self.baseline_refine_check.isChecked()
                and background_method.lower() != "none"
                and "manual" not in background_method.lower()
            )
            # Do NOT zero-clip background-subtracted data. Clipping biases noise and integrated areas.
            y_corr = y - bkg
            new_fit_context = {
                "x_fit": x, "y_raw": y, "background": bkg, "y_corr": y_corr,
                "centers": c, "shift_bound": self.shift_spin.value(),
                "fit_region": fr,
                "fit_window_source": fit_window_source,
                "ui_fit_mode": "free" if free_mode else "physics",
                "target": self.element_edit.text().strip(),
                "orbital": self.orbital_edit.text().strip(),
                "background_method": background_method,
                "manual_baseline_points": [
                    {"binding_energy_eV": float(px), "intensity": float(py)}
                    for px, py in (manual_points or [])
                ],
                "weighting_method": self.weighting_combo.currentText(),
                "lineshape_method": (
                    "Per-component shapes from Model Builder" if model_doc is not None
                    else self.lineshape_combo.currentText()
                ),
                "default_lineshape": self._selected_lineshape(),
                "physics_constraints_enabled": bool(
                    not free_mode and (
                        model_doc is not None or self.auto_doublet_check.isChecked()
                    )
                ),
                "shared_shape_enabled": bool(
                    not free_mode and self.shared_shape_check.isChecked()
                ),
                "baseline_refine_enabled": use_baseline_refine,
                "fwhm_limits": [self.fwhm_min_spin.value(), self.fwhm_max_spin.value()],
                "manual_constraints": self.constraints_box.toPlainText(),
                "charge_shift_eV": self.applied_charge_shift,
                "energy_reference": self.calibration_reference_edit.text().strip(),
                "energy_reference_measured_eV": self.measured_c1s.value(),
                "energy_reference_target_eV": self.target_c1s.value(),
                "model_schema": MODEL_SCHEMA if model_doc is not None else "legacy/generated",
                "model_name": model_doc.get("name", "") if model_doc else "",
                "model_reference": model_doc.get("reference", "") if model_doc else "",
                "model_notes": model_doc.get("notes", "") if model_doc else "",
                "model_components": copy.deepcopy(model_doc.get("components", [])) if model_doc else [],
                "model_validation_warnings": list(model_warnings),
            }

            ftol = float(self.ftol_edit.text())
            xtol = float(self.xtol_edit.text())
            if not (0.0 < ftol < 1.0 and 0.0 < xtol < 1.0):
                raise ValueError("ftol and xtol must be numeric values between 0 and 1.")
            s = {
                'ftol': ftol, 'xtol': xtol, 'max_iter': self.iter_spin.value(),
                'shape': self._selected_lineshape(),
                'weighting': 'poisson' if 'Poisson' in self.weighting_combo.currentText() else 'uniform',
                'constraints': self.constraints_box.toPlainText(),
                'db_info': None if free_mode else copy.deepcopy(self.db_fetched_info),
                'auto_doublet': bool(
                    not free_mode and self.auto_doublet_check.isChecked()
                ),
                'model_specs': copy.deepcopy(model_doc['components']) if model_doc is not None else None,
                'model_reference': model_doc.get('reference', '') if model_doc is not None else '',
                'fwhm_min': self.fwhm_min_spin.value(),
                'fwhm_max': self.fwhm_max_spin.value(),
                'shared_shape': bool(
                    not free_mode and self.shared_shape_check.isChecked()
                ),
                'baseline_refine': use_baseline_refine,
            }
            new_fit_context["optimizer_settings"] = {
                "requested_ftol": s["ftol"], "requested_xtol": s["xtol"],
                "requested_max_nfev": s["max_iter"],
                "final_ftol": min(s["ftol"], 1e-9),
                "final_xtol": min(s["xtol"], 1e-9),
                "final_max_nfev": max(s["max_iter"], 7000),
            }

            self.fit_context = new_fit_context
            self.fit_result = None
            self._set_manual_editor_checked(False)
            self._set_fit_running(True); self.progress_bar.setRange(0, 0)
            self.worker = FitWorker(self.fit_context, s)
            self.worker.progress.connect(self.log)
            self.worker.error.connect(self._on_fit_error)
            self.worker.finished.connect(self._on_fit_done)
            self.worker.start()
        except Exception as e:
            self._on_fit_error(str(e))

    def _on_fit_done(self, res):
        self._set_fit_running(False)
        self._pre_refit_snapshot = None
        self.fit_result = res
        self.analysis_tabs.setCurrentIndex(2)
        residual = self.fit_context["y_corr"] - res["y_fit"]
        self.setup_canvas.plot_fit(
            self.fit_context["x_fit"], self.fit_context["y_raw"], self.fit_context["background"],
            res.get("baseline_correction", np.zeros_like(res["y_fit"])), res["y_fit"],
            res["components"], residual, res["peak_results"],
            metrics=res["metrics"], state_fractions=res.get("state_fractions", {}),
            title=(
                f"{self.fit_context.get('target', '')} {self.fit_context.get('orbital', '')} XPS fit".strip()
                or "XPS peak fit"
            )
        )
        area_header = "Window/Full Area ± SE" if any(
            p.lineshape == 'doniach_sunjic' for p in res["peak_results"]
        ) else "Full Area ± SE"
        self.results_table.setHorizontalHeaderItem(6, QTableWidgetItem(area_header))

        self.results_table.setRowCount(len(res["peak_results"]))
        def with_se(value, stderr, digits):
            return f"{value:.{digits}f} ± {stderr:.{digits}f}" if stderr is not None else f"{value:.{digits}f}"

        for r, p in enumerate(res["peak_results"]):
            fraction_text = format_fraction_percent(
                p.area_fraction, decimals=2, include_symbol=False
            )
            if p.area_fraction is not None and p.area_fraction_stderr is not None:
                fraction_text += " ± " + format_fraction_percent(
                    p.area_fraction_stderr, decimals=2, include_symbol=False
                )
            if p.role == "partner":
                constraint_text = f"Doublet tie (+{p.offset_fitted:.3f} eV)" if p.offset_fitted is not None else "Doublet tie"
            elif p.center_mode == "fixed":
                constraint_text = "Fixed"
            elif p.center_mode == "bounded":
                constraint_text = f"Bounded [{p.center_min:.3f}, {p.center_max:.3f}]"
            elif p.center_mode == "tied":
                offset_text = with_se(p.offset_fitted, p.offset_stderr, 3) if p.offset_fitted is not None else "?"
                constraint_text = f"Tied to {p.tie_target} ({offset_text} eV)"
                if p.offset_min is not None and p.offset_max is not None:
                    constraint_text += f" in [{p.offset_min:.3f}, {p.offset_max:.3f}]"
            else:
                constraint_text = f"Free ±{self.fit_context.get('shift_bound', 0.0):.2f} eV"
            qa = p.bound_warning
            if not p.quantify:
                qa = (qa + "; " if qa else "") + "excluded from oxidation-state fraction"
            elif p.area_fraction is not None and p.area_fraction < 1e-4:
                qa = (qa + "; " if qa else "") + "effectively zero fitted area (<0.01%)"
            values = [
                p.label, p.state or "Unassigned", f"{p.center_input:.3f}",
                with_se(p.center_fitted, p.center_stderr, 3),
                with_se(p.fwhm_fitted, p.fwhm_stderr, 3),
                with_se(p.height_fitted, p.height_stderr, 2),
                with_se(p.area_fitted, p.area_stderr, 2), fraction_text,
                constraint_text, p.lineshape, p.role,
                f"{p.eta_fitted:.3f}", f"{p.asym_fitted:.3f}", qa
            ]
            for c, v in enumerate(values):
                it = QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if c == 8:
                    mode_colors = {
                        "free": ("#EFF6FF", "#1D4ED8"), "fixed": ("#F1F5F9", "#334155"),
                        "bounded": ("#FFFBEB", "#B45309"), "tied": ("#F5F3FF", "#6D28D9"),
                    }
                    bg, fg = mode_colors.get(p.center_mode, ("#F8FAFC", "#334155"))
                    it.setBackground(QBrush(QColor(bg))); it.setForeground(QBrush(QColor(fg)))
                elif c == 7 and p.area_fraction is None:
                    it.setBackground(QBrush(QColor("#F1F5F9"))); it.setForeground(QBrush(QColor("#64748B")))
                elif c == 13 and qa:
                    it.setBackground(QBrush(QColor("#FEF2F2"))); it.setForeground(QBrush(QColor("#B91C1C")))
                self.results_table.setItem(r, c, it)

        m = res["metrics"]
        quality_style = {
            "GOOD": ("#ECFDF5", "#047857", "#6EE7B7"),
            "REVIEW": ("#FFFBEB", "#B45309", "#FCD34D"),
            "POOR": ("#FEF2F2", "#B91C1C", "#FCA5A5"),
        }.get(m.fit_quality, ("#F8FAFC", "#334155", "#CBD5E1"))
        qbg, qfg, qborder = quality_style
        warning_count = len(res.get("warnings", []))
        self.fit_quality_banner.setText(
            f"Fit quality: {m.fit_quality}   |   R² = {m.r2:.4f}   |   "
            f"NRMSE = {m.nrmse_percent:.2f}%   |   DW = {m.durbin_watson:.2f}   |   "
            f"QA notes = {warning_count}"
        )
        self.fit_quality_banner.setStyleSheet(
            f"background:{qbg}; color:{qfg}; border:1px solid {qborder}; "
            "border-radius:7px; padding:9px 11px; font-weight:bold;"
        )
        self.metrics_label.setText(
            f"RMSE: {m.rmse:.4f} | NRMSE: {m.nrmse_percent:.2f}% | R²: {m.r2:.4f} | "
            f"Red. objective: {m.reduced_chi2:.4g} | AICc: {m.aicc:.2f} | BIC: {m.bic:.2f} | "
            f"DW: {m.durbin_watson:.2f} | Weighting: {m.weighting} | Quality: {m.fit_quality}"
        )
        self.log(
            f"Fit completed with {res.get('n_starts', 0)} multi-start trials and "
            f"{res.get('n_refined', 0)} high-precision refinements. Quality: {m.fit_quality}"
        )
        if res.get('state_fractions'):
            frac_text = " | ".join(
                f"{k}: {format_fraction_percent(v)}"
                + (" ± " + format_fraction_percent(res.get('state_fraction_stderr', {}).get(k))
                   if res.get('state_fraction_stderr', {}).get(k) is not None else "")
                for k, v in res['state_fractions'].items()
            )
            self.log("Integrated state fractions: " + frac_text)
            self.state_fraction_label.setText("Oxidation/chemical-state fractions: " + frac_text)
        else:
            self.state_fraction_label.setText(
                "Free fit: peak-area percentages are listed per component; no chemical state is inferred."
                if self.fit_context.get("ui_fit_mode") == "free" else
                "Oxidation/chemical-state fractions: not available"
            )
        for warning in res.get('warnings', []):
            self.log("WARNING: " + warning)
        self._initialize_studio_state(res)
        return_page = 1 if int(getattr(self, "_fit_origin_page", 0)) == 1 else 0
        self._show_workflow_page(return_page)
        if return_page == 0:
            self.status.showMessage(
                "Fit complete. Results are shown below; open Edit & Review only when you want manual peak adjustment.",
                8000,
            )

    def export_all(self):
        if not self.fit_result:
            return self.warn("Run a fit first.")
        if self.studio_dirty:
            final_name = (
                "Final Free Fit" if self._studio_free_editing_allowed()
                else "Final Constrained Fit"
            )
            return self.warn(
                f"The graph contains provisional manual edits. Run {final_name} before export "
                "so uncertainties and physical constraints are recalculated, or use "
                "Export Preview to keep the exact manual values."
            )
        try:
            return self._export_result_bundle(self.fit_result, "VALIDATED")
        except Exception as exc:
            self.warn(f"Export failed:\n\n{exc}")

    def export_current_preview(self):
        """Export exact interactive values without presenting them as optimizer-validated."""
        if self.fit_result is None or not self.studio_state:
            return self.warn("Run a fit before exporting an interactive preview.")
        if not self.studio_dirty:
            return self.export_all()
        try:
            preview = self._studio_preview_result()
            result = dict(self.fit_result)
            result.update(preview)
            result["baseline_correction"] = np.asarray(
                self.fit_result.get(
                    "baseline_correction",
                    np.zeros_like(np.asarray(preview["y_fit"], dtype=float)),
                ),
                dtype=float,
            )
            result["model_specs"] = self._studio_specs_from_state()
            result["peak_shapes"] = [
                canonical_lineshape(item.get("shape", "pseudo_voigt"))
                for item in self.studio_state
            ]
            result["state_fraction_stderr"] = {}
            result["mode"] = "manual_preview"
            result["success"] = False
            result["message"] = (
                "Operator-adjusted provisional preview; no final optimizer validation "
                "or parameter uncertainties were calculated."
            )
            result["n_starts"] = 0
            result["n_refined"] = 0
            provisional_warning = (
                "PROVISIONAL MANUAL PREVIEW: parameter uncertainties are unavailable; "
                "do not describe these values as a validated nonlinear least-squares result."
            )
            result["warnings"] = [provisional_warning] + [
                warning for warning in result.get("warnings", [])
                if warning != provisional_warning
            ]
            result["metrics"].warning_count = len(result["warnings"])
            return self._export_result_bundle(result, "PROVISIONAL_MANUAL")
        except Exception as exc:
            self.warn(f"Manual-preview export failed:\n\n{exc}")

    def _export_result_bundle(self, res: Dict[str, Any], result_status: str):
        validated = result_status == "VALIDATED"
        dialog_title = (
            "Select Directory to Export Validated Results"
            if validated else "Select Directory to Export Provisional Manual Preview"
        )
        dir_path = QFileDialog.getExistingDirectory(self, dialog_title)
        if not dir_path: return
        
        # Millisecond precision prevents two rapid exports from silently
        # overwriting one another in the same output directory.
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        suffix = "" if validated else "_ManualPreview"
        base_name = f"XPS_Analysis_{timestamp}{suffix}"
        base_path = Path(dir_path) / base_name
        
        ctx = self.fit_context
        
        baseline_correction = res.get("baseline_correction", np.zeros_like(res["y_fit"]))
        final_background = ctx["background"] + baseline_correction
        corrected_final = ctx["y_raw"] - final_background
        total_fit_raw = ctx["background"] + res["y_fit"]
        residual = ctx["y_raw"] - total_fit_raw
        data = {
            "Binding_Energy_eV": ctx["x_fit"],
            "Experimental_Intensity": ctx["y_raw"],
            "Initial_Background": ctx["background"],
            "Residual_Baseline_Correction": baseline_correction,
            "Final_Background": final_background,
            "Background_Subtracted_Intensity": corrected_final,
            "Total_Fit_Raw_Scale": total_fit_raw,
            "Residual_Data_Minus_Fit": residual,
        }
        for i, comp in enumerate(res["components"], 1): data[f"Peak_{i}"] = comp
        pd.DataFrame(data).to_csv(f"{base_path}_data.csv", index=False)
        origin_ready_frame = build_origin_ready_plot_frame(
            binding_energy=ctx["x_fit"],
            experimental_data=ctx["y_raw"],
            final_background=final_background,
            total_fit_raw_scale=total_fit_raw,
            residual_data_minus_fit=residual,
            components=res["components"],
            peak_results=res.get("peak_results", []),
        )
        origin_ready_frame.to_csv(
            f"{base_path}_OriginReady_Plot.csv", index=False
        )
        parameter_frame = pd.DataFrame([asdict(p) for p in res["peak_results"]])
        parameter_frame.insert(0, "Result_Status", result_status)
        parameter_frame.insert(1, "Uncertainties_Valid", validated)
        parameter_frame.to_csv(f"{base_path}_parameters.csv", index=False)
        manual_anchor_records = list(ctx.get("manual_baseline_points", []) or [])
        if manual_anchor_records:
            pd.DataFrame(manual_anchor_records).to_csv(
                f"{base_path}_manual_baseline_anchors.csv", index=False
            )
        quality_rows = [{
            "Result_Status": result_status,
            "Uncertainties_Valid": validated,
            "RMSE": res["metrics"].rmse, "R2": res["metrics"].r2, "RSS": res["metrics"].rss,
            "NRMSE_Percent": res["metrics"].nrmse_percent,
            "Objective_Value": res["metrics"].objective_value,
            "Reduced_Objective": res["metrics"].reduced_chi2,
            "Weighting": res["metrics"].weighting, "AIC": res["metrics"].aic,
            "AICc": res["metrics"].aicc, "BIC": res["metrics"].bic,
            "N_Varying_Parameters": res["metrics"].nvarys,
            "Max_Abs_Parameter_Correlation": res["metrics"].max_abs_corr,
            "Residual_Lag1_Correlation": res["metrics"].residual_lag1_corr,
            "Durbin_Watson": res["metrics"].durbin_watson,
            "Fit_Quality": res["metrics"].fit_quality,
            "Warnings": " | ".join(res.get("warnings", [])),
            **{f"Fraction_{k}": v for k, v in res.get("state_fractions", {}).items()},
            **{f"Fraction_SE_{k}": v for k, v in res.get("state_fraction_stderr", {}).items()}
        }]
        pd.DataFrame(quality_rows).to_csv(f"{base_path}_fit_quality.csv", index=False)

        fitted_model_document = {
            "schema": MODEL_SCHEMA,
            "result_status": result_status,
            "uncertainties_valid": validated,
            "name": ctx.get("model_name", "") or f"{ctx.get('target', '')} {ctx.get('orbital', '')} fitted model".strip(),
            "target": ctx.get("target", ""),
            "orbital": ctx.get("orbital", ""),
            "reference": ctx.get("model_reference", ""),
            "notes": ctx.get("model_notes", ""),
            "default_lineshape": ctx.get("default_lineshape", "pseudo_voigt"),
            "default_fwhm_min": float((ctx.get("fwhm_limits") or [0.35, 4.0])[0]),
            "default_fwhm_max": float((ctx.get("fwhm_limits") or [0.35, 4.0])[1]),
            "components": res.get("model_specs", []),
        }
        with open(f"{base_path}_model.json", "w", encoding="utf-8") as handle:
            json.dump(fitted_model_document, handle, ensure_ascii=False, indent=2)
        
        buf = BytesIO()
        self.canvas.figure.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        png_data = buf.getvalue()
        self.canvas.figure.savefig(f"{base_path}_plot.png", format='png', dpi=600, bbox_inches='tight')
        self.canvas.figure.savefig(f"{base_path}_plot.pdf", format='pdf', bbox_inches='tight')
        self.canvas.figure.savefig(f"{base_path}_plot.svg", format='svg', bbox_inches='tight')

        audit = {
            "software": APP_NAME,
            "software_version": APP_VERSION,
            "result_status": result_status,
            "uncertainties_valid": validated,
            "manual_preview_note": (
                "Exact operator-adjusted values; descriptive envelope metrics only. "
                "No final optimizer validation or parameter uncertainties."
                if not validated else ""
            ),
            "origin_ready_plot_export": {
                "file": f"{base_name}_OriginReady_Plot.csv",
                "binding_energy_order": "descending (high-to-low)",
                "component_curve_definition": "Final_Background + zero-based fitted component",
                "main_plot_columns": "Experimental_Data, Total_Fit, Final_Background, and all Peak_*_On_Background columns",
                "residual_plot_columns": "Residual_Data_Minus_Fit and Residual_Zero_Line",
            },
            "exported_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_file": str(self.file_path or ""),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "lmfit_version": getattr(lmfit, "__version__", "unknown"),
            "multi_start_random_seed": FIT_RANDOM_SEED,
            "target": ctx.get("target", ""),
            "orbital": ctx.get("orbital", ""),
            "reference_center_note": "Built-in centers are starting values only; chemical assignments require sample-appropriate references.",
            "charge_shift_eV": ctx.get("charge_shift_eV", 0.0),
            "energy_reference": ctx.get("energy_reference", ""),
            "energy_reference_measured_eV": ctx.get("energy_reference_measured_eV"),
            "energy_reference_target_eV": ctx.get("energy_reference_target_eV"),
            "fit_region_eV": list(ctx.get("fit_region", ())),
            "input_centers_eV": list(ctx.get("centers", [])),
            "background_method": ctx.get("background_method", ""),
            "background_definition": (
                "piecewise-linear interpolation through recorded user anchors"
                if manual_anchor_records else
                "static background calculated before peak optimization"
            ),
            "manual_baseline_points": manual_anchor_records,
            "residual_weighting": ctx.get("weighting_method", "Uniform"),
            "line_shape": ctx.get("lineshape_method", ""),
            "component_line_shapes": list(res.get("peak_shapes", [])),
            "area_definition": {
                "pseudo_voigt_asymmetric_pseudo_voigt_gaussian_lorentzian": "analytic full-line area",
                "doniach_sunjic": "finite fit-window integral (ideal DS tail is not integrable)",
            },
            "physics_constraints_enabled": ctx.get("physics_constraints_enabled", False),
            "ui_fit_mode": ctx.get("ui_fit_mode", "physics"),
            "shared_shape_enabled": ctx.get("shared_shape_enabled", False),
            "residual_baseline_refinement": ctx.get("baseline_refine_enabled", False),
            "center_shift_bound_eV": ctx.get("shift_bound"),
            "center_constraint_modes": {
                mode: sum(spec.get("center_mode") == mode for spec in res.get("model_specs", []))
                for mode in sorted(VALID_CENTER_MODES)
            },
            "fwhm_limits_eV": ctx.get("fwhm_limits", []),
            "manual_constraints": ctx.get("manual_constraints", ""),
            "model_schema": ctx.get("model_schema", ""),
            "model_name": ctx.get("model_name", ""),
            "model_reference": ctx.get("model_reference", ""),
            "model_notes": ctx.get("model_notes", ""),
            "validated_model_components": res.get("model_specs", []),
            "model_validation_warnings": ctx.get("model_validation_warnings", []),
            "fit_mode": res.get("mode", ""),
            "optimizer_settings": ctx.get("optimizer_settings", {}),
            "multi_start_trials": res.get("n_starts", 0),
            "high_precision_refinements": res.get("n_refined", 0),
            "fit_success": bool(res.get("success", False)),
            "optimizer_message": res.get("message", ""),
            "fit_metrics": asdict(res["metrics"]),
            "warnings": list(res.get("warnings", [])),
            "state_fractions": res.get("state_fractions", {}),
            "state_fraction_stderr": res.get("state_fraction_stderr", {}),
        }
        with open(f"{base_path}_audit.json", "w", encoding="utf-8") as f:
            json.dump(audit, f, ensure_ascii=False, indent=2)
            
        # Creating a 3-page report with A4 style and citation rules
        img_b64 = base64.b64encode(png_data).decode('utf-8')
        m = res["metrics"]
        
        table_rows = ""
        for p in res["peak_results"]:
            center_se = f" ± {p.center_stderr:.3f}" if p.center_stderr is not None else ""
            fwhm_se = f" ± {p.fwhm_stderr:.3f}" if p.fwhm_stderr is not None else ""
            area_se = f" ± {p.area_stderr:.2f}" if p.area_stderr is not None else ""
            area_pct = format_fraction_percent(
                p.area_fraction, decimals=2, include_symbol=False
            )
            if p.area_fraction is not None and p.area_fraction_stderr is not None:
                area_pct += " ± " + format_fraction_percent(
                    p.area_fraction_stderr, decimals=2, include_symbol=False
                )
            if p.role == "partner":
                center_constraint = f"Doublet tie (+{p.offset_fitted:.3f} eV)" if p.offset_fitted is not None else "Doublet tie"
            elif p.center_mode == "bounded":
                center_constraint = f"Bounded [{p.center_min:.3f}, {p.center_max:.3f}]"
            elif p.center_mode == "tied":
                center_constraint = f"Tied to {p.tie_target}"
                if p.offset_min is not None and p.offset_max is not None:
                    center_constraint += f"; offset [{p.offset_min:.3f}, {p.offset_max:.3f}] eV"
            else:
                center_constraint = p.center_mode.title()
            table_rows += (
                f"<tr><td>{html_lib.escape(p.label)}</td><td>{html_lib.escape(p.state or 'Unassigned')}</td>"
                f"<td>{p.center_fitted:.3f}{center_se}</td><td>{p.fwhm_fitted:.3f}{fwhm_se}</td>"
                f"<td>{p.area_fitted:.2f}{area_se}</td><td>{area_pct}</td>"
                f"<td>{html_lib.escape(center_constraint)}</td><td>{html_lib.escape(p.lineshape)}</td>"
                f"<td>{p.eta_fitted:.2f}</td><td>{p.asym_fitted:.2f}</td></tr>\n"
            )
            
        warning_html = "<br>".join([f"&#9888; {html_lib.escape(w)}" for w in res.get("warnings", [])]) or "No automatic fit-quality warnings."
        state_parts = []
        for k, v in res.get("state_fractions", {}).items():
            text = f"{html_lib.escape(k)}: {format_fraction_percent(v)}"
            se = res.get("state_fraction_stderr", {}).get(k)
            if se is not None:
                text += f" ± {format_fraction_percent(se)}"
            state_parts.append(text)
        state_html = " | ".join(state_parts) or "N/A"
        manual_anchor_summary = (
            f"{len(manual_anchor_records)} anchors (see manual-baseline CSV and audit JSON)"
            if manual_anchor_records else "N/A"
        )

        report_status_title = (
            "Validated XPS Fitting Report"
            if validated else "PROVISIONAL Manual XPS Preview"
        )
        report_status_notice = (
            ""
            if validated else
            "<div class='provisional'><strong>PROVISIONAL:</strong> These are exact operator-adjusted "
            "values. Parameter uncertainties and final optimizer validation are not available.</div>"
        )

        report_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>XPS Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Tahoma, Arial, sans-serif; margin: 0; background: #f1f5f9; color: #0f172a; }}
  .page {{ 
      background: white; width: 210mm; min-height: 297mm; padding: 20mm; 
      margin: 10mm auto; box-sizing: border-box; box-shadow: 0 4px 10px rgba(0,0,0,0.1); 
      page-break-after: always; display: flex; flex-direction: column;
  }}
  @media print {{
      body {{ background: white; }}
      .page {{ margin: 0; box-shadow: none; width: auto; min-height: auto; border: none; padding: 15mm; }}
  }}
  .cover {{ justify-content: center; align-items: center; text-align: center; }}
  .title {{ font-size: 3.5em; font-weight: bold; color: #1d4ed8; margin-bottom: 10px; }}
  .subtitle {{ font-size: 1.8em; color: #64748b; }}
  .date {{ font-size: 1.2em; margin-top: 40px; color: #475569; padding: 10px 20px; border: 2px solid #e2e8f0; border-radius: 8px; display: inline-block; }}
  
  .citation-box {{ margin-top: 60px; background: #FFFBEB; border-left: 5px solid #F59E0B; padding: 20px; text-align: left; border-radius: 6px; color: #92400E; max-width: 80%; font-size: 1.1em; line-height: 1.6; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
  
  h1 {{ color: #1e3a8a; border-bottom: 3px solid #e2e8f0; padding-bottom: 10px; margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 10.5px; table-layout: auto; }}
  th, td {{ border: 1px solid #cbd5e1; padding: 6px 4px; text-align: center; overflow-wrap: anywhere; }}
  th {{ background-color: #f8fafc; color: #0f172a; }}
  .metrics {{ background: #eff6ff; padding: 15px; border-radius: 8px; border: 1px solid #bfdbfe; margin-bottom: 20px; font-size: 15px; }}
  .provisional {{ background: #fff7ed; color: #9a3412; padding: 14px; border: 2px solid #fb923c; border-radius: 8px; margin: 16px 0; font-size: 15px; }}
  .footer {{ margin-top: auto; text-align: center; font-size: 12px; color: #94a3b8; padding-top: 20px; border-top: 1px solid #f1f5f9; }}
</style>
</head>
<body>
  <div class="page cover">
    <div class="title">Researcher Analysis Toolkit — XPS Fitting</div>
    <div class="subtitle">{report_status_title}</div>
    {report_status_notice}
    <div class="date">Date & Time:<br>{datetime.datetime.now().strftime("%Y-%m-%d &nbsp;|&nbsp; %H:%M:%S")}</div>
    
    <div class="citation-box">
        <h3 style="margin-top:0; color:#B45309; border-bottom: 1px solid #FCD34D; padding-bottom: 8px;">Reproducibility and citation</h3>
        <strong>XPS module version:</strong> {APP_VERSION}<br>
        <strong>Suggested citation:</strong> Behroozi et al., <i>RSC Adv.</i> (2026).
        <a href="https://doi.org/10.1039/D6RA08618G">10.1039/D6RA08618G</a><br><br>
        Report the background, line shape, parameter constraints, fit interval, and charge-reference method in any publication. Keep the accompanying audit JSON and numerical CSV files with the analysis record.
    </div>
  </div>

  <div class="page">
    <h1>Fitting Parameters & Results</h1>
    {report_status_notice}
    <div class="metrics">
        <strong>Global Fit Metrics:</strong><br><br>
        Target / region: {html_lib.escape(ctx.get('target', '') or 'Not specified')} / {html_lib.escape(ctx.get('orbital', '') or 'Not specified')}<br><br>
        Chemical model: {html_lib.escape(ctx.get('model_name', '') or res.get('mode', 'generated model'))}<br><br>
        RMSE: {m.rmse:.4f} &nbsp;&nbsp;|&nbsp;&nbsp; NRMSE: {m.nrmse_percent:.2f}% &nbsp;&nbsp;|&nbsp;&nbsp; R²: {m.r2:.4f}<br><br>
        Reduced objective: {m.reduced_chi2:.4g} &nbsp;&nbsp;|&nbsp;&nbsp; AICc: {m.aicc:.2f} &nbsp;&nbsp;|&nbsp;&nbsp; BIC: {m.bic:.2f}<br><br>
        Durbin-Watson: {m.durbin_watson:.3f} &nbsp;&nbsp;|&nbsp;&nbsp; Max |parameter correlation|: {m.max_abs_corr:.3f}<br><br>
        Background: {html_lib.escape(ctx.get('background_method', ''))} &nbsp;&nbsp;|&nbsp;&nbsp; Line shape: {html_lib.escape(ctx.get('lineshape_method', ''))}<br><br>
        Manual-background anchors: {html_lib.escape(manual_anchor_summary)}<br><br>
        Weighting: {html_lib.escape(m.weighting)}<br><br>
        Physics constraints: {'enabled' if ctx.get('physics_constraints_enabled') else 'disabled'} &nbsp;&nbsp;|&nbsp;&nbsp; Charge shift: {ctx.get('charge_shift_eV', 0.0):+.3f} eV<br><br>
        Energy reference: {html_lib.escape(ctx.get('energy_reference', '') or 'Not recorded')}<br><br>
        Model reference: {html_lib.escape(ctx.get('model_reference', '') or 'NOT RECORDED — add before publication')}<br><br>
        Result status: <strong>{html_lib.escape(result_status)}</strong><br><br>
        Fit Quality: <strong>{m.fit_quality}</strong>
    </div>
    <div class="metrics"><strong>State fractions:</strong> {state_html}<br><br><strong>Fit-quality review:</strong><br>{warning_html}</div>
    <table>
      <tr><th>Peak</th><th>State</th><th>Center ± SE</th><th>FWHM ± SE</th><th>Area ± SE</th><th>Area %</th><th>Center constraint</th><th>Shape</th><th>Eta</th><th>Asym</th></tr>
      {table_rows}
    </table>
    <div class="footer">Page 2 - Numerical Results</div>
  </div>

  <div class="page">
    <h1>Spectral Analysis Plot</h1>
    <div style="text-align: center; margin-top: 20px;">
        <img src="data:image/png;base64,{img_b64}" style="max-width: 100%; border: 1px solid #cbd5e1; border-radius: 8px;">
    </div>
    <div class="footer">Page 3 - Visualization</div>
  </div>
</body>
</html>"""
        
        # Saving the report file
        with open(f"{base_path}_report.html", "w", encoding="utf-8") as f:
            f.write(report_html)
            
        self.log(f"Exported all files to: {dir_path}")
        if validated:
            self.info(
                f"Exported validated results to:\n{dir_path}\n\n"
                "Includes full numerical data, an Origin-ready plot CSV, exact model JSON, audit JSON, "
                "600-dpi PNG, vector PDF/SVG, and HTML report."
            )
        else:
            self.info(
                f"Exported the exact manual preview to:\n{dir_path}\n\n"
                "Includes an Origin-ready plot CSV. Every numerical/report file is marked "
                "PROVISIONAL_MANUAL and parameter uncertainties are left unavailable."
            )

def main():
    
    try:
        # A unique identifier for your software
        myappid = 'researcher.xps.toolkit.v1' 
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass # In non-Windows operating systems, it is ignored
    # ---------------------------------------------------------

    app = QApplication(sys.argv)
    
    # Setting the default icon for all application windows (including warning messages)
    app.setWindowIcon(QIcon("ResearcherLogo.ico")) 
    
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 10))
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__": 
    main()
