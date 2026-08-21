# -*- coding: utf-8 -*-
"""
Researcher Analysis Toolkit — Tafel Analysis Module
OER / HER | Drag & Drop GUI | Column selection | PNG plots | PDF report

This version reports Tafel slopes from:
  - current i (standard Tafel)
  - 1st derivative d1  (per DTA concept)
  - 2nd derivative d2  (more sensitive; can be noisy)
  - pointwise instantaneous slope inside the confirmed fitting window

It also provides explicit reference-electrode-to-RHE conversion and
reaction-direction overpotential outputs for OER and HER.

Run:
  python Tafel_Analysis_corrected.py

Dependencies:
  pip install pyqt6 pandas numpy scipy matplotlib reportlab openpyxl
"""

import os
import sys
import math
import traceback
import datetime
import json
import textwrap
import re
import ctypes
from PyQt6.QtGui import QIcon

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import matplotlib
# Prefer interactive Qt backend for GUI features (manual window selection).
# Fallback to Agg for headless environments.
try:
    matplotlib.use("QtAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QPushButton, QLabel, QLineEdit, QCheckBox, QComboBox,
    QTextEdit, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QDialog, QDialogButtonBox, QGroupBox,
    QDoubleSpinBox, QSpinBox, QScrollArea, QMessageBox
)

# Matplotlib (Qt) helpers for manual window selection
try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
    from matplotlib.widgets import SpanSelector
except Exception:
    FigureCanvas = None
    NavigationToolbar = None
    SpanSelector = None

F_CONST = 96485.33212  # C/mol
R_CONST = 8.314462618  # J/mol/K


# ----------------------------
# Data classes
# ----------------------------
@dataclass
class FitResult:
    slope: float
    intercept: float
    r2: float
    stderr_slope: float
    n: int





# ---- Corrosion presets (for convenience) ----
# Atomic weights: IUPAC standard atomic weights. Densities: typical near-room-temperature values.
# Users should verify values for their specific material/alloy and temperature.
METAL_PRESETS = {
    # key: (label, atomic_weight_gmol, density_gcm3, default_valence_n)
    "Fe2+": ("Fe (Fe2+)", 55.845, 7.874, 2),
    "Fe3+": ("Fe (Fe3+)", 55.845, 7.874, 3),
    "Al3+": ("Al (Al3+)", 26.9815385, 2.70, 3),
    "Cu1+": ("Cu (Cu+)", 63.546, 8.96, 1),
    "Cu2+": ("Cu (Cu2+)", 63.546, 8.96, 2),
    "Zn2+": ("Zn (Zn2+)", 65.38, 7.14, 2),
    "Ni2+": ("Ni (Ni2+)", 58.6934, 8.90, 2),
    "Ti4+": ("Ti (Ti4+)", 47.867, 4.51, 4),
    "Mg2+": ("Mg (Mg2+)", 24.305, 1.74, 2),
}

def _placeholder_fitresult() -> "FitResult":
    """Create a NaN FitResult compatible with the local FitResult dataclass signature."""
    try:
        flds = list(getattr(FitResult, "__dataclass_fields__", {}).keys())
        # Common fields: slope, intercept, r2, stderr, n
        kwargs = {}
        for name in flds:
            if name == "n":
                kwargs[name] = 0
            else:
                kwargs[name] = float("nan")
        return FitResult(**kwargs)  # type: ignore
    except Exception:
        # Fallback: try positional for the common 5-field form
        try:
            return FitResult(float("nan"), float("nan"), float("nan"), float("nan"), 0)  # type: ignore
        except Exception:
            return FitResult(float("nan"), float("nan"), float("nan"), float("nan"))  # type: ignore


def _placeholder_dtaresult(e: np.ndarray, i: np.ndarray, warnings: list) -> "DTAResult":
    """
    Create a NaN-filled DTAResult compatible with the local DTAResult dataclass signature.
    This avoids version-specific field-name mismatches (e.g., alpha vs. alpha_apparent).
    """
    flds = list(getattr(DTAResult, "__dataclass_fields__", {}).keys())
    kwargs = {}
    for name in flds:
        n = name.lower()
        if n in ("e",):
            kwargs[name] = e
        elif n in ("i_raw", "raw_i", "i"):
            kwargs[name] = i
        elif n in ("i_smooth", "smooth_i"):
            kwargs[name] = i
        elif "i_norm" in n:
            kwargs[name] = np.full_like(i, np.nan, dtype=float)
        elif n in ("d1", "d2") or "deriv" in n or n.startswith("d1") or n.startswith("d2"):
            kwargs[name] = np.full_like(i, np.nan, dtype=float)
        elif "emax" in n:
            kwargs[name] = float("nan")
        elif "fit_start" in n or "fit_end" in n or n in ("elow", "ehigh"):
            kwargs[name] = float("nan")
        elif n.startswith("fit_") or "fit" in n:
            # fit objects
            kwargs[name] = _placeholder_fitresult()
        elif "warnings" in n:
            kwargs[name] = warnings
        elif n == "corr":
            kwargs[name] = None
        else:
            # scalar outputs (alpha, tafel slopes, scores, etc.)
            kwargs[name] = float("nan")
    return DTAResult(**kwargs)  # type: ignore
@dataclass
class CorrosionResult:
    has_both_branches: bool
    Ecorr: float
    Icorr: float  # A (or A/cm^2 if input is current density)
    fit_anodic: Optional[FitResult]
    fit_cathodic: Optional[FitResult]
    warnings: List[str]

    # Derived corrosion parameters (optional)
    beta_a_Vdec: float = float('nan')   # |dE/dlog10(i)| anodic
    beta_c_Vdec: float = float('nan')   # |dE/dlog10(i)| cathodic
    B_V: float = float('nan')           # Stern–Geary constant (V)
    Rp: float = float('nan')            # Ohm (if Icorr in A) or Ohm*cm^2 (if jcorr in A/cm^2)

    area_cm2: float = float('nan')      # if provided
    jcorr_Acm2: float = float('nan')    # Icorr/area
    EW_g_per_eq: float = float('nan')   # equivalent weight
    rho_g_cm3: float = float('nan')     # density
    corrosion_rate_mm_y: float = float('nan')
    corrosion_rate_mpy: float = float('nan')



@dataclass
class DTAResult:
    e: np.ndarray
    i_raw: np.ndarray
    i_smooth: np.ndarray

    i_norm: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    d1_norm: np.ndarray
    d2_norm: np.ndarray

    emax_d2: float

    fit_ln_i: FitResult
    fit_ln_d1: FitResult
    fit_ln_d2: FitResult

    fit_start: float
    fit_end: float

    alpha_from_i: float
    alpha_from_d1: float
    alpha_from_d2: float

    tafelslope_from_i_mVdec: float
    tafelslope_from_d1_mVdec: float
    tafelslope_from_d2_mVdec: float

    consistency_score_i_d1: float
    consistency_score_i_d2: float
    consistency_score_d1_d2: float

    warnings: List[str]

    corr: Optional[CorrosionResult] = None


# Nominal reference-electrode potentials versus SHE near 25 °C.  They are
# deliberately exposed in the GUI and written to every numerical output so a
# conversion is never hidden from the user.
REFERENCE_ELECTRODE_PRESETS: Dict[str, Tuple[str, Optional[float]]] = {
    "rhe": ("Input already referenced to RHE", None),
    "agcl_sat": ("Ag/AgCl (saturated KCl; +0.197 V vs SHE)", 0.197),
    "agcl_3m": ("Ag/AgCl (3.0 M KCl; +0.210 V vs SHE)", 0.210),
    "agcl_1m": ("Ag/AgCl (1.0 M KCl; +0.235 V vs SHE)", 0.235),
    "sce": ("SCE (saturated KCl; +0.241 V vs SHE)", 0.241),
    "custom": ("Custom reference potential vs SHE", None),
}


@dataclass
class PotentialConversion:
    """Auditable conversion from the imported potential axis to RHE and eta."""

    reaction_mode: str
    reference_key: str
    reference_label: str
    reference_offset_vs_she_V: float
    pH: float
    temperature_K: float
    nernst_pH_term_V: float
    measured_to_rhe_shift_V: float
    equilibrium_potential_rhe_V: float
    e_rhe: np.ndarray
    eta_signed: np.ndarray
    eta_reaction: np.ndarray

    def window_summary(self, res: "DTAResult") -> Dict[str, float]:
        mask = (res.e >= res.fit_start) & (res.e <= res.fit_end)
        if not np.any(mask):
            return {
                "fit_start_rhe_V": float("nan"),
                "fit_end_rhe_V": float("nan"),
                "eta_window_low_V": float("nan"),
                "eta_window_high_V": float("nan"),
            }
        erhe = self.e_rhe[mask]
        eta = self.eta_reaction[mask]
        return {
            "fit_start_rhe_V": float(np.nanmin(erhe)),
            "fit_end_rhe_V": float(np.nanmax(erhe)),
            "eta_window_low_V": float(np.nanmin(eta)),
            "eta_window_high_V": float(np.nanmax(eta)),
        }


def build_potential_conversion(
    e_input: np.ndarray,
    reaction_mode: str,
    reference_key: str,
    pH: float,
    temperature_K: float,
    custom_reference_offset_vs_she_V: Optional[float] = None,
    equilibrium_potential_rhe_V: Optional[float] = None,
) -> PotentialConversion:
    """
    Convert an imported potential axis to RHE and reaction-direction
    overpotential.

    For a reference electrode with potential E_ref versus SHE:
        E_RHE = E_input + E_ref + (2.303 RT/F) pH

    eta_signed = E_RHE - E_eq.  eta_reaction is positive in the expected
    reaction direction: eta_signed for OER and -eta_signed for HER.
    """
    mode = reaction_mode.strip().upper()
    if mode not in ("OER", "HER"):
        raise ValueError("RHE/overpotential conversion is available only for OER or HER mode.")
    if reference_key not in REFERENCE_ELECTRODE_PRESETS:
        raise ValueError("Select a valid input reference electrode before calculating overpotential.")

    label, preset_offset = REFERENCE_ELECTRODE_PRESETS[reference_key]
    e = np.asarray(e_input, dtype=float)
    if reference_key == "rhe":
        ref_offset = 0.0
        nernst_pH = 0.0
        shift = 0.0
    else:
        if reference_key == "custom":
            if custom_reference_offset_vs_she_V is None or not np.isfinite(custom_reference_offset_vs_she_V):
                raise ValueError("Enter the custom reference-electrode potential versus SHE.")
            ref_offset = float(custom_reference_offset_vs_she_V)
        else:
            ref_offset = float(preset_offset)
        nernst_pH = (2.303 * R_CONST * float(temperature_K) / F_CONST) * float(pH)
        shift = ref_offset + nernst_pH

    e_rhe = e + shift
    if equilibrium_potential_rhe_V is None:
        e_eq = 1.229 if mode == "OER" else 0.0
    else:
        e_eq = float(equilibrium_potential_rhe_V)
    eta_signed = e_rhe - e_eq
    eta_reaction = eta_signed if mode == "OER" else -eta_signed

    return PotentialConversion(
        reaction_mode=mode,
        reference_key=reference_key,
        reference_label=label,
        reference_offset_vs_she_V=float(ref_offset),
        pH=float(pH),
        temperature_K=float(temperature_K),
        nernst_pH_term_V=float(nernst_pH),
        measured_to_rhe_shift_V=float(shift),
        equilibrium_potential_rhe_V=float(e_eq),
        e_rhe=e_rhe,
        eta_signed=eta_signed,
        eta_reaction=eta_reaction,
    )


def instantaneous_tafel_slope_mVdec(
    res: "DTAResult",
    reaction_mode: str,
) -> np.ndarray:
    """Pointwise, reaction-oriented b_inst = dE/dlog10(i), in mV/dec."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_i = np.log10(res.i_norm)
        dlog_dE = np.gradient(log_i, res.e)
        b_signed = 1000.0 / dlog_dE
    b_reaction = b_signed if reaction_mode.strip().upper() == "OER" else -b_signed
    b_reaction = np.asarray(b_reaction, dtype=float)
    b_reaction[(~np.isfinite(b_reaction)) | (b_reaction <= 0)] = np.nan
    return b_reaction


def instantaneous_tafel_summary(
    res: "DTAResult",
    reaction_mode: str,
) -> Dict[str, float]:
    values = instantaneous_tafel_slope_mVdec(res, reaction_mode)
    mask = (
        (res.e >= res.fit_start)
        & (res.e <= res.fit_end)
        & np.isfinite(values)
    )
    vals = values[mask]
    if vals.size == 0:
        return {
            "instantaneous_n": 0,
            "instantaneous_median_mVdec": float("nan"),
            "instantaneous_mean_mVdec": float("nan"),
            "instantaneous_sd_mVdec": float("nan"),
            "instantaneous_p05_mVdec": float("nan"),
            "instantaneous_p95_mVdec": float("nan"),
            "instantaneous_cv_percent": float("nan"),
        }
    mean = float(np.nanmean(vals))
    sd = float(np.nanstd(vals, ddof=1)) if vals.size > 1 else 0.0
    return {
        "instantaneous_n": int(vals.size),
        "instantaneous_median_mVdec": float(np.nanmedian(vals)),
        "instantaneous_mean_mVdec": mean,
        "instantaneous_sd_mVdec": sd,
        "instantaneous_p05_mVdec": float(np.nanpercentile(vals, 5)),
        "instantaneous_p95_mVdec": float(np.nanpercentile(vals, 95)),
        "instantaneous_cv_percent": float(100.0 * sd / mean) if mean != 0 else float("nan"),
    }

# ----------------------------
# Helpers
# ----------------------------
def _ensure_numeric_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        s = s.astype(str).str.replace(",", ".", regex=False).str.strip()
    return pd.to_numeric(s, errors="coerce")


def _linear_fit(x: np.ndarray, y: np.ndarray) -> FitResult:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    n = len(x)
    if n < 3:
        raise ValueError("Not enough points for linear fit (need >=3).")

    x_mean = x.mean()
    y_mean = y.mean()
    ss_xx = np.sum((x - x_mean) ** 2)
    if not np.isfinite(ss_xx) or ss_xx <= 0:
        raise ValueError("Potential values do not span a finite fitting interval.")
    ss_xy = np.sum((x - x_mean) * (y - y_mean))
    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y_mean) ** 2)
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else 0.0)

    dof = max(n - 2, 1)
    s_err = math.sqrt(ss_res / dof)
    stderr_slope = s_err / math.sqrt(ss_xx) if ss_xx > 0 else float("inf")

    return FitResult(slope=slope, intercept=intercept, r2=r2, stderr_slope=stderr_slope, n=n)




def compute_equivalent_weight(atomic_weight_gmol: float, valence_n: float) -> float:
    """Equivalent weight (g/eq) for dissolution with valence n: EW = AW / n."""
    try:
        if not np.isfinite(atomic_weight_gmol) or not np.isfinite(valence_n) or valence_n <= 0:
            return float('nan')
        return float(atomic_weight_gmol / valence_n)
    except Exception:
        return float('nan')


def compute_ecorr_icorr(e: np.ndarray, i: np.ndarray) -> CorrosionResult:
    """
    Estimate Ecorr and Icorr from a polarization curve by fitting anodic and cathodic Tafel branches:

        E = m_a * log10(|I|) + c_a   (anodic, I>0)
        E = m_c * log10(|I|) + c_c   (cathodic, I<0)

    Intersection gives:
        log10(Icorr) = (c_c - c_a) / (m_a - m_c)
        Ecorr = m_a*log10(Icorr) + c_a
    """
    warnings = []
    e = np.asarray(e, dtype=float)
    i = np.asarray(i, dtype=float)

    anod = (i > 0) & np.isfinite(i) & np.isfinite(e)
    cath = (i < 0) & np.isfinite(i) & np.isfinite(e)

    if not np.any(anod) or not np.any(cath):
        return CorrosionResult(False, float("nan"), float("nan"), None, None, ["Both anodic (I>0) and cathodic (I<0) branches are required."])

    x_a = np.log10(np.abs(i[anod]))
    y_a = e[anod]
    x_c = np.log10(np.abs(i[cath]))
    y_c = e[cath]

    # remove invalid
    ma = np.isfinite(x_a) & np.isfinite(y_a)
    mc = np.isfinite(x_c) & np.isfinite(y_c)
    x_a, y_a = x_a[ma], y_a[ma]
    x_c, y_c = x_c[mc], y_c[mc]

    if len(x_a) < 6 or len(x_c) < 6:
        return CorrosionResult(False, float("nan"), float("nan"), None, None, ["Not enough valid points in one of the branches for fitting."])

    def pick_middle(x, y):
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        lo = np.nanpercentile(x, 25)
        hi = np.nanpercentile(x, 75)
        m = (x >= lo) & (x <= hi)
        if np.sum(m) < 6:
            m = np.ones_like(x, dtype=bool)
        return x[m], y[m]

    xa, ya = pick_middle(x_a, y_a)
    xc, yc = pick_middle(x_c, y_c)

    try:
        fit_a = _linear_fit(xa, ya)  # E vs log|I|
        fit_c = _linear_fit(xc, yc)
    except Exception as ex:
        return CorrosionResult(False, float("nan"), float("nan"), None, None, [f"Fit failed: {ex}"])

    denom = (fit_a.slope - fit_c.slope)
    if abs(denom) < 1e-12:
        return CorrosionResult(False, float("nan"), float("nan"), fit_a, fit_c, ["Anodic and cathodic slopes are too similar; intersection is ill-defined."])

    logIcorr = (fit_c.intercept - fit_a.intercept) / denom
    Icorr = 10 ** logIcorr
    Ecorr = fit_a.slope * logIcorr + fit_a.intercept

    if not np.isfinite(Icorr) or Icorr <= 0 or not np.isfinite(Ecorr):
        warnings.append("Computed Ecorr/Icorr is not finite; check data and fitting ranges.")

    res = CorrosionResult(True, float(Ecorr), float(Icorr), fit_a, fit_c, warnings)

    # Tafel slopes for corrosion (magnitudes)
    try:
        res.beta_a_Vdec = float(abs(fit_a.slope))
        res.beta_c_Vdec = float(abs(fit_c.slope))
        if np.isfinite(res.beta_a_Vdec) and np.isfinite(res.beta_c_Vdec) and (res.beta_a_Vdec + res.beta_c_Vdec) > 0 and np.isfinite(res.Icorr) and res.Icorr > 0:
            # Stern–Geary constant B (V)
            res.B_V = float((res.beta_a_Vdec * res.beta_c_Vdec) / (math.log(10.0) * (res.beta_a_Vdec + res.beta_c_Vdec)))
            # Rp = B / Icorr (units depend on whether Icorr is current or current density)
            res.Rp = float(res.B_V / res.Icorr)
    except Exception:
        pass

    return res


def infer_window_params(n_points: int) -> Tuple[int, int]:
    if n_points <= 80:
        return (10, max(20, int(0.45 * n_points)))
    if n_points <= 300:
        return (20, max(30, int(0.35 * n_points)))
    if n_points <= 1200:
        return (30, max(60, int(0.18 * n_points)))
    return (40, max(80, int(0.12 * n_points)))


def auto_select_tafel_window(
    e: np.ndarray,
    ln_i: np.ndarray,
    ln_d1: np.ndarray,
    r2_min: float = 0.995,
    mismatch_sigma: float = 2.0
) -> Tuple[Optional[Tuple[float, Tuple[int, int], FitResult, FitResult]], Tuple[int, int]]:
    """
    Auto-select window using ln(i) and ln(d1) consistency (robust).
    d2 is reported later but not used as hard criterion because it can be noisy.
    """
    n = len(e)
    min_points, max_points = infer_window_params(n)
    max_points = min(max_points, n - 3)
    min_points = min(min_points, max_points)

    best = None

    for w in range(min_points, max_points + 1):
        for start in range(0, n - w + 1):
            end = start + w
            x = e[start:end]
            y1 = ln_i[start:end]
            y2 = ln_d1[start:end]

            if not (np.all(np.isfinite(y1)) and np.all(np.isfinite(y2))):
                continue

            f1 = _linear_fit(x, y1)
            f2 = _linear_fit(x, y2)

            if f1.r2 < r2_min or f2.r2 < r2_min:
                continue
            # OER normally has positive dln|i|/dE and HER negative.  Both are
            # valid as long as current and first-derivative fits have the same
            # non-zero sign.
            if f1.slope == 0 or f2.slope == 0 or np.sign(f1.slope) != np.sign(f2.slope):
                continue

            denom = max(f1.stderr_slope, f2.stderr_slope, 1e-12)
            mismatch = abs(f1.slope - f2.slope) / denom
            if mismatch > mismatch_sigma:
                continue

            score = mismatch - 0.6 * (f1.r2 + f2.r2) - 0.002 * w

            if best is None or score < best[0]:
                best = (score, (start, end), f1, f2)

    return best, (min_points, max_points)


def _tafel_slope_mVdec_from_alpha(alpha: float, temperature_K: float) -> float:
    """
    Tafel slope magnitude (mV/dec) = ln(10) / (|alpha| * F/(RT)) * 1000
    """
    if not np.isfinite(alpha) or abs(alpha) <= 1e-30:
        # NaN is safer than infinity in CSV/PDF outputs and explicitly marks
        # an undefined slope.  The main current fit is rejected separately if
        # its slope is effectively zero.
        return float("nan")
    f = F_CONST / (R_CONST * temperature_K)
    return (math.log(10.0) / (abs(alpha) * f)) * 1000.0


# ----------------------------
# Core DTA analysis
# ----------------------------
def differential_tafel_analysis(
    e: np.ndarray,
    i: np.ndarray,
    reaction_mode: str,  # "OER", "HER", or "CORR"
    temperature_K: float = 298.15,
    sg_window: int = 21,
    sg_poly: int = 2,
    fit_range: Optional[Tuple[float, float]] = None,
    use_emax_d2_as_upper: bool = True,
    invert_current_manual: bool = False,
    manual_window: Optional[Tuple[float, float]] = None,
    manual_min_points: int = 4
) -> DTAResult:
    warnings: List[str] = []

    e = np.asarray(e, dtype=float)
    i = np.asarray(i, dtype=float)

    # sort by E
    idx = np.argsort(e)
    e = e[idx]
    i = i[idx]

    # remove NaNs
    mask = np.isfinite(e) & np.isfinite(i)
    e = e[mask]
    i = i[mask]
    if len(e) < 12:
        raise ValueError("Too few valid points after cleaning (need at least ~12).")

    mode = reaction_mode.strip().upper()

    # If corrosion mode is selected, only compute Ecorr/Icorr from both branches
    if mode.startswith('CORR'):
        res = _placeholder_dtaresult(e, i, warnings + ['Corrosion mode: DTA/Tafel parameters are not computed.'])
        try:
            res.corr = compute_ecorr_icorr(e, i)
        except Exception as _ex:
            res.corr = CorrosionResult(False, float('nan'), float('nan'), None, None, [str(_ex)])
        return res


    # SavGol window prep
    if sg_window % 2 == 0:
        sg_window += 1
    sg_window = min(sg_window, len(e) - (1 - (len(e) % 2)))
    sg_window = max(sg_window, 5)
    if sg_window >= len(e):
        sg_window = (len(e) - 1) if (len(e) - 1) % 2 == 1 else (len(e) - 2)
        sg_window = max(sg_window, 5)

    # smooth current
    try:
        i_smooth = savgol_filter(i, window_length=sg_window, polyorder=min(sg_poly, sg_window - 2))
    except Exception:
        warnings.append("Savitzky–Golay smoothing failed; using raw current.")
        i_smooth = i.copy()

    # Manual invert override
    if invert_current_manual:
        i_smooth = -i_smooth
        warnings.append("Manual current inversion applied.")

    # Reaction direction: positive current in reaction direction
    if mode == "HER":
        i_k = -i_smooth
    else:
        i_k = i_smooth.copy()

    if np.nanmedian(i_k) < 0:
        warnings.append("Current sign seems inverted; using absolute value for DTA stability.")
        i_k = np.abs(i_k)

    eps = np.nanmax(np.abs(i_k)) * 1e-12 + 1e-30
    i_k = np.where(i_k <= 0, eps, i_k)

    # Normalize I by max
    imax = np.nanmax(i_k)
    if not np.isfinite(imax) or imax <= 0:
        raise ValueError("Invalid current values; cannot normalize.")
    i_norm = i_k / imax

    # Derivatives (same E grid)
    d1 = np.gradient(i_k, e)
    d2 = np.gradient(d1, e)

    # Orient d1 along the kinetic direction before taking its logarithm.
    # For HER, |i| decreases as sorted E becomes more positive, so d|i|/dE
    # is negative and must be sign-reversed.  d2 is expected to be positive
    # for an exponential branch, but can need a robust sign correction.
    d1_adj = -d1 if mode == "HER" else d1.copy()
    d2_adj = d2.copy()

    if np.nanmax(d1_adj) <= 0 and np.nanmin(d1_adj) < 0:
        d1_adj = -d1_adj
    finite_d2 = d2_adj[np.isfinite(d2_adj)]
    if finite_d2.size and np.nanmedian(finite_d2) < 0:
        d2_adj = -d2_adj

    d1_adj = np.where(d1_adj <= 0, eps, d1_adj)
    d2_adj = np.where(d2_adj <= 0, eps, d2_adj)

    d1_norm = d1_adj / np.nanmax(d1_adj)
    d2_norm = d2_adj / np.nanmax(d2_adj)

    # Emax_D2 (diagnostic upper kinetic marker).  Numerical derivatives are
    # unreliable at scan endpoints, so exclude 5% at each edge before finding
    # the maximum.  This prevents initial transients from dominating Emax_D2.
    edge_n = max(2, int(round(0.05 * len(e))), sg_window // 2)
    if 2 * edge_n < len(e):
        d2_search = d2_norm[edge_n:len(e) - edge_n]
        emax_idx = edge_n + int(np.nanargmax(d2_search))
    else:
        emax_idx = int(np.nanargmax(d2_norm))
    emax_d2 = float(e[emax_idx])

    # Default fit range if user didn't set
    e_min = float(np.nanmin(e))
    e_max = float(np.nanmax(e))
    if fit_range is None:
        low = e_min + 0.05 * (e_max - e_min)
        high = e_min + 0.80 * (e_max - e_min)
        fit_range = (low, high)

    fit_low, fit_high = sorted((float(fit_range[0]), float(fit_range[1])))
    if use_emax_d2_as_upper:
        fit_high = min(fit_high, emax_d2)

    # ln arrays
    ln_i = np.log(i_norm)
    ln_d1 = np.log(d1_norm)
    ln_d2 = np.log(d2_norm)

    # ----------------------------
    # Manual window override (skip auto-scan)
    # ----------------------------
    if manual_window is not None:
        fit_low, fit_high = sorted((float(manual_window[0]), float(manual_window[1])))
        if use_emax_d2_as_upper and np.isfinite(emax_d2) and fit_high > emax_d2:
            fit_high = emax_d2
            warnings.append(
                f"Manual window upper limit was clipped to edge-excluded Emax(D2)={emax_d2:.6g} V."
            )
        if fit_high <= fit_low:
            raise ValueError(
                "The selected window lies above the Emax(D2) upper bound. "
                "Disable the Emax(D2) constraint or choose a lower window."
            )
        fit_mask_i = (e >= fit_low) & (e <= fit_high) & np.isfinite(ln_i)
        n_i = int(np.sum(fit_mask_i))
        if n_i < manual_min_points:
            raise ValueError(
                f"Not enough points in manual window for fitting (n={n_i}). "
                "Please widen the selected range or lower filtering/SG window."
            )
        fit_ln_i = _linear_fit(e[fit_mask_i], ln_i[fit_mask_i])

        # d1/d2 fits are optional; they may be NaN if derivatives are unstable
        fit_mask_d1 = fit_mask_i & np.isfinite(ln_d1)
        fit_mask_d2 = fit_mask_i & np.isfinite(ln_d2)

        if int(np.sum(fit_mask_d1)) >= manual_min_points:
            fit_ln_d1 = _linear_fit(e[fit_mask_d1], ln_d1[fit_mask_d1])
        else:
            fit_ln_d1 = FitResult(float('nan'), float('nan'), float('nan'), float('nan'), int(np.sum(fit_mask_d1)))

        if int(np.sum(fit_mask_d2)) >= manual_min_points:
            fit_ln_d2 = _linear_fit(e[fit_mask_d2], ln_d2[fit_mask_d2])
        else:
            fit_ln_d2 = FitResult(float('nan'), float('nan'), float('nan'), float('nan'), int(np.sum(fit_mask_d2)))

        fit_start = float(e[fit_mask_i].min())
        fit_end = float(e[fit_mask_i].max())
        warnings.append(f"Manual Tafel window: E={fit_start:.6g} to {fit_end:.6g} (n={n_i})")
    else:

        # Auto window selection based on ln(i) and ln(d1) consistency, limited
        # to the user-specified range and optional Emax(D2) upper bound.
        scan_mask = (
            (e >= fit_low) & (e <= fit_high)
            & np.isfinite(ln_i) & np.isfinite(ln_d1) & np.isfinite(ln_d2)
        )
        e_scan = e[scan_mask]
        ln_i_scan = ln_i[scan_mask]
        ln_d1_scan = ln_d1[scan_mask]
        ln_d2_scan = ln_d2[scan_mask]
        if len(e_scan) >= 12:
            best, (mp, xp) = auto_select_tafel_window(e_scan, ln_i_scan, ln_d1_scan)
        else:
            best, (mp, xp) = None, (0, 0)
        warnings.append(f"Auto-window scan: min_points={mp}, max_points={xp}")

        if best is None:
            raise ValueError(
                "No automatic Tafel window satisfied the linearity and derivative-consistency criteria. "
                "Select and confirm a fitting window manually."
            )
        else:
            _, (s, t), fit_ln_i, fit_ln_d1 = best
            # same window for d2 reporting
            fit_ln_d2 = _linear_fit(e_scan[s:t], ln_d2_scan[s:t])
            fit_start = float(e_scan[s])
            fit_end = float(e_scan[t - 1])
            warnings.append(f"Selected Tafel window: E={fit_start:.6g} to {fit_end:.6g} (n={t-s})")

    # A flat semi-log fit has no finite Tafel slope.  Stop with an actionable
    # message instead of exporting inf.
    if (not np.isfinite(fit_ln_i.slope)) or abs(fit_ln_i.slope) <= 1e-12:
        raise ValueError(
            "The selected window has an approximately zero semi-log slope; "
            "a finite Tafel slope cannot be calculated. Select another window."
        )

    # alpha and tafels
    f = F_CONST / (R_CONST * temperature_K)  # 1/V
    # Magnitudes are reported because HER has a negative dln|i|/dE slope on
    # an ascending potential axis.  These are apparent alpha values (n=1).
    alpha_from_i = abs(fit_ln_i.slope) / f
    alpha_from_d1 = abs(fit_ln_d1.slope) / f
    alpha_from_d2 = abs(fit_ln_d2.slope) / f

    taf_i = _tafel_slope_mVdec_from_alpha(alpha_from_i, temperature_K)
    taf_d1 = _tafel_slope_mVdec_from_alpha(alpha_from_d1, temperature_K)
    taf_d2 = _tafel_slope_mVdec_from_alpha(alpha_from_d2, temperature_K)

    # consistency scores (slope mismatch / uncertainty)
    def score(a: FitResult, b: FitResult) -> float:
        denom = max(a.stderr_slope, b.stderr_slope, 1e-12)
        return abs(a.slope - b.slope) / denom

    score_i_d1 = score(fit_ln_i, fit_ln_d1)
    score_i_d2 = score(fit_ln_i, fit_ln_d2)
    score_d1_d2 = score(fit_ln_d1, fit_ln_d2)

    # diagnostics
    if fit_ln_i.r2 < 0.98:
        warnings.append(f"ln(i_norm) fit R² is low ({fit_ln_i.r2:.3f}).")
    if fit_ln_d1.r2 < 0.98:
        warnings.append(f"ln(d1_norm) fit R² is low ({fit_ln_d1.r2:.3f}).")
    # d2 often noisier → softer threshold + warning
    if fit_ln_d2.r2 < 0.95:
        warnings.append(f"ln(d2_norm) fit R² is low ({fit_ln_d2.r2:.3f}); d2 may be too noisy (d2 can be noisy).")

    if score_i_d1 > 3.0:
        warnings.append("Slope mismatch between ln(i) and ln(d1) is large; Tafel region may be unreliable.")
    if score_i_d2 > 5.0:
        warnings.append("Slope mismatch involving d2 is large; interpret d2-based values with caution.")

    d2_noise = float(np.nanstd(d2_norm))
    if d2_noise > 0.25:
        warnings.append("Derivative noise is relatively high; consider increasing SG window or improving data quality.")

    return DTAResult(
        e=e,
        i_raw=i,
        i_smooth=i_smooth,
        i_norm=i_norm,
        d1=d1,
        d2=d2,
        d1_norm=d1_norm,
        d2_norm=d2_norm,
        emax_d2=emax_d2,
        fit_ln_i=fit_ln_i,
        fit_ln_d1=fit_ln_d1,
        fit_ln_d2=fit_ln_d2,
        fit_start=fit_start,
        fit_end=fit_end,
        alpha_from_i=float(alpha_from_i),
        alpha_from_d1=float(alpha_from_d1),
        alpha_from_d2=float(alpha_from_d2),
        tafelslope_from_i_mVdec=float(taf_i),
        tafelslope_from_d1_mVdec=float(taf_d1),
        tafelslope_from_d2_mVdec=float(taf_d2),
        consistency_score_i_d1=float(score_i_d1),
        consistency_score_i_d2=float(score_i_d2),
        consistency_score_d1_d2=float(score_d1_d2),
        warnings=warnings
    )



# ----------------------------
# Output helpers (no overwrite)
# ----------------------------
def _unique_path(path: str) -> str:
    """If path exists, append _1, _2, ... before extension."""
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path
    k = 1
    while True:
        p2 = f"{base}_{k}{ext}"
        if not os.path.exists(p2):
            return p2
        k += 1


def _make_unique_run_dir(root_outdir: str) -> str:
    """Create outputs under root_outdir/run_### to avoid overwriting previous runs."""
    os.makedirs(root_outdir, exist_ok=True)
    for k in range(1, 10000):
        run_dir = os.path.join(root_outdir, f"run_{k:03d}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
            return run_dir
    # fallback (should never happen)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root_outdir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir



def _get_desktop_dir() -> str:
    """Best-effort Desktop path (Windows/macOS/Linux)."""
    home = os.path.expanduser("~")
    cand = os.path.join(home, "Desktop")
    if os.path.isdir(cand):
        return cand
    return home


def _default_root_outdir_for_tafel(base_name: str) -> str:
    """Desktop/Researcher Output/Tafel Analysis/<base_name>"""
    desktop = _get_desktop_dir()
    return os.path.join(desktop, "Researcher Output", "Tafel Analysis", base_name)

def _next_run_dir(root_out: str) -> str:
    """Create and return next run_### directory inside root_out."""
    os.makedirs(root_out, exist_ok=True)
    for k in range(1, 10000):
        run_dir = os.path.join(root_out, f"run_{k:03d}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir, exist_ok=True)
            return run_dir
    # Fallback
    run_dir = os.path.join(root_out, f"run_{99999:05d}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir



def export_processed_data_csv(
    res: DTAResult,
    out_csv: str,
    i_abs: np.ndarray | None = None,
    j_density: np.ndarray | None = None,
    potential: Optional[PotentialConversion] = None,
    reaction_mode: str = "OER",
) -> str:
    """Export cleaned + derived arrays for transparency / re-use.

    Notes:
      - res.i_raw is whatever current-like column the user selected.
      - If the source file also contains absolute current and/or current density, pass them
        via i_abs / j_density to export them alongside the normalized arrays.
    """
    data = {
        "E": res.e,
        "I_raw_selected": res.i_raw,
        "I_smooth_selected": res.i_smooth,
        "I_norm": res.i_norm,
        "d1": res.d1,
        "d2": res.d2,
        "d1_norm": res.d1_norm,
        "d2_norm": res.d2_norm,
        "ln_i_norm": np.log(res.i_norm),
        "ln_d1_norm": np.log(res.d1_norm),
        "ln_d2_norm": np.log(res.d2_norm),
        "log10_i_norm": np.log10(res.i_norm),
        "instantaneous_tafel_slope_mVdec": instantaneous_tafel_slope_mVdec(res, reaction_mode),
        "in_selected_tafel_window": (
            (res.e >= res.fit_start) & (res.e <= res.fit_end)
        ).astype(int),
    }

    if potential is not None:
        data["E_RHE_V"] = potential.e_rhe
        data["eta_signed_V"] = potential.eta_signed
        data["eta_reaction_direction_V"] = potential.eta_reaction

    if i_abs is not None:
        data["I_abs_from_file"] = i_abs
    if j_density is not None:
        data["J_density_from_file"] = j_density

    df = pd.DataFrame(data)
    out_csv = _unique_path(out_csv)
    df.to_csv(out_csv, index=False)
    return out_csv


def export_tafel_numbers_csv(
    res: DTAResult,
    out_csv: str,
    potential: Optional[PotentialConversion] = None,
    reaction_mode: str = "OER",
) -> str:
    """Export compact numeric table of Tafel-related values."""
    out_csv = _unique_path(out_csv)
    row: Dict[str, Any] = {
        "fit_start": res.fit_start,
        "fit_end": res.fit_end,
        "Emax_D2": res.emax_d2,
        "slope_ln_i": res.fit_ln_i.slope,
        "stderr_slope_ln_i": res.fit_ln_i.stderr_slope,
        "r2_ln_i": res.fit_ln_i.r2,
        "slope_ln_d1": res.fit_ln_d1.slope,
        "stderr_slope_ln_d1": res.fit_ln_d1.stderr_slope,
        "r2_ln_d1": res.fit_ln_d1.r2,
        "slope_ln_d2": res.fit_ln_d2.slope,
        "stderr_slope_ln_d2": res.fit_ln_d2.stderr_slope,
        "r2_ln_d2": res.fit_ln_d2.r2,
        "alpha_from_i": res.alpha_from_i,
        "alpha_from_d1": res.alpha_from_d1,
        "alpha_from_d2": res.alpha_from_d2,
        "tafelslope_mVdec_from_i": res.tafelslope_from_i_mVdec,
        "tafelslope_mVdec_from_d1": res.tafelslope_from_d1_mVdec,
        "tafelslope_mVdec_from_d2": res.tafelslope_from_d2_mVdec,
        "consistency_i_d1": res.consistency_score_i_d1,
        "consistency_i_d2": res.consistency_score_i_d2,
        "consistency_d1_d2": res.consistency_score_d1_d2,
        "d1_diagnostic_valid": bool(
            np.isfinite(res.fit_ln_d1.r2)
            and res.fit_ln_d1.r2 >= 0.98
            and res.consistency_score_i_d1 <= 3.0
        ),
        "d2_diagnostic_valid": bool(
            np.isfinite(res.fit_ln_d2.r2)
            and res.fit_ln_d2.r2 >= 0.95
            and res.consistency_score_i_d2 <= 5.0
        ),
        "warnings": " | ".join(res.warnings),
    }
    row.update(instantaneous_tafel_summary(res, reaction_mode))
    if potential is not None:
        row.update({
            "input_reference": potential.reference_label,
            "reference_offset_vs_SHE_V": potential.reference_offset_vs_she_V,
            "pH": potential.pH,
            "nernst_pH_term_V": potential.nernst_pH_term_V,
            "measured_to_RHE_shift_V": potential.measured_to_rhe_shift_V,
            "equilibrium_potential_RHE_V": potential.equilibrium_potential_rhe_V,
        })
        row.update(potential.window_summary(res))
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    return out_csv


def export_tafel_fit_data_csv(
    res: DTAResult,
    out_csv: str,
    potential: Optional[PotentialConversion] = None,
    reaction_mode: str = "OER",
) -> str:
    """Export the *data points used for the Tafel fit* + fitted line.

    Saves the subset within [fit_start, fit_end] for:
      - ln(i_norm) vs E (the actual regression performed)
      - log10(i_norm) derived from ln for convenience (typical Tafel plot)

    Columns:
      E, i_norm, ln_i_norm, ln_fit, ln_residual, log10_i_norm, log10_fit, log10_residual
    """
    mask = (res.e >= res.fit_start) & (res.e <= res.fit_end)
    e_fit = res.e[mask]
    i_fit = res.i_norm[mask]
    ln_i = np.log(i_fit)

    # Regression is ln_i = m*E + b
    m_ln = res.fit_ln_i.slope
    b_ln = res.fit_ln_i.intercept
    ln_pred = m_ln * e_fit + b_ln
    ln_res = ln_i - ln_pred

    log10_i = ln_i / np.log(10.0)
    log10_pred = ln_pred / np.log(10.0)
    log10_res = log10_i - log10_pred

    df = pd.DataFrame({
        "E": e_fit,
        "i_norm": i_fit,
        "ln_i_norm": ln_i,
        "ln_fit": ln_pred,
        "ln_residual": ln_res,
        "log10_i_norm": log10_i,
        "log10_fit": log10_pred,
        "log10_residual": log10_res,
        "instantaneous_tafel_slope_mVdec": instantaneous_tafel_slope_mVdec(
            res, reaction_mode
        )[mask],
    })
    if potential is not None:
        df["E_RHE_V"] = potential.e_rhe[mask]
        df["eta_signed_V"] = potential.eta_signed[mask]
        df["eta_reaction_direction_V"] = potential.eta_reaction[mask]
    out_csv = _unique_path(out_csv)
    df.to_csv(out_csv, index=False)
    return out_csv



def export_tafel_window_data_csv(
    res: DTAResult,
    out_csv: str,
    potential: Optional[PotentialConversion] = None,
    reaction_mode: str = "OER",
) -> str:
    """Export the *entire Tafel-window* data (not only regression variables).

    This is the subset of the processed arrays within [fit_start, fit_end].
    It is useful if you want to inspect/plot only the chosen region later.

    Columns (as available):
      E, I_raw, I_smooth, I_norm, d1, d2, d1_norm, d2_norm,
      ln_i_norm, log10_i_norm
    """
    mask = (res.e >= res.fit_start) & (res.e <= res.fit_end)
    e = res.e[mask]
    i_raw = res.i_raw[mask]
    i_s = res.i_smooth[mask]
    i_n = res.i_norm[mask]
    d1 = res.d1[mask]
    d2 = res.d2[mask]
    d1n = res.d1_norm[mask]
    d2n = res.d2_norm[mask]

    # Use safe logs (i_norm in the chosen region should be >0 by construction)
    ln_i = np.log(i_n)
    log10_i = ln_i / np.log(10.0)

    df = pd.DataFrame({
        "E": e,
        "I_raw": i_raw,
        "I_smooth": i_s,
        "i_norm": i_n,
        "d1": d1,
        "d2": d2,
        "d1_norm": d1n,
        "d2_norm": d2n,
        "ln_i_norm": ln_i,
        "log10_i_norm": log10_i,
        "instantaneous_tafel_slope_mVdec": instantaneous_tafel_slope_mVdec(
            res, reaction_mode
        )[mask],
    })
    if potential is not None:
        df["E_RHE_V"] = potential.e_rhe[mask]
        df["eta_signed_V"] = potential.eta_signed[mask]
        df["eta_reaction_direction_V"] = potential.eta_reaction[mask]
    out_csv = _unique_path(out_csv)
    df.to_csv(out_csv, index=False)
    return out_csv


def write_figure_descriptions(out_txt: str, meta: Dict[str, Any], res: DTAResult) -> str:
    """Write detailed, human-readable explanations for each exported figure."""
    out_txt = _unique_path(out_txt)
    mode = meta.get("mode", "")
    # Common notes based on CHI sign convention etc.
    sign_note = ""
    if meta.get("chi_negative_oxidation", False):
        sign_note = ("NOTE: CHI instruments can display oxidation current as negative. "
                     "Interpret absolute trends and use overpotential/current direction carefully.\n\n")

    txt = []
    txt.append("FIGURE EXPLANATIONS")
    txt.append("="*70)
    txt.append(f"Source file: {os.path.basename(meta.get('source_file',''))}")
    txt.append(f"Mode: {mode}")
    txt.append("")
    if sign_note:
        txt.append(sign_note.strip())

    # Describe each plot saved by make_all_plots
    txt.append("Fig 0 — Dual-panel DTA figure (optional)")
    txt.append("Top panel shows normalized current i = I/Imax (relative intensity) together with the normalized first derivative d1 () versus E. "
               "Bottom panel shows the same quantities on a semilog y-axis to highlight exponential (Tafel-like) regions and deviations due to mass transport or background currents. "
               "Custom annotations are only added if enabled by the user, keeping the tool general.")
    txt.append("")

    txt.append("Fig 1 — Raw vs Smoothed current")
    txt.append("Shows the original current (I_raw) and the Savitzky–Golay smoothed current (I_smooth) versus potential E. "
               "Smoothing reduces noise before numerical differentiation. If the smoothed curve distorts peaks/edges, reduce SG window "
               "or inspect the raw data quality.")
    txt.append("")

    txt.append("Fig 2 — Normalized current and derivatives")
    txt.append("Shows i_norm (current normalized to a positive scale) and the first/second derivatives (d1, d2) and their normalized "
               "versions (d1_norm, d2_norm). d1 reflects the local slope of i(E) and d2 highlights curvature/inflection behavior; "
               "d2 is sensitive to noise, so SG parameters strongly affect it.")
    txt.append("")

    txt.append("Fig 3 — d2 marker (Emax_D2)")
    txt.append(f"Marks Emax_D2 = {res.emax_d2:.6g} V, where the normalized second derivative reaches its maximum after excluding the endpoint regions. "
               "This diagnostic curvature marker can optionally limit candidate Tafel windows but does not replace user review.")
    txt.append("")

    txt.append("Fig 4 — Tafel plot with selected window and fit")
    txt.append(f"Plots log10(i_norm) versus E and overlays the user-confirmed Tafel window "
               f"[{res.fit_start:.6g}, {res.fit_end:.6g}] V (highlighted). A linear fit is performed in ln(i_norm) vs E, "
               "then reported as a Tafel slope in mV/dec. If R² is low or the window is too small, manually adjust the window or "
               "improve data preprocessing.")
    txt.append("")

    txt.append("Fig 5 — Residuals in the Tafel window")
    txt.append("Shows residuals of the Tafel linear fit inside the chosen window. Random, structureless residuals indicate a good "
               "linear regime; systematic curvature suggests mixed kinetics, ohmic drop issues, or a wrong window.")
    txt.append("")

    txt.append("Saved CSV outputs")
    txt.append("- processed_data.csv: full processed arrays across the full scan.")
    txt.append("- tafel_numbers.csv: summary metrics (slopes, R², window limits, consistency scores).")
    txt.append("- tafel_fit_data.csv: only points used for the linear fit + fitted line + residuals.")
    txt.append("- tafel_window_data.csv: the entire chosen window subset (E, currents, derivatives, logs).")
    txt.append("")

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(textwrap.fill(p, width=110) if (len(p) > 110 and not p.startswith("-")) else p for p in txt))
        f.write("\n")

# ----------------------------
# Offline report (no API) — rule-based diagnostics
# ----------------------------
def _infer_scan_rate_from_filename(fname: str) -> Optional[float]:
    """Try to infer scan rate (V/s) from filename tokens like RATE0.05."""
    try:
        m = re.search(r"RATE\s*([0-9]*\.?[0-9]+)", fname, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def offline_review(meta: dict, res) -> dict:
    """Generate a rule-based (transparent) interpretation and recommendations."""
    fname = os.path.basename(meta.get("source_file", ""))
    scan_rate = _infer_scan_rate_from_filename(fname)

    slope = float(getattr(res, "tafelslope_from_i_mVdec", float("nan")))
    r2_i = float(getattr(res.fit_ln_i, "r2", float("nan")))
    r2_d1 = float(getattr(res.fit_ln_d1, "r2", float("nan")))
    r2_d2 = float(getattr(res.fit_ln_d2, "r2", float("nan")))

    flags = []
    recs = []

    # Slope-based heuristics
    if not math.isnan(slope):
        if slope > 180:
            flags.append("Very high Tafel slope (>180 mV/dec): likely dominated by non-kinetic artifacts (iR drop, mass transport, bubbles)." )
        elif slope > 120:
            flags.append("High Tafel slope (>120 mV/dec): non-kinetic effects likely (iR drop / transport / bubble masking)." )
        elif slope > 80:
            flags.append("Moderate Tafel slope (80–120 mV/dec): could be mixed control; verify iR-compensation and window selection." )
        else:
            flags.append("Tafel slope within typical kinetic range (<80 mV/dec)." )

    # Fit quality
    if not math.isnan(r2_i) and r2_i < 0.95:
        flags.append(f"Lower linearity in selected window (R² for ln(i) fit = {r2_i:.3f}). Consider adjusting the window.")
        recs.append("Move the Tafel window to a region with clearer log-linear behavior (activation-controlled).")

    # d2 instability is often a red flag for regime change
    if not math.isnan(r2_d2) and r2_d2 < 0.50:
        flags.append(f"Second-derivative fit is unstable/low R² (ln(d2) R² = {r2_d2:.3f}). This often indicates regime transition (transport/bubbles/noise)." )
        recs.append("Shift the window slightly to lower overpotential and/or improve smoothing/replicates.")

    # Scan rate
    if scan_rate is not None:
        if scan_rate >= 0.02:
            flags.append(f"Scan rate appears high (RATE≈{scan_rate:g} V/s). Capacitive current may inflate slopes.")
            recs.append("For Tafel extraction, prefer quasi-steady scans (e.g., 1–5 mV/s) or hold steps.")

    # Practical recommendations
    recs.append("Apply iR compensation (85–100% of Rs) if available (from EIS or high-frequency intercept)." )
    recs.append("Report reference electrode conversion (Ag/AgCl→RHE) and sign convention (CHI often shows anodic current negative)." )
    recs.append("Include Tafel window bounds and the number of points used in the fit.")

    # Compose text
    lines = []
    lines.append("OFFLINE REPORT (no API) — rule-based diagnostics")
    lines.append("="*60)
    lines.append(f"File: {fname}")
    lines.append(f"Mode: {meta.get('mode','')}")
    if scan_rate is not None:
        lines.append(f"Inferred scan rate: {scan_rate:g} V/s")
    lines.append("")
    lines.append("Key results")
    lines.append(f"- Tafel slope (from ln(i_norm) fit): {slope:.2f} mV/dec")
    lines.append(f"- Tafel window: [{res.fit_start:.4g}, {res.fit_end:.4g}] V")
    lines.append(f"- Emax(D2): {res.emax_d2:.4g} V")
    lines.append(f"- Fit R²: ln(i)={r2_i:.4f}, ln(d1)={r2_d1:.4f}, ln(d2)={r2_d2:.4f}")
    lines.append("")
    lines.append("Flags / interpretation")
    for fl in flags:
        lines.append(f"- {fl}")
    lines.append("")
    lines.append("Recommendations")
    for r in recs:
        lines.append(f"- {r}")
    text_out = "\n".join(lines)

    return {
        "file": fname,
        "mode": meta.get("mode", ""),
        "scan_rate_Vs": scan_rate,
        "tafelslope_mVdec": slope,
        "fit_start_V": float(res.fit_start),
        "fit_end_V": float(res.fit_end),
        "emax_d2_V": float(res.emax_d2),
        "r2_ln_i": r2_i,
        "r2_ln_d1": r2_d1,
        "r2_ln_d2": r2_d2,
        "flags": flags,
        "recommendations": recs,
        "text": text_out,
    }


def write_offline_report(run_dir: str, base: str, mode: str, meta: dict, res) -> dict:
    """Write offline report to TXT/JSON/CSV and return paths."""
    rep = offline_review(meta, res)

    txt_path = _unique_path(os.path.join(run_dir, f"{base}_OFFLINE_report_{mode}.txt"))
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(rep["text"].strip() + "\n")

    json_path = _unique_path(os.path.join(run_dir, f"{base}_OFFLINE_report_{mode}.json"))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in rep.items() if k != "text"}, f, ensure_ascii=False, indent=2)

    csv_path = _unique_path(os.path.join(run_dir, f"{base}_OFFLINE_summary_{mode}.csv"))
    import pandas as _pd
    _pd.DataFrame([{
        "file": rep["file"],
        "mode": rep["mode"],
        "scan_rate_Vs": rep["scan_rate_Vs"],
        "tafelslope_mVdec": rep["tafelslope_mVdec"],
        "fit_start_V": rep["fit_start_V"],
        "fit_end_V": rep["fit_end_V"],
        "emax_d2_V": rep["emax_d2_V"],
        "r2_ln_i": rep["r2_ln_i"],
        "r2_ln_d1": rep["r2_ln_d1"],
        "r2_ln_d2": rep["r2_ln_d2"],
        "n_flags": len(rep["flags"]),
    }]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    return {"offline_txt": txt_path, "offline_json": json_path, "offline_csv": csv_path}
# offline report saved


# ----------------------------
# Plotting + PDF report
# ----------------------------
def _save_plot(path: str, fig):
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_plot_data_csv(csv_path: str, data: dict):
    """Save numeric data used in a plot to CSV (one column per series)."""
    try:
        import pandas as _pd
        _pd.DataFrame(data).to_csv(csv_path, index=False, encoding="utf-8-sig")
    except Exception as _e:
        print(f"[WARN] Could not save plot data CSV: {csv_path} -> {_e}")



def make_corrosion_plots(res: DTAResult, outdir: str, title_prefix: str) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    os.makedirs(outdir, exist_ok=True)

    # Corrosion plot (E vs log10|I|)
    fig = plt.figure()
    i_use = res.i_smooth
    e_use = res.e
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(np.abs(i_use))
    y = e_use
    plt.scatter(x, y, s=12, label="Data")

    if (res.corr is not None) and res.corr.has_both_branches and (res.corr.fit_anodic is not None) and (res.corr.fit_cathodic is not None):
        xfin = x[np.isfinite(x)]
        if len(xfin) > 0:
            xa = np.linspace(np.nanmin(xfin), np.nanmax(xfin), 200)
            plt.plot(xa, res.corr.fit_anodic.slope * xa + res.corr.fit_anodic.intercept, linestyle="--", label="Anodic fit")
            plt.plot(xa, res.corr.fit_cathodic.slope * xa + res.corr.fit_cathodic.intercept, linestyle="--", label="Cathodic fit")
            if np.isfinite(res.corr.Icorr) and res.corr.Icorr > 0 and np.isfinite(res.corr.Ecorr):
                logIcorr = math.log10(res.corr.Icorr)
                plt.axhline(res.corr.Ecorr, linewidth=1.0)
                plt.axvline(logIcorr, linewidth=1.0)
                plt.scatter(logIcorr, res.corr.Ecorr, marker="o", label="Ecorr / Icorr")
                plt.annotate(
                    f"Ecorr={res.corr.Ecorr:.3g} V\nIcorr={res.corr.Icorr:.3g} A",
                    xy=(logIcorr, res.corr.Ecorr),
                    xytext=(logIcorr + 0.1, res.corr.Ecorr + 0.05),
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=9
                )

    plt.xlabel("Current Density, log10|I|")
    plt.ylabel("Potential, E")
    plt.title(f"{title_prefix} | Corrosion (E vs log|I|)")
    plt.legend()

    p = os.path.join(outdir, "01_corrosion_E_logI.png")
    _save_plot(p, fig)
    paths["corrosion_plot"] = p

    _save_plot_data_csv(os.path.join(outdir, "01_corrosion_E_logI_data.csv"), {
        "log10_absI": x,
        "E": y,
    })
    return paths


def make_all_plots(
    res: DTAResult,
    outdir: str,
    title_prefix: str,
    reaction_mode: str = "OER",
    export_dual_panel: bool = True,
    dual_panel_xlabel: str = "E (V) vs. RHE",
    dual_panel_annotations: Optional[List[Dict[str, Any]]] = None,
    potential: Optional[PotentialConversion] = None,
) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    paths: Dict[str, str] = {}
    # Instrument filenames can be extremely long.  Keep figures readable while
    # retaining the full source name in the PDF report and numerical metadata.
    if len(title_prefix) > 44:
        title_prefix = title_prefix[:41] + "..."

    # 0) Dual-panel DTA figure (optional; uses a dual-panel layout (linear + log-scale))
    #    Top: linear relative intensities; Bottom: semilog y-scale (ln-scale visualization)
    if export_dual_panel:
        fig = plt.figure(figsize=(6.2, 7.0))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(res.e, res.i_norm, label="Current")
        ax1.plot(res.e, res.d1_norm, label="1st derivative")
        ax1.set_ylabel("Rel. Intensity (vs. max)")
        ax1.set_title(f"{title_prefix} | Dual-panel DTA")
        ax1.legend(loc="best")
        ax1.tick_params(labelbottom=False)

        # Optional annotations (user-defined; keep tool general)
        if dual_panel_annotations:
            for ann in dual_panel_annotations:
                try:
                    text = str(ann.get("text", "")).strip()
                    if not text:
                        continue
                    x = float(ann.get("x"))
                    y = float(ann.get("y"))
                    ax1.text(x, y, text, fontsize=10)
                except Exception:
                    # ignore malformed annotations
                    pass

        ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
        ax2.plot(res.e, res.i_norm, label="Current")
        ax2.plot(res.e, res.d1_norm, label="1st derivative")
        ax2.set_yscale("log")
        ax2.set_ylabel("Rel. Intensity – ln scale")
        ax2.set_xlabel(dual_panel_xlabel)
        ax2.legend(loc="best")

        p = os.path.join(outdir, "00_dual_panel_dta.png")
        _save_plot(p, fig)
        paths["dual_panel"] = p
        _save_plot_data_csv(os.path.join(outdir, "00_dual_panel_data.csv"), {
            "E": res.e,
            "i_norm": res.i_norm,
            "d1_norm": res.d1_norm,
        })


    # 1) Raw + Smoothed
    fig = plt.figure()
    plt.plot(res.e, res.i_raw, label="Raw I")
    plt.plot(res.e, res.i_smooth, label="Smoothed I")
    plt.xlabel("Potential, E (V)")
    plt.ylabel("Current (I)")
    plt.title(f"{title_prefix} | Raw vs Smoothed")
    plt.legend()
    p = os.path.join(outdir, "01_raw_vs_smooth.png")
    _save_plot(p, fig)
    paths["raw_smooth"] = p
    _save_plot_data_csv(os.path.join(outdir, "01_raw_vs_smooth_data.csv"), {
        "E": res.e,
        "I_raw": res.i_raw,
        "I_smooth": res.i_smooth,
    })

    # 2) i_norm
    fig = plt.figure()
    plt.plot(res.e, res.i_norm)
    plt.xlabel("Potential, E (V)")
    plt.ylabel("i_norm = I/Imax")
    plt.title(f"{title_prefix} | Normalized current")
    p = os.path.join(outdir, "02_i_norm.png")
    _save_plot(p, fig)
    paths["i_norm"] = p
    _save_plot_data_csv(os.path.join(outdir, "02_i_norm_data.csv"), {
        "E": res.e,
        "i_norm": res.i_norm,
    })

    # 3) d1_norm & d2_norm with markers
    fig = plt.figure()
    plt.plot(res.e, res.d1_norm, label="d1_norm")
    plt.plot(res.e, res.d2_norm, label="d2_norm")
    plt.axvline(res.emax_d2, linestyle="--", label="Emax_D2")
    plt.axvspan(res.fit_start, res.fit_end, alpha=0.2, label="Selected Tafel window")
    plt.xlabel("Potential, E (V)")
    plt.ylabel("Normalized derivatives")
    plt.title(f"{title_prefix} | D1/D2 (normalized)")
    plt.legend()
    p = os.path.join(outdir, "03_d1_d2_norm.png")
    _save_plot(p, fig)
    paths["d1d2"] = p
    _save_plot_data_csv(os.path.join(outdir, "03_d1_d2_norm_data.csv"), {
        "E": res.e,
        "d1_norm": res.d1_norm,
        "d2_norm": res.d2_norm,
        "in_tafel_window": ((res.e >= res.fit_start) & (res.e <= res.fit_end)).astype(int),
        "is_emax_d2": np.isclose(res.e, res.emax_d2, atol=1e-12).astype(int),
    })

    # 4) ln(i_norm) vs E + fit
    fig = plt.figure()
    ln_i = np.log(res.i_norm)
    plt.plot(res.e, ln_i, label="ln(i_norm)")
    yfit = res.fit_ln_i.slope * res.e + res.fit_ln_i.intercept
    plt.plot(res.e, yfit, label=f"Fit | slope={res.fit_ln_i.slope:.4g}, R²={res.fit_ln_i.r2:.4f}")
    plt.axvspan(res.fit_start, res.fit_end, alpha=0.2, label="Selected Tafel window")
    plt.xlabel("Potential, E (V)")
    plt.ylabel("ln(i_norm)")
    plt.title(f"{title_prefix} | ln(i_norm) vs E")
    plt.legend()
    p = os.path.join(outdir, "04_ln_i_fit.png")
    _save_plot(p, fig)
    paths["ln_i_fit"] = p
    _save_plot_data_csv(os.path.join(outdir, "04_ln_i_fit_data.csv"), {
        "E": res.e,
        "ln_i_norm": ln_i,
        "ln_fit": (res.fit_ln_i.slope * res.e + res.fit_ln_i.intercept),
        "in_tafel_window": ((res.e >= res.fit_start) & (res.e <= res.fit_end)).astype(int),
    })

    # 5) ln(d1_norm) vs E + fit
    fig = plt.figure()
    ln_d1 = np.log(res.d1_norm)
    plt.plot(res.e, ln_d1, label="ln(d1_norm)")
    yfit = res.fit_ln_d1.slope * res.e + res.fit_ln_d1.intercept
    plt.plot(res.e, yfit, label=f"Fit | slope={res.fit_ln_d1.slope:.4g}, R²={res.fit_ln_d1.r2:.4f}")
    plt.axvspan(res.fit_start, res.fit_end, alpha=0.2, label="Selected Tafel window")
    plt.xlabel("Potential, E (V)")
    plt.ylabel("ln(d1_norm)")
    plt.title(f"{title_prefix} | ln(dI/dE) vs E")
    plt.legend()
    p = os.path.join(outdir, "05_ln_d1_fit.png")
    _save_plot(p, fig)
    paths["ln_d1_fit"] = p
    _save_plot_data_csv(os.path.join(outdir, "05_ln_d1_fit_data.csv"), {
        "E": res.e,
        "ln_d1_norm": ln_d1,
        "ln_fit": (res.fit_ln_d1.slope * res.e + res.fit_ln_d1.intercept),
        "in_tafel_window": ((res.e >= res.fit_start) & (res.e <= res.fit_end)).astype(int),
    })

    # 6) ln(d2_norm) vs E + fit (2nd derivative)
    fig = plt.figure()
    ln_d2 = np.log(res.d2_norm)
    plt.plot(res.e, ln_d2, label="ln(d2_norm)")
    yfit = res.fit_ln_d2.slope * res.e + res.fit_ln_d2.intercept
    plt.plot(res.e, yfit, label=f"Fit | slope={res.fit_ln_d2.slope:.4g}, R²={res.fit_ln_d2.r2:.4f}")
    plt.axvspan(res.fit_start, res.fit_end, alpha=0.2, label="Selected Tafel window")
    plt.xlabel("Potential, E (V)")
    plt.ylabel("ln(d2_norm)")
    plt.title(f"{title_prefix} | ln(d²I/dE²) vs E")
    plt.legend()
    p = os.path.join(outdir, "06_ln_d2_fit.png")
    _save_plot(p, fig)
    paths["ln_d2_fit"] = p
    _save_plot_data_csv(os.path.join(outdir, "06_ln_d2_fit_data.csv"), {
        "E": res.e,
        "ln_d2_norm": ln_d2,
        "ln_fit": (res.fit_ln_d2.slope * res.e + res.fit_ln_d2.intercept),
        "in_tafel_window": ((res.e >= res.fit_start) & (res.e <= res.fit_end)).astype(int),
    })

    # 7) Inverse Tafel representation: log10(i_norm) vs E.
    fig = plt.figure()
    log10_i = np.log10(res.i_norm)
    plt.plot(res.e, log10_i, label="Normalized current data")

    # Convert the ln-space regression exactly to log10-space.
    s_ln = res.fit_ln_i.slope
    b_ln = res.fit_ln_i.intercept
    ln10 = math.log(10.0)
    yfit_log10 = (s_ln / ln10) * res.e + (b_ln / ln10)

    # Plot only the fitted segment; extrapolating across the full scan can
    # visually imply that non-kinetic regions were included in the regression.
    wmask = (res.e >= res.fit_start) & (res.e <= res.fit_end)
    if np.any(wmask):
        plt.plot(
            res.e[wmask],
            yfit_log10[wmask],
            linewidth=3.0,
            label="Linear fit (selected window)",
        )

        # Annotate Tafel slope (mV/dec) on the plot in a fixed-position box (prevents overlap)
        taf_mVdec = res.tafelslope_from_i_mVdec
        label_txt = (
            f"Tafel slope = {taf_mVdec:.2f} mV/dec\n"
            f"R² = {res.fit_ln_i.r2:.4f}\n"
            f"Window: {res.fit_start:.4g} → {res.fit_end:.4g}"
        )
        if potential is not None:
            pwin = potential.window_summary(res)
            label_txt += (
                f"\nη window: {pwin['eta_window_low_V']:.4g} → "
                f"{pwin['eta_window_high_V']:.4g} V"
            )
        ax = plt.gca()
        ax.text(
            0.02, 0.98,
            label_txt,
            transform=ax.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
        )

        plt.axvspan(res.fit_start, res.fit_end, alpha=0.2, label="Selected Tafel window")

    plt.xlabel("Potential, E (V)")
    plt.ylabel(r"$\log_{10}(i_{\mathrm{norm}})$")
    plt.title(f"{title_prefix}\nInverse Tafel representation")

    # Mark Ecorr / Icorr on the Tafel plot (if available)
    try:
        if res is not None and (getattr(res, "corr", None) is not None):
            Ec = float(res.corr.Ecorr)
            Ic = float(res.corr.Icorr)
            if np.isfinite(Ec) and np.isfinite(Ic) and (Ic > 0):
                y_ec = math.log10(Ic)
                plt.scatter(Ec, y_ec, marker="o", label="Ecorr / Icorr")
                plt.annotate(
                    f"Ecorr={Ec:.3g} V\nIcorr={Ic:.3g} (norm)",
                    xy=(Ec, y_ec),
                    xytext=(Ec, y_ec + 0.25),
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=8
                )
    except Exception:
        pass

    plt.legend()
    p = os.path.join(outdir, "07_tafel_plot.png")
    _save_plot(p, fig)
    paths["tafel_plot"] = p
    tafel_plot_data = {
        "E": res.e,
        "log10_i_norm": log10_i,
        "log10_fit": yfit_log10,
        "Ecorr": np.full_like(res.e, res.corr.Ecorr if res.corr is not None else float("nan"), dtype=float),
        "Icorr_norm": np.full_like(res.e, res.corr.Icorr if res.corr is not None else float("nan"), dtype=float),
        "in_tafel_window": ((res.e >= res.fit_start) & (res.e <= res.fit_end)).astype(int),
    }
    if potential is not None:
        tafel_plot_data["E_RHE_V"] = potential.e_rhe
        tafel_plot_data["eta_signed_V"] = potential.eta_signed
        tafel_plot_data["eta_reaction_direction_V"] = potential.eta_reaction
    _save_plot_data_csv(os.path.join(outdir, "07_tafel_plot_data.csv"), tafel_plot_data)

    # 7b) Conventional Tafel orientation based on the imported potential:
    # E versus log10(i).  This preserves the earlier output requested by users
    # while keeping the regression line restricted to the confirmed window.
    fig = plt.figure()
    x_log = log10_i.copy()
    y_potential = res.e.copy()
    p_mask = np.isfinite(x_log) & np.isfinite(y_potential)
    plt.plot(x_log[p_mask], y_potential[p_mask], linewidth=1.2, label="Processed data")

    m = float(res.fit_ln_i.slope)
    c = float(res.fit_ln_i.intercept)
    wmask_p = (
        (res.e >= res.fit_start)
        & (res.e <= res.fit_end)
        & np.isfinite(x_log)
        & np.isfinite(y_potential)
    )
    if np.isfinite(m) and abs(m) > 1e-30 and np.any(wmask_p):
        xseg_p = np.linspace(np.nanmin(x_log[wmask_p]), np.nanmax(x_log[wmask_p]), 120)
        efit_p = (math.log(10.0) * xseg_p - c) / m
        plt.plot(
            x_log[wmask_p], y_potential[wmask_p],
            linewidth=3.0, alpha=0.65, label="Selected Tafel window"
        )
        plt.plot(xseg_p, efit_p, linewidth=2.4, label="Linear fit")
        ax_p = plt.gca()
        ax_p.text(
            0.02, 0.98,
            f"Tafel slope = {res.tafelslope_from_i_mVdec:.2f} mV/dec\n"
            f"R² = {res.fit_ln_i.r2:.5f}\n"
            f"E window: {res.fit_start:.4g} → {res.fit_end:.4g} V",
            transform=ax_p.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
        )

    plt.xlabel(r"$\log_{10}(i_{\mathrm{norm}})$")
    plt.ylabel("Potential, E (V)")
    plt.title(f"{title_prefix}\nTafel plot based on potential")
    plt.legend()
    p_potential = os.path.join(outdir, "07b_tafel_eta_logI.png")
    _save_plot(p_potential, fig)
    paths["tafel_potential"] = p_potential
    potential_plot_data = {
        "log10_i_norm": x_log,
        "potential_input_V": res.e,
        "in_tafel_window": wmask_p.astype(int),
    }
    if potential is not None:
        potential_plot_data["E_RHE_V"] = potential.e_rhe
        potential_plot_data["eta_signed_V"] = potential.eta_signed
        potential_plot_data["eta_reaction_direction_V"] = potential.eta_reaction
    _save_plot_data_csv(
        os.path.join(outdir, "07b_tafel_eta_logI_data.csv"),
        potential_plot_data,
    )
    
    # 7c) Publication-oriented Tafel representation.  Overpotential is used
    # only when the user has explicitly supplied the reference electrode and
    # pH conversion; otherwise the axis remains the imported potential.
    fig = plt.figure()
    with np.errstate(divide="ignore", invalid="ignore"):
        x_log = log10_i.copy()

    if potential is not None:
        y_axis = potential.eta_reaction.copy()
        if potential.reaction_mode == "HER":
            y_label = r"Reaction-direction overpotential, $|\eta_{\mathrm{HER}}|$ (V)"
        else:
            y_label = r"Overpotential, $\eta_{\mathrm{OER}}$ (V)"
        plot_title = f"{title_prefix}\nTafel plot based on overpotential"
    else:
        y_axis = res.e.copy()
        y_label = "Potential, E (V)"
        plot_title = f"{title_prefix}\nPotential vs normalized current"

    # Fit line from ln(i)=mE+c, converted only over the selected Tafel window.
    m = float(res.fit_ln_i.slope)
    c = float(res.fit_ln_i.intercept)
    wmask2 = (
        (res.e >= res.fit_start)
        & (res.e <= res.fit_end)
        & np.isfinite(x_log)
        & np.isfinite(y_axis)
    )
    if np.any(wmask2):
        y_low = float(np.nanmin(y_axis[wmask2]))
        y_high = float(np.nanmax(y_axis[wmask2]))
        y_pad = max(0.05, 0.20 * max(y_high - y_low, 1e-12))
        context_mask = (
            np.isfinite(x_log) & np.isfinite(y_axis)
            & (y_axis >= y_low - y_pad) & (y_axis <= y_high + y_pad)
        )
    else:
        context_mask = np.isfinite(x_log) & np.isfinite(y_axis)
    plt.plot(x_log[context_mask], y_axis[context_mask], linewidth=1.2, label="Processed data")

    if np.isfinite(m) and abs(m) > 1e-30 and np.any(wmask2):
        xseg = np.linspace(np.nanmin(x_log[wmask2]), np.nanmax(x_log[wmask2]), 120)
        efit = (math.log(10.0) * xseg - c) / m
        if potential is not None:
            efit_rhe = efit + potential.measured_to_rhe_shift_V
            eta_signed_fit = efit_rhe - potential.equilibrium_potential_rhe_V
            yfit_segment = eta_signed_fit if potential.reaction_mode == "OER" else -eta_signed_fit
        else:
            yfit_segment = efit

        plt.plot(
            x_log[wmask2], y_axis[wmask2],
            linewidth=3.0, alpha=0.65, label="Selected Tafel window"
        )
        plt.plot(xseg, yfit_segment, linewidth=2.4, label="Linear fit")

        extra = ""
        if potential is not None:
            pwin = potential.window_summary(res)
            extra = (
                f"\nη window: {pwin['eta_window_low_V']:.4g} → "
                f"{pwin['eta_window_high_V']:.4g} V"
            )
        ax2 = plt.gca()
        ax2.text(
            0.02, 0.98,
            f"Tafel slope = {res.tafelslope_from_i_mVdec:.2f} mV/dec\n"
            f"R² = {res.fit_ln_i.r2:.5f}{extra}",
            transform=ax2.transAxes,
            va="top", ha="left",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
        )

    plt.xlabel(r"$\log_{10}(i_{\mathrm{norm}})$")
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    p2 = os.path.join(outdir, "07b_tafel_overpotential.png")
    _save_plot(p2, fig)
    paths["tafel_overpotential"] = p2

    overpotential_plot_data = {
        "log10_i_norm": x_log,
        "potential_input_V": res.e,
        "plot_y_value_V": y_axis,
        "in_tafel_window": wmask2.astype(int),
    }
    if potential is not None:
        overpotential_plot_data["E_RHE_V"] = potential.e_rhe
        overpotential_plot_data["eta_signed_V"] = potential.eta_signed
        overpotential_plot_data["eta_reaction_direction_V"] = potential.eta_reaction
    _save_plot_data_csv(
        os.path.join(outdir, "07b_tafel_overpotential_data.csv"),
        overpotential_plot_data,
    )

    # 7c) True pointwise (instantaneous) Tafel slope.  Only the confirmed
    # fitting window is displayed because derivatives outside that region are
    # not kinetic Tafel results.
    slope_mode = potential.reaction_mode if potential is not None else reaction_mode
    b_inst = instantaneous_tafel_slope_mVdec(res, slope_mode)
    inst_mask = wmask2 & np.isfinite(b_inst)
    if np.any(inst_mask):
        fig = plt.figure()
        x_inst = potential.eta_reaction if potential is not None else res.e
        plt.plot(x_inst[inst_mask], b_inst[inst_mask], linewidth=1.5, label="Instantaneous Tafel slope")
        inst = instantaneous_tafel_summary(
            res, slope_mode
        )
        plt.axhline(
            inst["instantaneous_median_mVdec"],
            linestyle="--",
            label=f"Median = {inst['instantaneous_median_mVdec']:.2f} mV/dec",
        )
        plt.axhspan(
            inst["instantaneous_p05_mVdec"],
            inst["instantaneous_p95_mVdec"],
            alpha=0.15,
            label="5th–95th percentile",
        )
        if potential is not None:
            if potential.reaction_mode == "HER":
                plt.xlabel(r"Reaction-direction overpotential, $|\eta_{\mathrm{HER}}|$ (V)")
            else:
                plt.xlabel(r"Overpotential, $\eta_{\mathrm{OER}}$ (V)")
        else:
            plt.xlabel("Potential, E (V)")
        plt.ylabel("Instantaneous Tafel slope (mV/dec)")
        plt.title(f"{title_prefix}\nInstantaneous Tafel-slope diagnostic")
        plt.legend()
        p3 = os.path.join(outdir, "07c_instantaneous_tafel_slope.png")
        _save_plot(p3, fig)
        paths["tafel_instantaneous"] = p3
        inst_plot_data = {
            "potential_input_V": res.e,
            "instantaneous_tafel_slope_mVdec": b_inst,
            "in_tafel_window": wmask2.astype(int),
        }
        if potential is not None:
            inst_plot_data["E_RHE_V"] = potential.e_rhe
            inst_plot_data["eta_signed_V"] = potential.eta_signed
            inst_plot_data["eta_reaction_direction_V"] = potential.eta_reaction
        _save_plot_data_csv(
            os.path.join(outdir, "07c_instantaneous_tafel_slope_data.csv"),
            inst_plot_data,
        )

    # OER/HER analyses do not have corrosion parameters; avoid exporting an
    # irrelevant blank Ecorr/Icorr figure in those reports.
    if res.corr is None:
        return paths

    # 8) Ecorr/Icorr plot (E vs log10|I|)
    fig = plt.figure()
    i_use = res.i_smooth
    e_use = res.e
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.log10(np.abs(i_use))
    y = e_use

    plt.scatter(x, y, s=12, label="Data")

    if (res.corr is not None) and res.corr.has_both_branches and (res.corr.fit_anodic is not None) and (res.corr.fit_cathodic is not None):
        xfin = x[np.isfinite(x)]
        if len(xfin) > 0:
            xa = np.linspace(np.nanmin(xfin), np.nanmax(xfin), 200)
            ya_fit = res.corr.fit_anodic.slope * xa + res.corr.fit_anodic.intercept
            yc_fit = res.corr.fit_cathodic.slope * xa + res.corr.fit_cathodic.intercept
            plt.plot(xa, ya_fit, linestyle="--", label="Anodic fit")
            plt.plot(xa, yc_fit, linestyle="--", label="Cathodic fit")

            logIcorr = math.log10(res.corr.Icorr) if (res.corr.Icorr and res.corr.Icorr > 0) else float("nan")
            if np.isfinite(logIcorr) and np.isfinite(res.corr.Ecorr):
                plt.axhline(res.corr.Ecorr, linewidth=1.0)
                plt.axvline(logIcorr, linewidth=1.0)
                plt.scatter(logIcorr, res.corr.Ecorr, marker="o", label="Ecorr / Icorr")
                plt.annotate(
                    f"Ecorr={res.corr.Ecorr:.3g} V\nIcorr={res.corr.Icorr:.3g} A",
                    xy=(logIcorr, res.corr.Ecorr),
                    xytext=(logIcorr + 0.1, res.corr.Ecorr + 0.05),
                    arrowprops=dict(arrowstyle="->"),
                    fontsize=9
                )

    plt.xlabel("Current Density, log10|I|")
    plt.ylabel("Potential, E")
    plt.title(f"{title_prefix} | Ecorr / Icorr (E vs log|I|)")
    plt.legend()
    p = os.path.join(outdir, "08_ecorr_icorr.png")
    _save_plot(p, fig)
    paths["corrosion_plot"] = p

    return paths


def write_pdf_report(
    pdf_path: str,
    source_file: str,
    meta: Dict[str, Any],
    res: DTAResult,
    plot_paths: Dict[str, str],
    reaction_mode: str = ""
):
    """Write a PDF report.
    - OER/HER: full multi-page DTA report (existing style)
    - CORROSION: compact corrosion-only report (summary + plot)
    """
    c = canvas.Canvas(pdf_path, pagesize=A4)
    w, h = A4

    def header():
        c.setFont("Helvetica-Bold", 14)
        c.drawString(2*cm, h - 2*cm, "Differential Tafel Analysis (DTA) Report")
        c.setFont("Helvetica", 10)
        c.drawString(2*cm, h - 2.7*cm, f"Source: {os.path.basename(source_file)}")
        c.drawString(
            2*cm, h - 3.2*cm,
            f"Mode: {meta.get('mode','')} | T = {meta.get('T','')} K | SG window={meta.get('sg_window','')} poly={meta.get('sg_poly','')}"
        )
        c.drawString(
            2*cm, h - 3.7*cm,
            f"Columns: E='{meta.get('col_e','')}' | I='{meta.get('col_i','')}' | Use log column={meta.get('use_log','')}"
        )
        c.drawString(
            2*cm, h - 4.15*cm,
            f"Reference: {meta.get('reference_label','Not converted')} | pH={meta.get('pH','')} | "
            f"E_RHE shift={meta.get('measured_to_rhe_shift_V','')} V"
        )
        c.line(2*cm, h - 4.45*cm, w - 2*cm, h - 4.45*cm)

    def draw_kv(y, key, val):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(2*cm, y, f"{key}:")
        c.setFont("Helvetica", 10)
        c.drawString(7.0*cm, y, f"{val}")

    # -------- Cover page (reference & software info) --------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(2*cm, h - 2.5*cm, "Report Information")
    c.setFont("Helvetica", 11)
    lines = [
        "Software: Researcher Analysis Toolkit — Tafel Module",
        f"Report generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source file: {os.path.basename(source_file)}",
        "",
        "Citation placeholder (to be completed after publication):",
        "Article XXXXX",
        "",
        "Note: This cover page is the only place where the citation placeholder appears.",
    ]
    y0 = h - 3.5*cm
    for ln in lines:
        c.drawString(2*cm, y0, ln)
        y0 -= 0.65*cm
    c.showPage()

    # ---------- CORROSION MODE ----------
    if reaction_mode.strip().upper().startswith("CORR"):
        # Page 1: summary
        header()
        y = h - 5.2*cm
        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, "Corrosion Analysis Summary")
        y -= 0.8*cm
        c.setFont("Helvetica", 10)

        ok = (
            (res.corr is not None)
            and getattr(res.corr, "has_both_branches", False)
            and np.isfinite(getattr(res.corr, "Ecorr", float("nan")))
            and np.isfinite(getattr(res.corr, "Icorr", float("nan")))
        )
        if not ok:
            c.drawString(
                2*cm, y,
                "Ecorr/Icorr not available: requires both anodic and cathodic branches with I crossing zero."
            )
            c.showPage()
            c.save()
            return

        draw_kv(y, "Ecorr", f"{res.corr.Ecorr:.6g} V"); y -= 0.6*cm
        draw_kv(y, "Icorr", f"{res.corr.Icorr:.6g} A"); y -= 0.6*cm

        if np.isfinite(getattr(res.corr, "area_cm2", float("nan"))) and res.corr.area_cm2 > 0 and np.isfinite(getattr(res.corr, "jcorr_Acm2", float("nan"))):
            draw_kv(y, "jcorr", f"{res.corr.jcorr_Acm2:.6g} A/cm^2"); y -= 0.6*cm
        if np.isfinite(getattr(res.corr, "beta_a_Vdec", float("nan"))):
            draw_kv(y, "βa", f"{abs(res.corr.beta_a_Vdec)*1000:.3g} mV/dec"); y -= 0.6*cm
        if np.isfinite(getattr(res.corr, "beta_c_Vdec", float("nan"))):
            draw_kv(y, "βc", f"{abs(res.corr.beta_c_Vdec)*1000:.3g} mV/dec"); y -= 0.6*cm
        if np.isfinite(getattr(res.corr, "Rp", float("nan"))):
            rp_unit = "Ω·cm²" if (np.isfinite(getattr(res.corr, "area_cm2", float("nan"))) and res.corr.area_cm2 > 0 and np.isfinite(getattr(res.corr, "jcorr_Acm2", float("nan")))) else "Ω"
            draw_kv(y, "Rp", f"{res.corr.Rp:.6g} {rp_unit}"); y -= 0.6*cm
        if np.isfinite(getattr(res.corr, "corrosion_rate_mm_y", float("nan"))):
            draw_kv(y, "Corrosion rate", f"{res.corr.corrosion_rate_mm_y:.6g} mm/year"); y -= 0.6*cm
        if np.isfinite(getattr(res.corr, "corrosion_rate_mpy", float("nan"))):
            draw_kv(y, "Corrosion rate (mpy)", f"{res.corr.corrosion_rate_mpy:.6g} mpy"); y -= 0.6*cm

        # Page 2: plot if exists
        c.showPage()
        header()
        p = plot_paths.get("corrosion_plot")
        if p and os.path.exists(p):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(2*cm, h - 5.2*cm, "E vs log10(|I|) with Ecorr/Icorr")
            c.drawImage(p, 2*cm, 3*cm, width=w-4*cm, height=h-8.5*cm, preserveAspectRatio=True, anchor="c")

        c.showPage()
        c.save()
        return

    # ---------- OER / HER MODE ----------
    # NOTE: from here on, we keep your existing pages/logic; we only fixed corrosion handling.

    paths = plot_paths  # backward-compat alias for older code blocks
    header()
    y = h - 5.2*cm

    draw_kv(y, "E range", f"{res.e.min():.6g} to {res.e.max():.6g}")
    y -= 0.6*cm
    draw_kv(y, "Selected window", f"{res.fit_start:.6g} to {res.fit_end:.6g}")
    y -= 0.6*cm
    if np.isfinite(float(meta.get("fit_start_rhe_V", float("nan")))):
        draw_kv(
            y, "Window vs RHE",
            f"{meta.get('fit_start_rhe_V'):.6g} to {meta.get('fit_end_rhe_V'):.6g} V vs RHE"
        )
        y -= 0.6*cm
        eta_name = "eta_OER" if reaction_mode.strip().upper() == "OER" else "|eta_HER|"
        draw_kv(
            y, f"{eta_name} window",
            f"{meta.get('eta_window_low_V'):.6g} to {meta.get('eta_window_high_V'):.6g} V"
        )
        y -= 0.6*cm
        draw_kv(
            y, "Equilibrium potential",
            f"{meta.get('equilibrium_potential_rhe_V'):.6g} V vs RHE"
        )
        y -= 0.6*cm
    draw_kv(y, "Emax_D2", f"{res.emax_d2:.6g} (edge-excluded curvature marker)")
    y -= 0.8*cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Exponential slopes (from semi-log fits)")
    y -= 0.6*cm
    draw_kv(y, "ln(i_norm) slope", f"{res.fit_ln_i.slope:.6g} ± {res.fit_ln_i.stderr_slope:.2g} | R²={res.fit_ln_i.r2:.4f} | n={res.fit_ln_i.n}")
    y -= 0.6*cm
    draw_kv(y, "ln(d1_norm) slope", f"{res.fit_ln_d1.slope:.6g} ± {res.fit_ln_d1.stderr_slope:.2g} | R²={res.fit_ln_d1.r2:.4f} | n={res.fit_ln_d1.n}")
    y -= 0.6*cm
    draw_kv(y, "ln(d2_norm) slope", f"{res.fit_ln_d2.slope:.6g} ± {res.fit_ln_d2.stderr_slope:.2g} | R²={res.fit_ln_d2.r2:.4f} | n={res.fit_ln_d2.n}")
    y -= 0.9*cm

    c.setFont("Helvetica-Bold", 11)
    c.setFont("Helvetica", 9)
    for wline in textwrap.wrap("Tafel slope b (mV/dec) is calculated as b = 2.303 RT / (alpha F), where alpha is obtained from the slope of ln(i_n) vs potential.", width=110):
        c.drawString(2.1*cm, y, wline)
        y -= 0.42*cm
    y -= 0.15*cm
    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Charge-transfer coefficients (alpha) and Tafel slopes")
    y -= 0.6*cm

    draw_kv(y, "alpha (from i)", f"{res.alpha_from_i:.4f}  |  Tafel = {res.tafelslope_from_i_mVdec:.2f} mV/dec")
    y -= 0.6*cm
    d1_valid = bool(
        np.isfinite(res.fit_ln_d1.r2)
        and res.fit_ln_d1.r2 >= 0.98
        and res.consistency_score_i_d1 <= 3.0
    )
    d2_valid = bool(
        np.isfinite(res.fit_ln_d2.r2)
        and res.fit_ln_d2.r2 >= 0.95
        and res.consistency_score_i_d2 <= 5.0
    )
    d1_text = (
        f"{res.alpha_from_d1:.4f}  |  diagnostic Tafel = {res.tafelslope_from_d1_mVdec:.2f} mV/dec"
        if d1_valid else f"Not reported as Tafel result (diagnostic R²={res.fit_ln_d1.r2:.4f})"
    )
    draw_kv(y, "d1 diagnostic", d1_text)
    y -= 0.6*cm
    d2_text = (
        f"{res.alpha_from_d2:.4f}  |  diagnostic Tafel = {res.tafelslope_from_d2_mVdec:.2f} mV/dec"
        if d2_valid else f"Not reported as Tafel result (diagnostic R²={res.fit_ln_d2.r2:.4f})"
    )
    draw_kv(y, "d2 diagnostic", d2_text)
    y -= 0.8*cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Instantaneous Tafel-slope diagnostic")
    y -= 0.6*cm
    inst = instantaneous_tafel_summary(res, reaction_mode)
    draw_kv(
        y, "Median (5th–95th percentile)",
        f"{inst['instantaneous_median_mVdec']:.2f} "
        f"({inst['instantaneous_p05_mVdec']:.2f}–{inst['instantaneous_p95_mVdec']:.2f}) mV/dec"
    )
    y -= 0.6*cm
    draw_kv(
        y, "Mean ± SD",
        f"{inst['instantaneous_mean_mVdec']:.2f} ± {inst['instantaneous_sd_mVdec']:.2f} mV/dec "
        f"(n={int(inst['instantaneous_n'])})"
    )
    y -= 0.8*cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Consistency scores (lower is better)")
    y -= 0.6*cm
    draw_kv(y, "i vs d1", f"{res.consistency_score_i_d1:.2f}")
    y -= 0.6*cm
    draw_kv(y, "i vs d2", f"{res.consistency_score_i_d2:.2f}")
    y -= 0.6*cm
    draw_kv(y, "d1 vs d2", f"{res.consistency_score_d1_d2:.2f}")
    y -= 0.9*cm

    c.setFont("Helvetica-Bold", 11)
    c.drawString(2*cm, y, "Warnings / Notes")
    y -= 0.6*cm
    c.setFont("Helvetica", 10)
    if res.warnings:
        for wtxt in res.warnings[:12]:
            c.drawString(2.2*cm, y, f"- {wtxt}")
            y -= 0.5*cm
            if y < 3.5*cm:
                c.showPage()
                header()
                y = h - 4.8*cm
    else:
        c.drawString(2.2*cm, y, "- None")
        y -= 0.6*cm

    # Figures
    for key in [
        "dual_panel", "raw_smooth", "i_norm", "d1d2", "ln_i_fit",
        "ln_d1_fit", "ln_d2_fit", "tafel_plot", "tafel_potential", "tafel_overpotential",
        "tafel_instantaneous", "corrosion_plot"
    ]:
        if key not in plot_paths:
            continue
        c.showPage()
        header()

        fig_desc = {
            "raw_smooth": (
                "Raw vs Savitzky–Golay smoothed current. Use this plot to confirm noise reduction without distorting the trend. "
                "Large deviations may indicate an over-aggressive SG window or a drifting baseline."
            ),
            "dual_panel": (
                "Dual-panel DTA figure: normalized current i and normalized first derivative d1 shown together (top: linear; bottom: semilog y-scale). "
                "This matches the common DTA presentation where i = I/Imax (relative intensity,  ) and d1 is normalized to its maximum (). "
                "Use it to visually identify the region of fastest current increase and deviations due to mass-transport/background contributions."
            ),
            "i_norm": (
                "Normalized current: i_norm = I / Imax (relative intensity). "
                "Normalization enables comparison across datasets independent of absolute current scale. "
                "If your instrument reports anodic currents as negative (common in CHI), the analysis uses a positive kinetic scale internally."
            ),
            "d1d2": (
                "Normalized 1st and 2nd derivatives (d1_norm, d2_norm). The dashed line marks Emax(D2), often near the most rapid kinetic change. "
                f"The shaded region is the automatically selected Tafel window: [{res.fit_start:.4g}, {res.fit_end:.4g}] V."
            ),
            "ln_i_fit": (
                "ln(i_norm) vs E with global fit line. The reported kinetic slope is extracted only from the selected window. "
                f"Fit stats (within window): slope={res.fit_ln_i.slope:.4g} V^-1, R²={res.fit_ln_i.r2:.4f}."
            ),
            "ln_d1_fit": (
                "ln(d1_norm) vs E with fit. This serves as a diagnostic for kinetic consistency using the first derivative. "
                f"Fit stats (within window): slope={res.fit_ln_d1.slope:.4g} V^-1, R²={res.fit_ln_d1.r2:.4f}."
            ),
            "ln_d2_fit": (
                "ln(d2_norm) vs E with fit. Second-derivative analysis is sensitive to noise and regime changes (mass transport, bubbles). "
                f"Fit stats (within window): slope={res.fit_ln_d2.slope:.4g} V^-1, R²={res.fit_ln_d2.r2:.4f}."
            ),
            "tafel_plot": (
                "Tafel plot log10(i_norm) vs E with the fit segment highlighted inside the selected window. "
                f"Extracted Tafel slope (from ln-fit) = {res.tafelslope_from_i_mVdec:.2f} mV/dec; "
                f"Window: [{res.fit_start:.4g}, {res.fit_end:.4g}] V; Emax(D2)={res.emax_d2:.4g} V."
            ),
            "tafel_potential": (
                "Conventional Tafel orientation showing the imported potential E versus log10(i_norm). The full processed "
                "curve is retained for context, while the regression line is drawn only across the confirmed fitting window."
            ),
            "tafel_overpotential": (
                "Publication-oriented Tafel representation using reaction-direction overpotential calculated from the "
                "user-selected reference electrode, pH, and equilibrium potential. Only the selected-window fit segment "
                "is drawn; the conversion settings are recorded in the report and CSV outputs."
            ),
            "tafel_instantaneous": (
                "Pointwise instantaneous Tafel slope inside the confirmed fitting window. The dashed line is the median "
                "and the band is the 5th–95th percentile. This is a derivative-based diagnostic and should accompany, "
                "not replace, the regression-based Tafel slope."
            ),
        }

        img = ImageReader(plot_paths[key])
        margin = 2*cm
        img_w = w - 2*margin

        # Leave extra space at the bottom for multi-line captions
        img_h = h - 9*cm
        c.drawImage(img, margin, 4.0*cm, width=img_w, height=img_h,
                    preserveAspectRatio=True, anchor='c')

        c.setFont("Helvetica-Bold", 10)
        c.drawString(2*cm, 3.3*cm, f"Figure: {key}")

        # Wrapped description
        c.setFont("Helvetica", 9)
        desc = fig_desc.get(key, "")
        max_chars = 110
        lines = textwrap.wrap(desc, width=max_chars) if desc else []
        ycap = 2.8*cm
        for line in lines[:4]:
            c.drawString(2*cm, ycap, line)
            ycap -= 0.45*cm

    c.save()


# ----------------------------
# GUI
# ----------------------------


class DropLabel(QLabel):
    fileLoaded = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("\n\nDrop your data file here\n(CSV / TXT / XLSX)\n\nor click Browse\n")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QLabel{border:2px dashed #888; border-radius:10px; padding:20px;}")
        self.setAcceptDrops(True)
        self.file_path: Optional[str] = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            self.file_path = path
            self.setText(f"Loaded:\n{path}")
            try:
                self.fileLoaded.emit(path)
            except Exception:
                pass



class WindowSelectDialog(QDialog):
    """Interactive dialog to manually select a Tafel fit window by drawing an x-range on a plot."""

    def __init__(self, e: np.ndarray, logi: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manual Tafel Window Selection")
        self.resize(900, 600)

        self._selection: Optional[Tuple[float, float]] = None

        layout = QVBoxLayout(self)

        if FigureCanvas is None or SpanSelector is None:
            layout.addWidget(QLabel("Matplotlib Qt backend is not available.\n"
                                    "Please install/enable a Qt backend (QtAgg) or use E_low/E_high boxes."))
            btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            btns.rejected.connect(self.reject)
            btns.accepted.connect(self.accept)
            layout.addWidget(btns)
            return

        fig = plt.Figure(figsize=(8, 5), dpi=100)
        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        ax = fig.add_subplot(111)
        ax.plot(e, logi, linewidth=1.0)
        ax.set_xlabel("Potential, E")
        ax.set_ylabel("log10(|I|)")

        self.lbl = QLabel("Drag on the plot to select a window.")
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        layout.addWidget(self.lbl)

        def _onselect(xmin, xmax):
            if xmin is None or xmax is None:
                return
            lo, hi = (float(xmin), float(xmax)) if xmin <= xmax else (float(xmax), float(xmin))
            self._selection = (lo, hi)
            self.lbl.setText(f"Selected window: E_low = {lo:.6g}, E_high = {hi:.6g}")

        self._span = SpanSelector(ax, _onselect, "horizontal", useblit=True, interactive=True)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selection(self) -> Optional[Tuple[float, float]]:
        return self._selection


class ResearcherTafelWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Researcher Analysis Toolkit – Tafel Analysis")
        self.setWindowIcon(QIcon("Researcherlogo.png")) 

        self.resize(1040, 710)

        self.drop = DropLabel()
        try:
            self.drop.fileLoaded.connect(self.on_file_loaded)
        except Exception:
            pass

        self.btn_browse = QPushButton("Browse…")
        self.btn_browse.clicked.connect(self.browse_file)

        self.btn_load_cols = QPushButton("Load columns")
        self.btn_load_cols.clicked.connect(self.load_columns)

        self.btn_run = QPushButton("Run Analysis + Export PNG/PDF")
        self.btn_run.clicked.connect(self.run_analysis)

        self.mode = QComboBox()
        self.mode.addItems(["OER", "HER", "CORROSION"])

        # --- Corrosion inputs (only used when mode = CORROSION) ---
        self.corr_group = QGroupBox("Corrosion inputs (optional)")
        corr_form = QFormLayout()
        self.corr_group.setLayout(corr_form)

        self.area_edit = QLineEdit()
        self.area_edit.setPlaceholderText("e.g., 1.00  (cm^2)")
        corr_form.addRow("Area (cm^2) [optional]:", self.area_edit)

        self.metal_preset = QComboBox()
        self.metal_preset.addItem("Custom")
        for k, (label, aw, rho, n) in METAL_PRESETS.items():
            self.metal_preset.addItem(label, k)
        corr_form.addRow("Metal preset:", self.metal_preset)

        self.ew_edit = QLineEdit()
        self.ew_edit.setPlaceholderText("Equivalent weight (g/eq)")
        corr_form.addRow("EW (g/eq):", self.ew_edit)

        self.rho_edit = QLineEdit()
        self.rho_edit.setPlaceholderText("Density (g/cm^3)")
        corr_form.addRow("Density ρ (g/cm^3):", self.rho_edit)

        self.rate_unit = QComboBox()
        self.rate_unit.addItems(["mm/year", "mpy"])
        corr_form.addRow("Corrosion rate unit:", self.rate_unit)

        # preset behavior: auto-fill EW and density
        self.metal_preset.currentIndexChanged.connect(self._on_metal_preset_changed)
        self.mode.currentIndexChanged.connect(self._on_mode_changed)

        self.cmb_e = QComboBox()
        self.cmb_i = QComboBox()
        self.cmb_logi = QComboBox()
        self.cmb_logi.addItem("(none)")

        self.chk_use_log_col = QCheckBox("Use provided log10(|I|) column instead of I")
        self.chk_use_log_col.setChecked(False)

        self.chk_invert_current = QCheckBox("Invert current sign (manual override)")
        self.chk_invert_current.setChecked(False)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(200.0, 400.0)
        self.temperature.setValue(298.15)
        self.temperature.setDecimals(2)

        self.sg_window = QSpinBox()
        self.sg_window.setRange(5, 301)
        self.sg_window.setValue(21)

        self.sg_poly = QSpinBox()
        self.sg_poly.setRange(1, 5)
        self.sg_poly.setValue(2)

        # Explicit potential-reference conversion.  No overpotential is
        # calculated until the user selects the reference electrode.
        self.chk_overpotential = QCheckBox("Generate RHE and overpotential outputs")
        self.chk_overpotential.setChecked(True)

        self.reference_combo = QComboBox()
        self.reference_combo.addItem("Select reference electrode…", None)
        for ref_key, (ref_label, _) in REFERENCE_ELECTRODE_PRESETS.items():
            self.reference_combo.addItem(ref_label, ref_key)

        self.ph_value = QDoubleSpinBox()
        self.ph_value.setRange(-2.0, 16.0)
        self.ph_value.setValue(7.0)
        self.ph_value.setDecimals(3)

        self.custom_ref_offset = QDoubleSpinBox()
        self.custom_ref_offset.setRange(-2.0, 2.0)
        self.custom_ref_offset.setValue(0.197)
        self.custom_ref_offset.setDecimals(6)
        self.custom_ref_offset.setSuffix(" V vs SHE")
        self.custom_ref_offset.setEnabled(False)

        self.eq_potential_rhe = QDoubleSpinBox()
        self.eq_potential_rhe.setRange(-3.0, 3.0)
        self.eq_potential_rhe.setValue(1.229)
        self.eq_potential_rhe.setDecimals(4)
        self.eq_potential_rhe.setSuffix(" V vs RHE")
        self.reference_combo.currentIndexChanged.connect(self._on_reference_changed)

        self.fit_low = QLineEdit("")
        self.fit_high = QLineEdit("")

        self.btn_select_window = QPushButton("Select window on plot…")
        self.btn_select_window.clicked.connect(self.select_window_on_plot)
        self.chk_emax_upper = QCheckBox("Use edge-excluded Emax(D2) as an upper bound (optional)")
        self.chk_emax_upper.setChecked(False)

        # Dual-panel figure export (layout similar to the ) + optional annotations (kept general)
        self.chk_dual_panel_fig = QCheckBox("Export dual-panel DTA figure")
        self.chk_dual_panel_fig.setChecked(True)

        self.chk_dual_panel_annot = QCheckBox("Add custom annotations on dual-panel figure (optional)")
        self.chk_dual_panel_annot.setChecked(False)

        self.txt_ann1_text = QLineEdit("")
        self.txt_ann1_text.setPlaceholderText("Annotation 1 text (e.g., OH− → O₂)")
        self.txt_ann1_x = QLineEdit("")
        self.txt_ann1_x.setPlaceholderText("E position (x)")
        self.txt_ann1_y = QLineEdit("")
        self.txt_ann1_y.setPlaceholderText("Rel. intensity (y)")

        self.txt_ann2_text = QLineEdit("")
        self.txt_ann2_text.setPlaceholderText("Annotation 2 text (e.g., Cl− → Cl₂)")
        self.txt_ann2_x = QLineEdit("")
        self.txt_ann2_x.setPlaceholderText("E position (x)")
        self.txt_ann2_y = QLineEdit("")
        self.txt_ann2_y.setPlaceholderText("Rel. intensity (y)")

                # Results table + preview plot
        self.tbl_windows = QTableWidget(0, 6)
        self.tbl_windows.setHorizontalHeaderLabels(["Type", "E_start", "E_end", "n", "Slope (mV/dec)", "R²"])
        try:
            self.tbl_windows.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        except Exception:
            pass
        self.tbl_windows.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.tbl_windows.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tbl_windows.itemSelectionChanged.connect(self._on_window_row_selected)

        self.btn_pick_window = QPushButton("Pick window (manual)")
        self.btn_pick_window.clicked.connect(self.select_window_on_plot)
        self.btn_add_manual = QPushButton("Add manual window to table")
        self.btn_add_manual.clicked.connect(self._add_manual_window_to_table)

        self.preview_fig = plt.Figure(figsize=(8, 3.5), dpi=100)
        self.preview_canvas = FigureCanvas(self.preview_fig)
        self.preview_toolbar = NavigationToolbar(self.preview_canvas, self)
        self.preview_note = QLabel("Select a suggested window in the table to preview. Or pick a manual window and add it.")
        self.preview_note.setWordWrap(True)

        self.results = QTextEdit()
        self.results.setReadOnly(True)

        self.analysis_choice: Optional[str] = None  # 'auto' or 'manual'
        self.manual_window: Optional[Tuple[float, float]] = None
        self.manual_confirmed: bool = False

        self.manual_data_mode: Optional[str] = None  # 'lsv' or 'tafel'
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QVBoxLayout()
        right = QVBoxLayout()
        right_container = QWidget()                 
        right_container.setLayout(right)

        scroll = QScrollArea()                      
        scroll.setWidgetResizable(True)
        scroll.setWidget(right_container)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        left.addWidget(self.drop)
        row = QHBoxLayout()
        row.addWidget(self.btn_browse)
        row.addWidget(self.btn_load_cols)
        left.addLayout(row)
        left.addWidget(self.btn_run)
        left.addWidget(QLabel("Results / Preview"))
        left.addWidget(self.tbl_windows)
        _btnrow = QHBoxLayout()
        _btnrow.addWidget(self.btn_pick_window)
        _btnrow.addWidget(self.btn_add_manual)
        left.addLayout(_btnrow)
        left.addWidget(self.preview_toolbar)
        left.addWidget(self.preview_canvas)
        left.addWidget(self.preview_note)
        left.addWidget(self.results)

        g1 = QGroupBox("Reaction Mode")
        l1 = QVBoxLayout(g1)
        l1.addWidget(QLabel("Select mode: OER, HER, or Corrosion:"))
        l1.addWidget(self.mode)

        g2 = QGroupBox("Column Selection")
        l2 = QVBoxLayout(g2)
        l2.addWidget(QLabel("Potential column (E):"))
        l2.addWidget(self.cmb_e)
        l2.addWidget(QLabel("Current column (I or j):"))
        l2.addWidget(self.cmb_i)
        l2.addWidget(self.chk_use_log_col)
        l2.addWidget(QLabel("Optional: log10(|I|) column:"))
        l2.addWidget(self.cmb_logi)

        g3 = QGroupBox("DTA Parameters")
        l3 = QVBoxLayout(g3)
        rowt = QHBoxLayout()
        rowt.addWidget(QLabel("Temperature (K):"))
        rowt.addWidget(self.temperature)
        l3.addLayout(rowt)

        rowsg = QHBoxLayout()
        rowsg.addWidget(QLabel("SavGol window:"))
        rowsg.addWidget(self.sg_window)
        rowsg.addWidget(QLabel("poly:"))
        rowsg.addWidget(self.sg_poly)
        l3.addLayout(rowsg)

        self.potential_group = QGroupBox("Reference Electrode and Overpotential")
        lp = QFormLayout(self.potential_group)
        lp.addRow(self.chk_overpotential)
        lp.addRow("Input potential reference:", self.reference_combo)
        lp.addRow("Electrolyte pH:", self.ph_value)
        lp.addRow("Custom Eref:", self.custom_ref_offset)
        lp.addRow("Equilibrium potential:", self.eq_potential_rhe)
        note_ref = QLabel(
            "E_RHE = E_input + Eref(vs SHE) + 2.303RT/F × pH. "
            "Preset Eref values are nominal near 25 °C."
        )
        note_ref.setWordWrap(True)
        lp.addRow(note_ref)

        g4 = QGroupBox("Fit Range (optional)")
        l4 = QVBoxLayout(g4)
        l4.addWidget(QLabel("Enter limits or select a suggested/manual fitting window:"))
        rowf = QHBoxLayout()
        rowf.addWidget(QLabel("E_low:"))
        rowf.addWidget(self.fit_low)
        rowf.addWidget(QLabel("E_high:"))
        rowf.addWidget(self.fit_high)
        l4.addLayout(rowf)
        l4.addWidget(self.chk_emax_upper)
        l4.addWidget(self.btn_select_window)

        g4b = QGroupBox("Dual-panel Figure (optional)")
        l4b = QVBoxLayout(g4b)
        l4b.addWidget(self.chk_dual_panel_fig)
        l4b.addWidget(self.chk_dual_panel_annot)

        rowa1 = QHBoxLayout()
        rowa1.addWidget(self.txt_ann1_text)
        rowa1.addWidget(self.txt_ann1_x)
        rowa1.addWidget(self.txt_ann1_y)
        l4b.addLayout(rowa1)

        rowa2 = QHBoxLayout()
        rowa2.addWidget(self.txt_ann2_text)
        rowa2.addWidget(self.txt_ann2_x)
        rowa2.addWidget(self.txt_ann2_y)
        l4b.addLayout(rowa2)

        g5 = QGroupBox("Overrides")
        l5 = QVBoxLayout(g5)
        l5.addWidget(self.chk_invert_current)
        # ADD GROUPS TO RIGHT PANEL ----------
        right.addWidget(g1)
        right.addWidget(self.corr_group)
        right.addWidget(g2)
        right.addWidget(g3)
        right.addWidget(self.potential_group)
        right.addWidget(g4)
        right.addWidget(g4b)
        right.addWidget(g5)
        right.addStretch(1)  # باعث میشه اسکرول نرم‌تر و ظاهر بهتر بشه

        # ADD PANELS TO MAIN WINDOW ----------
        main.addLayout(left, 2)     # ستون چپ (بزرگ‌تر)
        main.addWidget(scroll, 1)   # ستون راست اسکرولی

        self.df: Optional[pd.DataFrame] = None
        self._on_reference_changed()
        self._on_mode_changed()


    def _collect_dual_panel_annotations(self) -> Optional[List[Dict[str, Any]]]:
        """Collect optional user-defined annotations for the dual-panel figure.
        Keeps the tool general: annotations are only added when explicitly enabled."""
        try:
            if getattr(self, "chk_dual_panel_annot", None) is None or not self.chk_dual_panel_annot.isChecked():
                return None
            anns: List[Dict[str, Any]] = []
            for idx in (1, 2):
                txt_w = getattr(self, f"txt_ann{idx}_text", None)
                x_w = getattr(self, f"txt_ann{idx}_x", None)
                y_w = getattr(self, f"txt_ann{idx}_y", None)
                if txt_w is None or x_w is None or y_w is None:
                    continue
                t = txt_w.text().strip()
                x = x_w.text().strip()
                y = y_w.text().strip()
                if not t or not x or not y:
                    continue
                anns.append({"text": t, "x": float(x), "y": float(y)})
            return anns if anns else None
        except Exception:
            return None



    # ----------------------------
    # File IO / GUI callbacks (added for stability)
    # ----------------------------
    def browse_file(self):
        """Open a file dialog and load a data file."""
        try:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Select data file",
                "",
                "Data files (*.csv *.txt *.tsv *.xlsx *.xls);;All files (*.*)"
            )
            if not path:
                return
            self.on_file_loaded(path)
        except Exception as e:
            try:
                self.results.append(f"Browse failed: {e}")
            except Exception:
                pass

    def _read_input_file(self, path: str) -> pd.DataFrame:
        """Read CSV/TSV/TXT/Excel into a DataFrame."""
        ext = os.path.splitext(path)[1].lower()
        if ext in [".xlsx", ".xls"]:
            return pd.read_excel(path)

        # CH Instruments text exports contain a metadata block followed by a
        # real header such as "Potential/V, Current/A".  Detect that header so
        # users can load the original instrument file without manual cleanup.
        encoding_used = None
        lines: List[str] = []
        for encoding in ("utf-8-sig", "cp1252", "latin1"):
            try:
                with open(path, "r", encoding=encoding) as handle:
                    lines = handle.readlines()
                encoding_used = encoding
                break
            except UnicodeDecodeError:
                continue

        if lines:
            for row_index, line in enumerate(lines):
                lower = line.lower()
                if "potential" in lower and ("current" in lower or "<i>" in lower):
                    try:
                        return pd.read_csv(
                            path,
                            skiprows=row_index,
                            sep=None,
                            engine="python",
                            encoding=encoding_used,
                        )
                    except Exception:
                        break

        # Generic text-like input: infer the delimiter with encoding fallbacks.
        last_error: Optional[Exception] = None
        for encoding in ("utf-8-sig", "cp1252", "latin1"):
            for sep in (None, ",", ";", "\t"):
                try:
                    return pd.read_csv(path, sep=sep, engine="python", encoding=encoding)
                except Exception as exc:
                    last_error = exc
        raise ValueError(f"Could not parse the input file: {last_error}")

    def load_columns(self):
        """Load dataframe columns and populate comboboxes."""
        try:
                    path = getattr(self.drop, "file_path", None)
                    if not path:
                        # also allow a cached path
                        path = getattr(self, "current_file", None)
                    if not path:
                        try:
                            self.results.append("No file selected.")
                        except Exception:
                            pass
                        return

                    self.current_file = path
                    df = self._read_input_file(path)

                    # coerce column names to strings
                    df.columns = [str(c).strip() for c in df.columns]
                    self.df = df

                    cols = list(df.columns)

                    # populate combos
                    self.cmb_e.clear()
                    self.cmb_i.clear()
                    self.cmb_logi.clear()
                    self.cmb_logi.addItem("(none)")
                    self.cmb_e.addItems(cols)
                    self.cmb_i.addItems(cols)
                    self.cmb_logi.addItems(cols)

                    # heuristic defaults
                    def _pick(cols, keys):
                        for k in keys:
                            for c in cols:
                                if k in c.lower():
                                    return c
                        return None

                    e_guess = _pick(cols, ["potential", "ewe", "e (v", "voltage", "e /", "e(" , "e " , "v"])
                    i_guess = _pick(cols, ["current", "i (", "i/", "j (", "current density", "amps", "a"])
                    log_guess = _pick(cols, ["log", "log10", "log(|", "log i", "logi"])

                    if e_guess and e_guess in cols:
                        self.cmb_e.setCurrentText(e_guess)
                    if i_guess and i_guess in cols:
                        self.cmb_i.setCurrentText(i_guess)
                    if log_guess and log_guess in cols:
                        self.cmb_logi.setCurrentText(log_guess)

                    try:
                        self.results.append(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
                    except Exception:
                        pass
        except Exception as e:
            try:
                self.results.append('Load columns failed: ' + str(e))
                self.results.append(traceback.format_exc())
            except Exception:
                pass
            return


    def _on_metal_preset_changed(self, *_):
        """Autofill EW and density from preset."""
        try:
            key = self.metal_preset.currentData()
            if not key or key == "Custom":
                return
            label, aw, rho, n = METAL_PRESETS.get(key, (None, None, None, None))
            if aw is None:
                return
            # Equivalent weight = atomic weight / valence
            ew = float(aw) / float(n)
            self.ew_edit.setText(f"{ew:.6g}")
            self.rho_edit.setText(f"{float(rho):.6g}")
        except Exception:
            return

    def _on_mode_changed(self, *_):
        """Show mode-specific controls and set the default reversible potential."""
        try:
            mode = self.mode.currentText().strip().upper()
            is_corr = mode.startswith("CORR")
            self.corr_group.setVisible(is_corr)
            if hasattr(self, "potential_group"):
                self.potential_group.setVisible(not is_corr)
            if hasattr(self, "eq_potential_rhe"):
                if mode == "OER":
                    self.eq_potential_rhe.setValue(1.229)
                elif mode == "HER":
                    self.eq_potential_rhe.setValue(0.0)
        except Exception:
            pass

    def _on_reference_changed(self, *_):
        """Enable the custom-reference field only for the custom preset."""
        try:
            key = self.reference_combo.currentData()
            self.custom_ref_offset.setEnabled(key == "custom")
            # pH is not used when the imported axis is already vs RHE.
            self.ph_value.setEnabled(key != "rhe")
        except Exception:
            pass

    def _choose_manual_input_mode(self) -> str:
        """Ask the user whether the loaded file is LSV or already a Tafel table."""
        msg = QMessageBox(self)
        msg.setWindowTitle("Manual mode input")
        msg.setText("In MANUAL mode, is your input an LSV (E vs I/j) that should be converted to Tafel,\n"
                    "or do you already have a Tafel table (E vs log10|I|)?")
        btn_lsv = msg.addButton("LSV → Tafel (recommended)", QMessageBox.ButtonRole.AcceptRole)
        btn_tafel = msg.addButton("Already Tafel (E vs log10|I|)", QMessageBox.ButtonRole.AcceptRole)
        msg.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(btn_lsv)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_tafel:
            return "tafel"
        if clicked == btn_lsv:
            return "lsv"
        return "lsv"

    def _precompute_ln_i_for_suggestions(self, e: np.ndarray, i: np.ndarray, reaction_mode: str):
        """Precompute the same smoothed, sign-aware current used by the main fit."""
        e = np.asarray(e, dtype=float)
        i = np.asarray(i, dtype=float)
        m = np.isfinite(e) & np.isfinite(i)
        e = e[m]
        i = i[m]
        if e.size == 0:
            return np.array([]), np.array([]), np.array([])
        # sort by E
        idx = np.argsort(e)
        e = e[idx]
        i = i[idx]

        sg_window = int(self.sg_window.value()) if hasattr(self, "sg_window") else 21
        sg_poly = int(self.sg_poly.value()) if hasattr(self, "sg_poly") else 2
        if sg_window % 2 == 0:
            sg_window += 1
        if sg_window >= len(e):
            sg_window = len(e) - 1 if (len(e) - 1) % 2 else len(e) - 2
        sg_window = max(5, sg_window)
        try:
            i_smooth = savgol_filter(i, window_length=sg_window, polyorder=min(sg_poly, sg_window - 2))
        except Exception:
            i_smooth = i.copy()

        if hasattr(self, "chk_invert_current") and self.chk_invert_current.isChecked():
            i_smooth = -i_smooth

        mode = reaction_mode.strip().upper()
        i_k = -i_smooth if mode == "HER" else i_smooth.copy()
        if np.nanmedian(i_k) < 0:
            i_k = np.abs(i_k)

        eps = np.nanmax(np.abs(i_k)) * 1e-12 + 1e-30
        i_k = np.where(i_k <= 0, eps, i_k)
        imax = np.nanmax(i_k)
        if not np.isfinite(imax) or imax <= 0:
            return np.array([]), np.array([]), np.array([])
        i_norm = i_k / imax

        # Cache an edge-excluded Emax(D2) marker for optional suggestion limits.
        d1 = np.gradient(i_k, e)
        d2 = np.gradient(d1, e)
        d2_adj = d2.copy()
        finite_d2 = d2_adj[np.isfinite(d2_adj)]
        if finite_d2.size and np.nanmedian(finite_d2) < 0:
            d2_adj = -d2_adj
        d2_adj = np.where(d2_adj <= 0, eps, d2_adj)
        edge_n = max(2, int(round(0.05 * len(e))), sg_window // 2)
        if 2 * edge_n < len(e):
            local = d2_adj[edge_n:len(e) - edge_n]
            emax_idx = edge_n + int(np.nanargmax(local))
        else:
            emax_idx = int(np.nanargmax(d2_adj))
        self._preview_emax_d2 = float(e[emax_idx])
        self._preview_expected_slope_sign = -1 if mode == "HER" else 1

        m2 = np.isfinite(i_norm) & (i_norm > 0)
        e = e[m2]
        i_norm = i_norm[m2]
        if e.size == 0:
            return np.array([]), np.array([]), np.array([])
        ln_i = np.log(i_norm)
        log10_i = np.log10(i_norm)
        return e, ln_i, log10_i

    def select_window_on_plot(self):
        """Open interactive window selection for manual fitting."""
        try:
            if getattr(self, "_preview_E", None) is None or getattr(self, "_preview_log10I", None) is None:
                # try to prepare suggestions/preview first
                self._prepare_suggestions_from_loaded_data()
            E = getattr(self, "_preview_E", None)
            ylog = getattr(self, "_preview_log10I", None)
            if E is None or ylog is None or E.size == 0:
                self.results.append("No preview data available. Load columns first.")
                return

            dlg = WindowSelectDialog(E, ylog, parent=self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            sel = dlg.selection()
            if not sel:
                return
            lo, hi = sel
            self.fit_low.setText(f"{lo:.6g}")
            self.fit_high.setText(f"{hi:.6g}")
            self.manual_window = (min(lo, hi), max(lo, hi))
            self.manual_confirmed = True
            self.analysis_choice = "manual"
            self.results.append(f"Manual window selected: {self.manual_window[0]:.6g} → {self.manual_window[1]:.6g}")
        except Exception as e:
            try:
                self.results.append(f"Manual window selection failed: {e}")
            except Exception:
                pass

    def run_analysis(self):
        """Run analysis using either automatic selection or the selected/manual window."""
        try:
            if getattr(self, "df", None) is None:
                self.load_columns()
            if getattr(self, "df", None) is None:
                self.results.append("No data loaded.")
                return

            # column mapping
            col_e = self.cmb_e.currentText()
            col_i = self.cmb_i.currentText()
            col_log = self.cmb_logi.currentText()
            use_log = self.chk_use_log_col.isChecked()

            if not col_e or col_e not in self.df.columns:
                self.results.append("Select a valid E column.")
                return

            E = _ensure_numeric_series(self.df[col_e]).to_numpy(dtype=float)

            if use_log:
                if not col_log or col_log == "(none)" or col_log not in self.df.columns:
                    self.results.append("Use log column is enabled but no log column is selected.")
                    return
                log10I = _ensure_numeric_series(self.df[col_log]).to_numpy(dtype=float)
                I = np.power(10.0, log10I)
            else:
                if not col_i or col_i not in self.df.columns:
                    self.results.append("Select a valid I column.")
                    return
                I = _ensure_numeric_series(self.df[col_i]).to_numpy(dtype=float)

            reaction_mode = self.mode.currentText().strip()
            T = float(self.temperature.value())
            sg_w = int(self.sg_window.value())
            sg_p = int(self.sg_poly.value())
            use_emax = bool(self.chk_emax_upper.isChecked())
            inv_manual = bool(self.chk_invert_current.isChecked())

            # Optional fit_range from boxes (used mainly for AUTO constraints)
            fit_range = None
            try:
                lo = float(self.fit_low.text().strip()) if self.fit_low.text().strip() else None
                hi = float(self.fit_high.text().strip()) if self.fit_high.text().strip() else None
                if lo is not None and hi is not None:
                    fit_range = (min(lo, hi), max(lo, hi))
            except Exception:
                fit_range = None

            manual_window = getattr(self, "manual_window", None)
            if manual_window is None and fit_range is not None:
                manual_window = fit_range
            if manual_window is None:
                self.results.append("Please select a Tafel window from the table or pick a manual window before running.")
                return
            self.manual_window = manual_window
            self.manual_confirmed = True
            # Run core analysis
            res = differential_tafel_analysis(
                E, I,
                reaction_mode=reaction_mode,
                temperature_K=T,
                sg_window=sg_w,
                sg_poly=sg_p,
                fit_range=fit_range,
                use_emax_d2_as_upper=use_emax,
                invert_current_manual=inv_manual,
                manual_window=manual_window,
                manual_min_points=8,
            )

            scan_rate = _infer_scan_rate_from_filename(
                os.path.basename(getattr(self, "current_file", ""))
            )
            if scan_rate is not None and scan_rate >= 0.02:
                res.warnings.append(
                    f"High scan rate detected ({scan_rate:g} V/s); capacitive current can bias a kinetic Tafel slope."
                )

            potential: Optional[PotentialConversion] = None
            mode_upper = reaction_mode.strip().upper()
            if mode_upper in ("OER", "HER") and self.chk_overpotential.isChecked():
                reference_key = self.reference_combo.currentData()
                if not reference_key:
                    raise ValueError(
                        "Select the input reference electrode (and verify pH) before "
                        "generating RHE/overpotential outputs."
                    )
                potential = build_potential_conversion(
                    res.e,
                    reaction_mode=mode_upper,
                    reference_key=str(reference_key),
                    pH=float(self.ph_value.value()),
                    temperature_K=T,
                    custom_reference_offset_vs_she_V=float(self.custom_ref_offset.value()),
                    equilibrium_potential_rhe_V=float(self.eq_potential_rhe.value()),
                )
                res.warnings.append(
                    f"Potential conversion: {potential.reference_label}; pH={potential.pH:g}; "
                    f"E_RHE = E_input + {potential.measured_to_rhe_shift_V:.6g} V."
                )
                if reference_key not in ("rhe", "custom") and abs(T - 298.15) > 5.0:
                    res.warnings.append(
                        "Reference-electrode preset is nominal near 25 °C; verify its temperature-specific potential."
                    )

            # output folder
            base = os.path.splitext(os.path.basename(getattr(self, "current_file", "data")))[0]
            root_out = _default_root_outdir_for_tafel(base)
            run_dir = _next_run_dir(root_out)

            # plots
            title_prefix = f"{reaction_mode} | {base}"
            plot_paths = make_all_plots(
                res,
                outdir=run_dir,
                title_prefix=title_prefix,
                reaction_mode=reaction_mode,
                export_dual_panel=bool(self.chk_dual_panel_fig.isChecked()),
                dual_panel_xlabel="Potential (V)",
                dual_panel_annotations=self._collect_dual_panel_annotations() if self.chk_dual_panel_annot.isChecked() else None,
                potential=potential,
            )

            # Export processed arrays, fit subset, and compact numerical results.
            try:
                export_processed_data_csv(
                    res, os.path.join(run_dir, "processed_data.csv"),
                    potential=potential, reaction_mode=reaction_mode,
                )
                export_tafel_numbers_csv(
                    res, os.path.join(run_dir, "tafel_numbers.csv"),
                    potential=potential, reaction_mode=reaction_mode,
                )
                export_tafel_fit_data_csv(
                    res, os.path.join(run_dir, "tafel_fit_data.csv"),
                    potential=potential, reaction_mode=reaction_mode,
                )
                export_tafel_window_data_csv(
                    res, os.path.join(run_dir, "tafel_window_data.csv"),
                    potential=potential, reaction_mode=reaction_mode,
                )
            except Exception as export_error:
                self.results.append(f"Numeric export warning: {export_error}")

            # pdf
            pdf_path = os.path.join(run_dir, "DTA_Report.pdf")
            meta = {
                "mode": reaction_mode,
                "T": T,
                "sg_window": sg_w,
                "sg_poly": sg_p,
                "col_e": col_e,
                "col_i": (col_log if use_log else col_i),
                "use_log": use_log,
                "fit_range": fit_range,
                "manual_window": manual_window,
                "source_file": getattr(self, "current_file", ""),
                "scan_rate_Vs": scan_rate,
                "reference_label": potential.reference_label if potential is not None else "Not converted",
                "reference_offset_vs_she_V": potential.reference_offset_vs_she_V if potential is not None else float("nan"),
                "pH": potential.pH if potential is not None else float("nan"),
                "measured_to_rhe_shift_V": potential.measured_to_rhe_shift_V if potential is not None else float("nan"),
                "equilibrium_potential_rhe_V": potential.equilibrium_potential_rhe_V if potential is not None else float("nan"),
            }
            if potential is not None:
                meta.update(potential.window_summary(res))
            write_pdf_report(pdf_path, getattr(self, "current_file", ""), meta, res, plot_paths, reaction_mode=reaction_mode)

            # summarize in UI
            self.results.append("")
            self.results.append(f"Saved to: {run_dir}")
            try:
                self.results.append(f"Tafel slope (from i): {res.tafelslope_from_i_mVdec:.3g} mV/dec | R²={res.fit_ln_i.r2:.4f} | n={res.fit_ln_i.n}")
                self.results.append(f"Input-potential window: {res.fit_start:.6g} → {res.fit_end:.6g} V")
                if potential is not None:
                    pwin = potential.window_summary(res)
                    self.results.append(
                        f"RHE window: {pwin['fit_start_rhe_V']:.6g} → {pwin['fit_end_rhe_V']:.6g} V vs RHE"
                    )
                    self.results.append(
                        f"Overpotential window: {pwin['eta_window_low_V']:.6g} → {pwin['eta_window_high_V']:.6g} V"
                    )
                inst = instantaneous_tafel_summary(res, reaction_mode)
                self.results.append(
                    f"Instantaneous slope (median; 5–95%): "
                    f"{inst['instantaneous_median_mVdec']:.3g} "
                    f"({inst['instantaneous_p05_mVdec']:.3g}–{inst['instantaneous_p95_mVdec']:.3g}) mV/dec"
                )
            except Exception:
                pass
        except Exception as e:
            self.results.append("Run failed:")
            self.results.append(str(e))
            self.results.append(traceback.format_exc())

    def on_file_loaded(self, path: str):
        """Called when a file is loaded via Browse or Drag&Drop."""
        # update UI label
        try:
            self.drop.file_path = path
            self.drop.setText(f"Loaded:\n{path}")
        except Exception:
            pass
        # Manual-only workflow (no automatic mode)
        self.analysis_choice = "manual"
        self.manual_window = None
        self.manual_confirmed = False
        self.manual_data_mode = None

        # Load columns immediately so we can build suggestions/table
        try:
            self.load_columns()
        except Exception:
            # we can still proceed; user may need to adjust column mapping
            pass

        if self.analysis_choice == "manual":
            # Decide whether input is LSV or already Tafel-ready
            try:
                self.manual_data_mode = self._choose_manual_input_mode()
            except Exception:
                self.manual_data_mode = "lsv"

            # Make the input-mode choice operational instead of informational.
            if hasattr(self, "chk_use_log_col"):
                self.chk_use_log_col.setChecked(self.manual_data_mode == "tafel")

            try:
                self._prepare_suggestions_from_loaded_data()
                self.results.append("Manual mode: choose a suggested window (table), or pick a manual window and add it to the table.")
            except Exception as e:
                self.results.append(f"Manual prep failed: {e}")
        else:
            # Automatic mode: clear table
            try:
                self.tbl_windows.setRowCount(0)
            except Exception:
                pass
            self.results.append("Automatic mode selected.")

        # Clear preview plot (do not block results text)
        try:
            self.preview_fig.clear()
            self.preview_canvas.draw()
        except Exception:
            pass

    
    def _prepare_suggestions_from_loaded_data(self) -> None:
        """Build suggested windows table and a preview plot from the loaded dataframe.
        Suggestions are computed using the SAME preprocessing as the main analysis, so the reported slope matches what you run.
        """
        if self.df is None:
            return

        col_e = self.cmb_e.currentText()
        if not col_e or col_e not in self.df.columns:
            return

        col_i = self.cmb_i.currentText()
        use_log = bool(getattr(self, 'chk_use_log_col', None) is not None and self.chk_use_log_col.isChecked())
        col_log = self.cmb_logi.currentText() if hasattr(self, "cmb_logi") else "(none)"

        E = _ensure_numeric_series(self.df[col_e]).to_numpy(dtype=float)

        if use_log:
            if not col_log or col_log == "(none)" or col_log not in self.df.columns:
                # cannot build suggestions without a valid log column
                return
            log10I_raw = _ensure_numeric_series(self.df[col_log]).to_numpy(dtype=float)
            I = np.power(10.0, log10I_raw)
        else:
            if not col_i or col_i not in self.df.columns:
                return
            I = _ensure_numeric_series(self.df[col_i]).to_numpy(dtype=float)

        reaction_mode = self.mode.currentText().strip() if hasattr(self, "mode") else "OER"
        E2, ln_i, log10_i = self._precompute_ln_i_for_suggestions(E, I, reaction_mode)
        if ln_i.size == 0:
            return

        # cache for preview plotting
        self._preview_E = E2
        self._preview_log10I = log10_i

        wins = self._compute_suggested_windows(E2, ln_i, max_rows=8)
        self._suggested_windows = wins

        # Fill table
        try:
            self.tbl_windows.setRowCount(0)
            for r, w in enumerate(wins):
                self.tbl_windows.insertRow(r)
                vals = [
                    w.get("type", "Auto-suggest"),
                    f'{w.get("e_start", float("nan")):.6g}',
                    f'{w.get("e_end", float("nan")):.6g}',
                    str(int(w.get("n", 0))),
                    f'{w.get("slope_mVdec", float("nan")):.2f}',
                    f'{w.get("r2", float("nan")):.4f}',
                ]
                for c, v in enumerate(vals):
                    self.tbl_windows.setItem(r, c, QTableWidgetItem(v))

            # auto-select first suggestion if present
            if wins:
                self.tbl_windows.selectRow(0)
        except Exception:
            pass
    def _compute_suggested_windows(self, E: np.ndarray, ln_i: np.ndarray, max_rows: int = 8) -> List[dict]:
            """Suggest Tafel windows by scanning fixed-size windows and ranking by R².
    
            IMPORTANT: This expects ln(i_norm) that matches the main analysis preprocessing.
            """
            wins: List[dict] = []
            n = int(E.size)
            if n < 12:
                return []
    
            for frac in (0.08, 0.10, 0.12, 0.15, 0.18, 0.22):
                w = max(8, int(frac * n))
                if w >= n:
                    continue
                step = max(1, w // 4)
                for i0 in range(0, n - w + 1, step):
                    i1 = i0 + w
                    x = E[i0:i1]
                    yln = ln_i[i0:i1]
                    if x.size < 4:
                        continue
                    msk = np.isfinite(x) & np.isfinite(yln)
                    if int(np.sum(msk)) < 4:
                        continue
    
                    fit = _linear_fit(x[msk], yln[msk])
                    if not np.isfinite(fit.r2) or not np.isfinite(fit.slope) or fit.slope == 0:
                        continue
                    if fit.r2 < 0.98:
                        continue

                    expected_sign = getattr(self, "_preview_expected_slope_sign", 1)
                    if expected_sign * fit.slope <= 0:
                        continue

                    if hasattr(self, "chk_emax_upper") and self.chk_emax_upper.isChecked():
                        emax = getattr(self, "_preview_emax_d2", float("nan"))
                        if np.isfinite(emax) and float(x[-1]) > emax:
                            continue

                    taf = abs(math.log(10.0) / fit.slope) * 1000.0
                    wins.append({
                        "type": "Auto-suggest",
                        "e_start": float(x[0]),
                        "e_end": float(x[-1]),
                        "n": int(np.sum(msk)),
                        "slope_mVdec": float(taf),
                        "r2": float(fit.r2),
                    })
    
            wins.sort(key=lambda d: (d["r2"], -abs(d["slope_mVdec"])), reverse=True)
    
            out: List[dict] = []
            seen = set()
            for w in wins:
                key = (round(w["e_start"], 4), round(w["e_end"], 4))
                if key in seen:
                    continue
                seen.add(key)
                out.append(w)
                if len(out) >= max_rows:
                    break
            return out
    
    def _populate_window_table(self, wins: List[dict]) -> None:
        self.tbl_windows.setRowCount(0)
        for r, w in enumerate(wins):
            self.tbl_windows.insertRow(r)
            vals = [
                str(w.get("type", "")),
                f'{w.get("e_start", float("nan")):.6g}',
                f'{w.get("e_end", float("nan")):.6g}',
                str(w.get("n", "")),
                f'{w.get("slope_mVdec", float("nan")):.2f}',
                f'{w.get("r2", float("nan")):.4f}',
            ]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                self.tbl_windows.setItem(r, c, it)

    def _on_window_row_selected(self) -> None:
        """When the user selects a suggested window row, adopt it as the current manual window."""
        try:
            row = self.tbl_windows.currentRow()
            if row < 0:
                return
            wins = getattr(self, "_suggested_windows", None)
            if not wins or row >= len(wins):
                return
            w = wins[row]

            e0 = float(w.get("e_start"))
            e1 = float(w.get("e_end"))

            # fill bounds (for transparency)
            if hasattr(self, "fit_low"):
                self.fit_low.setText(f"{e0:.6g}")
            if hasattr(self, "fit_high"):
                self.fit_high.setText(f"{e1:.6g}")

            # IMPORTANT: treat table selection as a confirmed manual window
            self.manual_window = (min(e0, e1), max(e0, e1))
            self.manual_confirmed = True
            self.analysis_choice = "manual"  # selecting a row implies manual override

            self._plot_preview_window(w)
        except Exception:
            return

    def _plot_preview_window(self, w: dict) -> None:
        E = getattr(self, "_preview_E", None)
        ylog = getattr(self, "_preview_log10I", None)
        if E is None or ylog is None or E.size == 0:
            return
        self.preview_fig.clear()
        ax = self.preview_fig.add_subplot(111)
        ax.plot(E, ylog, linewidth=1.0, label="log10(|I|)")
        a = w.get("e_start", None)
        b = w.get("e_end", None)
        if a is not None and b is not None:
            ax.axvspan(a, b, alpha=0.2, label="Window")
            m = (E >= a) & (E <= b)
            if np.any(m):
                # fit line
                yln = ylog[m] * math.log(10)
                x = E[m]
                aa, bb = np.polyfit(x, yln, 1)
                yhat_ln = aa * x + bb
                yhat = yhat_ln / math.log(10)
                ax.plot(x, yhat, linewidth=2.5, label="Fit")
                taf = w.get("slope_mVdec", float("nan"))
                r2 = w.get("r2", float("nan"))
                ax.text(
                    0.02, 0.98,
                    f"Tafel slope = {taf:.2f} mV/dec\nR² = {r2:.4f}\nWindow: {a:.4g} → {b:.4g}",
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85)
                )
        ax.set_xlabel("E (V)")
        ax.set_ylabel("log10(|I|)")
        ax.legend(loc="best")
        self.preview_canvas.draw()

    def _add_manual_window_to_table(self) -> None:
        """Add current fit_low/fit_high as a new 'Manual' row in the suggestions table."""
        try:
            E = getattr(self, "_preview_E", None)
            ylog = getattr(self, "_preview_log10I", None)
            if E is None or ylog is None or E.size == 0:
                QMessageBox.warning(self, "No data", "Load a file first to create suggestions.")
                return
            a = float(self.fit_low.text().strip())
            b = float(self.fit_high.text().strip())
            if a > b:
                a, b = b, a
            m = (E >= a) & (E <= b)
            if np.count_nonzero(m) < 8:
                QMessageBox.warning(self, "Too few points", "Select at least eight points for the Tafel fit.")
                return
            x = E[m]
            yln = ylog[m] * math.log(10)
            aa, bb = np.polyfit(x, yln, 1)
            yhat = aa * x + bb
            ss_res = float(np.nansum((yln - yhat) ** 2))
            ss_tot = float(np.nansum((yln - float(np.nanmean(yln))) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            taf = abs(math.log(10.0) / aa) * 1000.0 if aa != 0 else float("nan")
            wins = getattr(self, "_suggested_windows", [])
            wins = list(wins)
            wins.append({
                "type": "Manual",
                "e_start": float(x[0]),
                "e_end": float(x[-1]),
                "n": int(x.size),
                "slope_mVdec": float(taf),
                "r2": float(r2),
            })
            self._suggested_windows = wins
            self._populate_window_table(wins)
            self.tbl_windows.selectRow(len(wins) - 1)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add manual window: {e}")

def main():
    # This tells Windows that this is not a Python program and is a standalone application
    # so that it can show your custom icon in the taskbar
    try:
        myappid = 'researcher.Analysistoolkit.1.0' # A unique identifier for your software
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass # If the user's operating system is not Windows (such as Mac or Linux), this section is ignored.

    app = QApplication(sys.argv)
    
    # Set the icon for the entire application (this will also cause dialog boxes to get icons)
    app.setWindowIcon(QIcon("Researcherlogo.png"))
    
    win = ResearcherTafelWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
