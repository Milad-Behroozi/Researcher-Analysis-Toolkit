"""Researcher Analysis Toolkit — Manual EIS Fitting Pro."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import re
import sys
import copy
from datetime import datetime
from pathlib import Path
from fpdf import FPDF
from impedance.models.circuits.fitting import (
    buildCircuit, check_and_eval, extract_circuit_elements
)
from impedance.models.circuits.elements import circuit_elements, get_element_from_name
try:
    from impedance.validation import linKK
except ImportError:
    linKK = None
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageTk


def enable_high_dpi_awareness():
    """Enable per-monitor DPI awareness before the Tk root is created (Windows)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        # Windows 10+: modern per-monitor DPI awareness mode.
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except Exception:
        pass
    try:
        import ctypes
        # Windows 8.1 fallback.
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        import ctypes
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


# It must run before ``tk.Tk()``; calling it here is intentional.
enable_high_dpi_awareness()


ELEMENT_CATALOG = {
    "R": {
        "title": "Resistor",
        "category": "Basic",
        "description": "Frequency-independent ohmic resistance, commonly used for solution, film, contact, or charge-transfer resistance.",
        "formula": r"$Z=R$",
        "formula_text": "Z = R",
        "parameters": "R [Ohm]",
        "behavior": "A point on the real axis.",
        "color": "#2c3e50",
    },
    "C": {
        "title": "Capacitor",
        "category": "Basic",
        "description": "Ideal capacitive response. Use only when the phase behavior is sufficiently close to an ideal capacitor.",
        "formula": r"$Z=\frac{1}{j\omega C}$",
        "formula_text": "Z = 1/(j omega C)",
        "parameters": "C [F]",
        "behavior": "Vertical capacitive line; phase approaches -90 degrees.",
        "color": "#2980b9",
    },
    "L": {
        "title": "Inductor",
        "category": "Basic",
        "description": "Ideal inductance, often used to represent wiring or high-frequency inductive behavior.",
        "formula": r"$Z=j\omega L$",
        "formula_text": "Z = j omega L",
        "parameters": "L [H]",
        "behavior": "Inductive response below the Nyquist real axis when -Z'' is plotted.",
        "color": "#8e44ad",
    },
    "CPE": {
        "title": "Constant Phase Element",
        "category": "Basic",
        "description": "Non-ideal capacitance used for distributed relaxation, roughness, or heterogeneous interfaces.",
        "formula": r"$Z=\frac{1}{Q(j\omega)^\alpha}$",
        "formula_text": "Z = 1/[Q (j omega)^alpha]",
        "parameters": "Q [Ohm^-1 s^alpha], alpha [-], 0 < alpha <= 1",
        "behavior": "A depressed capacitive response; alpha = 1 reduces to an ideal capacitor.",
        "color": "#2471a3",
    },
    "W": {
        "title": "Semi-infinite Warburg",
        "category": "Diffusion",
        "description": "Semi-infinite linear diffusion. It should not be used when the spectrum bends to a finite-resistance or vertical low-frequency limit.",
        "formula": r"$Z=\frac{A_W}{\sqrt{\omega}}(1-j)$",
        "formula_text": "Z = Aw (1-j)/sqrt(omega)",
        "parameters": "Aw [Ohm s^-1/2]",
        "behavior": "A 45-degree line in the Nyquist plot.",
        "color": "#d35400",
    },
    "Wo": {
        "title": "Open finite-length Warburg",
        "category": "Diffusion",
        "description": "Finite diffusion with a blocking/reflective boundary. Appropriate only when finite diffusion and low-frequency accumulation are physically justified.",
        "formula": r"$Z=\frac{Z_0}{\sqrt{j\omega\tau}}\,\coth\!\left(\sqrt{j\omega\tau}\right)$",
        "formula_text": "Z = Z0*coth[sqrt(j omega tau)]/sqrt(j omega tau)",
        "parameters": "Z0 [Ohm], tau [s]",
        "behavior": "Approximately 45 degrees at intermediate frequency and near-vertical at low frequency.",
        "color": "#e67e22",
    },
    "Ws": {
        "title": "Short finite-length Warburg",
        "category": "Diffusion",
        "description": "Finite diffusion with a transmissive boundary. The low-frequency response approaches a finite real resistance.",
        "formula": r"$Z=\frac{Z_0}{\sqrt{j\omega\tau}}\,\tanh\!\left(\sqrt{j\omega\tau}\right)$",
        "formula_text": "Z = Z0*tanh[sqrt(j omega tau)]/sqrt(j omega tau)",
        "parameters": "Z0 [Ohm], tau [s]",
        "behavior": "Approximately 45 degrees at intermediate frequency and resistive at low frequency.",
        "color": "#ca6f1e",
    },
    "La": {
        "title": "Modified Inductance",
        "category": "Advanced",
        "description": "A non-ideal or distributed inductive element.",
        "formula": r"$Z=L(j\omega)^\alpha$",
        "formula_text": "Z = L (j omega)^alpha",
        "parameters": "L [H s], alpha [-]",
        "behavior": "A distributed inductive response.",
        "color": "#7d3c98",
    },
    "Zarc": {
        "title": "ZARC / Cole-Cole element",
        "category": "Advanced",
        "description": "An R-CPE relaxation parameterized directly by resistance, characteristic time, and dispersion exponent.",
        "formula": r"$Z=\frac{R}{1+(j\omega\tau)^\gamma}$",
        "formula_text": "Z = R/[1 + (j omega tau)^gamma]",
        "parameters": "R [Ohm], tau [s], gamma [-]",
        "behavior": "A depressed semicircle represented as one composite element.",
        "color": "#117864",
    },
    "G": {
        "title": "Gerischer",
        "category": "Advanced",
        "description": "Coupled reaction-diffusion response described by a semi-infinite Gerischer impedance.",
        "formula": r"$Z=\frac{R_G}{\sqrt{1+j\omega t_G}}$",
        "formula_text": "Z = RG/sqrt(1 + j omega tG)",
        "parameters": "RG [Ohm], tG [s]",
        "behavior": "Coupled kinetic and diffusion response.",
        "color": "#148f77",
    },
    "Gs": {
        "title": "Finite-length Gerischer",
        "category": "Advanced",
        "description": "Finite-length Gerischer response for coupled reaction and diffusion in a bounded domain.",
        "formula": r"$Z=\frac{R_G}{\sqrt{1+j\omega t_G}\tanh(\phi\sqrt{1+j\omega t_G})}$",
        "formula_text": "Z = RG/{sqrt(1+j omega tG) tanh[phi sqrt(1+j omega tG)]}",
        "parameters": "RG [Ohm], tG [s], phi [-]",
        "behavior": "Finite reaction-diffusion response.",
        "color": "#0e6655",
    },
    "TLMQ": {
        "title": "Transmission Line Model (CPE)",
        "category": "Advanced",
        "description": "Simplified porous-electrode transmission-line element with ionic resistance and a distributed surface CPE.",
        "formula": r"$Z=\sqrt{R_{ion}Z_S}\,\coth\!\sqrt{R_{ion}/Z_S}$",
        "formula_text": "Z = sqrt(Rion Zs) coth[sqrt(Rion/Zs)]",
        "parameters": "Rion [Ohm], Qs [F s^(gamma-1)], gamma [-]",
        "behavior": "Distributed porous-electrode response.",
        "color": "#1f618d",
    },
    "T": {
        "title": "Macrohomogeneous porous electrode",
        "category": "Advanced",
        "description": "Paasch-type macrohomogeneous porous-electrode model. Use only with a defensible physical assignment of its four parameters.",
        "formula": r"$Z=A\frac{\coth\beta}{\beta}+B\frac{1}{\beta\sinh\beta}$",
        "formula_text": "Z = A*coth(beta)/beta + B/[beta sinh(beta)]",
        "parameters": "A [Ohm m^2], B [Ohm m^2], a [-], b [s]",
        "behavior": "Macrohomogeneous porous-electrode response.",
        "color": "#1b4f72",
    },
    "K": {
        "title": "Lin-KK RC element",
        "category": "Validation",
        "description": "Internal RC relaxation used by linear Kramers-Kronig validation. It is normally not selected as a physical equivalent-circuit element.",
        "formula": r"$Z=\frac{R}{1+j\omega\tau_k}$",
        "formula_text": "Z = R/(1 + j omega tau_k)",
        "parameters": "R [Ohm], tau_k [s]",
        "behavior": "One ideal Debye relaxation used in a Lin-KK basis.",
        "color": "#566573",
    },
}

ELEMENT_ORDER = ["R", "C", "CPE", "L", "W", "Wo", "Ws", "La", "Zarc", "G", "Gs", "TLMQ", "T", "K"]


class HoverToolTip:
    """Small delayed tooltip that also works with ttk widgets."""

    def __init__(self, widget, text, delay=450, wraplength=430):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.wraplength = wraplength
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None):
        self._cancel()
        self._after_id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self):
        if self._tip is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        try:
            self._tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self._tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self._tip,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#fffbea",
            foreground="#17202a",
            padx=9,
            pady=7,
            wraplength=self.wraplength,
            font=("Segoe UI", 9),
        )
        label.pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip is not None:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def element_tooltip_text(element_type):
    meta = ELEMENT_CATALOG[element_type]
    return (
        f"{element_type} — {meta['title']}\n"
        f"{meta['description']}\n\n"
        f"{meta['formula_text']}\n"
        f"Parameters: {meta['parameters']}\n"
        f"Nyquist behavior: {meta['behavior']}"
    )

# =========================================================
# Helper Functions
# =========================================================
def _get_parameter_info(circuit_str, constants=None):
    """Return parameter metadata using impedance.py's own circuit definitions."""
    constants = constants or {}
    info = []

    for element_name in extract_circuit_elements(circuit_str):
        raw_element = get_element_from_name(element_name)
        element = check_and_eval(raw_element)

        if element.num_params > 1:
            for idx in range(element.num_params):
                full_name = f"{element_name}_{idx}"
                if element_name in constants or full_name in constants:
                    continue
                info.append({
                    "name": full_name,
                    "element": element_name,
                    "raw": raw_element,
                    "index": idx,
                    "unit": element.units[idx],
                })
        else:
            if element_name in constants:
                continue
            info.append({
                "name": element_name,
                "element": element_name,
                "raw": raw_element,
                "index": 0,
                "unit": element.units[0],
            })

    return info


def _extract_param_names(circuit_str, constants=None):
    """Extract parameter names through impedance.py (supports multi-parameter elements)."""
    return [item["name"] for item in _get_parameter_info(circuit_str, constants)]


def _is_shape_exponent(name):
    """Dimensionless 0..1 exponents stay linear; positive scale parameters use log space."""
    raw = get_element_from_name(name).upper()
    try:
        parameter_index = int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return False
    return (
        (raw in {"CPE", "LA"} and parameter_index == 1)
        or (raw in {"ZARC", "TLMQ"} and parameter_index == 2)
    )


def calculate_metrics(Z_exp, Z_fit, n_free_params=0):
    """Calculate raw and modulus-weighted residual metrics."""
    resid = np.asarray(Z_exp) - np.asarray(Z_fit)
    abs_z = np.maximum(np.abs(Z_exp), 1e-15)
    relative = np.abs(resid) / abs_z

    rmse = float(np.sqrt(np.mean(np.abs(resid) ** 2)))
    mean_rel_err = float(np.mean(relative))
    rel_max = float(np.max(relative))

    weighted_components = np.hstack([
        resid.real / abs_z,
        resid.imag / abs_z,
    ])
    weighted_sse = float(np.sum(weighted_components ** 2))
    dof = max(2 * len(Z_exp) - int(n_free_params), 1)
    reduced_weighted_error = weighted_sse / dof

    return rmse, mean_rel_err, rel_max, weighted_sse, reduced_weighted_error


def make_safe_bounds(param_names):
    """Broad physical safety bounds. Adaptive bounds are derived from these later."""
    lower_bounds = []
    upper_bounds = []

    for name in param_names:
        upper_name = name.upper()

        if _is_shape_exponent(name):
            lower_bounds.append(0.4)
            upper_bounds.append(1.0)
        elif upper_name.startswith("R"):
            lower_bounds.append(1e-9)
            upper_bounds.append(1e9)
        elif upper_name.startswith("CPE") and upper_name.endswith("_0"):
            lower_bounds.append(1e-14)
            upper_bounds.append(10.0)
        elif upper_name.startswith("C"):
            lower_bounds.append(1e-14)
            upper_bounds.append(10.0)
        elif upper_name.startswith("W"):
            lower_bounds.append(1e-12)
            upper_bounds.append(1e12)
        elif upper_name.startswith("L"):
            lower_bounds.append(1e-12)
            upper_bounds.append(1e6)
        else:
            lower_bounds.append(1e-14)
            upper_bounds.append(1e12)

    return np.asarray(lower_bounds, dtype=float), np.asarray(upper_bounds, dtype=float)


def make_adaptive_bounds(param_names, initial_guess):
    """Tighten very broad safety bounds around a physically informed starting point."""
    safe_lo, safe_hi = make_safe_bounds(param_names)
    initial_guess = np.asarray(initial_guess, dtype=float)
    lo = safe_lo.copy()
    hi = safe_hi.copy()

    for i, name in enumerate(param_names):
        if _is_shape_exponent(name):
            continue

        value = max(abs(initial_guess[i]), safe_lo[i] * 10.0, 1e-30)
        # Ten decades of search around a smart seed is still broad, but much
        # better conditioned than blindly spanning the full safety interval.
        lo[i] = max(safe_lo[i], value * 1e-5)
        hi[i] = min(safe_hi[i], value * 1e5)

        if not np.isfinite(hi[i]) or hi[i] <= lo[i]:
            lo[i], hi[i] = safe_lo[i], safe_hi[i]

    return lo, hi


def _clip_initial_guess(values, lower, upper):
    values = np.asarray(values, dtype=float).copy()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    # Add clearance in the parameter's natural geometry. Using a fraction of
    # the full *linear* span is unsafe for decade-wide bounds (e.g. it pushed a
    # valid Wo tau seed of milliseconds up to 1000 seconds).
    for i in range(len(values)):
        if lower[i] > 0 and upper[i] > 0:
            lo_i = lower[i] * (1.0 + 1e-9)
            hi_i = upper[i] * (1.0 - 1e-9)
        else:
            eps = max(abs(upper[i] - lower[i]) * 1e-9, 1e-15)
            lo_i, hi_i = lower[i] + eps, upper[i] - eps
        if hi_i <= lo_i:
            lo_i, hi_i = lower[i], upper[i]
        values[i] = min(max(values[i], lo_i), hi_i)
    return values


def _to_optimizer_space(values, param_names):
    values = np.asarray(values, dtype=float)
    transformed = np.empty_like(values)
    for i, (value, name) in enumerate(zip(values, param_names)):
        if _is_shape_exponent(name):
            transformed[i] = value
        else:
            transformed[i] = np.log10(max(value, 1e-300))
    return transformed


def _from_optimizer_space(values, param_names):
    values = np.asarray(values, dtype=float)
    physical = np.empty_like(values)
    for i, (value, name) in enumerate(zip(values, param_names)):
        if _is_shape_exponent(name):
            physical[i] = value
        else:
            physical[i] = 10.0 ** value
    return physical


def _bounds_to_optimizer_space(lower, upper, param_names):
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lo_t = np.empty_like(lower)
    hi_t = np.empty_like(upper)

    for i, name in enumerate(param_names):
        if _is_shape_exponent(name):
            lo_t[i], hi_t[i] = lower[i], upper[i]
        else:
            lo_t[i] = np.log10(max(lower[i], 1e-300))
            hi_t[i] = np.log10(max(upper[i], lower[i] * 10.0))

    return lo_t, hi_t


def _predict_impedance(circuit_str, frequencies, free_params, constants=None):
    """Evaluate an impedance.py circuit without changing any UI state."""
    constants = constants or {}
    expression = buildCircuit(
        circuit_str,
        np.asarray(frequencies, dtype=float),
        *np.asarray(free_params, dtype=float),
        constants=constants,
        eval_string='',
        index=0,
    )[0]
    return np.asarray(eval(expression, circuit_elements), dtype=complex)


def _weighted_residual_vector(Z_exp, Z_fit):
    abs_z = np.maximum(np.abs(Z_exp), 1e-15)
    return np.hstack([
        (Z_fit.real - Z_exp.real) / abs_z,
        (Z_fit.imag - Z_exp.imag) / abs_z,
    ])


def _weighted_sse(Z_exp, Z_fit):
    residual = _weighted_residual_vector(Z_exp, Z_fit)
    return float(np.dot(residual, residual))

# =========================================================
# Circuit Parser Helpers
# =========================================================
def split_top_level(s, sep):
    parts = []
    depth = 0
    current = ""

    for ch in s:
        if ch == "(":
            depth += 1
            current += ch
        elif ch == ")":
            depth -= 1
            current += ch
        elif ch == sep and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch

    if current.strip():
        parts.append(current.strip())

    return parts


def parse_circuit_expr(expr):
    expr = expr.strip()

    # Parallel block: p(...)
    if expr.startswith("p(") and expr.endswith(")"):
        inner = expr[2:-1]
        branches = split_top_level(inner, ",")
        return {
            "type": "parallel",
            "branches": [parse_circuit_expr(b) for b in branches]
        }

    # Series block: R0-R1-C1
    parts = split_top_level(expr, "-")
    if len(parts) > 1:
        return {
            "type": "series",
            "items": [parse_circuit_expr(p) for p in parts]
        }

    # Single element
    return {
        "type": "element",
        "name": expr
    }


def element_kind(name):
    upper = name.upper()

    if upper.startswith("CPE"):
        return "cpe"
    elif upper.startswith("R"):
        return "resistor"
    elif upper.startswith("C"):
        return "capacitor"
    elif upper.startswith("W"):
        return "warburg"
    elif upper.startswith("L"):
        return "inductor"
    else:
        return "unknown"


def tree_to_pretty_text(node):
    if node["type"] == "element":
        return node["name"]

    if node["type"] == "series":
        return " - ".join(tree_to_pretty_text(item) for item in node["items"])

    if node["type"] == "parallel":
        return "[ " + " || ".join(tree_to_pretty_text(b) for b in node["branches"]) + " ]"

    return "Unknown"


def tree_to_circuit_expr(node):
    """Serialize the graphical circuit tree to impedance.py circuit syntax."""
    if node is None:
        return ""
    if node["type"] == "element":
        return node["name"]
    if node["type"] == "series":
        return "-".join(tree_to_circuit_expr(item) for item in node["items"])
    if node["type"] == "parallel":
        return "p(" + ",".join(tree_to_circuit_expr(branch) for branch in node["branches"]) + ")"
    raise ValueError(f"Unknown circuit node type: {node.get('type')}")


def iter_element_names(node):
    if node is None:
        return
    if node["type"] == "element":
        yield node["name"]
        return
    key = "items" if node["type"] == "series" else "branches"
    for child in node[key]:
        yield from iter_element_names(child)


def normalize_circuit_tree(node):
    """Flatten redundant containers and remove empty containers after edits."""
    if node is None or node.get("type") == "element":
        return node

    key = "items" if node["type"] == "series" else "branches"
    children = []
    for child in node.get(key, []):
        child = normalize_circuit_tree(child)
        if child is None:
            continue
        if child.get("type") == node["type"]:
            children.extend(child[key])
        else:
            children.append(child)

    if not children:
        return None
    if len(children) == 1:
        return children[0]
    return {"type": node["type"], key: children}


def find_element_path(node, element_name, path=()):
    if node is None:
        return None
    if node["type"] == "element":
        return path if node["name"] == element_name else None
    key = "items" if node["type"] == "series" else "branches"
    for index, child in enumerate(node[key]):
        found = find_element_path(child, element_name, path + (index,))
        if found is not None:
            return found
    return None


def get_node_at_path(node, path):
    current = node
    for index in path:
        key = "items" if current["type"] == "series" else "branches"
        current = current[key][index]
    return current


def get_parent_at_path(node, path):
    if not path:
        return None, None, None
    parent = get_node_at_path(node, path[:-1])
    key = "items" if parent["type"] == "series" else "branches"
    return parent, key, path[-1]


def replace_node_at_path(root, path, replacement):
    if not path:
        return replacement
    parent, key, index = get_parent_at_path(root, path)
    parent[key][index] = replacement
    return root


def validate_circuit_expression(expression):
    """Validate syntax, unique labels, topology, and installed element support."""
    expression = re.sub(r"\s+", "", expression or "")
    if not expression:
        raise ValueError("Circuit expression is empty.")
    if expression.count("(") != expression.count(")"):
        raise ValueError("The circuit contains unbalanced parentheses.")

    tree = normalize_circuit_tree(parse_circuit_expr(expression))
    if tree is None:
        raise ValueError("Circuit expression is empty.")

    names = list(iter_element_names(tree))
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise ValueError("Each element label must be unique. Duplicate: " + ", ".join(duplicates))

    for name in names:
        if not re.fullmatch(r"[A-Za-z]+\d+", name):
            raise ValueError(
                f"Invalid element label '{name}'. Use an element type followed by an index, e.g. R0 or CPE1."
            )
        raw = get_element_from_name(name)
        try:
            check_and_eval(raw)
        except Exception as exc:
            raise ValueError(f"Element '{raw}' is not available in the installed impedance.py version.") from exc

    # Let impedance.py perform its own parameter extraction as a final parser check.
    _get_parameter_info(tree_to_circuit_expr(tree))
    return tree, tree_to_circuit_expr(tree)


def next_element_name(node, element_type):
    pattern = re.compile(rf"^{re.escape(element_type)}(\d+)$", re.IGNORECASE)
    used = []
    for name in iter_element_names(node):
        match = pattern.match(name)
        if match:
            used.append(int(match.group(1)))
    index = 0
    while index in used:
        index += 1
    return f"{element_type}{index}"


class ElementLibraryDialog(tk.Toplevel):
    """Searchable reference for every circuit element exposed by the program."""

    def __init__(self, master, on_insert=None):
        super().__init__(master)
        self.on_insert = on_insert
        self.title("EIS Element Library")
        self.transient(master)
        scale = getattr(master, "ui_scale", 1.0)
        screen_w = master.winfo_screenwidth()
        screen_h = master.winfo_screenheight()
        width = int(min(screen_w * 0.86, 1120 * scale))
        height = int(min(screen_h * 0.82, 720 * scale))
        width = max(min(820, max(screen_w - 50, 700)), width)
        height = max(min(560, max(screen_h - 90, 500)), height)
        self.geometry(f"{width}x{height}")
        self.minsize(min(760, max(screen_w - 80, 680)), min(520, max(screen_h - 120, 470)))

        top = ttk.Frame(self, padding=(12, 10))
        top.pack(fill="x")
        ttk.Label(top, text="Equivalent-Circuit Element Library", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(
            top,
            text="Select an element to see its physical meaning, impedance equation, parameters, and limiting behavior.",
            wraplength=900,
        ).pack(anchor="w", pady=(3, 0))

        pane = ttk.Panedwindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=12, pady=(0, 10))

        left = ttk.Frame(pane, padding=4)
        right = ttk.Frame(pane, padding=(14, 8))
        pane.add(left, weight=1)
        pane.add(right, weight=3)

        search_row = ttk.Frame(left)
        search_row.pack(fill="x", pady=(0, 6))
        ttk.Label(search_row, text="Filter:").pack(side="left")
        self.search_var = tk.StringVar()
        search_entry = ttk.Entry(search_row, textvariable=self.search_var)
        search_entry.pack(side="left", fill="x", expand=True, padx=(5, 0))
        self.search_var.trace_add("write", lambda *_: self._populate_tree())

        tree_wrap = ttk.Frame(left)
        tree_wrap.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_wrap, show="tree", selectmode="browse")
        scroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.tree.bind("<Double-1>", lambda _e: self._insert_selected())

        self.detail_title = ttk.Label(right, text="", font=("Segoe UI", 16, "bold"))
        self.detail_title.pack(anchor="w")
        self.availability_label = ttk.Label(right, text="")
        self.availability_label.pack(anchor="w", pady=(2, 10))

        self.detail_description = ttk.Label(right, text="", wraplength=720, justify="left")
        self.detail_description.pack(fill="x", anchor="w")

        formula_frame = ttk.LabelFrame(right, text="Impedance equation")
        formula_frame.pack(fill="x", pady=12)
        self.formula_fig = plt.Figure(figsize=(6.4, 1.15), dpi=100, facecolor="white")
        self.formula_ax = self.formula_fig.add_subplot(111)
        self.formula_ax.axis("off")
        self.formula_canvas = FigureCanvasTkAgg(self.formula_fig, master=formula_frame)
        self.formula_canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=5)

        self.parameter_label = ttk.Label(right, text="", wraplength=720, justify="left")
        self.parameter_label.pack(fill="x", anchor="w", pady=(0, 8))
        self.behavior_label = ttk.Label(right, text="", wraplength=720, justify="left")
        self.behavior_label.pack(fill="x", anchor="w")

        note = ttk.Label(
            right,
            text=(
                "Model-selection note: a low residual does not by itself justify an element. "
                "Use only elements that match the electrode physics and the observed frequency limits."
            ),
            wraplength=720,
            justify="left",
            foreground="#7d6608",
        )
        note.pack(fill="x", anchor="w", pady=(18, 4))

        bottom = ttk.Frame(self, padding=(12, 0, 12, 12))
        bottom.pack(fill="x")
        self.insert_button = ttk.Button(bottom, text="Insert / Select Element", command=self._insert_selected)
        if self.on_insert is None:
            self.insert_button.configure(state="disabled")
        self.insert_button.pack(side="right", padx=(6, 0))
        ttk.Button(bottom, text="Close", command=self.destroy).pack(side="right")

        self._populate_tree()
        self.after_idle(lambda: pane.sashpos(0, max(230, int(width * 0.27))))
        search_entry.focus_set()

    @staticmethod
    def _is_available(element_type):
        try:
            check_and_eval(element_type)
            return True
        except Exception:
            return False

    def _populate_tree(self):
        selected_type = self._selected_type()
        self.tree.delete(*self.tree.get_children())
        query = self.search_var.get().strip().lower()
        categories = ["Basic", "Diffusion", "Advanced", "Validation"]

        for category in categories:
            matching = []
            for element_type in ELEMENT_ORDER:
                meta = ELEMENT_CATALOG[element_type]
                haystack = " ".join([
                    element_type, meta["title"], meta["description"], meta["parameters"], meta["behavior"]
                ]).lower()
                if meta["category"] == category and (not query or query in haystack):
                    matching.append(element_type)
            if not matching:
                continue
            parent = self.tree.insert("", "end", text=category, open=True)
            for element_type in matching:
                suffix = "" if self._is_available(element_type) else "  (not installed)"
                item = self.tree.insert(parent, "end", text=f"{element_type} — {ELEMENT_CATALOG[element_type]['title']}{suffix}")
                self.tree.item(item, tags=(f"element:{element_type}",))
                if element_type == selected_type:
                    self.tree.selection_set(item)

        if not self.tree.selection():
            for parent in self.tree.get_children():
                children = self.tree.get_children(parent)
                if children:
                    self.tree.selection_set(children[0])
                    self.tree.focus(children[0])
                    break
        self._on_select()

    def _selected_type(self):
        selection = self.tree.selection() if hasattr(self, "tree") else ()
        if not selection:
            return None
        tags = self.tree.item(selection[0], "tags")
        for tag in tags:
            if tag.startswith("element:"):
                return tag.split(":", 1)[1]
        return None

    def _on_select(self, _event=None):
        element_type = self._selected_type()
        if element_type is None:
            return
        meta = ELEMENT_CATALOG[element_type]
        available = self._is_available(element_type)
        self.detail_title.configure(text=f"{element_type} — {meta['title']}")
        self.availability_label.configure(
            text="Available in the installed impedance.py package" if available else "Not available in the installed impedance.py package",
            foreground="#1e8449" if available else "#b03a2e",
        )
        self.detail_description.configure(text=meta["description"])
        self.parameter_label.configure(text=f"Parameters: {meta['parameters']}")
        self.behavior_label.configure(text=f"Nyquist / frequency behavior: {meta['behavior']}")
        self.formula_ax.clear()
        self.formula_ax.axis("off")
        try:
            self.formula_ax.text(0.5, 0.52, meta["formula"], ha="center", va="center", fontsize=19)
        except Exception:
            self.formula_ax.text(0.5, 0.52, meta["formula_text"], ha="center", va="center", fontsize=14)
        self.formula_canvas.draw_idle()
        if self.on_insert is not None:
            self.insert_button.configure(state="normal" if available and element_type != "K" else "disabled")

    def _insert_selected(self):
        element_type = self._selected_type()
        if element_type and self.on_insert is not None and self._is_available(element_type) and element_type != "K":
            self.on_insert(element_type)
            self.destroy()


class CircuitBuilderDialog(tk.Toplevel):
    """Graphical series/parallel circuit editor synchronized with impedance.py syntax."""

    def __init__(self, master, initial_expression, on_apply):
        super().__init__(master)
        self.title("Graphical Equivalent-Circuit Builder")
        self.transient(master)
        self.on_apply = on_apply
        self.ui_scale = getattr(master, "ui_scale", 1.0)
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = int(min(screen_w * 0.94, 1480 * self.ui_scale))
        height = int(min(screen_h * 0.90, 900 * self.ui_scale))
        width = max(min(980, max(screen_w - 40, 760)), width)
        height = max(min(650, max(screen_h - 80, 560)), height)
        self.geometry(f"{width}x{height}")
        self.minsize(min(920, max(screen_w - 60, 740)), min(620, max(screen_h - 110, 540)))

        self.circuit_tree = None
        self.selected_name = None
        self.active_element_type = "R"
        self.history = []
        self.redo_stack = []
        self.zoom = 1.0
        self._internal_expression_update = False
        self._expression_after_id = None
        self._drag_ghost = None
        self._drag_element_type = None

        try:
            self.circuit_tree, canonical = validate_circuit_expression(initial_expression)
        except Exception:
            self.circuit_tree, canonical = validate_circuit_expression("R0-p(R1,CPE1)")
        names = list(iter_element_names(self.circuit_tree))
        self.selected_name = names[-1] if names else None

        self.expression_var = tk.StringVar(value=canonical)
        self.drop_mode_var = tk.StringVar(value="series")
        self.status_var = tk.StringVar(value="Circuit is valid.")
        self.selected_info_var = tk.StringVar(value="")
        self._build_ui()
        self.expression_var.trace_add("write", self._schedule_expression_sync)
        self.after_idle(self.redraw)

    def _build_ui(self):
        header = tk.Frame(self, bg="#2c3e50")
        header.pack(fill="x")
        tk.Label(
            header,
            text="Graphical Equivalent-Circuit Builder",
            bg="#2c3e50",
            fg="white",
            font=("Segoe UI", 16, "bold"),
        ).pack(side="left", padx=16, pady=10)
        tk.Label(
            header,
            text="Series / parallel topology with synchronized circuit syntax",
            bg="#2c3e50",
            fg="#d6eaf8",
            font=("Segoe UI", 9),
        ).pack(side="left", padx=(8, 0), pady=12)

        expr_frame = ttk.LabelFrame(self, text="Circuit Expression")
        expr_frame.pack(fill="x", padx=12, pady=(10, 6))
        self.expression_entry = ttk.Entry(expr_frame, textvariable=self.expression_var, font=("Courier New", 11))
        self.expression_entry.pack(side="left", fill="x", expand=True, padx=(9, 6), pady=8)
        ttk.Button(expr_frame, text="Validate & Redraw", command=self.load_expression).pack(side="left", padx=(0, 8), pady=8)
        self.expression_entry.bind("<Return>", lambda _e: self.load_expression())

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=4)

        palette_panel = ttk.Frame(body, padding=(4, 2))
        canvas_panel = ttk.Frame(body, padding=(4, 2))
        action_panel = ttk.Frame(body, padding=(4, 2))
        body.add(palette_panel, weight=1)
        body.add(canvas_panel, weight=4)
        body.add(action_panel, weight=1)

        ttk.Label(palette_panel, text="Element Palette", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(2, 5))
        ttk.Label(
            palette_panel,
            text="Click an element, then add it in series or parallel. You can also drag it onto the drawing area.",
            wraplength=220,
            justify="left",
        ).pack(fill="x", anchor="w", pady=(0, 6))

        palette_notebook = ttk.Notebook(palette_panel)
        palette_notebook.pack(fill="both", expand=True)
        for category in ["Basic", "Diffusion", "Advanced"]:
            tab = ttk.Frame(palette_notebook, padding=6)
            palette_notebook.add(tab, text=category)
            row = 0
            for element_type in ELEMENT_ORDER:
                meta = ELEMENT_CATALOG[element_type]
                if meta["category"] != category or element_type == "K":
                    continue
                available = ElementLibraryDialog._is_available(element_type)
                button = ttk.Button(
                    tab,
                    text=f"{element_type}   {meta['title']}",
                    command=lambda t=element_type: self.select_palette_element(t),
                    state="normal" if available else "disabled",
                )
                button.grid(row=row, column=0, sticky="ew", pady=3)
                tab.columnconfigure(0, weight=1)
                if available:
                    button.bind("<ButtonPress-1>", lambda e, t=element_type: self._start_palette_drag(e, t), add="+")
                    button.bind("<B1-Motion>", self._move_palette_drag, add="+")
                    button.bind("<ButtonRelease-1>", self._finish_palette_drag, add="+")
                HoverToolTip(button, element_tooltip_text(element_type))
                row += 1

        ttk.Button(
            palette_panel,
            text="ⓘ Open Full Element Library",
            command=lambda: ElementLibraryDialog(self, on_insert=self.select_palette_element),
        ).pack(fill="x", pady=(7, 0))

        canvas_toolbar = ttk.Frame(canvas_panel)
        canvas_toolbar.pack(fill="x", pady=(0, 5))
        ttk.Label(canvas_toolbar, text="Drop action:").pack(side="left")
        ttk.Radiobutton(canvas_toolbar, text="Series", variable=self.drop_mode_var, value="series").pack(side="left", padx=(5, 2))
        ttk.Radiobutton(canvas_toolbar, text="Parallel to selected", variable=self.drop_mode_var, value="parallel").pack(side="left", padx=2)
        ttk.Button(canvas_toolbar, text="−", width=3, command=lambda: self.change_zoom(-0.1)).pack(side="right", padx=(2, 0))
        ttk.Button(canvas_toolbar, text="+", width=3, command=lambda: self.change_zoom(0.1)).pack(side="right", padx=2)
        ttk.Label(canvas_toolbar, text="Zoom").pack(side="right", padx=(4, 2))

        canvas_wrap = ttk.Frame(canvas_panel)
        canvas_wrap.pack(fill="both", expand=True)
        self.circuit_canvas = tk.Canvas(
            canvas_wrap,
            bg="white",
            highlightthickness=1,
            highlightbackground="#aeb6bf",
            cursor="arrow",
        )
        hscroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self.circuit_canvas.xview)
        vscroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self.circuit_canvas.yview)
        self.circuit_canvas.configure(xscrollcommand=hscroll.set, yscrollcommand=vscroll.set)
        self.circuit_canvas.grid(row=0, column=0, sticky="nsew")
        vscroll.grid(row=0, column=1, sticky="ns")
        hscroll.grid(row=1, column=0, sticky="ew")
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.circuit_canvas.bind("<Button-1>", self._canvas_background_click)
        self.circuit_canvas.bind("<Configure>", lambda _e: self.redraw())

        selected_bar = ttk.Frame(canvas_panel)
        selected_bar.pack(fill="x", pady=(5, 0))
        ttk.Label(selected_bar, textvariable=self.selected_info_var, wraplength=750, justify="left").pack(anchor="w")

        ttk.Label(action_panel, text="Build Tools", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(2, 6))
        self.active_label = ttk.Label(action_panel, text="Selected palette element: R", wraplength=210, justify="left")
        self.active_label.pack(fill="x", pady=(0, 6))
        ttk.Button(action_panel, text="Add in Series", command=self.add_series).pack(fill="x", pady=3)
        ttk.Button(action_panel, text="Add in Parallel", command=self.add_parallel).pack(fill="x", pady=3)
        ttk.Separator(action_panel).pack(fill="x", pady=7)
        ttk.Button(action_panel, text="Duplicate Selected", command=self.duplicate_selected).pack(fill="x", pady=3)
        ttk.Button(action_panel, text="Delete Selected", command=self.delete_selected).pack(fill="x", pady=3)
        move_row = ttk.Frame(action_panel)
        move_row.pack(fill="x", pady=3)
        ttk.Button(move_row, text="← Move", command=lambda: self.move_selected(-1)).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(move_row, text="Move →", command=lambda: self.move_selected(1)).pack(side="left", fill="x", expand=True, padx=(2, 0))
        ttk.Separator(action_panel).pack(fill="x", pady=7)
        history_row = ttk.Frame(action_panel)
        history_row.pack(fill="x", pady=3)
        ttk.Button(history_row, text="Undo", command=self.undo).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(history_row, text="Redo", command=self.redo).pack(side="left", fill="x", expand=True, padx=(2, 0))
        ttk.Button(action_panel, text="Clear Circuit", command=self.clear_circuit).pack(fill="x", pady=3)
        ttk.Button(action_panel, text="Auto Arrange / Fit View", command=self.auto_arrange).pack(fill="x", pady=3)
        ttk.Button(action_panel, text="Export Schematic", command=self.export_schematic).pack(fill="x", pady=3)

        template_frame = ttk.LabelFrame(action_panel, text="Templates")
        template_frame.pack(fill="x", pady=(10, 3))
        self.template_var = tk.StringVar(value="Rs-(R||CPE)")
        templates = [
            "Rs-(R||CPE)",
            "Rs-(R||CPE)-CPE",
            "Rs-(R||CPE)-Wo",
            "Two time constants",
        ]
        ttk.Combobox(template_frame, textvariable=self.template_var, values=templates, state="readonly").pack(fill="x", padx=5, pady=5)
        ttk.Button(template_frame, text="Load Template", command=self.load_template).pack(fill="x", padx=5, pady=(0, 5))

        status = ttk.Label(
            self,
            textvariable=self.status_var,
            anchor="w",
            justify="left",
            foreground="#1e8449",
            padding=(14, 4),
        )
        status.pack(fill="x")
        self.status_label = status

        bottom = ttk.Frame(self, padding=(12, 6, 12, 12))
        bottom.pack(fill="x")
        ttk.Button(bottom, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(bottom, text="Apply Circuit to Main Window", command=self.apply_to_main).pack(side="right", padx=(0, 7))

        self.after_idle(lambda: self._set_initial_sashes(body))

    def _set_initial_sashes(self, pane):
        try:
            width = max(self.winfo_width(), 980)
            pane.sashpos(0, max(215, int(width * 0.17)))
            pane.sashpos(1, max(720, int(width * 0.81)))
        except Exception:
            pass

    def select_palette_element(self, element_type):
        if element_type not in ELEMENT_CATALOG:
            return
        self.active_element_type = element_type
        self.active_label.configure(text=f"Selected palette element: {element_type} — {ELEMENT_CATALOG[element_type]['title']}")
        self.status_var.set(f"{element_type} selected. Add it in series or parallel, or drag it onto the canvas.")
        self.status_label.configure(foreground="#1f618d")

    def _start_palette_drag(self, event, element_type):
        self.select_palette_element(element_type)
        self._drag_element_type = element_type
        try:
            self._drag_ghost = tk.Toplevel(self)
            self._drag_ghost.overrideredirect(True)
            self._drag_ghost.attributes("-topmost", True)
            tk.Label(
                self._drag_ghost,
                text=f"  {element_type}  ",
                bg="#d6eaf8",
                fg="#154360",
                relief="solid",
                borderwidth=1,
                font=("Segoe UI", 10, "bold"),
            ).pack()
            self._drag_ghost.geometry(f"+{event.x_root + 12}+{event.y_root + 12}")
        except Exception:
            self._drag_ghost = None

    def _move_palette_drag(self, event):
        if self._drag_ghost is not None:
            self._drag_ghost.geometry(f"+{event.x_root + 12}+{event.y_root + 12}")

    def _finish_palette_drag(self, event):
        try:
            x0 = self.circuit_canvas.winfo_rootx()
            y0 = self.circuit_canvas.winfo_rooty()
            x1 = x0 + self.circuit_canvas.winfo_width()
            y1 = y0 + self.circuit_canvas.winfo_height()
            inside = x0 <= event.x_root <= x1 and y0 <= event.y_root <= y1
            if inside and self._drag_element_type:
                cx = self.circuit_canvas.canvasx(event.x_root - x0)
                cy = self.circuit_canvas.canvasy(event.y_root - y0)
                overlapping = self.circuit_canvas.find_overlapping(cx - 4, cy - 4, cx + 4, cy + 4)
                for item in reversed(overlapping):
                    for tag in self.circuit_canvas.gettags(item):
                        if tag.startswith("element::"):
                            self.selected_name = tag.split("::", 1)[1]
                            break
                if self.drop_mode_var.get() == "parallel":
                    self.add_parallel()
                else:
                    self.add_series()
        finally:
            if self._drag_ghost is not None:
                try:
                    self._drag_ghost.destroy()
                except Exception:
                    pass
            self._drag_ghost = None
            self._drag_element_type = None

    def _schedule_expression_sync(self, *_args):
        if self._internal_expression_update:
            return
        if self._expression_after_id is not None:
            try:
                self.after_cancel(self._expression_after_id)
            except Exception:
                pass
        self._expression_after_id = self.after(650, self.load_expression)

    def _set_expression_from_tree(self):
        self._internal_expression_update = True
        try:
            self.expression_var.set(tree_to_circuit_expr(self.circuit_tree))
        finally:
            self._internal_expression_update = False

    def _commit_tree(self, new_tree, selected_name=None, record=True):
        new_tree = normalize_circuit_tree(new_tree)
        old_expression = tree_to_circuit_expr(self.circuit_tree)
        new_expression = tree_to_circuit_expr(new_tree)
        if record and old_expression != new_expression:
            self.history.append(old_expression)
            self.history = self.history[-100:]
            self.redo_stack.clear()
        self.circuit_tree = new_tree
        names = list(iter_element_names(new_tree)) if new_tree is not None else []
        if selected_name in names:
            self.selected_name = selected_name
        elif self.selected_name not in names:
            self.selected_name = names[-1] if names else None
        self._set_expression_from_tree()
        self.redraw()
        if new_tree is None:
            self.status_var.set("Circuit is empty. Add an element before applying it.")
            self.status_label.configure(foreground="#b03a2e")
        else:
            self.status_var.set("Circuit updated and syntax synchronized.")
            self.status_label.configure(foreground="#1e8449")

    def load_expression(self):
        self._expression_after_id = None
        expression = self.expression_var.get().strip()
        try:
            tree, canonical = validate_circuit_expression(expression)
            if tree_to_circuit_expr(self.circuit_tree) != canonical:
                self._commit_tree(tree, record=True)
            else:
                self.circuit_tree = tree
                self._set_expression_from_tree()
                self.redraw()
            self.status_var.set("Circuit expression is valid and synchronized.")
            self.status_label.configure(foreground="#1e8449")
        except Exception as exc:
            self.status_var.set(f"Invalid circuit: {exc}")
            self.status_label.configure(foreground="#b03a2e")

    def add_series(self):
        element_type = self.active_element_type
        if not ElementLibraryDialog._is_available(element_type):
            self.status_var.set(f"{element_type} is not available in the installed impedance.py package.")
            self.status_label.configure(foreground="#b03a2e")
            return
        tree = copy.deepcopy(self.circuit_tree)
        name = next_element_name(tree, element_type)
        new_node = {"type": "element", "name": name}

        if tree is None:
            tree = new_node
        elif self.selected_name:
            path = find_element_path(tree, self.selected_name)
            if path is None:
                tree = {"type": "series", "items": [tree, new_node]}
            else:
                parent, key, index = get_parent_at_path(tree, path)
                if parent is not None and parent["type"] == "series":
                    parent[key].insert(index + 1, new_node)
                else:
                    selected_node = copy.deepcopy(get_node_at_path(tree, path))
                    tree = replace_node_at_path(
                        tree,
                        path,
                        {"type": "series", "items": [selected_node, new_node]},
                    )
        elif tree.get("type") == "series":
            tree["items"].append(new_node)
        else:
            tree = {"type": "series", "items": [tree, new_node]}
        self._commit_tree(tree, selected_name=name)

    def add_parallel(self):
        element_type = self.active_element_type
        if not ElementLibraryDialog._is_available(element_type):
            self.status_var.set(f"{element_type} is not available in the installed impedance.py package.")
            self.status_label.configure(foreground="#b03a2e")
            return
        tree = copy.deepcopy(self.circuit_tree)
        name = next_element_name(tree, element_type)
        new_node = {"type": "element", "name": name}
        if tree is None:
            tree = new_node
        else:
            target_name = self.selected_name
            path = find_element_path(tree, target_name) if target_name else ()
            if path is None:
                path = ()
            parent, key, index = get_parent_at_path(tree, path)
            if parent is not None and parent["type"] == "parallel":
                parent[key].insert(index + 1, new_node)
            else:
                selected_node = copy.deepcopy(get_node_at_path(tree, path)) if path else copy.deepcopy(tree)
                replacement = {"type": "parallel", "branches": [selected_node, new_node]}
                tree = replace_node_at_path(tree, path, replacement)
        self._commit_tree(tree, selected_name=name)

    def duplicate_selected(self):
        if not self.selected_name or self.circuit_tree is None:
            self.status_var.set("Select an element to duplicate.")
            self.status_label.configure(foreground="#b9770e")
            return
        self.active_element_type = get_element_from_name(self.selected_name)
        self.add_series()

    def delete_selected(self):
        if not self.selected_name or self.circuit_tree is None:
            self.status_var.set("Select an element to delete.")
            self.status_label.configure(foreground="#b9770e")
            return
        tree = copy.deepcopy(self.circuit_tree)
        path = find_element_path(tree, self.selected_name)
        if path is None:
            return
        if not path:
            tree = None
        else:
            parent, key, index = get_parent_at_path(tree, path)
            del parent[key][index]
        self._commit_tree(tree, selected_name=None)

    def move_selected(self, direction):
        if not self.selected_name or self.circuit_tree is None:
            self.status_var.set("Select an element to move.")
            self.status_label.configure(foreground="#b9770e")
            return
        tree = copy.deepcopy(self.circuit_tree)
        path = find_element_path(tree, self.selected_name)
        if not path:
            self.status_var.set("The selected element is already the complete circuit.")
            return
        parent, key, index = get_parent_at_path(tree, path)
        new_index = index + direction
        if new_index < 0 or new_index >= len(parent[key]):
            self.status_var.set("The selected element cannot move farther in this branch.")
            return
        parent[key][index], parent[key][new_index] = parent[key][new_index], parent[key][index]
        self._commit_tree(tree, selected_name=self.selected_name)

    def clear_circuit(self):
        if self.circuit_tree is not None and messagebox.askyesno("Clear Circuit", "Remove every element from the builder?", parent=self):
            self._commit_tree(None)

    def undo(self):
        if not self.history:
            return
        current = tree_to_circuit_expr(self.circuit_tree)
        previous = self.history.pop()
        self.redo_stack.append(current)
        if previous:
            tree, _ = validate_circuit_expression(previous)
        else:
            tree = None
        self._commit_tree(tree, record=False)

    def redo(self):
        if not self.redo_stack:
            return
        current = tree_to_circuit_expr(self.circuit_tree)
        upcoming = self.redo_stack.pop()
        self.history.append(current)
        if upcoming:
            tree, _ = validate_circuit_expression(upcoming)
        else:
            tree = None
        self._commit_tree(tree, record=False)

    def load_template(self):
        templates = {
            "Rs-(R||CPE)": "R0-p(R1,CPE1)",
            "Rs-(R||CPE)-CPE": "R0-p(R1,CPE1)-CPE2",
            "Rs-(R||CPE)-Wo": "R0-p(R1,CPE1)-Wo1",
            "Two time constants": "R0-p(R1,CPE1)-p(R2,CPE2)",
        }
        expression = templates[self.template_var.get()]
        tree, _ = validate_circuit_expression(expression)
        self._commit_tree(tree, selected_name=list(iter_element_names(tree))[-1])

    def apply_to_main(self):
        try:
            _tree, canonical = validate_circuit_expression(tree_to_circuit_expr(self.circuit_tree))
        except Exception as exc:
            messagebox.showerror("Invalid Circuit", str(exc), parent=self)
            return
        self.on_apply(canonical)
        self.destroy()

    def change_zoom(self, delta):
        self.zoom = min(1.65, max(0.65, self.zoom + delta))
        self.redraw()

    def auto_arrange(self):
        self.zoom = 1.0
        self.redraw()
        self.circuit_canvas.xview_moveto(0.0)
        self.circuit_canvas.yview_moveto(0.0)

    def export_schematic(self):
        if self.circuit_tree is None:
            messagebox.showwarning("Empty Circuit", "Build a circuit before exporting it.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Equivalent-Circuit Schematic",
            defaultextension=".svg",
            filetypes=[("Scalable Vector Graphics", "*.svg"), ("PNG image", "*.png"), ("PDF document", "*.pdf")],
        )
        if not path:
            return
        try:
            self.master.generate_circuit_image(tree_to_circuit_expr(self.circuit_tree), path)
            messagebox.showinfo("Schematic Exported", f"Circuit schematic saved to:\n{path}", parent=self)
        except Exception as exc:
            messagebox.showerror("Export Error", str(exc), parent=self)

    def _canvas_background_click(self, event):
        overlapping = self.circuit_canvas.find_overlapping(
            self.circuit_canvas.canvasx(event.x - 2),
            self.circuit_canvas.canvasy(event.y - 2),
            self.circuit_canvas.canvasx(event.x + 2),
            self.circuit_canvas.canvasy(event.y + 2),
        )
        if not any(any(tag.startswith("element::") for tag in self.circuit_canvas.gettags(item)) for item in overlapping):
            self.selected_name = None
            self.redraw()

    def _select_canvas_element(self, element_name):
        self.selected_name = element_name
        self.redraw()

    def _hover_canvas_element(self, element_name, entering=True):
        if entering:
            raw = get_element_from_name(element_name)
            meta = ELEMENT_CATALOG.get(raw)
            if meta:
                self.selected_info_var.set(
                    f"{element_name}: {meta['title']}   |   {meta['formula_text']}   |   {meta['parameters']}"
                )
            self.circuit_canvas.configure(cursor="hand2")
        else:
            self._update_selected_info()
            self.circuit_canvas.configure(cursor="arrow")

    def _update_selected_info(self):
        if not self.selected_name:
            self.selected_info_var.set("Click an element to select it. Drag the canvas scrollbars or use zoom for larger circuits.")
            return
        raw = get_element_from_name(self.selected_name)
        meta = ELEMENT_CATALOG.get(raw)
        if meta:
            self.selected_info_var.set(
                f"Selected: {self.selected_name} — {meta['title']}   |   {meta['formula_text']}   |   {meta['parameters']}"
            )

    def _measure_node(self, node):
        z = self.zoom
        element_w, element_h = 126 * z, 64 * z
        series_gap = 22 * z
        parallel_pad = 64 * z
        branch_gap = 26 * z
        if node["type"] == "element":
            return element_w, element_h
        if node["type"] == "series":
            sizes = [self._measure_node(item) for item in node["items"]]
            return sum(w for w, _h in sizes) + series_gap * (len(sizes) - 1), max(h for _w, h in sizes)
        sizes = [self._measure_node(branch) for branch in node["branches"]]
        return max(w for w, _h in sizes) + 2 * parallel_pad, sum(h for _w, h in sizes) + branch_gap * (len(sizes) - 1)

    def redraw(self):
        if not hasattr(self, "circuit_canvas"):
            return
        canvas = self.circuit_canvas
        canvas.delete("all")
        viewport_w = max(canvas.winfo_width(), 650)
        viewport_h = max(canvas.winfo_height(), 420)

        if self.circuit_tree is None:
            canvas.create_text(
                viewport_w / 2,
                viewport_h / 2,
                text="Circuit is empty\nSelect an element and choose Add in Series, or drag an element here.",
                fill="#7f8c8d",
                font=("Segoe UI", 13, "bold"),
                justify="center",
            )
            canvas.configure(scrollregion=(0, 0, viewport_w, viewport_h))
            self._update_selected_info()
            return

        width, height = self._measure_node(self.circuit_tree)
        margin = 58 * self.zoom
        total_w = max(viewport_w, width + 2 * margin)
        total_h = max(viewport_h, height + 2 * margin)

        grid = max(int(24 * self.zoom), 16)
        for gx in range(0, int(total_w) + grid, grid):
            canvas.create_line(gx, 0, gx, total_h, fill="#f4f6f7", tags=("grid",))
        for gy in range(0, int(total_h) + grid, grid):
            canvas.create_line(0, gy, total_w, gy, fill="#f4f6f7", tags=("grid",))

        x = max(margin, (total_w - width) / 2)
        y = max(margin, (total_h - height) / 2)
        start, end = self._draw_node(self.circuit_tree, x, y)
        terminal_r = max(4, int(4 * self.zoom))
        lead = 24 * self.zoom
        canvas.create_line(start[0] - lead, start[1], start[0], start[1], fill="#17202a", width=2)
        canvas.create_oval(start[0] - lead - terminal_r, start[1] - terminal_r, start[0] - lead + terminal_r, start[1] + terminal_r, outline="#17202a", width=2, fill="white")
        canvas.create_line(end[0], end[1], end[0] + lead, end[1], fill="#17202a", width=2)
        canvas.create_oval(end[0] + lead - terminal_r, end[1] - terminal_r, end[0] + lead + terminal_r, end[1] + terminal_r, outline="#17202a", width=2, fill="white")
        canvas.tag_lower("grid")
        canvas.configure(scrollregion=(0, 0, total_w, total_h))
        self._update_selected_info()

    def _draw_node(self, node, x, y):
        if node["type"] == "element":
            width, height = self._measure_node(node)
            cy = y + height / 2
            self._draw_element_symbol(node["name"], x, cy, width, height)
            return (x, cy), (x + width, cy)

        if node["type"] == "series":
            _width, height = self._measure_node(node)
            cursor_x = x
            previous_end = None
            first_start = None
            last_end = None
            gap = 22 * self.zoom
            for child in node["items"]:
                child_w, child_h = self._measure_node(child)
                child_y = y + (height - child_h) / 2
                child_start, child_end = self._draw_node(child, cursor_x, child_y)
                if previous_end is not None:
                    self.circuit_canvas.create_line(previous_end[0], previous_end[1], child_start[0], child_start[1], fill="#17202a", width=2)
                if first_start is None:
                    first_start = child_start
                previous_end = child_end
                last_end = child_end
                cursor_x += child_w + gap
            return first_start, last_end

        width, height = self._measure_node(node)
        pad = 64 * self.zoom
        branch_gap = 26 * self.zoom
        sizes = [self._measure_node(branch) for branch in node["branches"]]
        max_child_w = max(w for w, _h in sizes)
        left_bus_x = x + pad * 0.45
        right_bus_x = x + width - pad * 0.45
        branch_y = y
        branch_centers = []
        branch_endpoints = []

        for branch, (branch_w, branch_h) in zip(node["branches"], sizes):
            child_x = x + pad + (max_child_w - branch_w) / 2
            start, end = self._draw_node(branch, child_x, branch_y)
            branch_centers.append(start[1])
            branch_endpoints.append((start, end))
            branch_y += branch_h + branch_gap

        center_y = y + height / 2
        self.circuit_canvas.create_line(x, center_y, left_bus_x, center_y, fill="#17202a", width=2)
        self.circuit_canvas.create_line(right_bus_x, center_y, x + width, center_y, fill="#17202a", width=2)
        self.circuit_canvas.create_line(left_bus_x, min(branch_centers), left_bus_x, max(branch_centers), fill="#17202a", width=2)
        self.circuit_canvas.create_line(right_bus_x, min(branch_centers), right_bus_x, max(branch_centers), fill="#17202a", width=2)
        for start, end in branch_endpoints:
            self.circuit_canvas.create_line(left_bus_x, start[1], start[0], start[1], fill="#17202a", width=2)
            self.circuit_canvas.create_line(end[0], end[1], right_bus_x, end[1], fill="#17202a", width=2)
            self.circuit_canvas.create_oval(left_bus_x - 2.5, start[1] - 2.5, left_bus_x + 2.5, start[1] + 2.5, fill="#17202a", outline="")
            self.circuit_canvas.create_oval(right_bus_x - 2.5, end[1] - 2.5, right_bus_x + 2.5, end[1] + 2.5, fill="#17202a", outline="")
        return (x, center_y), (x + width, center_y)

    def _draw_element_symbol(self, name, x, cy, width, height):
        canvas = self.circuit_canvas
        raw = get_element_from_name(name)
        kind = element_kind(name)
        tag = f"element::{name}"
        tags = ("circuit_element", tag)
        lead = 19 * self.zoom
        x1 = x + lead
        x2 = x + width - lead
        color = ELEMENT_CATALOG.get(raw, {}).get("color", "#2c3e50")
        line_width = max(2, int(2 * self.zoom))

        if name == self.selected_name:
            canvas.create_rectangle(
                x + 3,
                cy - height * 0.43,
                x + width - 3,
                cy + height * 0.43,
                fill="#eaf2f8",
                outline="#3498db",
                width=max(2, int(2 * self.zoom)),
                tags=tags,
            )

        canvas.create_line(x, cy, x1, cy, fill="#17202a", width=line_width, tags=tags)
        canvas.create_line(x2, cy, x + width, cy, fill="#17202a", width=line_width, tags=tags)

        if kind == "resistor":
            points = [x1, cy]
            segments = 8
            step = (x2 - x1) / segments
            for i in range(1, segments):
                offset = (-1 if i % 2 else 1) * 10 * self.zoom
                points.extend([x1 + i * step, cy + offset])
            points.extend([x2, cy])
            canvas.create_line(*points, fill=color, width=line_width, tags=tags)
        elif kind == "capacitor":
            gap = 6 * self.zoom
            plate = 17 * self.zoom
            canvas.create_line(x1, cy, (x1 + x2) / 2 - gap, cy, fill="#17202a", width=line_width, tags=tags)
            canvas.create_line((x1 + x2) / 2 + gap, cy, x2, cy, fill="#17202a", width=line_width, tags=tags)
            canvas.create_line((x1 + x2) / 2 - gap, cy - plate, (x1 + x2) / 2 - gap, cy + plate, fill=color, width=line_width, tags=tags)
            canvas.create_line((x1 + x2) / 2 + gap, cy - plate, (x1 + x2) / 2 + gap, cy + plate, fill=color, width=line_width, tags=tags)
        elif kind == "cpe":
            gap = 7 * self.zoom
            plate = 17 * self.zoom
            mid = (x1 + x2) / 2
            canvas.create_line(x1, cy, mid - gap, cy, fill="#17202a", width=line_width, tags=tags)
            canvas.create_line(mid + gap, cy, x2, cy, fill="#17202a", width=line_width, tags=tags)
            canvas.create_line(mid - gap, cy - plate, mid - gap, cy + plate, fill=color, width=line_width, tags=tags)
            canvas.create_line(mid + gap + 4 * self.zoom, cy - plate, mid + gap - 4 * self.zoom, cy + plate, fill=color, width=line_width, tags=tags)
        elif kind == "inductor":
            points = []
            samples = 44
            for i in range(samples + 1):
                px = x1 + (x2 - x1) * i / samples
                py = cy - 9 * self.zoom * np.sin(4 * np.pi * i / samples)
                points.extend([px, py])
            canvas.create_line(*points, fill=color, width=line_width, smooth=True, tags=tags)
        elif kind == "warburg":
            box_h = 22 * self.zoom
            canvas.create_rectangle(x1, cy - box_h / 2, x2, cy + box_h / 2, outline=color, width=line_width, fill="white", tags=tags)
            canvas.create_line(x1 + 7 * self.zoom, cy + box_h / 2, x1 + 18 * self.zoom, cy - box_h / 2, fill=color, width=line_width, tags=tags)
            canvas.create_line(x1 + 16 * self.zoom, cy + box_h / 2, x1 + 27 * self.zoom, cy - box_h / 2, fill=color, width=line_width, tags=tags)
            canvas.create_text((x1 + x2) / 2 + 9 * self.zoom, cy, text=raw, fill=color, font=("Segoe UI", max(8, int(9 * self.zoom)), "bold"), tags=tags)
        else:
            box_h = 24 * self.zoom
            canvas.create_rectangle(x1, cy - box_h / 2, x2, cy + box_h / 2, outline=color, width=line_width, fill="white", tags=tags)
            canvas.create_text((x1 + x2) / 2, cy, text=raw, fill=color, font=("Segoe UI", max(8, int(9 * self.zoom)), "bold"), tags=tags)

        canvas.create_text(
            x + width / 2,
            cy - 24 * self.zoom,
            text=name,
            fill="#17202a",
            font=("Segoe UI", max(8, int(10 * self.zoom)), "bold"),
            tags=tags,
        )
        canvas.tag_bind(tag, "<Button-1>", lambda _e, n=name: (self._select_canvas_element(n), "break")[1])
        canvas.tag_bind(tag, "<Enter>", lambda _e, n=name: self._hover_canvas_element(n, True))
        canvas.tag_bind(tag, "<Leave>", lambda _e, n=name: self._hover_canvas_element(n, False))

# =========================================================
# Main Application UI
# =========================================================
class ResearcherEISPro(tk.Tk):
    def __init__(self):
        super().__init__()
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                    "researcher.analysis.toolkit.eis.pro"
                )
            except Exception:
                pass

        # Use the monitor-reported DPI rather than assuming a 96-DPI display.
        try:
            self.monitor_dpi = float(self.winfo_fpixels("1i"))
        except Exception:
            self.monitor_dpi = 96.0
        self.ui_scale = min(2.25, max(0.90, self.monitor_dpi / 96.0))
        self.tk.call("tk", "scaling", max(1.0, self.monitor_dpi / 72.0))

        self.title("Researcher Analysis Toolkit - Manual EIS Fitting Pro")
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        initial_w = min(int(screen_w * 0.94), int(1550 * self.ui_scale))
        initial_h = min(int(screen_h * 0.90), int(960 * self.ui_scale))
        initial_w = max(min(1040, max(screen_w - 40, 760)), initial_w)
        initial_h = max(min(650, max(screen_h - 80, 560)), initial_h)
        self.geometry(f"{initial_w}x{initial_h}")
        self.minsize(min(980, max(screen_w - 60, 760)), min(620, max(screen_h - 100, 540)))

        # Improve default readability without changing the established visual style.
        style = ttk.Style(self)
        try:
            style.configure("TButton", padding=(5, 4))
            style.configure("TCombobox", padding=2)
            style.configure("TLabelframe.Label", font=("Segoe UI", 9, "bold"))
        except Exception:
            pass
        
        # Apply application icon (ico format)
        try:
            self.iconbitmap('Researcherlogo.ico') 
        except Exception:
            pass # Ignore if the ico file is not found

        # Paths & Constants
        self.software_name = "Researcher Analysis Toolkit - Manual EIS Fitting Pro"
        self.base_output_path = os.path.join(os.path.expanduser("~"), "Desktop", "Researcher Output", "Nyquist and Bode Analysis")
        
        # Data variables
        self.df = None
        self.f = None
        self.Z = None
        self.Phase_exp = None
        self.Mod_exp = None
        self.Z_fit = None
        
        # Backup variables for reset functionality
        self.f_orig = None
        self.Z_orig = None
        self.Phase_orig = None
        self.Mod_orig = None
        
        self.prune_mode = False
        self.param_vars = {}
        self.current_circuit_str = ""

        # Fitting diagnostics (backend only; UI layout remains unchanged)
        self.param_errors = {}
        self.fit_bounds = {}
        self.fit_diagnostics = {}
        self.lin_kk_summary = {}
        self.imag_sign_auto_flipped = False

        self.setup_ui()

    def setup_ui(self):
        # 1. Header
        header_height = int(min(132, max(76, 80 * self.ui_scale)))
        logo_size = int(min(100, max(56, 60 * self.ui_scale)))
        header_frame = tk.Frame(self, bg="#2c3e50", height=header_height)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)

        # High-quality logo resizing using Pillow
        try:
            original_image = Image.open("Researcherlogo.png")
            resized_image = original_image.resize((logo_size, logo_size), Image.Resampling.LANCZOS)
            self.logo_img = ImageTk.PhotoImage(resized_image)
            tk.Label(header_frame, image=self.logo_img, bg="#2c3e50").pack(side="left", padx=int(18 * self.ui_scale), pady=8)
        except Exception as e:
            print(f"Logo error: {e}")
            # Fallback to a drawn canvas logo if image is missing
            self.logo_canvas = tk.Canvas(header_frame, width=logo_size, height=logo_size, bg="#2c3e50", highlightthickness=0)
            self.logo_canvas.pack(side="left", padx=int(18 * self.ui_scale), pady=8)
            inset = max(4, int(5 * self.ui_scale))
            self.logo_canvas.create_oval(inset, inset, logo_size - inset, logo_size - inset, outline="white", width=2)
            self.logo_canvas.create_text(logo_size / 2, logo_size / 2, text="RE", fill="white", font=("Arial", 14, "bold"))

        logical_screen_width = self.winfo_screenwidth() / max(self.ui_scale, 0.1)
        title_size = 20 if logical_screen_width >= 1300 else (17 if logical_screen_width >= 1000 else 14)
        tk.Label(
            header_frame,
            text=self.software_name,
            fg="white",
            bg="#2c3e50",
            font=("Segoe UI", title_size, "bold"),
        ).pack(side="left", pady=20)

        # 2. Main Layout (non-collapsible two-column grid)
        # A two-column grid is intentionally used instead of PanedWindow.
        # Some Windows/HiDPI Tk combinations can initialize a PanedWindow pane
        # at zero width. A grid column with a hard minimum cannot collapse.
        self.main_pane = ttk.Frame(self)
        self.main_pane.pack(fill="both", expand=True, padx=10, pady=10)
        self.main_pane.rowconfigure(0, weight=1)
        
        # =======================================================
        # Professional scroll system for the left panel
        # =======================================================
        initial_left_width = int(max(360, 390 * self.ui_scale))
        left_container = ttk.Frame(self.main_pane, width=initial_left_width)
        self.left_container = left_container
        self.main_pane.columnconfigure(0, weight=0, minsize=initial_left_width)
        self.main_pane.columnconfigure(1, weight=1, minsize=int(max(520, 520 * self.ui_scale)))
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.left_canvas = tk.Canvas(left_container, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_container, orient="vertical", command=self.left_canvas.yview)
        
        self.left_frame = ttk.Frame(self.left_canvas)

        def configure_inner_frame(event):
            self.left_canvas.itemconfig(self.canvas_window, width=event.width)

        self.left_frame.bind("<Configure>", lambda e: self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all")))
        self.canvas_window = self.left_canvas.create_window((0, 0), window=self.left_frame, anchor="nw")
        self.left_canvas.bind("<Configure>", configure_inner_frame)
        self.left_canvas.configure(yscrollcommand=left_scrollbar.set)

        self.left_canvas.pack(side="left", fill="both", expand=True)
        left_scrollbar.pack(side="right", fill="y")

        # Scroll only when the pointer is over the left pane; do not globally
        # unbind mouse-wheel handlers used by plots or the circuit builder.
        def _on_mousewheel(event):
            try:
                px, py = self.winfo_pointerx(), self.winfo_pointery()
                x0, y0 = self.left_canvas.winfo_rootx(), self.left_canvas.winfo_rooty()
                inside = x0 <= px <= x0 + self.left_canvas.winfo_width() and y0 <= py <= y0 + self.left_canvas.winfo_height()
                if inside:
                    delta = int(-event.delta / 120) if getattr(event, "delta", 0) else 0
                    if delta:
                        self.left_canvas.yview_scroll(delta, "units")
                        return "break"
            except Exception:
                pass

        self.bind_all("<MouseWheel>", _on_mousewheel, add="+")
        # =======================================================

        # --- Data Loading ---
        load_frame = ttk.LabelFrame(self.left_frame, text="1. Data Input & Mapping")
        load_frame.pack(fill="x", padx=5, pady=5)
        
        top_load = ttk.Frame(load_frame)
        top_load.pack(fill="x", padx=5, pady=5)
        ttk.Button(top_load, text="📂 Load EIS Data", command=self.load_data).pack(side="left", padx=5)
        self.lbl_file = ttk.Label(top_load, text="No file loaded", font=("Arial", 9, "italic"))
        self.lbl_file.pack(side="left", padx=5)

        ttk.Label(load_frame, text="Freq/Hz:").pack(anchor="w", padx=5)
        self.cb_freq = ttk.Combobox(load_frame, state="readonly")
        self.cb_freq.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(load_frame, text="Z'/ohm:").pack(anchor="w", padx=5)
        self.cb_real = ttk.Combobox(load_frame, state="readonly")
        self.cb_real.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(load_frame, text='Z"/ohm:').pack(anchor="w", padx=5)
        self.cb_imag = ttk.Combobox(load_frame, state="readonly")
        self.cb_imag.pack(fill="x", padx=5, pady=2)

        ttk.Button(load_frame, text="Apply Columns & Plot", command=self.apply_columns).pack(fill="x", padx=10, pady=10)

        # --- Data Cleaning ---
        clean_frame = ttk.LabelFrame(self.left_frame, text="2. Data Pruning (Nyquist Plot)")
        clean_frame.pack(fill="x", padx=5, pady=5)
        self.btn_prune = ttk.Button(clean_frame, text="✂️ Enable Prune Mode", command=self.toggle_prune_mode, state="disabled")
        self.btn_prune.pack(side="left", fill="x", expand=True, padx=2, pady=5)
        self.btn_reset_data = ttk.Button(clean_frame, text="↩️ Reset Data", command=self.reset_data, state="disabled")
        self.btn_reset_data.pack(side="right", fill="x", expand=True, padx=2, pady=5)

        # --- Circuit Frame ---
        circ_frame = ttk.LabelFrame(self.left_frame, text="3. Circuit Definition")
        circ_frame.pack(fill="x", padx=5, pady=5)
        self.ent_circuit = ttk.Entry(circ_frame, font=("Courier New", 12))
        self.ent_circuit.pack(fill="x", padx=10, pady=5)
        self.ent_circuit.insert(0, "R0-p(R1,CPE1)")

        element_row = ttk.Frame(circ_frame)
        element_row.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Label(element_row, text="Elements:").pack(side="left", padx=(0, 3))
        for element_type in ["R", "C", "CPE", "L", "W", "Wo", "Ws"]:
            button = ttk.Button(
                element_row,
                text=element_type,
                width=max(3, len(element_type) + 1),
                command=lambda t=element_type: self.open_circuit_builder(t),
            )
            button.pack(side="left", padx=1)
            HoverToolTip(button, element_tooltip_text(element_type))

        circuit_tools = ttk.Frame(circ_frame)
        circuit_tools.pack(fill="x", padx=8, pady=(0, 4))
        ttk.Button(circuit_tools, text="Graphical Builder", command=self.open_circuit_builder).pack(side="left", fill="x", expand=True, padx=(0, 2))
        ttk.Button(circuit_tools, text="Element Library", command=self.open_element_library).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(circuit_tools, text="Validate", command=lambda: self.validate_main_circuit(show_success=True)).pack(side="left", fill="x", expand=True, padx=(2, 0))

        self.lbl_circuit_status = ttk.Label(circ_frame, text="Circuit syntax ready.", foreground="#1e8449")
        self.lbl_circuit_status.pack(fill="x", padx=10, pady=(0, 3))
        self.ent_circuit.bind("<KeyRelease>", self._on_circuit_text_changed)
        self.btn_fit = ttk.Button(circ_frame, text="🎯 Run Smart Initial Fit", command=self.initial_fit, state="disabled")
        self.btn_fit.pack(fill="x", padx=10, pady=5)

        # --- Parameters Frame ---
        self.param_frame = ttk.LabelFrame(self.left_frame, text="4. Interactive Parameters")
        self.param_frame.pack(fill="both", expand=True, padx=5, pady=5)

        act_frame = ttk.Frame(self.left_frame)
        act_frame.pack(fill="x", padx=5, pady=5)
        self.btn_refit = ttk.Button(act_frame, text="🔄 Re-Fit with Locks", command=self.refit_with_locks, state="disabled")
        self.btn_refit.pack(fill="x", padx=2, pady=2)

        self.lbl_metrics = ttk.Label(self.left_frame, text="Metrics:\n-", justify="left", font=("Arial", 9))
        self.lbl_metrics.pack(fill="x", padx=10, pady=5)

        # --- Export Actions ---
        export_frame = ttk.LabelFrame(self.left_frame, text="5. Final Output")
        export_frame.pack(fill="x", padx=5, pady=5)
        
        # Only the main Export Pro Report button remains
        ttk.Button(export_frame, text="💾 Export Pro Report (PDF) & Excel", command=self.export_pro_results).pack(fill="x", padx=10, pady=10)

        # Right Panel (Plots)
        self.right_frame = ttk.Frame(self.main_pane)
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.fig, (self.ax_nyq, self.ax_bode) = plt.subplots(2, 1, figsize=(8, 10))
        self.ax_phase_twin = self.ax_bode.twinx()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.plot_toolbar = NavigationToolbar2Tk(self.canvas, self.right_frame, pack_toolbar=False)
        self.plot_toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
        self.fig.tight_layout()

        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        self._show_startup_plot_placeholder()
        self.after_idle(self._apply_initial_layout)
        self.after(180, self._apply_initial_layout)
        self.after(700, self._apply_initial_layout)

    def _apply_initial_layout(self):
        """Reassert the non-collapsible control-column width after Windows mapping."""
        try:
            total_width = self.main_pane.winfo_width()
            if total_width < 500:
                self.after(120, self._apply_initial_layout)
                return
            desired = int(max(360, 390 * self.ui_scale))
            maximum_reasonable = max(320, int(total_width * 0.42))
            desired = min(desired, maximum_reasonable)
            self.main_pane.columnconfigure(0, minsize=desired)
            self.left_container.configure(width=desired)
        except Exception:
            pass

    def _show_startup_plot_placeholder(self):
        """Show an intentional startup state instead of two unexplained empty axes."""
        self.ax_nyq.clear()
        self.ax_bode.clear()
        self.ax_phase_twin.clear()
        self.ax_nyq.set_axis_off()
        self.ax_bode.set_axis_off()
        self.ax_phase_twin.set_axis_off()
        self.ax_nyq.text(
            0.5,
            0.48,
            "Load an EIS data file to begin",
            ha="center",
            va="center",
            transform=self.ax_nyq.transAxes,
            fontsize=17,
            fontweight="bold",
            color="#566573",
        )
        self.ax_bode.text(
            0.5,
            0.52,
            "Map frequency, Z' and Z\" columns, then click Apply Columns & Plot",
            ha="center",
            va="center",
            transform=self.ax_bode.transAxes,
            fontsize=11,
            color="#7f8c8d",
        )
        self.canvas.draw_idle()

    def open_element_library(self):
        ElementLibraryDialog(self)

    def open_circuit_builder(self, preselect=None):
        dialog = CircuitBuilderDialog(
            self,
            self.ent_circuit.get().strip() or "R0-p(R1,CPE1)",
            self._apply_builder_circuit,
        )
        if preselect:
            dialog.select_palette_element(preselect)

    def _apply_builder_circuit(self, expression):
        old_expression = self.ent_circuit.get().strip()
        self.ent_circuit.delete(0, tk.END)
        self.ent_circuit.insert(0, expression)
        self.lbl_circuit_status.configure(text="Valid circuit received from graphical builder.", foreground="#1e8449")
        if expression != old_expression and self.Z_fit is not None:
            self._invalidate_fit(clear_parameters=True)

    def validate_main_circuit(self, show_success=False):
        try:
            _tree, canonical = validate_circuit_expression(self.ent_circuit.get())
            if canonical != self.ent_circuit.get().strip():
                self.ent_circuit.delete(0, tk.END)
                self.ent_circuit.insert(0, canonical)
            count = len(list(iter_element_names(_tree)))
            self.lbl_circuit_status.configure(
                text=f"Valid circuit: {count} element{'s' if count != 1 else ''}.",
                foreground="#1e8449",
            )
            if show_success:
                messagebox.showinfo("Circuit Valid", f"The circuit is valid:\n{canonical}")
            return canonical
        except Exception as exc:
            self.lbl_circuit_status.configure(text=f"Invalid circuit: {exc}", foreground="#b03a2e")
            if show_success:
                messagebox.showerror("Invalid Circuit", str(exc))
            return None

    def _on_circuit_text_changed(self, _event=None):
        expression = self.ent_circuit.get().strip()
        if self.current_circuit_str and expression != self.current_circuit_str and self.Z_fit is not None:
            self._invalidate_fit(clear_parameters=True)
        try:
            _tree, _canonical = validate_circuit_expression(expression)
            self.lbl_circuit_status.configure(text="Circuit syntax is valid.", foreground="#1e8449")
        except Exception:
            self.lbl_circuit_status.configure(text="Circuit is being edited or is not yet valid.", foreground="#b9770e")

    # =========================================================
    # Logic Methods
    # =========================================================
    def load_data(self):
        fp = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv;*.txt;*.dat;*.xlsx;*.xls")])
        if not fp:
            return
        try:
            if fp.lower().endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(fp)
            else:
                self.df = pd.read_csv(fp, sep=None, engine='python')
                
            cols = [""] + list(self.df.columns)
            self.cb_freq['values'] = cols
            self.cb_real['values'] = cols
            self.cb_imag['values'] = cols
            
            for c in self.df.columns:
                c_lower = c.strip().lower()
                if "freq" in c_lower:
                    self.cb_freq.set(c)
                elif "z'" in c_lower and "ohm" in c_lower:
                    self.cb_real.set(c)
                elif 'z"' in c_lower and "ohm" in c_lower:
                    self.cb_imag.set(c)
            
            self.lbl_file.config(text=os.path.basename(fp))
            messagebox.showinfo("Success", "Data loaded successfully! Please check the mapped columns and click Apply.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file:\n{e}")

    def _invalidate_fit(self, clear_parameters=False):
        """Invalidate stale fit results whenever the experimental data changes."""
        self.Z_fit = None
        self.param_errors = {}
        self.fit_bounds = {}
        self.fit_diagnostics = {}
        self.lin_kk_summary = {}

        if clear_parameters:
            for widget in self.param_frame.winfo_children():
                widget.destroy()
            self.param_vars.clear()
            self.current_circuit_str = ""
            self.btn_refit.config(state="disabled")
        else:
            self.btn_refit.config(state="normal" if self.param_vars else "disabled")

        self.lbl_metrics.config(text="Metrics:\nFit invalidated - run fitting again.")

    def apply_columns(self):
        if self.df is None:
            return
        try:
            f_col = self.cb_freq.get()
            r_col = self.cb_real.get()
            i_col = self.cb_imag.get()

            if not f_col or not r_col or not i_col:
                messagebox.showerror("Missing Data", "Freq, Z', and Z\" columns are required.")
                return

            f = self.df[f_col].values.astype(float)
            zr = self.df[r_col].values.astype(float)
            zi = self.df[i_col].values.astype(float)

            # More conservative sign detection than using the mean alone.
            finite_zi = zi[np.isfinite(zi)]
            self.imag_sign_auto_flipped = False
            if finite_zi.size:
                positive_fraction = np.mean(finite_zi > 0)
                negative_fraction = np.mean(finite_zi < 0)
                if positive_fraction >= 0.75 and negative_fraction <= 0.25:
                    zi = -zi
                    self.imag_sign_auto_flipped = True

            mask = (
                np.isfinite(f) &
                np.isfinite(zr) &
                np.isfinite(zi) &
                (f > 0)
            )

            self.f = f[mask]
            self.Z = zr[mask] + 1j * zi[mask]

            if len(self.f) < 4:
                raise ValueError("At least four valid positive-frequency points are required.")

            self.Mod_exp = np.abs(self.Z)
            self.Phase_exp = np.angle(self.Z, deg=True)

            self.f_orig = self.f.copy()
            self.Z_orig = self.Z.copy()
            self.Mod_orig = self.Mod_exp.copy()
            self.Phase_orig = self.Phase_exp.copy()

            self._invalidate_fit(clear_parameters=True)
            self.btn_fit.config(state="normal")
            self.btn_prune.config(state="normal")
            self.btn_reset_data.config(state="normal")

            self.plot_data(self.f, self.Z, self.Mod_exp, self.Phase_exp)
        except Exception as e:
            messagebox.showerror("Error", f"Error mapping data:\n{e}")

    def toggle_prune_mode(self):
        self.prune_mode = not self.prune_mode
        if self.prune_mode:
            self.btn_prune.config(text="🔴 Prune Mode ON")
            self.ax_nyq.set_title("Nyquist Plot [CLICK TO DELETE POINTS]", color="red", fontweight='bold')
        else:
            self.btn_prune.config(text="✂️ Enable Prune Mode")
            self.ax_nyq.set_title("Nyquist Plot", color="black", fontweight='bold')
        self.canvas.draw()

    def reset_data(self):
        if self.f_orig is None:
            return
        self.f = self.f_orig.copy()
        self.Z = self.Z_orig.copy()
        self.Mod_exp = self.Mod_orig.copy()
        self.Phase_exp = self.Phase_orig.copy()
        self._invalidate_fit(clear_parameters=False)
        self.plot_data(self.f, self.Z, self.Mod_exp, self.Phase_exp)

    def on_plot_click(self, event):
        if not self.prune_mode or event.inaxes != self.ax_nyq:
            return
        if self.Z is None or len(self.Z) == 0:
            return
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
        distances = np.sqrt((self.Z.real - click_x)**2 + (-self.Z.imag - click_y)**2)
        closest_idx = np.argmin(distances)

        self.f = np.delete(self.f, closest_idx)
        self.Z = np.delete(self.Z, closest_idx)
        self.Mod_exp = np.delete(self.Mod_exp, closest_idx)
        self.Phase_exp = np.delete(self.Phase_exp, closest_idx)

        self._invalidate_fit(clear_parameters=False)
        self.plot_data(self.f, self.Z, self.Mod_exp, self.Phase_exp)
        self.ax_nyq.set_title("Nyquist Plot [CLICK TO DELETE POINTS]", color="red", fontweight='bold')
        self.canvas.draw()

    def plot_data(self, f, Z, mod_exp, phase_exp, Z_fit=None):
        self.ax_nyq.clear()
        self.ax_bode.clear()
        self.ax_phase_twin.clear()
        self.ax_nyq.set_axis_on()
        self.ax_bode.set_axis_on()
        self.ax_phase_twin.set_axis_on()
        
        self.ax_nyq.plot(Z.real, -Z.imag, 'o', color='blue', label='Data', markersize=5)
        if Z_fit is not None:
            self.ax_nyq.plot(Z_fit.real, -Z_fit.imag, '-', color='red', label='Fit', linewidth=2)
            
        self.ax_nyq.set_xlabel("Z' (Ohm)")
        self.ax_nyq.set_ylabel("-Z'' (Ohm)")
        self.ax_nyq.set_title("Nyquist Plot", fontweight='bold')
        self.ax_nyq.grid(True, linestyle='--', alpha=0.5)
        self.ax_nyq.legend()
        
        self.ax_bode.loglog(f, mod_exp, 'o', color='blue', label='|Z| Data', markersize=5)
        self.ax_phase_twin.semilogx(f, phase_exp, '^', color='green', label='Phase Data', markersize=5, alpha=0.7)
        
        if Z_fit is not None:
            self.ax_bode.loglog(f, np.abs(Z_fit), '-', color='red', label='|Z| Fit', linewidth=2)
            self.ax_phase_twin.semilogx(f, np.angle(Z_fit, deg=True), '--', color='darkorange', label='Phase Fit', linewidth=2)
            
        self.ax_bode.set_xlabel("Frequency (Hz)")
        self.ax_bode.set_ylabel("|Z| (Ohm)", color='blue')
        self.ax_phase_twin.set_ylabel("Phase (deg)", color='green')
        self.ax_bode.set_title("Bode Plot", fontweight='bold')
        self.ax_bode.grid(True, linestyle='--', alpha=0.5)
        
        lines_1, labels_1 = self.ax_bode.get_legend_handles_labels()
        lines_2, labels_2 = self.ax_phase_twin.get_legend_handles_labels()
        self.ax_bode.legend(lines_1 + lines_2, labels_1 + labels_2, loc='best')

        self.fig.tight_layout()
        self.canvas.draw()

    def build_param_ui(self, param_names, initial_values=None):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.param_vars.clear()

        unit_map = {}
        element_map = {}
        try:
            for item in _get_parameter_info(self.current_circuit_str):
                unit_map[item["name"]] = item["unit"]
                element_map[item["name"]] = item["raw"]
        except Exception:
            pass
        
        ttk.Label(self.param_frame, text="Element", font=("", 10, "bold")).grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(self.param_frame, text="Value", font=("", 10, "bold")).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(self.param_frame, text="Unit", font=("", 10, "bold")).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(self.param_frame, text="Lock 🔒", font=("", 10, "bold")).grid(row=0, column=3, padx=5, pady=5)
        self.param_frame.columnconfigure(1, weight=1)
        
        for i, name in enumerate(param_names, start=1):
            val = initial_values[i-1] if initial_values is not None else 1.0
            name_label = ttk.Label(self.param_frame, text=name)
            name_label.grid(row=i, column=0, sticky="w", padx=5)
            raw = element_map.get(name, get_element_from_name(name))
            if raw in ELEMENT_CATALOG:
                HoverToolTip(name_label, element_tooltip_text(raw))
            var_val = tk.StringVar(value=f"{val:.6e}")
            ttk.Entry(self.param_frame, textvariable=var_val, width=18).grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            ttk.Label(self.param_frame, text=unit_map.get(name, ""), foreground="#566573").grid(row=i, column=2, sticky="w", padx=3)
            var_lock = tk.BooleanVar(value=False)
            ttk.Checkbutton(self.param_frame, variable=var_lock).grid(row=i, column=3, padx=5)
            self.param_vars[name] = {'value': var_val, 'lock': var_lock}

    # =========================================================
    # Smart Fitting Engine (backend only - UI layout unchanged)
    # =========================================================
    def _robust_end_estimates(self):
        """Robust high/low frequency real-axis estimates using medians."""
        n = len(self.f)
        count = max(3, min(7, int(np.ceil(n * 0.08))))
        order = np.argsort(self.f)
        lf_idx = order[:count]
        hf_idx = order[-count:]

        r_series = float(np.median(self.Z.real[hf_idx]))
        r_total = float(np.median(self.Z.real[lf_idx]))
        r_series = max(r_series, 1e-6)
        r_pol = max(r_total - r_series, max(0.02 * np.nanmax(np.abs(self.Z.real)), 1e-3))
        return r_series, r_total, r_pol

    def _detect_time_constants(self, expected_count):
        """Find characteristic frequencies from -Z'' versus log(f)."""
        if expected_count <= 0:
            return [], []

        order = np.argsort(self.f)
        f_sorted = self.f[order]
        y = (-self.Z.imag)[order]

        # Very small smoothing to reduce single-point noise without moving peaks much.
        if len(y) >= 7:
            kernel = np.ones(3) / 3.0
            y_smooth = np.convolve(y, kernel, mode="same")
        else:
            y_smooth = y.copy()

        y_range = float(np.nanmax(y_smooth) - np.nanmin(y_smooth))
        prominence = max(0.04 * y_range, 1e-12)
        min_distance = max(1, len(y_smooth) // max(expected_count * 4, 4))
        peaks, props = find_peaks(y_smooth, prominence=prominence, distance=min_distance)

        candidates = []
        for idx in peaks:
            candidates.append((float(y_smooth[idx]), float(f_sorted[idx])))

        # Highest semicircle peaks first, then sort selected processes by frequency.
        candidates.sort(key=lambda item: item[0], reverse=True)
        candidates = candidates[:expected_count]
        candidates.sort(key=lambda item: item[1], reverse=True)

        peak_heights = [max(item[0], 1e-12) for item in candidates]
        peak_freqs = [item[1] for item in candidates]

        # If peaks overlap too strongly to detect separately, distribute fallback
        # characteristic frequencies logarithmically through the measured window.
        if len(peak_freqs) < expected_count:
            f_hi = float(np.max(self.f))
            f_lo = float(np.min(self.f))
            fallback = np.logspace(np.log10(f_hi), np.log10(f_lo), expected_count + 2)[1:-1]
            for freq in fallback:
                if len(peak_freqs) >= expected_count:
                    break
                if not peak_freqs or np.min(np.abs(np.log10(np.asarray(peak_freqs) / freq))) > 0.35:
                    peak_freqs.append(float(freq))
                    nearest = int(np.argmin(np.abs(np.log10(self.f / freq))))
                    peak_heights.append(max(float(-self.Z.imag[nearest]), 1e-12))

        peak_freqs = peak_freqs[:expected_count]
        peak_heights = peak_heights[:expected_count]

        paired = sorted(zip(peak_freqs, peak_heights), key=lambda item: item[0], reverse=True)
        return [p[0] for p in paired], [p[1] for p in paired]

    def _smart_initial_guess(self, param_names):
        r_series, _, r_pol = self._robust_end_estimates()

        resistor_names = [n for n in param_names if n.upper().startswith("R")]
        nonseries_resistors = resistor_names[1:] if len(resistor_names) > 1 else []

        cpe_q_names = [n for n in param_names if n.upper().startswith("CPE") and n.upper().endswith("_0")]
        capacitor_names = [
            n for n in param_names
            if n.upper().startswith("C") and not n.upper().startswith("CPE")
        ]
        finite_diffusion_tau_names = [
            n for n in param_names
            if get_element_from_name(n).upper() in {"WO", "WS"} and n.endswith("_1")
        ]
        ws_z0_names = [
            n for n in param_names
            if get_element_from_name(n).upper() == "WS" and n.endswith("_0")
        ]
        process_count = max(
            len(nonseries_resistors),
            len(cpe_q_names) + len(capacitor_names) + len(finite_diffusion_tau_names),
            1,
        )
        peak_freqs, peak_heights = self._detect_time_constants(process_count)

        if not peak_freqs:
            peak_freqs = [float(np.sqrt(np.max(self.f) * np.min(self.f)))]
            peak_heights = [max(float(np.nanmax(-self.Z.imag)), 1.0)]

        # For an ideal RC semicircle, peak height is approximately R/2.
        branch_r = np.asarray([max(2.0 * h, 1e-9) for h in peak_heights], dtype=float)
        if branch_r.size < max(len(nonseries_resistors), 1):
            branch_r = np.pad(branch_r, (0, max(len(nonseries_resistors), 1) - branch_r.size), constant_values=1.0)
        branch_r = branch_r[:max(len(nonseries_resistors), 1)]
        if branch_r.sum() <= 0:
            branch_r[:] = 1.0
        branch_r = r_pol * branch_r / branch_r.sum()
        if ws_z0_names:
            # A short finite-length Warburg contributes a finite low-frequency
            # resistance. Do not assign the entire real-axis span to R branches.
            branch_r *= 0.5

        # Make sure every process has a characteristic frequency.
        if len(peak_freqs) < max(len(cpe_q_names), len(capacitor_names), len(nonseries_resistors), 1):
            needed = max(len(cpe_q_names), len(capacitor_names), len(nonseries_resistors), 1)
            peak_freqs = list(np.logspace(np.log10(np.max(self.f)), np.log10(np.min(self.f)), needed + 2)[1:-1])

        cpe_n_default = 0.85
        guess_map = {}

        if resistor_names:
            guess_map[resistor_names[0]] = r_series
        for i, name in enumerate(nonseries_resistors):
            guess_map[name] = float(branch_r[min(i, len(branch_r) - 1)])

        # CPE Q = 1 / [R * (2*pi*f_peak)^n] for an R||CPE process.
        for i, q_name in enumerate(cpe_q_names):
            r_i = float(branch_r[min(i, len(branch_r) - 1)])
            f_i = max(float(peak_freqs[min(i, len(peak_freqs) - 1)]), 1e-12)
            q_i = 1.0 / (max(r_i, 1e-12) * (2.0 * np.pi * f_i) ** cpe_n_default)
            guess_map[q_name] = max(q_i, 1e-14)
            n_name = q_name[:-2] + "_1"
            if n_name in param_names:
                guess_map[n_name] = cpe_n_default

        for i, c_name in enumerate(capacitor_names):
            r_i = float(branch_r[min(i, len(branch_r) - 1)])
            f_i = max(float(peak_freqs[min(i, len(peak_freqs) - 1)]), 1e-12)
            guess_map[c_name] = max(1.0 / (2.0 * np.pi * f_i * max(r_i, 1e-12)), 1e-14)

        # Generic but data-scaled guesses for diffusion/inductive/other elements.
        z_scale = max(float(np.nanmedian(np.abs(self.Z))), 1e-6)
        f_mid = max(float(np.sqrt(np.max(self.f) * np.min(self.f))), 1e-12)
        f_diffusion = min(float(np.max(self.f)), max(float(np.min(self.f)) * 10.0, float(np.min(self.f))))
        for name in param_names:
            upper = name.upper()
            if name in guess_map:
                continue
            if upper.startswith("W"):
                raw = get_element_from_name(name).upper()
                if raw in {"WO", "WS"} and upper.endswith("_1"):
                    guess_map[name] = 1.0 / (2.0 * np.pi * max(f_diffusion, 1e-12))
                elif raw == "WS" and upper.endswith("_0"):
                    guess_map[name] = max(0.5 * r_pol / max(len(ws_z0_names), 1), 1e-12)
                elif upper.endswith("_1"):
                    guess_map[name] = 1.0 / (2.0 * np.pi * f_mid)
                else:
                    guess_map[name] = z_scale
            elif upper.startswith("L"):
                hf = int(np.argmax(self.f))
                guess_map[name] = max(self.Z.imag[hf] / (2.0 * np.pi * self.f[hf]), 1e-9)
            elif _is_shape_exponent(name):
                guess_map[name] = cpe_n_default
            else:
                guess_map[name] = 1.0

        initial = np.asarray([guess_map[name] for name in param_names], dtype=float)
        safe_lo, safe_hi = make_safe_bounds(param_names)
        return _clip_initial_guess(initial, safe_lo, safe_hi)

    def _log_balanced_indices(self, max_points=60):
        """Select points approximately uniformly in log-frequency for fast search."""
        n = len(self.f)
        if n <= max_points:
            return np.arange(n, dtype=int)

        log_f = np.log10(self.f)
        order = np.argsort(log_f)
        sorted_log = log_f[order]
        targets = np.linspace(sorted_log[0], sorted_log[-1], max_points)
        chosen = []

        for target in targets:
            pos = np.searchsorted(sorted_log, target)
            candidates = []
            if pos < len(order):
                candidates.append(pos)
            if pos > 0:
                candidates.append(pos - 1)
            best_pos = min(candidates, key=lambda p: abs(sorted_log[p] - target))
            chosen.append(int(order[best_pos]))

        return np.asarray(sorted(set(chosen)), dtype=int)

    def _fit_once(self, frequencies, impedance, circuit_str, param_names, initial_guess,
                  bounds, constants=None, loss="linear", max_nfev=2500,
                  tolerance=1e-7, f_scale=1.0):
        """Weighted nonlinear least-squares in transformed parameter space."""
        constants = constants or {}
        lower, upper = bounds
        initial_guess = _clip_initial_guess(initial_guess, lower, upper)
        x0 = _to_optimizer_space(initial_guess, param_names)
        lo_t, hi_t = _bounds_to_optimizer_space(lower, upper, param_names)
        x0 = np.clip(x0, lo_t + 1e-10, hi_t - 1e-10)
        abs_z = np.maximum(np.abs(impedance), 1e-15)

        def residual(opt_values):
            physical = _from_optimizer_space(opt_values, param_names)
            z_model = _predict_impedance(circuit_str, frequencies, physical, constants)
            return np.hstack([
                (z_model.real - impedance.real) / abs_z,
                (z_model.imag - impedance.imag) / abs_z,
            ])

        result = least_squares(
            residual,
            x0,
            bounds=(lo_t, hi_t),
            method="trf",
            x_scale="jac",
            loss=loss,
            f_scale=f_scale,
            ftol=tolerance,
            xtol=tolerance,
            gtol=tolerance,
            max_nfev=max_nfev,
        )

        params = _from_optimizer_space(result.x, param_names)
        z_fit = _predict_impedance(circuit_str, frequencies, params, constants)
        score = _weighted_sse(impedance, z_fit)

        # Covariance approximation from the final weighted Jacobian.
        errors = np.full(len(param_names), np.nan, dtype=float)
        condition_number = np.nan
        max_correlation = np.nan
        correlation_pair = ""
        covariance = None

        try:
            jac = np.asarray(result.jac, dtype=float)
            jtj = jac.T @ jac
            condition_number = float(np.linalg.cond(jtj))
            dof = max(len(result.fun) - len(param_names), 1)
            residual_variance = float(np.sum(result.fun ** 2) / dof)
            covariance_x = np.linalg.pinv(jtj) * residual_variance
            covariance = covariance_x.copy()
            sigma_x = np.sqrt(np.maximum(np.diag(covariance_x), 0.0))

            for i, name in enumerate(param_names):
                if _is_shape_exponent(name):
                    errors[i] = sigma_x[i]
                else:
                    errors[i] = np.log(10.0) * params[i] * sigma_x[i]

            if len(param_names) > 1:
                diag = np.sqrt(np.maximum(np.diag(covariance_x), 1e-300))
                corr = covariance_x / np.outer(diag, diag)
                np.fill_diagonal(corr, 0.0)
                flat = int(np.nanargmax(np.abs(corr)))
                i, j = np.unravel_index(flat, corr.shape)
                max_correlation = float(abs(corr[i, j]))
                correlation_pair = f"{param_names[i]} / {param_names[j]}"
        except Exception:
            pass

        return {
            "params": params,
            "errors": errors,
            "score": score,
            "result": result,
            "condition_number": condition_number,
            "max_correlation": max_correlation,
            "correlation_pair": correlation_pair,
            "covariance": covariance,
        }

    def _perturb_guess(self, base_guess, param_names, bounds, rng, spread=0.75):
        lower, upper = bounds
        guess = np.asarray(base_guess, dtype=float).copy()

        for i, name in enumerate(param_names):
            if _is_shape_exponent(name):
                guess[i] += rng.normal(0.0, 0.08 if spread >= 0.5 else 0.035)
            else:
                local_spread = spread
                if name.upper().startswith("R") and i == 0:
                    local_spread *= 0.45
                guess[i] *= 10.0 ** rng.uniform(-local_spread, local_spread)

        return _clip_initial_guess(guess, lower, upper)

    def _solutions_consistent(self, candidates, param_names):
        if len(candidates) < 3:
            return False
        best = sorted(candidates, key=lambda c: c["score"])[:3]
        best_score = max(best[0]["score"], 1e-30)
        normalized_residual = float(np.sqrt(best_score / max(2 * len(self.f), 1)))
        score_spread = (best[-1]["score"] - best[0]["score"]) / best_score
        x_values = np.vstack([_to_optimizer_space(c["params"], param_names) for c in best])
        typical_param_spread = float(np.median(np.std(x_values, axis=0))) if x_values.size else np.inf
        # Several starts can agree on the same poor local basin. Only stop the
        # global search early when they agree *and* already reproduce the data.
        return normalized_residual < 0.025 and score_spread < 0.03 and typical_param_spread < 0.15

    def _run_parallel_trials(self, guesses, frequencies, impedance, circuit_str,
                             param_names, bounds, constants=None):
        candidates = []
        max_workers = max(1, min(4, os.cpu_count() or 1, len(guesses)))

        def worker(guess):
            return self._fit_once(
                frequencies, impedance, circuit_str, param_names, guess, bounds,
                constants=constants, loss="linear", max_nfev=2500, tolerance=1e-7,
            )

        if max_workers == 1:
            for guess in guesses:
                try:
                    candidates.append(worker(guess))
                except Exception:
                    pass
            return candidates

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(worker, guess) for guess in guesses]
            for future in as_completed(futures):
                try:
                    candidates.append(future.result())
                except Exception:
                    pass
        return candidates

    def _adaptive_multistart(self, circuit_str, param_names, smart_guess, bounds,
                             constants=None, local_mode=False):
        """Adaptive multi-start: quick search first, expand only if solutions disagree."""
        rng = np.random.default_rng(20260723)
        search_idx = self._log_balanced_indices(max_points=60)
        f_search = self.f[search_idx]
        z_search = self.Z[search_idx]

        if local_mode:
            stage_sizes = [5]
            spread = 0.25
        else:
            stage_sizes = [8, 12, 10]
            spread = 0.80

        candidates = []
        starts_used = 0

        for stage_i, stage_size in enumerate(stage_sizes):
            guesses = []
            if stage_i == 0:
                guesses.append(np.asarray(smart_guess, dtype=float))
            while len(guesses) < stage_size:
                guesses.append(self._perturb_guess(smart_guess, param_names, bounds, rng, spread=spread))

            stage_candidates = self._run_parallel_trials(
                guesses, f_search, z_search, circuit_str, param_names, bounds, constants
            )
            starts_used += len(guesses)

            # Rank each quick-search candidate on the full experimental spectrum.
            for candidate in stage_candidates:
                try:
                    z_full = _predict_impedance(circuit_str, self.f, candidate["params"], constants)
                    candidate["score"] = _weighted_sse(self.Z, z_full)
                    candidates.append(candidate)
                except Exception:
                    pass

            candidates.sort(key=lambda c: c["score"])
            if local_mode or self._solutions_consistent(candidates, param_names):
                break

            # Expand the search radius slightly only when necessary.
            spread = min(spread + 0.15, 1.15)

        if not candidates:
            raise RuntimeError("All multi-start fitting attempts failed. Check the circuit and data range.")

        best_seed = candidates[0]["params"]

        # Final high-precision refinement on every data point.
        final_fit = self._fit_once(
            self.f, self.Z, circuit_str, param_names, best_seed, bounds,
            constants=constants, loss="linear", max_nfev=20000, tolerance=1e-11,
        )

        # Robust fallback only when a few points dominate the residual. It is used
        # only to find an alternative basin; the accepted final result is linear-weighted.
        z_final = _predict_impedance(circuit_str, self.f, final_fit["params"], constants)
        relative = np.abs(self.Z - z_final) / np.maximum(np.abs(self.Z), 1e-15)
        median_rel = max(float(np.median(relative)), 1e-8)
        robust_used = False

        if float(np.max(relative)) > max(0.03, 6.0 * median_rel):
            try:
                robust_seed = self._fit_once(
                    self.f, self.Z, circuit_str, param_names, final_fit["params"], bounds,
                    constants=constants, loss="soft_l1", max_nfev=8000,
                    tolerance=1e-8, f_scale=max(median_rel, 0.01),
                )
                alternative = self._fit_once(
                    self.f, self.Z, circuit_str, param_names, robust_seed["params"], bounds,
                    constants=constants, loss="linear", max_nfev=16000, tolerance=1e-11,
                )
                if alternative["score"] < final_fit["score"]:
                    final_fit = alternative
                    robust_used = True
            except Exception:
                pass

        final_fit["starts_used"] = starts_used
        final_fit["robust_used"] = robust_used
        return final_fit

    def _run_lin_kk_validation(self):
        self.lin_kk_summary = {}
        if linKK is None or self.f is None or len(self.f) < 8:
            return
        try:
            M, mu, z_kk, res_real, res_imag = linKK(
                np.asarray(self.f, dtype=float),
                np.asarray(self.Z, dtype=complex),
                c=0.85,
                max_M=min(40, max(8, len(self.f) // 2)),
                fit_type="real",
                add_cap=False,
            )
            kk_residual_rms = float(np.sqrt(np.mean(np.asarray(res_real) ** 2 + np.asarray(res_imag) ** 2)))
            self.lin_kk_summary = {
                "M": int(M),
                "mu": float(mu),
                "residual_rms": kk_residual_rms,
            }
        except Exception as exc:
            self.lin_kk_summary = {"error": str(exc)}

    def _bound_hit_names(self, param_names, params, bounds):
        lower, upper = bounds
        x = _to_optimizer_space(params, param_names)
        lo_t, hi_t = _bounds_to_optimizer_space(lower, upper, param_names)
        hits = []
        for i, name in enumerate(param_names):
            span = max(hi_t[i] - lo_t[i], 1e-12)
            distance = min(abs(x[i] - lo_t[i]), abs(hi_t[i] - x[i])) / span
            if distance < 0.005:
                hits.append(name)
        return hits

    def initial_fit(self):
        if self.f is None:
            return
        canonical = self.validate_main_circuit(show_success=False)
        if canonical is None:
            messagebox.showerror("Fit Error", "Please correct the circuit expression before fitting.")
            return
        self.current_circuit_str = canonical

        try:
            names = _extract_param_names(self.current_circuit_str)
            if not names:
                raise ValueError("No free parameters were detected in the circuit.")

            initial_guess = self._smart_initial_guess(names)
            bounds = make_adaptive_bounds(names, initial_guess)
            initial_guess = _clip_initial_guess(initial_guess, *bounds)

            self.btn_refit.config(state="disabled")
            self.lbl_metrics.config(text="Fitting in progress... smart multi-start search.")
            self.update_idletasks()

            fit = self._adaptive_multistart(
                self.current_circuit_str, names, initial_guess, bounds,
                constants=None, local_mode=False,
            )

            final_params = np.asarray(fit["params"], dtype=float)
            self.Z_fit = _predict_impedance(self.current_circuit_str, self.f, final_params)
            self.param_errors = {
                name: float(err) if np.isfinite(err) else np.nan
                for name, err in zip(names, fit["errors"])
            }
            self.fit_bounds = {name: (float(bounds[0][i]), float(bounds[1][i])) for i, name in enumerate(names)}

            self._run_lin_kk_validation()
            bound_hits = self._bound_hit_names(names, final_params, bounds)
            self.fit_diagnostics = {
                "starts_used": int(fit.get("starts_used", 0)),
                "robust_used": bool(fit.get("robust_used", False)),
                "condition_number": float(fit.get("condition_number", np.nan)),
                "max_correlation": float(fit.get("max_correlation", np.nan)),
                "correlation_pair": fit.get("correlation_pair", ""),
                "bound_hits": bound_hits,
            }

            self.build_param_ui(names, final_params)
            self.plot_data(self.f, self.Z, self.Mod_exp, self.Phase_exp, self.Z_fit)
            self.update_metrics(self.Z_fit, names, final_params, fit["errors"], bounds, len(names))
            self.btn_refit.config(state="normal")
            messagebox.showinfo("Fit Complete", "Smart multi-start fitting completed successfully!")

        except Exception as e:
            self.btn_refit.config(state="normal" if self.param_vars else "disabled")
            messagebox.showerror("Fit Error", str(e))

    def refit_with_locks(self):
        if not self.current_circuit_str or not self.param_vars:
            return

        all_names = _extract_param_names(self.current_circuit_str)
        constants = {}
        all_values = {}

        try:
            for name in all_names:
                value = float(self.param_vars[name]['value'].get())
                all_values[name] = value
                if self.param_vars[name]['lock'].get():
                    constants[name] = value

            free_names = _extract_param_names(self.current_circuit_str, constants=constants)

            # If everything is locked, simply evaluate the requested circuit.
            if not free_names:
                self.Z_fit = _predict_impedance(self.current_circuit_str, self.f, [], constants)
                for name in all_names:
                    self.param_errors[name] = 0.0
                self.fit_diagnostics = {
                    "starts_used": 0,
                    "robust_used": False,
                    "condition_number": np.nan,
                    "max_correlation": np.nan,
                    "correlation_pair": "",
                    "bound_hits": [],
                }
                self.plot_data(self.f, self.Z, self.Mod_exp, self.Phase_exp, self.Z_fit)
                self.update_metrics(self.Z_fit, [], np.asarray([]), np.asarray([]), (np.asarray([]), np.asarray([])), 0)
                return

            initial_guess = np.asarray([all_values[name] for name in free_names], dtype=float)
            bounds = make_adaptive_bounds(free_names, initial_guess)
            initial_guess = _clip_initial_guess(initial_guess, *bounds)

            self.lbl_metrics.config(text="Re-fitting... local multi-start search around current parameters.")
            self.update_idletasks()

            fit = self._adaptive_multistart(
                self.current_circuit_str, free_names, initial_guess, bounds,
                constants=constants, local_mode=True,
            )
            free_params = np.asarray(fit["params"], dtype=float)
            self.Z_fit = _predict_impedance(self.current_circuit_str, self.f, free_params, constants)

            free_idx = 0
            for name in all_names:
                if name in constants:
                    self.param_errors[name] = 0.0
                else:
                    value = free_params[free_idx]
                    self.param_vars[name]['value'].set(f"{value:.6e}")
                    err = fit["errors"][free_idx]
                    self.param_errors[name] = float(err) if np.isfinite(err) else np.nan
                    free_idx += 1

            self.fit_bounds = {
                name: (float(bounds[0][i]), float(bounds[1][i]))
                for i, name in enumerate(free_names)
            }
            bound_hits = self._bound_hit_names(free_names, free_params, bounds)
            self.fit_diagnostics = {
                "starts_used": int(fit.get("starts_used", 0)),
                "robust_used": bool(fit.get("robust_used", False)),
                "condition_number": float(fit.get("condition_number", np.nan)),
                "max_correlation": float(fit.get("max_correlation", np.nan)),
                "correlation_pair": fit.get("correlation_pair", ""),
                "bound_hits": bound_hits,
            }
            self._run_lin_kk_validation()

            self.plot_data(self.f, self.Z, self.Mod_exp, self.Phase_exp, self.Z_fit)
            self.update_metrics(self.Z_fit, free_names, free_params, fit["errors"], bounds, len(free_names))

        except Exception as e:
            messagebox.showerror("Re-Fit Error", str(e))

    def update_metrics(self, Z_fit, param_names=None, params=None, errors=None,
                       bounds=None, n_free_params=0):
        rmse, mre, rel_max, weighted_sse, reduced_weighted = calculate_metrics(
            self.Z, Z_fit, n_free_params=n_free_params
        )

        lines = [
            f"Circuit: {self.current_circuit_str}",
            f"RMSE: {rmse:.4e}",
            f"Mean Rel Error: {mre:.2%}",
            f"Max Rel Error: {rel_max:.2%}",
            f"Weighted SSE: {weighted_sse:.4e}",
            f"Reduced weighted error: {reduced_weighted:.4e}",
        ]

        starts = self.fit_diagnostics.get("starts_used")
        if starts is not None:
            lines.append(f"Multi-start trials: {starts}")
        if self.fit_diagnostics.get("robust_used"):
            lines.append("Robust basin recovery: used")

        bound_hits = self.fit_diagnostics.get("bound_hits", [])
        lines.append("Bounds: " + ("OK" if not bound_hits else "hit -> " + ", ".join(bound_hits)))

        cond = self.fit_diagnostics.get("condition_number", np.nan)
        if np.isfinite(cond):
            lines.append(f"Jacobian condition: {cond:.3e}")

        corr = self.fit_diagnostics.get("max_correlation", np.nan)
        pair = self.fit_diagnostics.get("correlation_pair", "")
        if np.isfinite(corr):
            corr_text = f"Max |parameter corr.|: {corr:.3f}"
            if pair:
                corr_text += f" ({pair})"
            lines.append(corr_text)

        # Flag parameters whose one-sigma error is at least as large as the value.
        poorly_determined = []
        if param_names is not None and params is not None and errors is not None:
            for name, value, err in zip(param_names, params, errors):
                if np.isfinite(err) and abs(value) > 0 and abs(err / value) >= 1.0:
                    poorly_determined.append(name)
        if poorly_determined:
            lines.append("High uncertainty: " + ", ".join(poorly_determined))

        if self.lin_kk_summary:
            if "error" in self.lin_kk_summary:
                lines.append("Lin-KK: unavailable for this dataset")
            else:
                lines.append(
                    f"Lin-KK: M={self.lin_kk_summary['M']}, "
                    f"mu={self.lin_kk_summary['mu']:.3f}, "
                    f"resid RMS={self.lin_kk_summary['residual_rms']:.3e}"
                )

        self.lbl_metrics.config(text="\n".join(lines))

    # =========================================================
    # Reporting System & Diagram Generator
    # =========================================================
    def generate_circuit_image(self, circuit_str, save_path):
        """
        Generate circuit schematic from impedance.py circuit string.
        Supports nested circuits such as:
        R0-p(R1-W1,CPE1)
        R0-p(R1,CPE1)-p(R2-W1,CPE2)
        """
        try:
            import schemdraw
            import schemdraw.elements as elm

            tree, circuit_str = validate_circuit_expression(circuit_str)

            def draw_element(d, name, length=2.2, label_loc="top"):
                kind = element_kind(name)

                if kind == "resistor":
                    d += elm.Resistor().right().label(name, loc=label_loc).length(length)

                elif kind == "capacitor":
                    d += elm.Capacitor().right().label(name, loc=label_loc).length(length)

                elif kind == "cpe":
                    # schemdraw provides the standard CPE symbol in current releases.
                    cpe_cls = getattr(elm, "CPE", elm.Capacitor)
                    d += cpe_cls().right().label(name, loc=label_loc).length(length)

                elif kind == "warburg":
                    # A boxed impedance symbol avoids confusing diffusion with an ohmic resistor.
                    warburg_cls = getattr(elm, "RBox", elm.ResistorIEC)
                    d += warburg_cls().right().label(name, loc=label_loc).length(length)

                elif kind == "inductor":
                    d += elm.Inductor().right().label(name, loc=label_loc).length(length)

                else:
                    d += elm.Resistor().right().label(name, loc=label_loc).length(length)

            def draw_node(d, node):
                if node["type"] == "element":
                    draw_element(d, node["name"])
                    return

                if node["type"] == "series":
                    for index, item in enumerate(node["items"]):
                        draw_node(d, item)
                        if index < len(node["items"]) - 1:
                            d += elm.Line().right().length(0.25)
                    return

                if node["type"] == "parallel":
                    draw_parallel(d, node["branches"])
                    return

            def draw_branch(d, node, total_length=5.5, label_loc="top"):
                if node["type"] == "element":
                    draw_element(d, node["name"], length=total_length, label_loc=label_loc)

                elif node["type"] == "series":
                    items = node["items"]
                    n = max(len(items), 1)
                    connector = 0.12
                    each_length = max((total_length - connector * (n - 1)) / n, 0.5)

                    for index, item in enumerate(items):
                        if item["type"] == "element":
                            draw_element(d, item["name"], length=each_length, label_loc=label_loc)
                        else:
                            draw_node(d, item)
                        if index < len(items) - 1:
                            d += elm.Line().right().length(connector)

                elif node["type"] == "parallel":
                    draw_parallel(d, node["branches"])

            def draw_parallel(d, branches):
                if len(branches) == 0:
                    return

                if len(branches) == 1:
                    draw_node(d, branches[0])
                    return

                vertical_gap = 1.45
                branch_length = 5.5

                # entrance
                d += elm.Line().right().length(0.4)
                left_junction = d.here
                right_junction = None

                offsets = [((len(branches) - 1) / 2.0 - index) * vertical_gap for index in range(len(branches))]
                for branch, offset in zip(branches, offsets):
                    d.move_from(left_junction)
                    if offset > 0:
                        d += elm.Line().up().length(offset)
                    elif offset < 0:
                        d += elm.Line().down().length(abs(offset))

                    draw_branch(
                        d,
                        branch,
                        total_length=branch_length,
                        label_loc="top" if offset >= 0 else "bottom",
                    )

                    if offset > 0:
                        d += elm.Line().down().length(offset)
                    elif offset < 0:
                        d += elm.Line().up().length(abs(offset))
                    if right_junction is None:
                        right_junction = d.here

                d.move_from(right_junction)

                # exit
                d += elm.Line().right().length(0.4)

            with schemdraw.Drawing(show=False) as d:
                d.config(fontsize=13, lw=2, font="sans-serif", bgcolor="white")

                d += elm.Dot(open=True)
                d += elm.Line().right().length(0.4)

                draw_node(d, tree)

                d += elm.Line().right().length(0.4)
                d += elm.Dot(open=True)

                d.save(save_path, dpi=300)

        except Exception:
            # Safe fallback: show accurate topology text instead of a wrong schematic
            try:
                tree = parse_circuit_expr(circuit_str)
                pretty = tree_to_pretty_text(tree)
            except Exception:
                pretty = circuit_str

            fig, ax = plt.subplots(figsize=(8, 2), facecolor="white")
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"Equivalent circuit topology:\n{pretty}",
                fontsize=16,
                ha="center",
                va="center",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#e8f4f8",
                    edgecolor="#2980b9",
                    linewidth=2
                )
            )
            fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
            plt.close(fig)

    def _rgba_to_rgb_white(self, png_path, jpg_path):
        """Smart function to fix FPDF bug: Convert transparent image to absolute white background"""
        img = Image.open(png_path)
        img.load() 
        if img.mode == 'RGBA' or img.mode == 'LA':
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3]) # Whiten the alpha channel
            background.save(jpg_path, 'JPEG', quality=95)
        else:
            img.convert('RGB').save(jpg_path, 'JPEG', quality=95)

    def export_pro_results(self):
        if self.Z_fit is None:
            messagebox.showwarning("Warning", "Please run fitting first!")
            return

        try:
            # 1. Create Circuit Base Directory
            safe_circ_name = self.current_circuit_str.replace('(', '_').replace(')', '_').replace(',', '-')
            circuit_base_dir = os.path.join(self.base_output_path, safe_circ_name)
            Path(circuit_base_dir).mkdir(parents=True, exist_ok=True)

            # 2. Find the next run_00X folder (Safe Loop to prevent hanging)
            counter = 1
            while counter < 1000:
                run_folder_name = f"run_manual_{counter:03d}"
                final_dir = os.path.join(circuit_base_dir, run_folder_name)
                if not os.path.exists(final_dir):
                    os.makedirs(final_dir)
                    break
                counter += 1
            
            if counter >= 1000:
                messagebox.showerror("Error", "Too many export folders! Please clean up your directory.")
                return

            # 3. Export Excel
            excel_path = os.path.join(final_dir, "Fitting_Data_Sheet.xlsx")
            params_list = []
            for k, v in self.param_vars.items():
                value = float(v['value'].get())
                error = self.param_errors.get(k, np.nan)
                rel_error = (abs(error / value) * 100.0) if np.isfinite(error) and value != 0 else np.nan
                bound = self.fit_bounds.get(k)
                at_bound = False
                if bound is not None:
                    lo, hi = bound
                    if _is_shape_exponent(k):
                        span = max(hi - lo, 1e-15)
                        at_bound = min(abs(value - lo), abs(hi - value)) / span < 0.005
                    elif value > 0 and lo > 0 and hi > lo:
                        log_span = max(np.log10(hi) - np.log10(lo), 1e-15)
                        at_bound = min(abs(np.log10(value) - np.log10(lo)), abs(np.log10(hi) - np.log10(value))) / log_span < 0.005

                params_list.append({
                    "Parameter": k,
                    "Value": value,
                    "Std Error (1 sigma)": error,
                    "Relative Error (%)": rel_error,
                    "Status": "Locked" if v['lock'].get() else "Fitted",
                    "At Bound": at_bound,
                })

            residual = self.Z - self.Z_fit
            rel_residual = np.abs(residual) / np.maximum(np.abs(self.Z), 1e-15)

            quality_rows = []
            for key, value in self.fit_diagnostics.items():
                quality_rows.append({"Metric": key, "Value": str(value)})
            for key, value in self.lin_kk_summary.items():
                quality_rows.append({"Metric": f"LinKK_{key}", "Value": str(value)})
            quality_rows.append({"Metric": "Imaginary sign auto-flipped", "Value": str(self.imag_sign_auto_flipped)})

            with pd.ExcelWriter(excel_path) as writer:
                pd.DataFrame(params_list).to_excel(writer, sheet_name="Fitting Parameters", index=False)
                pd.DataFrame({
                    "Freq": self.f,
                    "Z_Real": self.Z.real,
                    "Z_Imag": self.Z.imag,
                    "Fit_Real": self.Z_fit.real,
                    "Fit_Imag": self.Z_fit.imag,
                    "Residual_Real": residual.real,
                    "Residual_Imag": residual.imag,
                    "Relative_Residual": rel_residual,
                }).to_excel(writer, sheet_name="Data Trace", index=False)
                pd.DataFrame(quality_rows).to_excel(writer, sheet_name="Fit Quality", index=False)

            # 4. Save Plots as Permanent PNG & convert to Temp JPG for PDF
            plot_png = os.path.join(final_dir, "Nyquist_Bode_Plots.png")
            temp_plot_jpg = os.path.join(final_dir, "temp_plot.jpg")
            self.fig.savefig(plot_png, dpi=200, facecolor='white', bbox_inches='tight')
            self._rgba_to_rgb_white(plot_png, temp_plot_jpg)
            
            # 5. Save Circuit as Permanent PNG & convert to Temp JPG for PDF
            circ_png = os.path.join(final_dir, "Circuit_Schematic.png")
            temp_circ_jpg = os.path.join(final_dir, "temp_circ.jpg")
            self.generate_circuit_image(self.current_circuit_str, circ_png)
            self._rgba_to_rgb_white(circ_png, temp_circ_jpg)

            # 6. Generate PDF
            pdf_path = os.path.join(final_dir, "Official_Researcher_Report.pdf")
            pdf = FPDF()
            
            # ==========================================
            # Page 1: Cover Page
            # ==========================================
            pdf.add_page()
                
            pdf.set_font("Helvetica", "B", 20)
            pdf.cell(0, 100, "", ln=True)
            pdf.cell(0, 20, self.software_name, ln=True, align="C")
            pdf.set_font("Helvetica", "", 14)
            pdf.cell(0, 20, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
            pdf.cell(0, 20, "Official Analysis Report", ln=True, align="C")

            pdf.ln(30)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 7, "Toolkit version: 1.0.0", ln=True, align="C")

            doi = "10.1039/D6RA08618G"
            pdf.set_font("Helvetica", "U", 11)
            pdf.set_text_color(31, 78, 121)
            pdf.cell(
                0, 8,
                f"Suggested citation · Behroozi et al., RSC Adv. (2026). {doi}",
                ln=True,
                align="C",
                link=f"https://doi.org/{doi}",
            )
            pdf.set_text_color(0, 0, 0)
            
            # ==========================================
            # Page 2: Circuit & Parameters
            # ==========================================
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Fitting Analysis Results", ln=True)
            pdf.set_font("Helvetica", "", 12)
            pdf.cell(0, 10, "Status: Manually Fitted and Verified", ln=True)
            
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 10, "Equivalent Circuit Schematic:", ln=True)
            
            if os.path.exists(temp_circ_jpg):
                pdf.image(temp_circ_jpg, x=15, y=pdf.get_y(), w=130) 
                pdf.set_y(pdf.get_y() + 75) 
            else:
                pdf.cell(0, 10, f"Circuit: {self.current_circuit_str}", ln=True)
                pdf.ln(10)
            
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(40, 10, "Parameter", 1)
            pdf.cell(60, 10, "Value", 1)
            pdf.ln()
            pdf.set_font("Helvetica", "", 12)
            for k, v in self.param_vars.items():
                pdf.cell(40, 10, str(k), 1)
                pdf.cell(60, 10, str(v['value'].get()), 1)
                pdf.ln()

            # ==========================================
            # Page 3: Plots
            # ==========================================
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Fitting Plots (Nyquist & Bode)", ln=True)
            pdf.ln(5)

            if os.path.exists(temp_plot_jpg):
                pdf.image(temp_plot_jpg, x=10, y=pdf.get_y() + 5, w=190)
            
            # save pdf
            pdf.output(pdf_path)
            
            # Clean up ONLY temporary jpg files, keep the PNGs
            for f in [temp_plot_jpg, temp_circ_jpg]:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception:
                    pass

            messagebox.showinfo("Export Successful", f"All files (PDF, Excel, Images) saved successfully to:\n{final_dir}")

        except PermissionError:
            messagebox.showerror("File Error", "The PDF or Excel file is currently open in another program. Please close it and try again.")
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred during export:\n{str(e)}")
if __name__ == "__main__":
    app = ResearcherEISPro()
    app.mainloop()
