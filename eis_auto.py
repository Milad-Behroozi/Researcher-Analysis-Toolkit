import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from impedance.models.circuits import CustomCircuit
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import os
import sys
import warnings
import random
import json
import re
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import subprocess
from auto_library import maybe_store, is_auto_learning_enabled
import auto_library
import time
import threading
import ctypes

from eis_physics import (
    PHYSICS_ENGINE_VERSION,
    analyze_eis_physics,
    annotate_model_confidence,
    annotate_result_physics,
    build_physics_guided_candidates,
    circuit_element_families,
    circuit_relaxation_count,
    physics_compatibility,
    run_lin_kk_diagnostic,
)

from scipy.signal import find_peaks, savgol_filter

SOFTWARE_NAME = "Researcher Analysis Toolkit — Auto EIS Fitting Pro"
SOFTWARE_VERSION = "1.3.0-physics-preview"

# Fixed seed for reproducible candidate generation and multistart fitting.
# Users can change this value deliberately, but repeated runs with the same
# input and settings should otherwise produce the same candidate ordering.
DEFAULT_RANDOM_SEED = 20260821

# Relative residuals below this level are treated as numerical equivalence for
# information-criterion ranking.  This prevents machine-precision differences
# in noise-free synthetic data from rewarding an unnecessarily complex model.
INFORMATION_CRITERION_RELATIVE_FLOOR = 1e-6


# =========================================================
# Output folder helpers (Desktop/Researcher Output/...)
# =========================================================

def _get_desktop_dir():
    """Return user's Desktop folder (Windows/macOS/Linux)."""
    home = os.path.expanduser("~")
    # Windows: prefer ~/Desktop, else USERPROFILE/Desktop
    desktop = os.path.join(home, "Desktop")
    if not os.path.isdir(desktop):
        desktop = os.path.join(os.environ.get("USERPROFILE", home), "Desktop")
    return desktop

def _safe_folder_name(name: str) -> str:
    """Sanitize a name for a folder (cross-platform friendly)."""
    name = (name or "").strip()
    if not name:
        return "untitled"
    # Replace forbidden characters on Windows and avoid trailing dots/spaces
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name

def build_output_run_folder(source_file_path: str, analysis_name: str = "Nyquist and Bode Analysis") -> str:
    """
    Build output folder as:
        Desktop/Researcher Output/<analysis_name>/<Raw File Name>/run_00x
    Where run_00x auto-increments (run_001, run_002, ...).
    """
    base_name = os.path.splitext(os.path.basename(source_file_path))[0]
    base_name = _safe_folder_name(base_name)

    desktop = _get_desktop_dir()
    base_dir = os.path.join(desktop, "Researcher Output", analysis_name, base_name)
    os.makedirs(base_dir, exist_ok=True)

    # next run index
    run_re = re.compile(r"^run_(\d{3})$")
    max_i = 0
    try:
        for nm in os.listdir(base_dir):
            m = run_re.match(nm)
            if m:
                try:
                    max_i = max(max_i, int(m.group(1)))
                except Exception:
                    pass
    except Exception:
        pass

    run_idx = max_i + 1
    run_folder = os.path.join(base_dir, f"run_{run_idx:03d}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder

# =========================================================
# Helper for library import
# =========================================================

def ensure_library_installed(module_name, pip_name=None):
    if pip_name is None:
        pip_name = module_name
    try:
        __import__(module_name)
        return True
    except ImportError:
        print(f"🔧 Installing missing library: {pip_name} ...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            __import__(module_name)
            print(f"✅ {pip_name} installed successfully.")
            return True
        except Exception as e:
            print(f"❌ Failed to install {pip_name}: {e}")
            return False

try:
    import autoeis as ae
except ImportError:
    ae = None
    print("⚠️ AutoEIS is unavailable; optional Level 5 is disabled.")


# =========================================================
# 3. Import Schemdraw (for drawing circuits)
# =========================================================
try:
    import schemdraw
    import schemdraw.elements as elm
    HAS_SCHEMDRAW = True
except ImportError:
    HAS_SCHEMDRAW = False
    print("⚠️ Schemdraw is unavailable; circuit diagrams will be skipped.")

import matplotlib.patches as patches

def draw_circuit_to_file(circuit_str, filename):
    """
    Final version:
    1. Labels are placed outside the elements.
    2. CPE is displayed as CPE (not converted to Q).
    3. Font is Times New Roman.
    """
    try:
        # Font settings for the circuit
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]

        # Overall dimensions
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.set_aspect('equal')
        ax.axis('off')

        # Overall appearance settings
        LINE_WIDTH = 2.5      
        FONT_SIZE = 18        
        TEXT_OFFSET = 0.7     # Distance of text from element (to be placed outside)
        
        # Circuit dimension settings
        H_SERIES = 3.5        
        H_PARALLEL = 4.0      
        V_GAP = 2.5           
        SPLIT_LEAD = 1.5      
        # --- (ADDED) Parser that understands nested parentheses ---
        def split_top_level(s, delimiter):
            parts = []
            depth = 0
            buf = []
            for ch in s:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1

                if ch == delimiter and depth == 0:
                    part = "".join(buf).strip()
                    if part:
                        parts.append(part)
                    buf = []
                else:
                    buf.append(ch)

            last = "".join(buf).strip()
            if last:
                parts.append(last)

            return parts
        # --- توابع کمکی رسم ---
        def draw_resistor(ax, start, end, label=None):
            x0, y0 = start; x1, y1 = end
            dx, dy = x1 - x0, y1 - y0
            length = (dx**2 + dy**2)**0.5
            
            # Zigzag settings
            n_zig = 6
            w_zig = 0.25 # Wider zigzag
            gap = 0.8    # Gap in the middle of the wire for the component
            
            if length > gap:
                margin = (length - gap) / 2
                sx, sy = x0 + margin*(dx/length), y0 + margin*(dy/length)
                ex, ey = x1 - margin*(dx/length), y1 - margin*(dy/length)
                ax.plot([x0, sx], [y0, sy], 'k-', lw=LINE_WIDTH)
                ax.plot([ex, x1], [ey, y1], 'k-', lw=LINE_WIDTH)
            else:
                sx, sy, ex, ey = x0, y0, x1, y1

            # Drawing the zigzag
            seg_len = gap / n_zig
            perp_x, perp_y = -dy/length, dx/length
            x_vals, y_vals = [sx], [sy]
            for i in range(1, n_zig + 1):
                progress = i * seg_len - (seg_len/2)
                amp = w_zig if i % 2 != 0 else -w_zig
                mid_x = sx + (ex-sx)*(progress/gap) + perp_x*amp
                mid_y = sy + (ey-sy)*(progress/gap) + perp_y*amp
                x_vals.extend([mid_x])
                y_vals.extend([mid_y])
            x_vals.append(ex); y_vals.append(ey)
            
            ax.plot(x_vals, y_vals, 'k-', lw=LINE_WIDTH)
            
            # --- Resistor / Series element label ---
            if label:
                # Convert Q → CPE
                if label.startswith("Q"):
                    label = label.replace("Q", "CPE")

                # متن را کمی بالاتر می‌بریم
                ax.text((x0+x1)/2, (y0+y1)/2 + TEXT_OFFSET, label,
                        fontsize=FONT_SIZE + 2,
                        ha='center', va='center',
                        weight='bold', family='serif',
                        color='blue')

        def draw_capacitor(ax, start, end, label=None, style='C'):
            x0, y0 = start
            x1, y1 = end

            dx, dy = x1 - x0, y1 - y0
            length = (dx*dx + dy*dy) ** 0.5
            if length == 0:
                return

            # --- Plate settings ---
            plate_gap = 0.45                 # Distance between two plates
            margin = max(0.20, (length - plate_gap) / 2.0)
            plate_w = 0.55                   # Plate length (the larger, the clearer)

            # Unit vector along the line and perpendicular to the line
            ux, uy = dx/length, dy/length
            perp_x, perp_y = -uy, ux

            # Plate points on the line
            cx1, cy1 = x0 + margin*ux, y0 + margin*uy
            cx2, cy2 = x1 - margin*ux, y1 - margin*uy

            # Wires to plates
            ax.plot([x0, cx1], [y0, cy1], 'k-', lw=LINE_WIDTH)
            ax.plot([cx2, x1], [cy2, y1], 'k-', lw=LINE_WIDTH)

            # Plates
            ax.plot([cx1 + perp_x*plate_w, cx1 - perp_x*plate_w],
                    [cy1 + perp_y*plate_w, cy1 - perp_y*plate_w], 'k-', lw=LINE_WIDTH)
            ax.plot([cx2 + perp_x*plate_w, cx2 - perp_x*plate_w],
                    [cy2 + perp_y*plate_w, cy2 - perp_y*plate_w], 'k-', lw=LINE_WIDTH)

            # Label
            if label:
                if label.startswith("Q"):
                    label = label.replace("Q", "CPE")
                ax.text((x0+x1)/2, (y0+y1)/2 + TEXT_OFFSET, label,
                        fontsize=FONT_SIZE + 2, ha='center', va='center',
                        weight='bold', family='serif', color='blue')


        # --- (ADDED) branch renderer (handles nested p(...) and series inside branches) ---
        def _is_res_like(tok: str) -> bool:
            t = (tok or "").strip()
            return t.startswith("R") or t.startswith("W") or t.startswith("Wo") or t.startswith("Ws") or t.startswith("L")

        def _draw_element_on_span(ax, x0, y0, x1, y1, token: str):
            t = (token or "").strip()
            if not t:
                ax.plot([x0, x1], [y0, y1], 'k-', lw=LINE_WIDTH)
                return
            if _is_res_like(t):
                draw_resistor(ax, (x0, y0), (x1, y1), t)
            else:
                draw_capacitor(ax, (x0, y0), (x1, y1), t)

        def draw_branch(ax, x0, y0, x1, y1, branch_str: str, inner_vgap: float = None):
            """Render a branch between (x0,y0) and (x1,y1).
            Supports:
              - single element: R1, CPE1, C1, W1 ...
              - series: R1-CPE1 or R1-W1-CPE2 ...
              - nested parallel: p(R2,CPE2) or p(R1,p(R2,CPE2))
            """
            s = (branch_str or "").strip()
            if not s:
                ax.plot([x0, x1], [y0, y1], 'k-', lw=LINE_WIDTH)
                return

            # Nested parallel inside a branch
            # --- Nested parallel inside a branch (GENERALIZED to N branches) ---
            if s.startswith("p(") and s.endswith(")"):
                content = s[2:-1]
                brs = split_top_level(content, ",")
                if len(brs) < 2:
                    _draw_element_on_span(ax, x0, y0, x1, y1, s)
                    return

                n = len(brs)

                # compact inner parallel block
                vgap_base = inner_vgap if inner_vgap is not None else (V_GAP * 0.70)

                # If there are too many branches, reduce the vertical distance slightly so the circuit doesn't get too tall
                vgap = max(0.9, vgap_base * (2.0 / max(n, 2))**0.5)

                lead = min(1.0, max(0.6, SPLIT_LEAD * 0.65))
                split_x = x0 + lead
                end_x = x1
                mid_y = y0

                # --- NEW: indent for nested loop so it "closes earlier" than outer end ---
                INNER_INDENT = 1.0  # Suggested: 0.8 to 1.4
                inner_end_x = end_x - INNER_INDENT
                # Prevent the inner loop from becoming too short or reversed
                inner_end_x = max(split_x + 0.8, inner_end_x)

                # lead to split
                ax.plot([x0, split_x], [mid_y, mid_y], 'k-', lw=LINE_WIDTH)

                # y positions for all branches (top to bottom)
                offsets = [((n - 1) / 2.0 - i) * vgap for i in range(n)]

                for i, b in enumerate(brs):
                    yy = mid_y + offsets[i]

                    # vertical down/up from split node to branch level
                    ax.plot([split_x, split_x], [mid_y, yy], 'k-', lw=LINE_WIDTH)

                    # render this branch (supports nested again)
                    # IMPORTANT: use inner_end_x (NOT end_x) to show "loop inside loop"
                    draw_branch(ax, split_x, yy, inner_end_x, yy, b, inner_vgap=vgap * 0.70)

                    # connect back to inner node (NOT the outer end)
                    ax.plot([inner_end_x, inner_end_x], [yy, mid_y], 'k-', lw=LINE_WIDTH)

                # --- NEW: one short connection from inner loop to outer end ---
                ax.plot([inner_end_x, end_x], [mid_y, mid_y], 'k-', lw=LINE_WIDTH)

                return



            # Series inside a branch
            if "-" in s:
                toks = split_top_level(s, "-")
                if not toks:
                    ax.plot([x0, x1], [y0, y1], 'k-', lw=LINE_WIDTH)
                    return
                span = (x1 - x0) if (x1 - x0) != 0 else 1.0
                seg = span / max(len(toks), 1)
                cx = x0
                for i, tok in enumerate(toks):
                    nx = x0 + (i + 1) * seg
                    _draw_element_on_span(ax, cx, y0, nx, y1, tok)
                    cx = nx
                return

            # Single element
            _draw_element_on_span(ax, x0, y0, x1, y1, s)
            return

        # --- منطق اصلی رسم ---
        start_node = (0, 0)
        curr_x = 0
        
        elements = split_top_level(circuit_str, "-")
        
        # رسم نود شروع
        ax.plot(0, 0, 'ko', markersize=6)

        for el in elements:
            if el.startswith("p("):
                content = el[2:-1]
                branches = split_top_level(content, ",")

                n = len(branches)
                if n < 2:
                    # fallback: treat it like a normal element span
                    next_x = curr_x + H_PARALLEL
                    draw_branch(ax, curr_x, 0, next_x, 0, el)
                    curr_x = next_x
                    continue

                split_x = curr_x + SPLIT_LEAD
                end_x = split_x + H_PARALLEL

                # Line before the split
                ax.plot([curr_x, split_x], [0, 0], 'k-', lw=LINE_WIDTH)

                # Adaptive vertical spacing for many branches
                vgap = max(1.2, V_GAP * (2.0 / max(n, 2))**0.5)

                offsets = [((n - 1) / 2.0 - i) * vgap for i in range(n)]

                for i, b in enumerate(branches):
                    yy = offsets[i]

                    ax.plot([split_x, split_x], [0, yy], 'k-', lw=LINE_WIDTH)
                    draw_branch(ax, split_x, yy, end_x, yy, b)
                    ax.plot([end_x, end_x], [yy, 0], 'k-', lw=LINE_WIDTH)

                curr_x = end_x

                
            else:
                # --- Series element ---
                next_x = curr_x + H_SERIES
                # Fix name for display (remove _0, etc.)
                disp_label = el.replace("CPE", "Q")
                
                if "R" in el or "W" in el:
                    draw_resistor(ax, (curr_x, 0), (next_x, 0), disp_label)
                else:
                    draw_capacitor(ax, (curr_x, 0), (next_x, 0), disp_label)
                curr_x = next_x

        # Line and end node
        ax.plot([curr_x, curr_x+0.8], [0, 0], 'k-', lw=LINE_WIDTH)
        ax.plot(curr_x+0.8, 0, 'ko', markersize=6)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return True
    except Exception as e:
        print(f"Failed to draw circuit: {e}")
        return False


# =========================================================
# Global settings
# =========================================================

ERROR_THRESHOLD = 0.30

LEVEL_INFO = {
    # The second value is a hard candidate cap, not a request to generate
    # thousands of arbitrary strings.  Physics-guided construction normally
    # produces fewer candidates than these limits.
    "L1": ("Level 1 Fast Physics-Guided Search", 12),
    "L2": ("Level 2 Extended Physics-Guided Search", 20),
    "L3": ("Level 3 Standard Physics-Guided Search (Recommended)", 32),
    "L4": ("Level 4 Advanced Physics-Guided Search", 48),
    "L5": ("Level 5 AutoEIS-Assisted Search", 64),
}

# =========================================================
# GUI helpers
# =========================================================

def select_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select EIS data file (table: CSV / TXT / DAT / Excel)",
        filetypes=[
            ("Table files", "*.csv;*.txt;*.dat;*.tsv;*.xlsx;*.xls"),
            ("CSV files", "*.csv"),
            ("Text files", "*.txt;*.dat;*.tsv"),
            ("Excel files", "*.xlsx;*.xls"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return file_path

def gui_select_columns_and_level(df, autoeis_available):
    root = tk.Tk()
    root.title(f"{SOFTWARE_NAME} – Column & Level Selection")

    cols = list(df.columns)

    tk.Label(root, text="Select columns for EIS data:", font=("Segoe UI", 10, "bold")).grid(
        row=0, column=0, columnspan=2, pady=(10, 5)
    )

    # Frequency
    tk.Label(root, text="Frequency (Hz):").grid(row=1, column=0, sticky="e", padx=5, pady=3)
    freq_var = tk.StringVar(value=cols[0])
    freq_cb = ttk.Combobox(root, textvariable=freq_var, values=cols, state="readonly")
    freq_cb.grid(row=1, column=1, sticky="w", padx=5, pady=3)

    # Z real
    tk.Label(root, text="Z' (Real):").grid(row=2, column=0, sticky="e", padx=5, pady=3)
    zreal_var = tk.StringVar(value=cols[1] if len(cols) > 1 else cols[0])
    zreal_cb = ttk.Combobox(root, textvariable=zreal_var, values=cols, state="readonly")
    zreal_cb.grid(row=2, column=1, sticky="w", padx=5, pady=3)

    # Z imag
    tk.Label(root, text="Z'' (Imag):").grid(row=3, column=0, sticky="e", padx=5, pady=3)
    zimag_var = tk.StringVar(value=cols[2] if len(cols) > 2 else cols[0])
    zimag_cb = ttk.Combobox(root, textvariable=zimag_var, values=cols, state="readonly")
    zimag_cb.grid(row=3, column=1, sticky="w", padx=5, pady=3)

    # Phase optional
    phase_options = ["<None>"] + cols
    tk.Label(root, text="Phase (deg, optional):").grid(row=4, column=0, sticky="e", padx=5, pady=3)
    phase_var = tk.StringVar(value="<None>")
    phase_cb = ttk.Combobox(root, textvariable=phase_var, values=phase_options, state="readonly")
    phase_cb.grid(row=4, column=1, sticky="w", padx=5, pady=3)

    # Analysis level
    tk.Label(root, text="Analysis Level:", font=("Segoe UI", 10, "bold")).grid(
        row=5, column=0, columnspan=2, pady=(10, 5)
    )

    level_var = tk.StringVar(value="L1")

    level_texts = {
        "L1": "Level 1 Basic Topology Search",
        "L2": "Level 2 Intermediate Topology Search",
        "L3": "Level 3 Standard Hierarchical Search (Recommended)",
        "L4": "Level 4 Advanced Topology Exploration",
        "L5": "Level 5 AI-Assisted Topology Optimization",
    }

    row_level = 6
    for key in ["L1", "L2", "L3", "L4", "L5"]:
        if key == "L5" and not autoeis_available:
            continue
        tk.Radiobutton(root, text=level_texts[key], variable=level_var, value=key).grid(
            row=row_level, column=0, columnspan=2, sticky="w", padx=10, pady=2
        )
        row_level += 1

    result = {"ok": False}

    def on_ok():
        result["freq"] = cols.index(freq_var.get())
        result["zreal"] = cols.index(zreal_var.get())
        result["zimag"] = cols.index(zimag_var.get())
        if phase_var.get() == "<None>":
            result["phase"] = None
        else:
            result["phase"] = cols.index(phase_var.get())
        result["level"] = level_var.get()
        result["ok"] = True
        root.destroy()

    def on_cancel():
        result["ok"] = False
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.grid(row=row_level, column=0, columnspan=2, pady=10)

    tk.Button(btn_frame, text="OK", width=10, command=on_ok).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side="left", padx=5)

    root.mainloop()

    if not result["ok"]:
        print("Operation cancelled by user.")
        sys.exit()

    return result["freq"], result["zreal"], result["zimag"], result["phase"], result["level"]

# =========================================================
# Circuit helpers  (data-based initial guess + multi-start)
# =========================================================

def _extract_param_names(circuit_str):
    names = []
    i = 0
    while i < len(circuit_str):
        if circuit_str[i] in ["R", "C", "W", "L", "G"]:
            if circuit_str[i:i+3] == "CPE":
                j = i + 3
                while j < len(circuit_str) and circuit_str[j].isdigit():
                    j += 1
                base = circuit_str[i:j]          
                names.append(base + "_0")        
                names.append(base + "_1")        
                i = j
            elif circuit_str[i:i+2] in ("Wo", "Ws"):
                # Finite-length Warburg elements in impedance.py have two
                # fitted parameters.  The previous parser read only "W",
                # producing an initial-guess length mismatch and causing all
                # Wo/Ws candidates to fail before fitting.
                j = i + 2
                while j < len(circuit_str) and circuit_str[j].isdigit():
                    j += 1
                base = circuit_str[i:j]
                names.append(base + "_0")
                names.append(base + "_1")
                i = j
            else:
                j = i + 1
                while j < len(circuit_str) and circuit_str[j].isdigit():
                    j += 1
                names.append(circuit_str[i:j])   
                i = j
        else:
            i += 1
    return names


def build_data_based_initial_guess(circuit_str, f, Z):
    names = _extract_param_names(circuit_str)
    Zreal = np.real(Z)
    Zimag = np.imag(Z)
    
    # 1. Basic Resistance Estimation
    R_min = float(np.min(Zreal))
    R_max = float(np.max(Zreal))
    dR = max(R_max - R_min, 1e-3)

    # 2. INTELLIGENT FREQUENCY DETECTION (MULTI-PEAK)
    # Find the main peak (Low Frequency)
    idx_max_imag = np.argmax(-Zimag)
    f_peak_low = f[idx_max_imag]
    if f_peak_low <= 0:
        f_pos = f[f > 0]
        f_peak_low = np.sqrt(np.min(f_pos) * np.max(f_pos))
        
    # Set a high frequency guess point (e.g., geometric mean of peak and max)
    f_max = np.max(f)
    f_peak_high = np.sqrt(f_peak_low * f_max) 
    if f_peak_high < f_peak_low * 10: # Ensure separation
        f_peak_high = f_max / 2.0

    # 3. Parameter Assignment
    R_names = [n for n in names if n.startswith("R")]
    n_R = len(R_names)
    
    # Count CPEs to distribute them
    cpe_indices = [i for i, n in enumerate(names) if n.startswith("CPE") and n.endswith("_0")]
    
    guess = [1.0] * len(names) # Placeholder

    for i, n in enumerate(names):
        val = 1.0
        
        # --- Resistances ---
        if n == "R0":
            val = max(R_min, 1e-4)
        elif n.startswith("R"):
            if n_R > 1:
                val = max(dR / (n_R - 1), 1e-3)
            else:
                val = max(dR, 1e-3)

        # --- CPE Magnitude (Q) ---
        elif n.startswith("CPE") and n.endswith("_0"):
            # Distribute CPEs: First one to Low Freq, Second one to High Freq
            is_second_cpe = (len(cpe_indices) > 1) and (i == cpe_indices[-1])
            
            target_f = f_peak_high if is_second_cpe else f_peak_low
            omega_peak = 2.0 * np.pi * target_f
            
            if n_R > 0:
                R_eff = max(dR / max(n_R - 1, 1), 1e-3)
            else:
                R_eff = max(dR, 1e-3)
                
            # Q = 1 / ((w)^n * R) assuming n=0.85
            val = 1.0 / ((omega_peak ** 0.85) * R_eff)

        # --- CPE Exponent (n) ---
        elif n.startswith("CPE") and n.endswith("_1"):
            val = 0.85

        # --- Capacitors ---
        elif n.startswith("C"):
            target_f = f_peak_high if (len(cpe_indices) > 0) else f_peak_low
            omega_peak = 2.0 * np.pi * target_f
            R_eff = max(dR, 1e-3)
            val = 1.0 / (omega_peak * R_eff)
            
        # --- Warburg / Inductance ---
        elif (n.startswith("Wo") or n.startswith("Ws")) and n.endswith("_0"):
            val = dR * 0.5
        elif (n.startswith("Wo") or n.startswith("Ws")) and n.endswith("_1"):
            # Characteristic diffusion time (s).
            val = 1.0 / (2.0 * np.pi * max(f_peak_low, 1e-30))
        elif n.startswith("W"):
            val = dR * 0.5
        elif n.startswith("L"):
            val = 1e-7
        elif n.startswith("G"):
            val = 1.0 / dR
            
        guess[i] = val

    return guess

def _fit_circuit_with_guess(circuit_str, f, Z, init_guess, max_iter=200):
    try:
        param_names = _extract_param_names(circuit_str)
        if len(param_names) != len(init_guess):
            return np.inf, None, None

        # Broad physical bounds.  In particular, a CPE exponent is not an
        # unconstrained positive parameter: n must remain between 0 and 1.
        # The 0.4 lower bound matches the broad safety range documented for
        # the toolkit and avoids numerically meaningless exponents.
        lower_bounds = []
        upper_bounds = []
        bounded_guess = []
        for name, value in zip(param_names, init_guess):
            if name.startswith("CPE") and name.endswith("_1"):
                lo, hi = 0.4, 1.0
            else:
                lo, hi = 1e-30, np.inf
            lower_bounds.append(lo)
            upper_bounds.append(hi)
            if np.isfinite(hi):
                value = min(max(float(value), lo + 1e-9), hi - 1e-9)
            else:
                value = max(float(value), lo)
            bounded_guess.append(value)
        
        circuit = CustomCircuit(circuit_str, initial_guess=bounded_guess)

        # Numerical safety: some candidates can overflow (especially CPE) and produce NaN/Inf.
        # We silence RuntimeWarnings locally and skip invalid candidates without breaking the search.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                try:
                    # Use the same modulus weighting for optimization and for
                    # the downstream model-selection diagnostics.
                    circuit.fit(
                        f, Z,
                        maxfev=max_iter,
                        bounds=(lower_bounds, upper_bounds),
                        weight_by_modulus=True,
                    )
                except TypeError:
                    # Compatibility fallback for older impedance.py releases.
                    circuit.fit(f, Z, maxfev=max_iter, bounds=(lower_bounds, upper_bounds))
                Z_fit = circuit.predict(f)

        if Z_fit is None or (not np.all(np.isfinite(Z_fit))):
            return None, None, None

        absZ = np.abs(Z)
        resid = Z - Z_fit
        abs_resid = np.abs(resid)

        # Modulus Weighting (Chi-squared style)
        weight = 1.0 / (absZ + 1e-12)

        num = np.sum((weight * abs_resid) ** 2)
        denom = np.sum((weight * absZ) ** 2)
        if denom <= 0:
            denom = np.sum((weight * absZ) ** 2 + 1e-12)

        error = np.sqrt(num / denom) # RMSE (relative)

        # --- Extra statistical metrics (for GUI ranking) ---
        eps = 1e-12

        # Absolute RMSE on complex residual magnitude
        rmse_abs = float(np.sqrt(np.mean(abs_resid ** 2)))

        # Pointwise relative residual: |Zexp - Zfit| / |Zexp|
        rel_err = abs_resid / (absZ + eps)

        # True maximum pointwise relative residual.  The previous definition
        # divided every residual by max(|Z|), which understated low-impedance
        # deviations and was not a maximum relative error.
        rel_max = float(np.max(rel_err))

        # Mean of |Zexp - Zfit| / |Zexp|
        mean_rel_err = float(np.mean(rel_err))

        # Chi-squared (same definition as in save_fit_outputs)
        chi_sq = float(np.sum(rel_err ** 2))
        # ----------------------------------------------------

        if np.isnan(error) or np.isinf(error):
            return np.inf, None, None

        # Information criteria are calculated from the same modulus-weighted
        # residual basis used for ranking.  A complex datum contributes two
        # real observations.  AICc is used instead of plain AIC to penalize
        # over-parameterized circuits for finite EIS datasets.
        weighted_components = np.concatenate([
            np.real(resid) / (absZ + eps),
            np.imag(resid) / (absZ + eps),
        ])
        residual_sum_sq = float(np.sum(weighted_components ** 2))
        n_data = int(weighted_components.size)
        k_params = len(circuit.parameters_)

        if n_data > k_params:
            rss_floor = n_data * (INFORMATION_CRITERION_RELATIVE_FLOOR ** 2)
            rss_safe = max(residual_sum_sq, rss_floor, np.finfo(float).tiny)
            aic = n_data * np.log(rss_safe / n_data) + 2 * k_params
            if n_data > (k_params + 1):
                aicc = aic + (2.0 * k_params * (k_params + 1)) / (n_data - k_params - 1)
            else:
                aicc = np.inf
            bic = n_data * np.log(rss_safe / n_data) + k_params * np.log(n_data)
        else:
            aicc = np.inf
            bic = np.inf

        # The first field remains in the historical "AIC" position for API
        # compatibility, but now contains the more appropriate AICc value.
        return error, circuit, (aicc, bic, rel_max, chi_sq, rmse_abs, mean_rel_err)
    except Exception:
        return np.inf, None, None

def evaluate_circuit(circuit_str, f, Z, max_iter=200):
    base_guess = build_data_based_initial_guess(circuit_str, f, Z)
    return _fit_circuit_with_guess(circuit_str, f, Z, base_guess, max_iter=max_iter)


def _rng_for_circuit(circuit_str, random_seed=DEFAULT_RANDOM_SEED):
    """Return a stable RNG whose stream does not depend on Python hash randomization."""
    checksum = sum((i + 1) * ord(ch) for i, ch in enumerate(str(circuit_str)))
    seed = (int(random_seed) + checksum) % (2**32)
    return np.random.default_rng(seed)


def refine_circuit_fit(circuit_str, f, Z, n_restarts=100, target_rel_max=0.01,
                       max_iter=50000, random_seed=DEFAULT_RANDOM_SEED):
    """
    Optimization loop:
    - Minimizes RMSE (standard least squares)
    - But STOPS if rel_max < target_rel_max (excellent fit everywhere)
    """
    base_guess = build_data_based_initial_guess(circuit_str, f, Z)
    best_err = np.inf
    best_model = None
    best_info = None
    best_rel_max = np.inf
    rng = _rng_for_circuit(circuit_str, random_seed=random_seed)

    print(f"\n>>> Multi-start refined fitting for {circuit_str}")
    # target_rel_max default 0.01 means 1% max deviation
    print(f"    Target rel_max: {target_rel_max:.3f} | Max Restarts: {n_restarts}")

    for k in range(n_restarts):
        if k == 0:
            guess = base_guess
        else:
            guess = []
            for i, val in enumerate(base_guess):
                if val <= 0:
                    val = 1e-6
                factor = 10 ** rng.uniform(-1.0, 1.0)
                guess.append(val * factor)
        
        # Log occasionally
        if k > 0 and k % 10 == 0:
            pass # print(f"    ... attempt {k}/{n_restarts} ...")

        err, model, info = _fit_circuit_with_guess(circuit_str, f, Z, guess, max_iter=max_iter)
        
        if np.isfinite(err):
            # Extract rel_max from info
            if info is not None and len(info) >= 3:
                current_rel_max = info[2]
            else:
                current_rel_max = np.inf

            # We still optimize for RMSE (err), but we track rel_max
            if err < best_err:
                best_err, best_model, best_info = err, model, info
                best_rel_max = current_rel_max
                print(f"       -> New best error: {best_err:.6e} | rel_max: {best_rel_max:.4f}")
                
                # *** Stop condition based on rel_max ***
                if best_rel_max <= target_rel_max:
                    print(f"    ✅ Target rel_max {target_rel_max} reached! Stopping early.")
                    break

    if best_model is None:
        best_err, best_model, best_info = evaluate_circuit(circuit_str, f, Z, max_iter=max_iter)

    return best_err, best_model, best_info

def random_arc_circuit(max_segments=4, rng=None):
    rng = rng or random.Random(DEFAULT_RANDOM_SEED)
    base_elements = ["CPE", "W", "Wo", "Ws"] 
    n_segments = rng.randint(1, max_segments)
    circuit = "R0"
    idx = 1
    for _ in range(n_segments):
        seg_type = rng.choice(["simple", "randles"])
        e = rng.choice(base_elements)
        if seg_type == "simple":
            circuit += f"-{e}{idx}"
            idx += 1
        else:
            circuit += f"-p(R{idx},{e}{idx})"
            idx += 1
    return circuit

# =========================================================
# Nested-circuit helpers (ADDED for Level 3 & Level 4)
# =========================================================

def _random_branch(idx: int, allow_warburg: bool = True, allow_series_len: int = 2,
                   rng=None):
    """Build a branch string that can be used inside p(..., ...).
    Branch can be a single element (e.g., R1) or a short series (e.g., R1-CPE1 or R1-W1).
    """
    rng = rng or random.Random(DEFAULT_RANDOM_SEED)
    elems = []
    # Always start with a resistor in branches to keep things numerically stable
    elems.append(f"R{idx}")
    # add one more element
    choose_pool = ["CPE", "C"]
    if allow_warburg:
        choose_pool += ["W"]
    t = rng.choice(choose_pool)
    if t == "CPE":
        elems.append(f"CPE{idx}")
    elif t == "C":
        elems.append(f"C{idx}")
    else:
        elems.append(f"W{idx}")

    # Optionally add another element (rare) to increase nesting richness
    if allow_series_len >= 3 and rng.random() < 0.25:
        idx2 = idx + 1
        t2 = rng.choice(["CPE", "C"])
        elems.append(f"{t2}{idx2}")

    return "-".join(elems) if len(elems) > 1 else elems[0]


def generate_nested_circuit_library(shape_mode: str = "arc", target_size: int = 1500,
                                    random_seed: int = DEFAULT_RANDOM_SEED):
    """Generate additional candidates that include *nested* parallel blocks.
    These are ONLY appended for Level 3/4 as requested.
    """
    circuits = set()
    rng = random.Random(int(random_seed))

    # --- Deterministic nested templates (safe + common) ---
    # Note: branches inside p(...) can themselves contain a parallel p(...).
    templates = [
        "R0-p(R1,p(R2,CPE2))",
        "R0-p(R1,p(R2,C2))",
        "R0-p(p(R1,CPE1),p(R2,CPE2))",
        "R0-p(p(R1,C1),p(R2,C2))",
        "R0-p(R1-CPE1,p(R2,CPE2))",
        "R0-p(R1-C1,p(R2,C2))",
        "R0-p(R1-W1,p(R2,CPE2))",
        "R0-p(R1-W1,p(R2,C2))",
        "R0-p(p(R1-W1,CPE1),p(R2,CPE2))",
        "R0-p(p(R1-W1,C1),p(R2,C2))",
        # 2-level nesting with an external arc
        "R0-p(R1,CPE1)-p(R2,p(R3,CPE3))",
        "R0-p(R1,C1)-p(R2,p(R3,C3))",
        "R0-p(R1,CPE1)-p(p(R2,CPE2),p(R3,CPE3))",
        "R0-p(R1,C1)-p(p(R2,C2),p(R3,C3))",
    ]
    for t in templates:
        circuits.add(t)

    # Linear mode: keep it small and more resistive/warburg-like
    if shape_mode == "linear":
        circuits |= {
            "R0-p(R1-W1,p(R2,CPE2))",
            "R0-p(R1-W1,p(R2,C2))",
            "R0-R1-p(R2,p(R3,CPE3))",
            "R0-R1-p(R2,p(R3,C3))",
        }

    # --- Random nested generation ---
    # Hard caps to prevent explosion in Level 4
    max_attempts = max(int(target_size * 6), 2000)
    attempts = 0
    while len(circuits) < int(target_size) and attempts < max_attempts:
        attempts += 1

        # choose indices deterministically so tokens are valid (R1, CPE1, ...)
        base_idx = rng.randint(1, 6)
        inner_idx = base_idx + 1
        inner_parallel_idx = base_idx + 2

        # inner parallel block
        inner = f"p({_random_branch(inner_idx, allow_warburg=(shape_mode!='arc'), rng=rng)},CPE{inner_parallel_idx})"
        if rng.random() < 0.45:
            inner = f"p({_random_branch(inner_idx, allow_warburg=True, rng=rng)},C{inner_parallel_idx})"

        # outer parallel
        outer_left = _random_branch(base_idx, allow_warburg=True, rng=rng)
        outer = f"p({outer_left},{inner})"

        # build full circuit
        c = f"R0-{outer}"

        # optionally chain another arc
        if shape_mode == "arc" and rng.random() < 0.35:
            j = inner_parallel_idx + 1
            c += f"-p(R{j},CPE{j})"
            if rng.random() < 0.25:
                c += f"-W{j}"

        circuits.add(c)

    return sorted(circuits)

# =========================================================
# Feature extraction & rule-based circuit suggestion
# =========================================================

def extract_eis_features(f, Z):
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)

    order = np.argsort(f)
    f_sorted = f[order]
    Z_sorted = Z[order]
    Zreal = Z_sorted.real
    Zimag = Z_sorted.imag

    y = -Zimag
    num_peaks = 0
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            num_peaks += 1

    tail_angle_deg = 0.0
    if len(Zreal) >= 3:
        n_tail = min(8, len(Zreal))
        xr = Zreal[-n_tail:]
        yi = -Zimag[-n_tail:]
        A = np.vstack([xr - xr.mean(), np.ones_like(xr)]).T
        m, b = np.linalg.lstsq(A, yi, rcond=None)[0]
        tail_angle_deg = float(np.degrees(np.arctan(m)))

    has_inductive = False #bool(np.any(Zimag > 0))
    phase_deg = np.angle(Z_sorted, deg=True)
    max_phase = float(np.min(phase_deg))

    return {
        "num_peaks": int(num_peaks),
        "tail_angle_deg": float(tail_angle_deg),
        "has_inductive": has_inductive,
        "max_phase_deg": float(max_phase),
    }

def suggest_circuits_by_rules(features, shape_mode):
    n_peaks = features["num_peaks"]
    
    circuits = []
    
    priority = [
        "R0-p(R1,CPE1)-W1",        
        "R0-p(R1-W1,CPE1)",        
        "R0-p(R1,CPE1)",           
        "R0-p(R1,CPE1)-p(R2,CPE2)" 
    ]
    circuits.extend(priority)

    if shape_mode == "arc":
        if n_peaks <= 1:
            base = [
                "R0-p(R1,CPE1)-p(R2,CPE2)",
                "R0-p(R1,C1)-p(R2,C2)",
                "R0-R1-C1-p(R2,CPE2)",
                "R0-R1-CPE1-p(R2,C2)",
                "R0-p(R1,CPE1)-p(R2,CPE2)-W1",
                "R0-p(R1,C1)-p(R2,C2)-W1",
                "R0-R1-p(R2,CPE2)-W1",
                "R0-R1-p(R2,C2)-W1",
                "R0-L1-p(R1,CPE1)",
                # Complex 3-loop
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
                "R0-p(R1,C1)-p(R2,C2)-p(R3,C3)",
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-W1",
            ]
            circuits += base

        elif n_peaks == 2:
            base = [
                "R0-p(R1,CPE1)-p(R2,CPE2)",
                "R0-p(R1,C1)-p(R2,C2)",
                "R0-p(R1,CPE1)-p(R2,CPE2)-W1",
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)", 
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-W1", 
            ]
            circuits += base
    
    else: 
        base = [
            "R0-R1-W1", "R0-R1-CPE1", "R0-R1-C1", "R0-R1-CPE1-CPE2"
        ]
        circuits += base

    seen = set()
    unique = []
    for c in circuits:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


# ---------------------------------------------------------
# Advisory Bode-phase estimate of resolvable time constants
# ---------------------------------------------------------
def estimate_time_constants_from_phase(f_hz, phase_deg, smooth_window=11,
                                      base_thr_ratio=0.08,
                                      second_peak_ratio=0.30,
                                      min_decade_sep_from_main=0.50,
                                      merge_decade_sep=0.25,
                                      min_peak_phase_deg=5.0,
                                      max_time_constants=4):
    """Conservatively estimate resolvable time constants from Bode phase.

    A time constant is associated with a *phase-magnitude maximum*, not with a
    maximum of ``|d(phi)/d(log f)|``.  The old derivative approach counted the
    two flanks of one phase peak as two loops and amplified small experimental
    ripples.  Here the phase is smoothed, peaks are required to have meaningful
    prominence and separation in log-frequency, and the result is capped at the
    topology depth supported by Levels 1--4.

    This estimate is deliberately advisory: overlapping processes may not be
    resolvable from phase alone.  Circuit ranking therefore uses information
    criteria and fit diagnostics before this estimate.
    """
    info = {}
    try:
        f = np.asarray(f_hz, dtype=float)
        phi = np.asarray(phase_deg, dtype=float)
        mask = np.isfinite(f) & np.isfinite(phi) & (f > 0)
        f = f[mask]
        phi = phi[mask]
        if f.size < 10:
            return 1, info

        order = np.argsort(f)
        f = f[order]
        phi = phi[order]
        logf = np.log10(np.clip(f, 1e-30, np.inf))

        # Capacitive EIS commonly has negative phase.  If the imported sign is
        # reversed, fall back to absolute phase so the advisory remains useful.
        phase_mag = np.clip(-phi, 0.0, np.inf)
        if float(np.nanmax(phase_mag)) < float(min_peak_phase_deg):
            phase_mag = np.abs(phi)

        w = max(5, int(smooth_window or 11))
        if w % 2 == 0:
            w += 1
        if w >= phase_mag.size:
            w = phase_mag.size if phase_mag.size % 2 == 1 else phase_mag.size - 1
        if w >= 5:
            phase_s = savgol_filter(phase_mag, window_length=w, polyorder=2, mode="interp")
        else:
            phase_s = phase_mag

        log_step = float(np.nanmedian(np.diff(logf)))
        min_sep_dec = max(float(min_decade_sep_from_main), float(merge_decade_sep))
        min_distance = max(1, int(round(min_sep_dec / max(abs(log_step), 1e-12))))
        dynamic_range = max(float(np.nanmax(phase_s) - np.nanmin(phase_s)), 0.0)
        prominence = max(1.5, float(base_thr_ratio) * dynamic_range)

        peaks, props = find_peaks(
            phase_s,
            height=float(min_peak_phase_deg),
            prominence=prominence,
            distance=min_distance,
        )

        if peaks.size:
            strengths = np.asarray(props.get("prominences", phase_s[peaks]), dtype=float)
            strongest = float(np.max(strengths))
            keep_mask = strengths >= max(float(second_peak_ratio) * strongest, prominence)
            peaks = peaks[keep_mask]
            strengths = strengths[keep_mask]

            # Retain the most prominent resolvable processes, then restore
            # frequency order for reporting.
            if peaks.size > int(max_time_constants):
                top = np.argsort(strengths)[-int(max_time_constants):]
                peaks, strengths = peaks[top], strengths[top]
            order_peaks = np.argsort(logf[peaks])
            peaks, strengths = peaks[order_peaks], strengths[order_peaks]
        else:
            strengths = np.asarray([], dtype=float)

        peak_freqs = [float(f[i]) for i in peaks]
        peak_strengths = [float(v) for v in strengths]
        peak_taus = [float(1.0 / (2.0 * np.pi * pf)) for pf in peak_freqs if pf > 0]
        n_tc = max(1, int(len(peak_freqs)))

        info = {
            "peak_freqs_hz": peak_freqs,
            "peak_strengths": peak_strengths,
            "peak_taus_s": peak_taus,
            "max_score": float(np.nanmax(phase_s)) if phase_s.size else 0.0,
            "threshold": float(prominence),
            "method": "smoothed_phase_maxima",
            "advisory_only": True,
            "params": {
                "base_thr_ratio": float(base_thr_ratio),
                "second_peak_ratio": float(second_peak_ratio),
                "min_decade_sep_from_main": float(min_decade_sep_from_main),
                "merge_decade_sep": float(merge_decade_sep),
                "smooth_window": int(w),
                "min_peak_phase_deg": float(min_peak_phase_deg),
                "max_time_constants": int(max_time_constants),
            },
        }
        return n_tc, info
    except Exception:
        return 1, info

def _circuit_complexity_score(c_str):
    """Lower is simpler. Rough parameter proxy."""
    # Count element tokens
    nR = len(re.findall(r"\bR\d+\b", c_str))
    nCPE = len(re.findall(r"\bCPE\d+\b", c_str))
    nC = len(re.findall(r"\bC\d+\b", c_str))
    nW = len(re.findall(r"\bW\d+\b|\bWo\d+\b|\bWs\d+\b", c_str))
    nL = len(re.findall(r"\bL\d+\b", c_str))
    # Parallel blocks add complexity
    nP = c_str.count("p(")
    return nR + nCPE + nC + 2*nW + 2*nL + nP


def _count_parameter_bound_hits(circuit_str, model, tol_fraction=0.005):
    """Count fitted parameters that are effectively pinned at a safety bound."""
    if model is None or not hasattr(model, "parameters_"):
        return 999999
    names = _extract_param_names(circuit_str)
    values = np.asarray(model.parameters_, dtype=float)
    if len(names) != values.size:
        return 999999

    hits = 0
    for name, value in zip(names, values):
        if not np.isfinite(value):
            hits += 1
        elif name.startswith("CPE") and name.endswith("_1"):
            lo, hi = 0.4, 1.0
            tol = float(tol_fraction) * (hi - lo)
            if value <= lo + tol or value >= hi - tol:
                hits += 1
        elif value <= 1e-28:
            # Positive parameters collapsed onto the broad numerical floor.
            hits += 1
    return int(hits)


def circuit_selection_key(result):
    """Physics-aware ranking key for a fitted circuit result.

    Hard physical conflicts are excluded before numerical closeness is
    considered.  Parameter identifiability and structured residuals then
    precede physics-adjusted information criteria.  This prevents an
    over-flexible but physically unsupported topology from winning only
    because it shaves a small amount from RMSE.
    """
    status_bad = 0 if result.get("status") == "OK" else 1
    hard_reject = 1 if result.get("hard_reject", False) else 0
    physical_rank = {
        "consistent": 0,
        "plausible": 1,
        "weak": 2,
        "inconsistent": 3,
    }.get(str(result.get("physical_status", "inconsistent")), 3)
    identifiability_rank = int(result.get("identifiability_rank", 3))
    residual_rank = int(result.get("residual_structure_rank", 2))
    bound_hits = int(result.get("bound_hits", 999999))

    def metric(name):
        value = float(result.get(name, np.inf))
        return value if np.isfinite(value) else np.inf

    return (
        status_bad,
        hard_reject,
        physical_rank,
        0 if bound_hits == 0 else 1,
        bound_hits,
        identifiability_rank,
        residual_rank,
        metric("physics_adjusted_bic"),
        metric("physics_adjusted_aicc"),
        metric("error"),
        metric("mean_rel_err"),
        metric("rel_max"),
        int(result.get("loop_mismatch", 999999)),
        _circuit_complexity_score(result.get("circuit", "")),
        result.get("circuit", ""),
    )


def rank_circuit_results(results):
    return sorted(list(results or []), key=circuit_selection_key)


def select_best_circuit_result(results):
    ranked = rank_circuit_results(results)
    for result in ranked:
        if result.get("status") == "OK" and not result.get("hard_reject", False):
            return result
    return None


def _update_result_from_fit(result, circuit_str, error, model, info):
    """Update every ranking metric after a refined fit (not only RMSE)."""
    if model is None or not np.isfinite(error) or info is None or len(info) < 6:
        return result
    aicc, bic, rel_max, chi_sq, rmse_abs, mean_rel_err = info[:6]
    result.update({
        "error": float(error),
        "aic": float(aicc),
        "aicc": float(aicc),
        "bic": float(bic),
        "rel_max": float(rel_max),
        "chi_sq": float(chi_sq),
        "rmse_abs": float(rmse_abs),
        "mean_rel_err": float(mean_rel_err),
        "bound_hits": _count_parameter_bound_hits(circuit_str, model),
        "status": "OK",
        "model": model,
        "info": info,
    })
    return result


def save_candidate_ranking(results, output_folder,
                           filename="Candidate_Circuit_Ranking.csv"):
    """Export the complete automatic-selection audit trail."""
    if not output_folder:
        return None
    os.makedirs(output_folder, exist_ok=True)
    ranked = rank_circuit_results(results)
    rows = []
    for rank, r in enumerate(ranked, 1):
        rows.append({
            "Rank": int(rank),
            "Circuit": r.get("circuit", ""),
            "Status": r.get("status", ""),
            "Parallel_blocks": r.get("loops", np.nan),
            "Phase_loop_mismatch_advisory": r.get("loop_mismatch", np.nan),
            "Parameter_bound_hits": r.get("bound_hits", np.nan),
            "Physical_status": r.get("physical_status", ""),
            "Physical_reasons": "; ".join(r.get("physical_reasons", []) or []),
            "Identifiability_status": r.get("identifiability_status", ""),
            "Identifiability_flags": "; ".join(r.get("identifiability_flags", []) or []),
            "Residual_lag1_structure": r.get("residual_structure", np.inf),
            "BIC": r.get("bic", np.inf),
            "AICc": r.get("aicc", r.get("aic", np.inf)),
            "Physics_adjusted_BIC": r.get("physics_adjusted_bic", np.inf),
            "Physics_adjusted_AICc": r.get("physics_adjusted_aicc", np.inf),
            "Delta_AICc_physics": r.get("delta_aicc_physics", np.inf),
            "Akaike_weight": r.get("akaike_weight", 0.0),
            "Modulus_weighted_RMS": r.get("error", np.inf),
            "Maximum_relative_residual": r.get("rel_max", np.inf),
            "Mean_relative_error": r.get("mean_rel_err", np.inf),
            "Absolute_RMSE_ohm": r.get("rmse_abs", np.inf),
            "Chi_squared_relative": r.get("chi_sq", np.inf),
            "Source": r.get("source", ""),
            "Random_seed": int(DEFAULT_RANDOM_SEED),
        })
    path = os.path.join(output_folder, filename)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_physics_selection_audit(
    output_folder,
    *,
    features,
    lin_kk,
    confidence,
    results,
    level,
    allowed_elements,
    filename="Physics_Guided_Selection_Audit.json",
):
    """Write a machine-readable audit record for automatic topology ranking."""
    if not output_folder:
        return None
    os.makedirs(output_folder, exist_ok=True)
    best = select_best_circuit_result(results)
    payload = {
        "software_version": SOFTWARE_VERSION,
        "physics_engine_version": PHYSICS_ENGINE_VERSION,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "analysis_level": level,
        "allowed_element_families": sorted(allowed_elements),
        "candidate_count": len(results),
        "detected_features": features,
        "lin_kk": lin_kk,
        "selection_confidence": confidence,
        "top_candidate": None if best is None else {
            "circuit": best.get("circuit"),
            "physical_status": best.get("physical_status"),
            "identifiability_status": best.get("identifiability_status"),
            "physics_adjusted_aicc": best.get("physics_adjusted_aicc"),
            "akaike_weight": best.get("akaike_weight"),
            "bound_hits": best.get("bound_hits"),
        },
        "interpretation_notice": (
            "Automatic topology ranking is advisory. Equivalent circuits may "
            "be non-unique over a finite frequency window."
        ),
    }
    path = os.path.join(output_folder, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, ensure_ascii=False)
    return path

def prioritize_circuits_by_bode_phase(circuits, expected_loops):
    """Return a stable unique list; phase is no longer a hard pre-filter.

    The function name is retained for compatibility with the existing GUI
    pipeline.  ``expected_loops`` is stored as an advisory diagnostic in each
    result and is used only as a late tie-break by ``circuit_selection_key``.
    """
    return list(dict.fromkeys(list(circuits)))

def generate_impedance_circuit_library(shape_mode="linear", target_size=5000,
                                       random_seed=DEFAULT_RANDOM_SEED):
    circuits = set()
    rng = random.Random(int(random_seed))
    base_elements = ["CPE", "W", "Wo", "Ws"]

    circuits.add("R0-p(R1,CPE1)-W1")
    circuits.add("R0-p(R1-W1,CPE1)")
    circuits.add("R0-p(R1,CPE1)-p(R2,CPE2)")
    circuits.add("R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)")

    max_random_attempts = target_size * 5
    attempts = 0
    while len(circuits) < target_size and attempts < max_random_attempts:
        c = random_arc_circuit(max_segments=4, rng=rng)
        circuits.add(c)
        attempts += 1

    return sorted(circuits)

# =========================================================
# Residual-based analysis
# =========================================================

def analyze_residuals_shape(f, Z, Z_fit):
    f = np.asarray(f, dtype=float)
    Z = np.asarray(Z, dtype=complex)
    Z_fit = np.asarray(Z_fit, dtype=complex)

    absZ = np.abs(Z)
    resid = Z - Z_fit
    abs_resid = np.abs(resid)

    max_absZ = np.max(absZ) if np.max(absZ) > 0 else 1.0
    rel_max = float(np.max(abs_resid) / max_absZ)

    order = np.argsort(f)
    r_sorted = abs_resid[order]
    n_peaks = 0
    for i in range(1, len(r_sorted) - 1):
        if r_sorted[i] > r_sorted[i - 1] and r_sorted[i] > r_sorted[i + 1]:
            n_peaks += 1

    return rel_max, int(n_peaks)


def propose_more_complex_circuits(base_circuit):
    suggestions = []
    if base_circuit == "R0-p(R1,CPE1)-W1":
        suggestions.append("R0-p(R1,CPE1)-p(R2,CPE2)-W1")
    
    generic = [
        "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
        "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-W1"
    ]
    for g in generic:
        if g not in suggestions:
            suggestions.append(g)

    seen = set()
    uniq = []
    for c in suggestions:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq

# =========================================================
# GUI: progress bar evaluation
# =========================================================

def evaluate_circuit_list_with_progress(circuit_list, f, Z, source_label="library", expected_loops=None):
    if not circuit_list:
        return []

    # Preserve a stable candidate order; phase-loop count is advisory only.
    try:
        circuit_list = prioritize_circuits_by_bode_phase(circuit_list, expected_loops)
    except Exception:
        circuit_list = list(circuit_list)

    win = tk.Tk()
    win.title(f"Evaluating {source_label} circuits")

    tk.Label(win, text=f"Evaluating {len(circuit_list)} {source_label} circuits ...").pack(
        padx=10, pady=(10, 5)
    )
    progress = ttk.Progressbar(win, length=300, mode="determinate", maximum=len(circuit_list))
    progress.pack(padx=10, pady=(0, 10))

    status_label = tk.Label(win, text="Please wait...")
    status_label.pack(padx=10, pady=(0, 10))

    # =======================
    # (ADDED) Spinner + Timer (UI Heartbeat)
    # =======================
    spinner_frames = ["|", "/", "-", "\\"]
    spinner_state = {
        "running": False,
        "start": 0.0,
        "i": 0,
        "cur_i": 0,
        "total": len(circuit_list),
        "cur_c": "",
        "source": source_label,
    }

    def _spinner_tick():
        if not spinner_state["running"]:
            return

        spinner_state["i"] += 1
        elapsed = int(time.time() - spinner_state["start"])
        mm, ss = elapsed // 60, elapsed % 60
        frame = spinner_frames[spinner_state["i"] % len(spinner_frames)]

        status_label.config(
            text=f"{frame}  {spinner_state['cur_i']}/{spinner_state['total']}  "
                 f"[{spinner_state['source']}]  Fitting: {spinner_state['cur_c']}   "
                 f"{mm:02d}:{ss:02d}"
        )
        win.after(250, _spinner_tick)

    win.update()

    # =======================
    # (ADDED) CMD spinner + timer (do not cancel)
    # =======================
    spinner_frames = ["⟳", "⟲"]
    _cmd_spinner_running = True
    _cmd_spinner_start = time.time()

    def _cmd_spinner_loop():
        k = 0
        while _cmd_spinner_running:
            elapsed = int(time.time() - _cmd_spinner_start)
            mm, ss = elapsed // 60, elapsed % 60
            frame = spinner_frames[k % len(spinner_frames)]
            # One-line live update in CMD
            sys.stdout.write(
                f"\r{frame}  Continue... don't cancel   |   {mm:02d}:{ss:02d}"
            )
            sys.stdout.flush()
            k += 1
            time.sleep(0.25)
        # clear line at the end (optional)
        sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()

    _cmd_thread = threading.Thread(target=_cmd_spinner_loop, daemon=True)
    _cmd_thread.start()

    results = []
    for i, c in enumerate(circuit_list, 1):
        err, fit_model, info = evaluate_circuit(c, f, Z, max_iter=200)
        
        # info now contains (aic, bic, rel_max, chi_sq, rmse_abs, mean_rel_err)
        if info is not None and len(info) >= 6:
            aic, bic, rel_max, chi_sq, rmse_abs, mean_rel_err = info
        else:
            aic, bic, rel_max = np.inf, np.inf, np.inf
            chi_sq, rmse_abs, mean_rel_err = np.inf, np.inf, np.inf

        n_loops = c.count("p(")
        loop_mismatch = abs(int(n_loops) - int(expected_loops)) if expected_loops is not None else 0

        status = "OK" if (fit_model is not None and np.isfinite(err)) else "FAILED"
        bound_hits = _count_parameter_bound_hits(c, fit_model) if status == "OK" else 999999
        results.append({
            "circuit": c,
            "loops": int(n_loops),
            "loop_mismatch": int(loop_mismatch),
            "error": float(err),
            "aic": float(aic),
            "aicc": float(aic),
            "bic": float(bic),
            "rel_max": float(rel_max),
            "chi_sq": float(chi_sq),
            "rmse_abs": float(rmse_abs),
            "mean_rel_err": float(mean_rel_err),
            "bound_hits": int(bound_hits),
            "source": source_label,
            "status": status,
        })

        progress["value"] = i
        status_label.config(text=f"{i}/{len(circuit_list)} evaluated")
        win.update_idletasks()

    # =======================
    # (ADDED) stop CMD spinner
    # =======================
    _cmd_spinner_running = False
    try:
        _cmd_thread.join(timeout=1.0)
    except Exception:
        pass
    print("✅ Done. Continuing...\n")

    win.destroy()
    return results

# =========================================================
# GUI: circuit selection from results
# =========================================================

def gui_select_circuit_from_results(results):
    if not results:
        return None

    # Statistically penalized ranking first.  Phase-derived loop count is a
    # soft final tie-break because overlapping processes are not always
    # resolvable from Bode phase alone.
    results_sorted = rank_circuit_results(results)

    root = tk.Tk()
    root.title("Select Equivalent Circuit")

    tk.Label(root,
             text="Select a circuit (Click any header to sort):",
             font=("Segoe UI", 10, "bold")).pack(padx=10, pady=(10, 3))

    search_frame = tk.Frame(root)
    search_frame.pack(padx=10, pady=(0, 5), fill="x")

    tk.Label(search_frame, text="Filter Text:").pack(side="left")
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var, width=40)
    search_entry.pack(side="left", padx=5)

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=5, fill="both", expand=True)

    # ---Change: Added Loops column. ---
    columns = ("idx", "loops", "circuit", "error", "chi_sq", "rmse_abs", "mean_rel_err", "source", "status")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=20)
    
    tree.heading("idx", text="#")
    tree.heading("loops", text="Loops", command=lambda: sort_tree(tree, "loops", numeric=True))
    tree.heading("circuit", text="Circuit String")
    tree.heading("error", text="Error (RMS)", command=lambda: sort_tree(tree, "error", numeric=True))
    tree.heading("chi_sq", text="χ²", command=lambda: sort_tree(tree, "chi_sq", numeric=True))
    tree.heading("rmse_abs", text="RMSE", command=lambda: sort_tree(tree, "rmse_abs", numeric=True))
    tree.heading("mean_rel_err", text="|Zexp-Zfit|/|Zexp|", command=lambda: sort_tree(tree, "mean_rel_err", numeric=True))
    tree.heading("source", text="Source")
    tree.heading("status", text="Status")

    tree.column("idx", width=40, anchor="center")
    tree.column("loops", width=55, anchor="center")
    tree.column("circuit", width=320, anchor="w")
    tree.column("error", width=95, anchor="e")
    tree.column("chi_sq", width=95, anchor="e")
    tree.column("rmse_abs", width=95, anchor="e")
    tree.column("mean_rel_err", width=140, anchor="e")
    tree.column("source", width=80, anchor="center")
    tree.column("status", width=70, anchor="center")

    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")

    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)

    # ---Helper function for sorting by clicking on column headers---
    sort_state = {"col": None, "descending": False}

    def sort_tree(tv, col, numeric=False):
        # Toggle direction if the same column is clicked
        if sort_state["col"] == col:
            sort_state["descending"] = not sort_state["descending"]
        else:
            sort_state["col"] = col
            sort_state["descending"] = False  # first click: ascending

        rows = []
        for iid in tv.get_children(''):
            val = tv.set(iid, col)
            if numeric:
                try:
                    v = float(val)
                except Exception:
                    v = float("inf")
                rows.append((v, iid))
            else:
                rows.append((val, iid))

        rows.sort(key=lambda x: x[0], reverse=sort_state["descending"])
        for index, (_, iid) in enumerate(rows):
            tv.move(iid, '', index)

    def populate_tree(filter_text=""):
        tree.delete(*tree.get_children())
        filter_text = filter_text.lower()
        display_idx = 1

        for true_idx, r in enumerate(results_sorted):
            c_str = r["circuit"]
            # --- Smart loop count calculation ---
            # Each "p(" represents a parallel block (loop)
            n_loops = int(r.get("loops", c_str.count("p(")))
            
            text_concat = f"{c_str} {r['source']} {r['status']}"
            
            # ---Combined filter: if user enters a number, check loop count---
            if filter_text.isdigit():
                if int(filter_text) != n_loops:
                    continue
            # ---Else perform a regular text search---
            elif filter_text and filter_text not in text_concat.lower():
                continue

            err_str = "inf" if not np.isfinite(r["error"]) else f"{r['error']:.3e}"

            chi_str = "inf" if not np.isfinite(r.get("chi_sq", np.inf)) else f"{r['chi_sq']:.3e}"
            rmse_str = "inf" if not np.isfinite(r.get("rmse_abs", np.inf)) else f"{r['rmse_abs']:.3e}"
            mre_str  = "inf" if not np.isfinite(r.get("mean_rel_err", np.inf)) else f"{r['mean_rel_err']:.3e}"

            tree.insert(
                "",
                "end",
                iid=str(true_idx),
                values=(display_idx, n_loops, c_str, err_str, chi_str, rmse_str, mre_str, r['source'], r['status']),
            )
            display_idx += 1

    def on_search(*args):
        populate_tree(search_var.get())

    search_var.trace_add("write", on_search)
    populate_tree()
    search_entry.focus_set()

    selected_circuit = {"value": None}

    def on_fit():
        sel = tree.selection()
        if not sel:
            messagebox.showwarning("No selection", "Please select a circuit.")
            return
        true_idx = int(sel[0])
        r = results_sorted[true_idx]
        selected_circuit["value"] = r["circuit"]
        root.destroy()

    def on_cancel():
        selected_circuit["value"] = None
        root.destroy()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=10)

    tk.Button(btn_frame, text="Fit Selected", width=15, command=on_fit).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Cancel", width=10, command=on_cancel).pack(side="left", padx=5)

    root.mainloop()
    return selected_circuit["value"]
    
# =========================================================
# Reporting helpers
# =========================================================

def _param_unit_and_description(name: str):
    unit = ""
    desc = "Param"

    if name.startswith("R"):
        unit = "Ohm"
        if name == "R0":
            desc = "Series R" # was shortened
        else:
            desc = "Charge Transfer R" # was shortened
    elif name.startswith("CPE") and name.endswith("_0"):
        unit = "Ohm^-1 s^n" # was shortened
        desc = "CPE Q"
    elif name.startswith("CPE") and name.endswith("_1"):
        unit = "0 < n < 1"
        desc = "CPE n"
    elif name.startswith("C"):
        unit = "F"
        desc = "Capacitance"
    elif name.startswith("L"):
        unit = "H"
        desc = "Inductance"
    elif name.startswith("W") or name.startswith("Wo") or name.startswith("Ws"):
        unit = "Ohm s^-0.5"
        desc = "Warburg"
    elif name.startswith("G"):
        unit = "S"
        desc = "Conductance"

    return unit, desc

# =========================================================
# Helper: Generate clean parameter names from circuit string
# =========================================================
def generate_param_names_from_circuit(circuit_str):
    """
    Parses the circuit string (e.g. 'R0-p(R1,CPE1)') and returns 
    the list of parameter names in order (e.g. ['R0', 'R1', 'CPE1_0', 'CPE1_1']).
    """
    import re  # Needs regex to parse string
    
    # Find all elements like R0, R1, CPE1, W2, etc.
    # Regex pattern: look for letters followed by numbers
    tokens = re.findall(r'([A-Za-z]+[0-9]+)', circuit_str)
    
    param_names = []
    for tok in tokens:
        if tok.startswith("CPE"):
            # CPE has 2 parameters: Q (magnitude) and n (exponent)
            param_names.append(f"{tok}_0") # Q
            param_names.append(f"{tok}_1") # n
        else:
            # R, C, L, W, etc. have 1 parameter
            param_names.append(tok)
            
    return param_names

# =========================================================
# (Here starts your save_fit_outputs function...)
# def save_fit_outputs(...):
# =========================================================
# def save_fit_outputs(...):
# =========================================================

def save_fit_outputs(f, Z, Z_fit, Phase_raw, chosen_fit, chosen_circuit,
                     chosen_error, chosen_aic, chosen_bic,
                     analysis_level, shape_mode, autoeis_used,
                     source_file, base_output_folder, fit_index,
                     expected_loops=None, phase_info=None):
    
    # --- 1. Font settings Times New Roman ---
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["mathtext.fontset"] = "stix"

    fit_folder_name = f"Fit_{fit_index:02d}_{chosen_circuit.replace('(', '').replace(')', '').replace(',', '-')}"
    fit_folder = os.path.join(base_output_folder, fit_folder_name)
    os.makedirs(fit_folder, exist_ok=True)
    
    # --- Initial calculations (outside try block to prevent errors) ---
    absZ = np.abs(Z)
    z_fit_mag = np.abs(Z_fit)
    z_fit_phase = np.angle(Z_fit, deg=True)
    phase_exp = Phase_raw if Phase_raw is not None else np.angle(Z, deg=True)
    
    # Error calculations
    resid = Z - Z_fit
    rmse_val = np.sqrt(np.mean(np.abs(resid)**2))
    
    # Relative error
    rel_err_array = np.abs(resid) / (absZ + 1e-12)
    mean_rel_err = np.mean(rel_err_array)
    rel_error_percent = rel_err_array * 100

    # Chi-squared
    chi_sq = np.sum((np.abs(resid) / (absZ + 1e-12))**2)
    names_raw = _extract_param_names(chosen_circuit)
    vals = chosen_fit.parameters_
    # --- 2. SAVE CSVs & PLOTS ---
    try:
        # Empirical phase (if the phase column is not given, we calculate from the Z angle)
        if Phase_raw is None:
            phase_exp = np.angle(Z, deg=True)
        else:
            phase_exp = Phase_raw

        # ==========================
        # 2.1 Nyquist Data (CSV)
        # ==========================
        df_nyquist = pd.DataFrame({
            "Freq (Hz)": f,

            # Experimental data (same as original Z)
            "Z'_Data": Z.real,
            "-Z''_Data": -Z.imag,

            # Fitted model
            "Z'_Fit": Z_fit.real,
            "-Z''_Fit": -Z_fit.imag,

            "Error (%)": rel_error_percent
        })
        nyquist_path = os.path.join(fit_folder, "Nyquist_Data.csv")
        df_nyquist.to_csv(nyquist_path, index=False)

        # ==========================
        # 2.2 Bode Magnitude (CSV)
        # ==========================
        df_bode_mag = pd.DataFrame({
            "Freq (Hz)": f,

            # Experimental magnitude
            "|Z|_Data": absZ,

            # Fitted magnitude
            "|Z|_Fit": z_fit_mag,

            "Error (%)": rel_error_percent
        })
        bode_mag_path = os.path.join(fit_folder, "Bode_Magnitude_Data.csv")
        df_bode_mag.to_csv(bode_mag_path, index=False)

        # ==========================
        # 2.3 Bode Phase (CSV)
        # ==========================
        df_bode_phase = pd.DataFrame({
            "Freq (Hz)": f,

            # Experimental phase
            "Phase_Data": phase_exp,

            # Fitted phase
            "Phase_Fit": z_fit_phase
        })
        bode_phase_path = os.path.join(fit_folder, "Bode_Phase_Data.csv")
        df_bode_phase.to_csv(bode_phase_path, index=False)

        # ==========================
        # 3. SAVE PLOT IMAGES (PNG)
        # ==========================

        # 3.1 Nyquist Plot
        fig_nyq, ax_nyq = plt.subplots(figsize=(6, 5))
        ax_nyq.minorticks_off()
        ax_nyq.grid(True, which='major')
        ax_nyq.plot(Z.real, -Z.imag, 'o', ms=4, label='Exp')
        ax_nyq.plot(Z_fit.real, -Z_fit.imag, '-', lw=2, label='Fit')
        ax_nyq.set_xlabel("Z' (Ohm)")
        ax_nyq.set_ylabel("-Z'' (Ohm)")
        ax_nyq.set_title("Nyquist Plot")
        ax_nyq.grid(True, linestyle='--', alpha=0.6)
        ax_nyq.legend()
        nyq_img_path = os.path.join(fit_folder, "Nyquist_Plot.png")
        plt.tight_layout()
        fig_nyq.savefig(nyq_img_path, dpi=300)
        plt.close(fig_nyq)

                # 3.2 Bode Magnitude Plot
        fig_mag, ax_mag = plt.subplots(figsize=(6, 5))
        ax_mag.loglog(f, absZ, 'o', ms=4, label='|Z| Exp')
        ax_mag.loglog(f, z_fit_mag, '-', lw=2, label='|Z| Fit')
        ax_mag.set_xlabel("Frequency (Hz)")
        ax_mag.set_ylabel("|Z| (Ohm)")
        ax_mag.set_title("Bode Magnitude")

        # Only major grid, no minor
        ax_mag.minorticks_off()
        ax_mag.grid(True, which='major', linestyle='--', alpha=0.6)

        ax_mag.legend()
        mag_img_path = os.path.join(fit_folder, "Bode_Magnitude_Data.png")
        plt.tight_layout()
        fig_mag.savefig(mag_img_path, dpi=300)
        plt.close(fig_mag)

                # 3.3 Bode Phase Plot
        fig_phase, ax_phase = plt.subplots(figsize=(6, 5))
        ax_phase.semilogx(f, phase_exp, 'o', ms=4, label='Phase Exp')
        ax_phase.semilogx(f, z_fit_phase, '-', lw=2, label='Phase Fit')
        ax_phase.set_xlabel("Frequency (Hz)")
        ax_phase.set_ylabel("Phase (deg)")
        ax_phase.set_title("Bode Phase")

        # Only major grid, no minor
        ax_phase.minorticks_off()
        ax_phase.grid(True, which='major', linestyle='--', alpha=0.6)

        ax_phase.legend()
        phase_img_path = os.path.join(fit_folder, "Bode_Phase_Data.png")
        plt.tight_layout()
        fig_phase.savefig(phase_img_path, dpi=300)
        plt.close(fig_phase)    

        print(f"   -> Data & plots saved to: {fit_folder}")

    except Exception as e:
        print(f"❌ Error saving CSVs/plots: {e}")

        # d) Parameter file
        param_list = []
        for n, v in zip(names_raw, vals):
            unit = ""
            if n.startswith("R"): unit = "Ohm"
            elif "CPE" in n:
                if "_0" in n: unit = "S.s^n"
                elif "_1" in n: unit = "0 < n <= 1"
            elif n.startswith("C"): unit = "F"
            elif "W" in n: unit = "Ohm.s^-0.5"
            param_list.append({"Element": n, "Value": v, "Unit": unit})
        
        pd.DataFrame(param_list).to_csv(os.path.join(fit_folder, "Fitted_Parameters.csv"), index=False)
        
        print(f"   -> CSV files saved in {fit_folder}")

    except Exception as e:
        print(f"CSV Save Error: {e}")

    # --- 3. Generate 3-page PDF ---
    pdf_path = os.path.join(fit_folder, "Official_Report.pdf")
    try:
        with PdfPages(pdf_path) as pdf:
            
            # Page 1
            fig1 = plt.figure(figsize=(8.27, 11.69))
            gs1 = fig1.add_gridspec(20, 1)
            
            ax_head = fig1.add_subplot(gs1[0:4, :]); ax_head.axis('off')
            ax_head.text(0.5, 0.8, "EIS Analysis Report", fontsize=24, weight='bold', ha='center')
            info_text = (f"File: {os.path.basename(source_file)}\n"
                         f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                         f"Software: {SOFTWARE_NAME} v{SOFTWARE_VERSION}")
            ax_head.text(0.1, 0.1, info_text, fontsize=14, linespacing=1.6)
            ax_head.axhline(y=0, color='black', lw=1)

            ax_circ = fig1.add_subplot(gs1[4:12, :]); ax_circ.axis('off')
            img_path = os.path.join(fit_folder, "schematic.png")
            if draw_circuit_to_file(chosen_circuit, img_path):
                img = plt.imread(img_path); ax_circ.imshow(img)
            else: ax_circ.text(0.5, 0.5, "Schematic gen failed", ha='center')

            ax_err = fig1.add_subplot(gs1[13:16, :]); ax_err.axis('off')
            err_text = (
                f"Statistical Metrics:\n"
                f"--------------------\n"
                f"RMSE : {rmse_val:.5e}\n"
                rf"$\chi^2$ (Chi-squared) : {chi_sq:.5e}\n"
                f"Mean Rel. Error ($|Z_{{exp}} - Z_{{fit}}| / |Z_{{exp}}|$): {mean_rel_err:.4%}\n"
                f"AICc: {chosen_aic:.1f} | BIC: {chosen_bic:.1f}"
            )
            # ---- Add Bode-phase loops/time-constants to report text ----
            if phase_info is None:
                phase_info = {}
            pf = phase_info.get("peak_freqs_hz", [])
            taus = phase_info.get("taus_s", [])
            bode_line = f"\nBode phase loops: {expected_loops} | f_peak: {pf} | tau(s): {taus}"
            err_text = err_text + bode_line

            ax_err.text(0.5, 0.5, err_text, ha='center', va='center', fontsize=14,
                        bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="black"))

            ax_cite = fig1.add_subplot(gs1[17:20, :]); ax_cite.axis('off')
            cite_txt = "CITATION REQUIRED:\nPlease cite: Author Name, Journal Title (2025)."
            ax_cite.text(0.5, 0.5, cite_txt, ha='center', fontsize=12, style='italic', color='black',
                         bbox=dict(boxstyle="round,pad=1", fc="#ffe6e6", ec="red", lw=2))
            
            pdf.savefig(fig1); plt.close(fig1)

            # Page 2
            fig2 = plt.figure(figsize=(8.27, 11.69))
            ax2 = fig2.add_subplot(111); ax2.axis('off')
            ax2.set_title("Fitted Parameters", fontsize=20, weight='bold', pad=20)
            
            table_data = []
            for n, v in zip(names_raw, vals):
                disp_name, unit = n, ""
                if n.startswith("R"): unit = "Ohm"
                elif "CPE" in n:
                    if "_0" in n: disp_name = n.replace("_0", " (Q)"); unit = "S.s^n"
                    elif "_1" in n: disp_name = n.replace("_1", " (n)"); unit = "0 < n <= 1"
                elif n.startswith("C"): unit = "F"
                elif "W" in n: unit = "Ohm.s^-0.5"
                table_data.append([disp_name, f"{v:.4e}", unit])
                
            table = ax2.table(cellText=table_data, colLabels=["Element", "Value", "Unit"], loc='center', cellLoc='center')
            table.set_fontsize(14); table.scale(1, 2)
            pdf.savefig(fig2); plt.close(fig2)

            # Page 3
            fig3 = plt.figure(figsize=(8.27, 11.69))
            gs3 = fig3.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.3)
            
            # 1. Nyquist
            ax_nyq = fig3.add_subplot(gs3[0])
            ax_nyq.plot(Z.real, -Z.imag, 'bo', ms=4, label='Data')
            ax_nyq.plot(Z_fit.real, -Z_fit.imag, 'r-', lw=2, label='Fit')
            ax_nyq.set_xlabel("Z' (Ohm)", fontsize=12)
            ax_nyq.set_ylabel("-Z'' (Ohm)", fontsize=12)
            ax_nyq.set_title("Nyquist Plot", weight='bold', fontsize=14)
            ax_nyq.legend()
            ax_nyq.grid(True, linestyle='--', alpha=0.6)
            ax_nyq.set_aspect('auto') # تنظیم برای هم‌اندازه شدن

            # 2. Bode Magnitude
            ax_mag = fig3.add_subplot(gs3[1])
            ax_mag.loglog(f, absZ, 'bo', ms=4)
            ax_mag.loglog(f, z_fit_mag, 'r-', lw=2)
            ax_mag.set_ylabel("|Z| (Ohm)", fontsize=12)
            ax_mag.set_title("Bode Magnitude", weight='bold', fontsize=14)
            ax_mag.minorticks_off()
            ax_mag.grid(True, which='major', linestyle='-', alpha=0.8, color='gray')

            # 3. Bode Phase
            ax_ph = fig3.add_subplot(gs3[2])
            ax_ph.semilogx(f, phase_exp, 'bo', ms=4, label='Data')
            ax_ph.semilogx(f, z_fit_phase, 'r-', lw=2, label='Fit')
            ax_ph.set_xlabel("Frequency (Hz)", fontsize=12)
            ax_ph.set_ylabel("Phase (deg)", fontsize=12)
            ax_ph.set_title("Bode Phase", weight='bold', fontsize=14)
            ax_ph.minorticks_off()
            ax_ph.grid(True, which='major', linestyle='-', alpha=0.8)
            
            pdf.savefig(fig3); plt.close(fig3)

        print(f"✅ PDF saved: {pdf_path}")
        
    except Exception as e:
        print(f"❌ Report failed: {e}")
        import traceback; traceback.print_exc()

    # ==========================
    #  🔹 SAVE METADATA JSON
    # ==========================
    try:
        meta = {
            "circuit": chosen_circuit,
            "parameters": {
                name: float(val)
                for name, val in zip(names_raw, vals)
            },
            "metrics": {
                "rmse": float(rmse_val),
                "mean_relative_error": float(mean_rel_err),
                "aicc": float(chosen_aic),
                "bic": float(chosen_bic),
            },
            "files": {
                "nyquist_csv": os.path.basename(nyquist_path),
                "bode_mag_csv": os.path.basename(bode_mag_path),
                "bode_phase_csv": os.path.basename(bode_phase_path),
                "nyquist_png": os.path.basename(nyq_img_path),
                "bode_mag_png": os.path.basename(mag_img_path),
                "bode_phase_png": os.path.basename(phase_img_path),
                "pdf_report": os.path.basename(pdf_path),
            },
            "analysis": {
                "level": analysis_level,
                "shape_mode": shape_mode,
                "autoeis_used": bool(autoeis_used),
                "bode_phase_estimated_loops": int(expected_loops) if expected_loops is not None else None,
                "bode_phase_estimate_is_advisory": True,
                "bode_phase_peak_freqs_hz": [float(x) for x in (phase_info or {}).get("peak_freqs_hz", [])],
                "bode_phase_taus_s": [float(x) for x in (phase_info or {}).get("peak_taus_s", [])],
                "automatic_ranking": "bound diagnostics, BIC, AICc, weighted errors, phase tie-break",
                "random_seed": int(DEFAULT_RANDOM_SEED),
                "information_criterion_relative_floor": float(INFORMATION_CRITERION_RELATIVE_FLOOR),
            },
            "source_file": os.path.basename(source_file),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "software": {
                "name": SOFTWARE_NAME,
                "version": SOFTWARE_VERSION
            }
        }

        meta_path = os.path.join(fit_folder, "Fit_Metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"✅ Metadata JSON saved: {meta_path}")

    except Exception as e:
        print(f"⚠️ Metadata export failed: {e}")

    return pdf_path, fit_folder
    
# =========================================================
# Main program
# =========================================================

def legacy_main():
    print("=" * 53)
    print(f"  {SOFTWARE_NAME}  –  Version {SOFTWARE_VERSION}")
    print("=" * 53)
    print("\nPrepare your EIS data file (table format).")

    file_path = select_file()
    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"\nSelected File: {file_path}")

    output_folder = build_output_run_folder(file_path, analysis_name="Nyquist and Bode Analysis")
    print(f"Output Folder: {output_folder}")

    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in [".csv", ".txt", ".dat", ".tsv"]:
            try:
                df = pd.read_csv(file_path)
            except Exception:
                df = pd.read_csv(file_path, sep=None, engine="python")
        elif ext in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path)
            except Exception:
                df = pd.read_csv(file_path, sep=None, engine="python")
    except Exception as e:
        print(f"❌ Failed to read file as table: {e}")
        return

    idx_freq, idx_real, idx_imag, idx_phase, analysis_level = gui_select_columns_and_level(df, ae is not None)
    level_label, level_target_size = LEVEL_INFO[analysis_level]
    print(f"\n>>> Selected analysis level: {level_label}")

    try:
        f = np.array(df.iloc[:, idx_freq].values, dtype=float)
        Z_real = np.array(df.iloc[:, idx_real].values, dtype=float)
        Z_imag = np.array(df.iloc[:, idx_imag].values, dtype=float)
        if idx_phase is not None:
            Phase_raw = np.array(df.iloc[:, idx_phase].values, dtype=float)
        else:
            Phase_raw = None
    except ValueError:
        print("❌ Error: Selected columns contain non-numeric data.")
        sys.exit()

    Z = Z_real + 1j * Z_imag
    mask = f > 0
    f = f[mask]
    Z = Z[mask]
    if idx_phase is not None and Phase_raw is not None:
        Phase_raw = Phase_raw[mask]
    else:
        Phase_raw = None

    
    # ---------------------------------------------------------
    # Advisory estimate of resolvable time constants from Bode phase
    # ---------------------------------------------------------
    try:
        phase_for_features = Phase_raw if Phase_raw is not None else np.angle(Z, deg=True)
        expected_loops, phase_info = estimate_time_constants_from_phase(f, phase_for_features, second_peak_ratio=0.30)
        print(f"→ Bode-Phase estimated loops/time-constants = {expected_loops} | peak f (Hz) ≈ {phase_info.get('peak_freqs_hz', [])} | tau (s) ≈ {phase_info.get('peak_taus_s', [])}")
    except Exception as _e:
        expected_loops, phase_info = None, {}
        print(f"⚠️ Could not estimate loops from Bode phase: {_e}")

    # --- IMPORTANT: from here on, NO INDENT (aligned to the left) ---
    delta_real = float(max(Z.real) - min(Z.real))
    delta_imag = float(max(-Z.imag) - min(-Z.imag))

    if delta_real == 0:
        delta_real = 1e-9

    aspect_ratio = delta_imag / delta_real
    print(f"\nShape diagnostic: ΔReal={delta_real:.3g}, ΔImag={delta_imag:.3g}, aspect_ratio={aspect_ratio:.2f}")

    if aspect_ratio > 3.0:
        shape_mode = "linear"
        print("→ Nyquist almost vertical → using LINEAR library (no Randles arcs) for levels 1–4.")
    else:
        shape_mode = "arc"
        print("→ Nyquist arc-like → using ARC library (Randles allowed) for levels 1–4.")

    autoeis_impedance_circuits = set()
    autoeis_used = False

    if analysis_level == "L5":
        if ae is None:
            print("❌ Level5 is not installed, but Level 5 was selected. Exiting.")
            sys.exit()
        try:
            print("\n" + "*" * 60)
            print(">>> [Level 5] Running Level5 full analysis (candidate circuits)...")
            try:
                imp_data = ae.utils.ImpedanceData(f, Z)
                imp_data = imp_data.preprocess()
                freq_ae, Z_ae = imp_data.freq, imp_data.Z
            except Exception:
                freq_ae, Z_ae = f, Z

            kwargs = {
                "iters": 40,
                "complexity": 16,
                "population_size": 120,
                "generations": 40,
                "terminals": "RLPWC",
                "tol": 1e-2,
                "parallel": True,
            }
            circuits_unfiltered = ae.core.generate_equivalent_circuits(freq_ae, Z_ae, **kwargs)
            if circuits_unfiltered is None or len(circuits_unfiltered) == 0:
                raise RuntimeError("Level5 did not return any circuits")

            raw_csv = os.path.join(output_folder, "Level5_Candidates_Raw.csv")
            circuits_unfiltered.to_csv(raw_csv, index=False)
            print(f"✅ Level5 raw candidates saved to: {raw_csv}")

            circuits = ae.core.filter_implausible_circuits(circuits_unfiltered)
            if circuits is None or len(circuits) == 0:
                print("⚠️ Level5: no plausible circuits after filtering, using raw pool.")
                circuits = circuits_unfiltered

            filt_csv = os.path.join(output_folder, "Level5_Candidates_Filtered.csv")
            circuits.to_csv(filt_csv, index=False)
            print(f"✅ Level5 filtered candidates saved to: {filt_csv}")

            for _, row in circuits.iterrows():
                ae_circuit = row["circuitstring"]
                try:
                    imp_circuit = ae.parser.convert_to_impedance_format(ae_circuit)
                except Exception:
                    imp_circuit = ae_circuit
                autoeis_impedance_circuits.add(imp_circuit)

            if len(autoeis_impedance_circuits) > 0:
                autoeis_used = True
                print(f"✅ Level5 provided {len(autoeis_impedance_circuits)} candidate circuits for impedance.py.")
            else:
                print("⚠️ Level5 did not yield any usable circuits for impedance.py.")
        except Exception as e:
            print(f"❌ Error while running Level5: {e}")
            print("⚠️ Will fall back to library/rule-based only.")
            autoeis_used = False
            autoeis_impedance_circuits = set()

        print("\n" + "*" * 60)
    print(">>> Building candidate circuit sets ...")

    library_circuits = []
    rule_based_circuits = []

    if analysis_level in ("L1", "L2", "L3", "L4"):
        target_size = level_target_size
        imp_library = generate_impedance_circuit_library(
            shape_mode=shape_mode,
            target_size=target_size
        )
        library_circuits = sorted(set(imp_library))
        print(f"   -> Library circuits (random/shape-based): {len(library_circuits)}")

        # ---- (ADDED) Nested circuits for Level 3 & Level 4 ----
        # IMPORTANT: this does NOT modify existing candidate logic; it only appends extra candidates.
        if analysis_level in ("L3", "L4"):
            try:
                # Nested candidates scale with level, but are hard-capped to keep runtime reasonable.
                nested_target = 1500 if analysis_level == "L3" else 3500
                nested_circuits = generate_nested_circuit_library(shape_mode=shape_mode, target_size=nested_target)
                # keep only truly new items
                existing_set = set(library_circuits)
                nested_circuits = [c for c in nested_circuits if c not in existing_set]
                if nested_circuits:
                    library_circuits.extend(nested_circuits)
                    library_circuits = sorted(set(library_circuits))
                    print(f"   -> Nested circuits (Level3/4 add-on): {len(nested_circuits)}  |  total library: {len(library_circuits)}")
            except Exception as _ne:
                print(f"⚠️ Nested-circuit add-on failed: {_ne}")
        # -------------------------------------------------------

        # ---- Auto-Library (A): add previously successful circuits (persistent) ----
        auto_circuits = []
        if auto_library is not None and is_auto_learning_enabled("Researcher"):
            try:
                auto_circuits = auto_library.get_suggestions(max_n=60)
            except Exception:
                auto_circuits = []

        # Keep only circuits not already in library list
        auto_circuits = [c for c in auto_circuits if c not in set(library_circuits)]
        if auto_circuits:
            print(f"   -> Auto-library circuits (learned): {len(auto_circuits)}")
        # -------------------------------------------------------------------------

        features = extract_eis_features(f, Z)
        print(
            f"Feature summary: peaks={features['num_peaks']}, "
            f"tail_angle={features['tail_angle_deg']:.1f}°, "
            f"inductive={features['has_inductive']}"
        )
        rule_based_circuits = suggest_circuits_by_rules(features, shape_mode)

        FORCE_ADVANCED_CIRCUITS = True
        if FORCE_ADVANCED_CIRCUITS and shape_mode == "arc":
            advanced_circuits = [
                "R0-p(R1,CPE1)-p(R2,CPE2)",
                "R0-p(R1,C1)-p(R2,C2)",
                "R0-R1-p(R2,CPE2)",
                "R0-R1-p(R2,C2)",
                "R0-p(R1,CPE1)-p(R2,CPE2)-W1",
                "R0-p(R1,C1)-p(R2,C2)-W1",
                "R0-R1-p(R2,CPE2)-W1",
                "R0-R1-p(R2,C2)-W1",
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
                "R0-p(R1,C1)-p(R2,C2)-p(R3,C3)",
                "R0-R1-p(R2,CPE2)-p(R3,CPE3)",
                "R0-R1-p(R2,C2)-p(R3,C3)",
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-W1",
                "R0-p(R1,C1)-p(R2,C2)-p(R3,C3)-W1",
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-p(R4,CPE4)",
                "R0-p(R1,C1)-p(R2,C2)-p(R3,C3)-p(R4,C4)",
                "R0-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-p(R4,CPE4)-p(R5,CPE5)",
                "R0-L1-p(R1,CPE1)-p(R2,CPE2)",
                "R0-L1-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)",
                "R0-L1-p(R1,CPE1)-p(R2,CPE2)-p(R3,CPE3)-W1",
            ]
            for ac in advanced_circuits:
                if ac not in rule_based_circuits:
                    rule_based_circuits.append(ac)

        famous_circuits = [
            "R0-p(R1,CPE1)",
            "R0-p(R1,C1)",
            "R0-p(R1,CPE1)-W1",
            "R0-p(R1,C1)-W1",
        ]
        for fc in famous_circuits:
            if fc not in rule_based_circuits:
                rule_based_circuits.append(fc)

    print(f"Total library circuits (L1–4): {len(library_circuits)}")
    print(f"Total rule-based circuits (L1–4): {len(rule_based_circuits)}")
    if analysis_level == "L5":
        print(f"Total Level5 circuits (L5): {len(autoeis_impedance_circuits)}")

    simple_circuits = [
        "R0-p(R1,CPE1)",
        "R0-p(R1,C1)",
        "R0-R1-C1",
        "R0-R1-CPE1",
        "R0-p(R1,CPE1)-W1",
    ]
    fallback_arc = "R0-p(R1,CPE1)"

    print("\n" + "*" * 60)
    print(">>> Evaluating candidate circuits (with GUI progress) ...")

    all_results = []

    if analysis_level in ("L1", "L2", "L3", "L4") and library_circuits:
        all_results.extend(
            evaluate_circuit_list_with_progress(library_circuits, f, Z, source_label="library", expected_loops=expected_loops)
        )

        if auto_circuits:
            all_results.extend(
                evaluate_circuit_list_with_progress(auto_circuits, f, Z, source_label="auto", expected_loops=expected_loops)
            )

    if analysis_level in ("L1", "L2", "L3", "L4") and rule_based_circuits:
        finite_lib = [r for r in all_results if r["source"] == "library" and np.isfinite(r["error"])]
        best_lib_err = min([r["error"] for r in finite_lib], default=np.inf)
        if (not np.isfinite(best_lib_err)) or (best_lib_err > ERROR_THRESHOLD):
            all_results.extend(
                evaluate_circuit_list_with_progress(rule_based_circuits, f, Z, source_label="rule-based", expected_loops=expected_loops)
            )

    if analysis_level == "L5" and autoeis_impedance_circuits:
        all_results.extend(
            evaluate_circuit_list_with_progress(list(autoeis_impedance_circuits), f, Z, source_label="Level5")
        )

    finite_so_far = [r for r in all_results if np.isfinite(r["error"])]
    best_err_so_far = min([r["error"] for r in finite_so_far], default=np.inf)
    if (not np.isfinite(best_err_so_far)) or (best_err_so_far > ERROR_THRESHOLD):
        all_results.extend(
            evaluate_circuit_list_with_progress(simple_circuits, f, Z, source_label="simple")
        )

    if all(r["circuit"] != fallback_arc for r in all_results):
        all_results.extend(
            evaluate_circuit_list_with_progress([fallback_arc], f, Z, source_label="fallback")
        )

    # -------------------- AUTO-GENERATION (B): create new circuits if needed --------------------
    # Trigger only if the best available model is still poor.
    try:
        finite_tmp = [r for r in all_results if np.isfinite(r.get("error", np.inf))]
        finite_tmp.sort(key=lambda r: r.get("error", np.inf))
        if finite_tmp:
            best_r = finite_tmp[0]
            best_rel_max = float(best_r.get("rel_max", np.inf))
            best_mean_rel = float(best_r.get("mean_rel_err", np.inf))

            # Thresholds are intentionally conservative to avoid exploding candidates
            if (best_rel_max > 0.20) or (best_mean_rel > 0.20):
                print("\n⚠️ Best model is still poor. Auto-generating additional circuits (B) ...")
                seed_circuits = [best_r["circuit"]]
                # also consider 2nd best if available
                if len(finite_tmp) > 1:
                    seed_circuits.append(finite_tmp[1]["circuit"])

                generated = []
                for sc in seed_circuits:
                    try:
                        generated.extend(propose_more_complex_circuits(sc))
                    except Exception:
                        pass

                # unique + remove existing
                existing = {r["circuit"] for r in all_results}
                generated = [c for c in dict.fromkeys(generated) if c not in existing]

                # hard cap
                MAX_GENERATED = 80
                generated = generated[:MAX_GENERATED]

                if generated:
                    extra_results = evaluate_circuit_list_with_progress(generated, f, Z, source_label="generated")
                    all_results.extend(extra_results)
                    print(f"✅ Added {len(generated)} generated circuits to candidates.")
                else:
                    print("ℹ️ No new generated circuits (all were duplicates or none produced).")
    except Exception:
        pass
    # -------------------------------------------------------------------------------------------


    if len(all_results) == 0:
        print("\n❌ No circuits were evaluated.")
        return

    finite_results = [r for r in all_results if np.isfinite(r["error"])]
    finite_results.sort(key=lambda r: r["error"])

    # ---------- OPTIMIZATION: Reduced ranking candidates for speed ----------
    N_REFINE_FOR_RANKING = 20  # Changed from 50 to 12
    # ----------------------------------------------------------------------

    print(f"\n>>> Refining top {min(N_REFINE_FOR_RANKING, len(finite_results))} circuits for better ranking ...")

    for r in finite_results[:N_REFINE_FOR_RANKING]:
        c_str = r["circuit"]
        print(f"   -> refining circuit for ranking: {c_str}")
        # Faster refinement: fewer restarts (3), fewer iters (1000)
        err_ref, model_ref, info_ref = refine_circuit_fit(
            c_str, f, Z, n_restarts=3, target_rel_max=1e-4, max_iter=1000
        )
        if model_ref is not None and np.isfinite(err_ref):
            _update_result_from_fit(r, c_str, err_ref, model_ref, info_ref)

    all_results = rank_circuit_results(all_results)
    print("✔ Ranking updated using AICc/BIC and refined fitting.\n")
    try:
        ranking_csv = save_candidate_ranking(all_results, output_folder)
        print(f"✅ Candidate ranking saved: {ranking_csv}")
    except Exception as ranking_error:
        print(f"⚠️ Candidate ranking export failed: {ranking_error}")

    if not all_results:
        print("\n❌ No circuits were evaluated.")
        return

    fit_counter = 1
    while True:
        chosen_circuit = gui_select_circuit_from_results(all_results)
        if chosen_circuit is None:
            break

        print("\n" + "=" * 60)
        print(f" Detailed multi-start fitting for selected circuit: {chosen_circuit}")
        print("=" * 60)

        # Full power fit for the selected circuit
        # target_rel_max set to 0.01 (1%) for stop condition
        chosen_error, chosen_fit, chosen_info = refine_circuit_fit(
            chosen_circuit, f, Z, n_restarts=30, max_iter=100000, #target_rel_max=1e-1, 
        )

        if chosen_info is not None:
            chosen_aic, chosen_bic, chosen_rel_max = chosen_info[:3]
        else:
            chosen_aic, chosen_bic, chosen_rel_max = np.inf, np.inf, np.inf

        if (chosen_fit is None) or (not np.isfinite(chosen_error)):
            print("❌ Detailed fitting failed for the selected circuit.")
            messagebox.showerror("Fit failed", f"Detailed fitting failed for:\n{chosen_circuit}")
            break

        print(f"\n✅ Final selected circuit: {chosen_circuit}")
        print(f"   Relative error (RMSE): {chosen_error:.4e}")
        print(f"   Max Residual (rel_max): {chosen_rel_max:.4f}")
        print(f"   AICc: {chosen_aic:.2f},  BIC: {chosen_bic:.2f}")

        # ---- Auto-Library (A): store successful circuits for future runs ----
        if auto_library is not None and is_auto_learning_enabled("Researcher"):
            try:
                # chosen_info may contain extra metrics after index 3
                chosen_chi_sq = chosen_info[3] if len(chosen_info) > 3 else float("inf")
                chosen_rmse_abs = chosen_info[4] if len(chosen_info) > 4 else float("inf")
                chosen_mean_rel = chosen_info[5] if len(chosen_info) > 5 else float("inf")

                auto_library.maybe_store(
                    circuit_str=chosen_circuit,
                    metrics={
                        "error": float(chosen_error),
                        "rel_max": float(chosen_rel_max),
                        "aic": float(chosen_aic),
                        "aicc": float(chosen_aic),
                        "bic": float(chosen_bic),
                        "chi_sq": float(chosen_chi_sq),
                        "rmse_abs": float(chosen_rmse_abs),
                        "mean_rel_err": float(chosen_mean_rel),
                    },
                )
            except Exception:
                pass
        # -------------------------------------------------------------------


        try:
            print("\nFitted parameters:")
            if hasattr(chosen_fit, "parameter_names_"):
                names = chosen_fit.parameter_names_
            else:
                names = chosen_fit.parameters_
            for name, val in zip(names, chosen_fit.parameters_):
                print(f"  {str(name):12s} = {val: .3e}")
        except Exception:
            pass

        Z_fit_plot = chosen_fit.predict(f)

        rel_max_resid, n_peaks_resid = analyze_residuals_shape(f, Z, Z_fit_plot)
        print(f"\nResidual analysis: rel_max = {rel_max_resid:.3f}, peaks = {n_peaks_resid}")

        if (rel_max_resid > 0.15) and (n_peaks_resid >= 1):
            msg = (
                "Residuals still contain a strong structured feature:\n"
                f"  max(|resid|)/max(|Z|) ≈ {rel_max_resid:.2f}, peaks ≈ {n_peaks_resid}\n\n"
                "This suggests that the selected circuit is missing at least one time constant.\n\n"
                "The program can automatically generate and evaluate more complex circuits\n"
                "derived from the current model and add them to the candidate list.\n\n"
                "Do you want to add these suggested circuits?"
            )
            if messagebox.askyesno("Residual-based suggestion", msg):
                new_cands = propose_more_complex_circuits(chosen_circuit)
                existing_circs = {r["circuit"] for r in all_results}
                new_cands = [c for c in new_cands if c not in existing_circs]

                if new_cands:
                    extra_results = evaluate_circuit_list_with_progress(
                        new_cands, f, Z, source_label="resid-suggested"
                    )
                    all_results.extend(extra_results)
                    messagebox.showinfo(
                        "New circuits added",
                        f"{len(new_cands)} new circuits were evaluated and added.\n"
                        "You can select them in the next circuit-selection window."
                    )
                else:
                    messagebox.showinfo(
                        "No new circuits",
                        "All residual-based suggested circuits have already been evaluated."
                    )

        pdf_path, fit_folder = save_fit_outputs(
            f,
            Z,
            Z_fit_plot,
            Phase_raw,
            chosen_fit,
            chosen_circuit,
            chosen_error,
            chosen_aic,
            chosen_bic,
            analysis_level,
            shape_mode,
            autoeis_used,
            file_path,
            output_folder,
            fit_counter,
            expected_loops=expected_loops,
            phase_info=phase_info
        )
        fit_counter += 1

        # ==============================
        # Auto-learning: store learned circuit (Auto-Library)
        # ==============================
        try:
            if is_auto_learning_enabled("Researcher"):
                resid = Z - Z_fit_plot
                absZ = np.abs(Z)
                rel_err_array = np.abs(resid) / (absZ + 1e-12)
                mean_rel_err = float(np.mean(rel_err_array))
                rmse_val = float(np.sqrt(np.mean(np.abs(resid) ** 2)))
                rel_max_val = float(np.max(rel_err_array))
                chi_sq_val = float(np.sum(rel_err_array ** 2))
    
                n_loops = chosen_circuit.count("p(")
    
                maybe_store(
                    circuit_str=chosen_circuit,
                    metrics={
                        "error": float(chosen_error) if chosen_error is not None else float("inf"),
                        "rmse": rmse_val,
                        "chi_sq": chi_sq_val,
                        "mean_rel_err": mean_rel_err,
                        "rel_max": rel_max_val,
                    },
                    tags={
                        "loops": int(n_loops),
                        "analysis_level": analysis_level,
                        "shape_mode": shape_mode,
                        "source": "user-selected",
                    },
                    app_name="Researcher",
                    filename="auto_library.json",
                )
        except Exception as e:
            # never break the main workflow because of auto-learning
            print(f"[Auto-learning] store skipped: {e}")
        messagebox.showinfo(
            "Fit completed",
            f"Fitting for circuit:\n{chosen_circuit}\n\n"
            f"Report folder:\n{fit_folder}\n\nPDF report:\n{pdf_path}",
        )

        again = messagebox.askyesno(
            "Another circuit?",
            "Do you want to select and fit another circuit using the same candidates?",
        )
        if not again:
            break

    print("\n       ::    ::   ::::::.")
    print("       :::  :::   ::    ::")
    print("       :: :: ::   ::::::.")
    print("       ::    ::   ::    ::")
    print("       ::    ::   ::::::'")
    print("=============================")
    print("  JOB FINISHED")
    print("=============================")


# =========================================================
# Corva-style single-window GUI (embedded plots + table + progress)
# =========================================================
# Notes:
# - This GUI is an ADD-ON layer. It calls the same core functions used by legacy_main()
#   (evaluate_circuit, refine_circuit_fit, save_fit_outputs, etc.).
# - legacy_main() is preserved for compatibility.

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class _ScrollableFrame(ttk.Frame):
    """A vertically scrollable frame for long right-side panels."""
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._vsb = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)

        self._inner = ttk.Frame(self._canvas)
        self._window_id = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")

        self._canvas.grid(row=0, column=0, sticky="nsew")
        self._vsb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

    @property
    def inner(self):
        return self._inner

    def _on_inner_configure(self, _evt=None):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, evt):
        # keep inner frame width in sync
        self._canvas.itemconfig(self._window_id, width=evt.width)
    def _bind_mousewheel(self, event=None):
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event=None):
        self._canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        # Windows mousewheel: event.delta is multiple of 120
        try:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass


def _safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _make_param_table_df(circuit_str, fitted_params):
    names_raw = _extract_param_names(circuit_str)
    rows = []
    for n, v in zip(names_raw, fitted_params):
        unit = ""
        if n.startswith("R"):
            unit = "Ohm"
        elif "CPE" in n:
            if "_0" in n:
                unit = "S·s^n"
            elif "_1" in n:
                unit = "0<n<=1"
        elif n.startswith("C"):
            unit = "F"
        elif "W" in n:
            unit = "Ohm·s^-0.5"
        elif n.startswith("L"):
            unit = "H"
        rows.append({"Element": n, "Value": float(v), "Unit": unit})
    return pd.DataFrame(rows)


def _evaluate_circuit_list_with_callback(
    circuit_list,
    f,
    Z,
    expected_loops=None,
    update_cb=None,
    *,
    max_iter=600,
    physics_features=None,
    allowed_elements=None,
    source_map=None,
):
    """Fit a compact candidate set and attach v2 physical diagnostics."""
    if not circuit_list:
        return []

    try:
        circuit_list = prioritize_circuits_by_bode_phase(circuit_list, expected_loops)
    except Exception:
        circuit_list = list(circuit_list)

    results = []
    total = len(circuit_list)
    for i, c in enumerate(circuit_list, 1):
        if update_cb:
            update_cb(stage="scan", done=i-1, total=total, circuit=c, msg="Scanning library...")
        err, fit_model, info = evaluate_circuit(c, f, Z, max_iter=max_iter)

        if info is not None and len(info) >= 6:
            aic, bic, rel_max, chi_sq, rmse_abs, mean_rel_err = info
        else:
            aic, bic, rel_max = np.inf, np.inf, np.inf
            chi_sq, rmse_abs, mean_rel_err = np.inf, np.inf, np.inf

        n_loops = c.count("p(")
        loop_mismatch = abs(int(n_loops) - int(expected_loops)) if expected_loops is not None else 0
        status = "OK" if (fit_model is not None and np.isfinite(err)) else "FAILED"
        bound_hits = _count_parameter_bound_hits(c, fit_model) if status == "OK" else 999999

        result = {
            "circuit": c,
            "loops": int(n_loops),
            "loop_mismatch": int(loop_mismatch),
            "error": float(err),
            "aic": float(aic),
            "aicc": float(aic),
            "bic": float(bic),
            "rel_max": float(rel_max),
            "chi_sq": float(chi_sq),
            "rmse_abs": float(rmse_abs),
            "mean_rel_err": float(mean_rel_err),
            "bound_hits": int(bound_hits),
            "source": (source_map or {}).get(c, "physics-guided"),
            "status": status,
            "model": fit_model,   # keep for quick preview if needed
            "info": info,
        }
        if physics_features is not None:
            annotate_result_physics(
                result,
                physics_features,
                f,
                Z,
                allowed_elements=allowed_elements,
            )
        results.append(result)

        if update_cb:
            update_cb(stage="scan", done=i, total=total, circuit=c, msg=f"Scanned {i}/{total}")

    return results


def _refine_circuit_fit_with_callback(circuit_str, f, Z, n_restarts=100,
                                      target_rel_max=0.01, max_iter=50000,
                                      update_cb=None,
                                      random_seed=DEFAULT_RANDOM_SEED,
                                      initial_guess=None):
    """A callback-enabled wrapper that keeps original logic intact."""
    expected_parameter_count = len(_extract_param_names(circuit_str))
    try:
        supplied_guess = np.asarray(initial_guess, dtype=float)
    except Exception:
        supplied_guess = np.asarray([], dtype=float)
    if (
        supplied_guess.size == expected_parameter_count
        and np.all(np.isfinite(supplied_guess))
        and np.all(supplied_guess > 0)
    ):
        base_guess = supplied_guess.tolist()
    else:
        base_guess = build_data_based_initial_guess(circuit_str, f, Z)
    best_err = np.inf
    best_model = None
    best_info = None
    best_rel_max = np.inf
    rng = _rng_for_circuit(circuit_str, random_seed=random_seed)

    for k in range(n_restarts):
        if k == 0:
            guess = base_guess
        else:
            guess = []
            for val in base_guess:
                if val <= 0:
                    val = 1e-6
                factor = 10 ** rng.uniform(-1.0, 1.0)
                guess.append(val * factor)

        if update_cb:
            update_cb(stage="refine", done=k, total=n_restarts, circuit=circuit_str,
                      msg=f"Refine attempt {k+1}/{n_restarts} | best rel_max={best_rel_max:.4f}")

        err, model, info = _fit_circuit_with_guess(circuit_str, f, Z, guess, max_iter=max_iter)

        if np.isfinite(err):
            current_rel_max = info[2] if (info is not None and len(info) >= 3) else np.inf

            if err < best_err:
                best_err, best_model, best_info = err, model, info
                best_rel_max = current_rel_max

                if update_cb:
                    update_cb(stage="refine", done=k+1, total=n_restarts, circuit=circuit_str,
                              msg=f"New best | error={best_err:.3e} | rel_max={best_rel_max:.4f}")

                if best_rel_max <= target_rel_max:
                    break

    if best_model is None:
        best_err, best_model, best_info = evaluate_circuit(circuit_str, f, Z, max_iter=max_iter)

    if update_cb:
        update_cb(stage="refine", done=n_restarts, total=n_restarts, circuit=circuit_str,
                  msg=f"Refine done | error={float(best_err):.3e}")

    return best_err, best_model, best_info


class CorvaEISApp(tk.Tk):
    def __init__(self):
        super().__init__()
        try:
            myappid = 'istinye.researcher.eis.1.0' 
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            self.iconbitmap("Researcherlogo.ico")
        except Exception:
            pass
        self.title(f"{SOFTWARE_NAME}")
        self.geometry("1200x750")
        self.minsize(1100, 680)
        # Icon addition
        try:
            self.iconbitmap("Researcherlogo.ico")
        except Exception:
            pass

        # Data state
        self.file_path = None
        self.df = None
        self.original_df = None
        self._source_row_ids = []
        self.edit_history = []
        self.data_edit_log = []
        self.data_dirty = False
        self.analysis_running = False
        self._active_cell_editor = None
        self._pick_data_mode = False
        self._vector_row_positions = np.array([], dtype=int)
        self._selected_plot_artist = None
        self.f = None
        self.Z = None
        self.Phase_raw = None
        self.output_folder = None
        self.shape_mode = None
        self.expected_loops = None
        self.phase_info = {}
        self.physics_features = {}
        self.lin_kk_result = {}
        self.selection_confidence = {}
        self.candidate_count = 0
        self.search_level = "L3"
        self.search_allowed_elements = {"R", "C", "CPE", "W", "L"}
        self.autoeis_used = False
        self.all_results = []
        self.chosen_circuit = None
        self.chosen_fit = None
        self.chosen_error = None
        self.chosen_aic = None
        self.chosen_bic = None

        # UI vars
        self.analysis_level = tk.StringVar(value="L3")
        self.target_rel_max = tk.DoubleVar(value=0.01)
        self.n_restarts = tk.IntVar(value=20)

        self.freq_col = tk.StringVar(value="")
        self.zreal_col = tk.StringVar(value="")
        self.zimag_col = tk.StringVar(value="")
        self.phase_col = tk.StringVar(value="<None>")

        # Top
        self._build_topbar()

        # Main layout: left nav | center content | right panel
        self.paned = ttk.Panedwindow(self, orient="horizontal")
        self.paned.pack(fill="both", expand=True)

        self.left_nav = ttk.Frame(self.paned, width=180)
        self.center = ttk.Frame(self.paned)
        self.right = _ScrollableFrame(self.paned, width=320)

        self.paned.add(self.left_nav, weight=0)
        self.paned.add(self.center, weight=1)
        self.paned.add(self.right, weight=0)

        self._build_left_nav()
        self._build_center()
        self._build_right_panel()
        self._build_bottom_status()

    def _build_topbar(self):
        bar = ttk.Frame(self)
        bar.pack(fill="x")
     
        ttk.Button(bar, text="Open file", command=self.open_file).pack(side="left", padx=6, pady=6)
        ttk.Button(bar, text="Run", command=self.run_full_pipeline).pack(side="left", padx=6, pady=6)
        ttk.Button(bar, text="Export", command=self.export_current).pack(side="left", padx=6, pady=6)

        self.file_label = ttk.Label(bar, text="No file loaded")
        self.file_label.pack(side="left", padx=10)

        ttk.Button(bar, text="Legacy workflow", command=self._run_legacy).pack(side="right", padx=6, pady=6)

    def _build_left_nav(self):
        ttk.Label(self.left_nav, text="Sections", font=("Segoe UI", 10, "bold")).pack(padx=10, pady=(10, 6), anchor="w")
        for name in ["Data Editor", "Nyquist", "Bode", "Parameters", "Circuits"]:
            ttk.Button(self.left_nav, text=name, command=lambda n=name: self._show_tab(n)).pack(fill="x", padx=10, pady=4)

    def _build_center(self):
        self.notebook = ttk.Notebook(self.center)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # Editable working-data table.  The source file is never overwritten;
        # edits are applied only to self.df and can be undone or restored.
        self.tab_data = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_data, text="Data Editor")

        data_tools = ttk.Frame(self.tab_data)
        data_tools.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Button(data_tools, text="Delete selected rows",
                   command=self.delete_selected_data_rows).pack(side="left", padx=(0, 6))
        ttk.Button(data_tools, text="Undo",
                   command=self.undo_data_edit).pack(side="left", padx=6)
        ttk.Button(data_tools, text="Restore original",
                   command=self.restore_original_data).pack(side="left", padx=6)
        self.pick_data_btn = ttk.Button(data_tools, text="Pick point on Nyquist",
                                        command=self.toggle_pick_data_point)
        self.pick_data_btn.pack(side="left", padx=6)
        ttk.Button(data_tools, text="Save cleaned data",
                   command=self.save_cleaned_data).pack(side="right", padx=(6, 0))

        ttk.Label(
            self.tab_data,
            text=("Double-click a cell to edit it. Select one or more rows and press Delete to remove them. "
                  "Rows with invalid mapped values are shown in red. All changes affect only the working copy."),
            wraplength=900,
        ).pack(fill="x", padx=10, pady=(0, 5))

        data_table_frame = ttk.Frame(self.tab_data)
        data_table_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.data_tree = ttk.Treeview(data_table_frame, columns=(), show="headings",
                                      selectmode="extended")
        data_vsb = ttk.Scrollbar(data_table_frame, orient="vertical", command=self.data_tree.yview)
        data_hsb = ttk.Scrollbar(data_table_frame, orient="horizontal", command=self.data_tree.xview)
        self.data_tree.configure(yscrollcommand=data_vsb.set, xscrollcommand=data_hsb.set)
        self.data_tree.grid(row=0, column=0, sticky="nsew")
        data_vsb.grid(row=0, column=1, sticky="ns")
        data_hsb.grid(row=1, column=0, sticky="ew")
        data_table_frame.rowconfigure(0, weight=1)
        data_table_frame.columnconfigure(0, weight=1)
        self.data_tree.bind("<Double-1>", self._begin_cell_edit)
        self.data_tree.bind("<Delete>", lambda _event: self.delete_selected_data_rows())
        self.data_tree.bind("<<TreeviewSelect>>", self._on_data_table_selection)
        self.data_tree.tag_configure("invalid", background="#ffe3e3")

        # Nyquist
        self.tab_nyquist = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_nyquist, text="Nyquist")
        self.fig_nyq, self.ax_nyq = plt.subplots(figsize=(6, 5))
        self.canvas_nyq = FigureCanvasTkAgg(self.fig_nyq, master=self.tab_nyquist)
        self.canvas_nyq.get_tk_widget().pack(fill="both", expand=True)
        self.canvas_nyq.get_tk_widget().bind(
            "<Delete>", lambda _event: self.delete_selected_data_rows()
        )
        NavigationToolbar2Tk(self.canvas_nyq, self.tab_nyquist).update()
        self._nyquist_click_cid = self.canvas_nyq.mpl_connect(
            "button_press_event", self._on_nyquist_data_click
        )

        # Bode
        self.tab_bode = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_bode, text="Bode")
        self.fig_bode, (self.ax_mag, self.ax_phase) = plt.subplots(nrows=2, figsize=(6, 6))
        self.canvas_bode = FigureCanvasTkAgg(self.fig_bode, master=self.tab_bode)
        self.canvas_bode.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas_bode, self.tab_bode).update()

        # Parameters
        self.tab_params = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_params, text="Parameters")
        self.param_tree = ttk.Treeview(self.tab_params, columns=("Element", "Value", "Unit"), show="headings", height=20)
        for c in ("Element", "Value", "Unit"):
            self.param_tree.heading(c, text=c)
        self.param_tree.column("Element", width=160, anchor="w")
        self.param_tree.column("Value", width=180, anchor="e")
        self.param_tree.column("Unit", width=120, anchor="center")
        vsb = ttk.Scrollbar(self.tab_params, orient="vertical", command=self.param_tree.yview)
        self.param_tree.configure(yscrollcommand=vsb.set)
        self.param_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        vsb.pack(side="left", fill="y", pady=8)

        # Circuits
        self.tab_circuits = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_circuits, text="Circuits")
        top = ttk.Frame(self.tab_circuits)
        top.pack(fill="x", padx=8, pady=(8, 4))

        ttk.Label(top, text="Filter:").pack(side="left")
        self.filter_var = tk.StringVar(value="")
        ent = ttk.Entry(top, textvariable=self.filter_var, width=45)
        ent.pack(side="left", padx=6)
        ent.bind("<KeyRelease>", lambda _e: self._refresh_circuit_table())

        # Extra filters / ranking
        self.only_cpe_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Only CPE", variable=self.only_cpe_var,
                        command=self._refresh_circuit_table).pack(side="left", padx=(10, 0))
        # These selections constrain the next search and also filter the
        # current table.  Family-aware matching keeps C separate from CPE.
        self.elem_vars = {}
        elem_box = ttk.Frame(top)
        elem_box.pack(side="left", padx=(10, 0))
        ttk.Label(elem_box, text="Search elements:").pack(side="left")
        for el in ["R", "C", "CPE", "W", "L"]:
            v = tk.BooleanVar(value=True)
            self.elem_vars[el] = v
            ttk.Checkbutton(elem_box, text=el, variable=v, command=self._refresh_circuit_table).pack(side="left", padx=2)
        btn_detect = ttk.Button(top, text="Detect loops from phase", command=self.detect_loops_from_phase)
        btn_detect.pack(side="left", padx=(10, 0))


        ttk.Label(top, text="Sort:").pack(side="left", padx=(10, 0))
        self.sort_mode_var = tk.StringVar(value="auto")
        self.sort_cb = ttk.Combobox(top, textvariable=self.sort_mode_var,
                                   values=["auto", "physics_aicc", "bic", "aicc", "error", "rel_max", "bound_hits", "loop_mismatch"],
                                   state="readonly", width=14)
        self.sort_cb.pack(side="left", padx=6)
        self.sort_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_circuit_table())

        ttk.Label(top, text="Show:").pack(side="left", padx=(10, 0))
        self.max_show_var = tk.IntVar(value=30)
        sp = ttk.Spinbox(top, from_=1, to=500, textvariable=self.max_show_var, width=6,
                         command=self._refresh_circuit_table)
        sp.pack(side="left", padx=6)

        ttk.Button(top, text="Fit selected", command=self.fit_selected_circuit).pack(side="right")

        cols = (
            "loops", "circuit", "physical", "identifiability", "weight",
            "bic", "aicc", "bound_hits", "error", "rel_max", "status",
        )
        self.circuit_tree = ttk.Treeview(self.tab_circuits, columns=cols, show="headings", height=18)
        col_titles = {
            "loops": "Loops",
            "circuit": "Circuit",
            "physical": "Physical",
            "identifiability": "Identifiability",
            "weight": "Weight",
            "bic": "BIC",
            "aicc": "AICc",
            "bound_hits": "Bound Hits",
            "error": "Fit Error",
            "rel_max": "Max Rel. Err",
            "status": "Status",
        }
        for c in cols:
            self.circuit_tree.heading(c, text=col_titles.get(c, c))
        self.circuit_tree.column("loops", width=55, anchor="center")
        self.circuit_tree.column("circuit", width=380, anchor="w")
        self.circuit_tree.column("physical", width=90, anchor="center")
        self.circuit_tree.column("identifiability", width=95, anchor="center")
        self.circuit_tree.column("weight", width=70, anchor="e")
        self.circuit_tree.column("bic", width=85, anchor="e")
        self.circuit_tree.column("aicc", width=85, anchor="e")
        self.circuit_tree.column("bound_hits", width=75, anchor="center")
        self.circuit_tree.column("error", width=90, anchor="e")
        self.circuit_tree.column("rel_max", width=90, anchor="e")
        self.circuit_tree.column("status", width=70, anchor="center")

        vsb2 = ttk.Scrollbar(self.tab_circuits, orient="vertical", command=self.circuit_tree.yview)
        hsb2 = ttk.Scrollbar(self.tab_circuits, orient="horizontal", command=self.circuit_tree.xview)
        self.circuit_tree.configure(yscrollcommand=vsb2.set, xscrollcommand=hsb2.set)

        # Row coloring by fit quality
        self.circuit_tree.tag_configure("good", background="#d8f3dc")
        self.circuit_tree.tag_configure("ok", background="#fff3bf")
        self.circuit_tree.tag_configure("bad", background="#ffe3e3")
        # Double-click = quick fit + update plots
        self.circuit_tree.bind("<Double-1>", lambda _e: self.fit_selected_circuit())
        self.circuit_tree.pack(side="left", fill="both", expand=True, padx=(8, 0), pady=8)
        vsb2.pack(side="left", fill="y", pady=8)
        hsb2.pack(side="bottom", fill="x", padx=8, pady=(0, 8))

        self.result_box = tk.Text(self.tab_circuits, height=6, wrap="word")
        self.result_box.pack(fill="x", padx=8, pady=(0, 8))

    def _build_right_panel(self):
        p = self.right.inner
        ttk.Label(p, text="Controls", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(10, 6))

        # Columns section (no popups)
        frm_cols = ttk.LabelFrame(p, text="Columns")
        frm_cols.pack(fill="x", padx=10, pady=6)
        self.cb_freq = ttk.Combobox(frm_cols, textvariable=self.freq_col, values=[], state="readonly")
        self.cb_zr = ttk.Combobox(frm_cols, textvariable=self.zreal_col, values=[], state="readonly")
        self.cb_zi = ttk.Combobox(frm_cols, textvariable=self.zimag_col, values=[], state="readonly")
        self.cb_ph = ttk.Combobox(frm_cols, textvariable=self.phase_col, values=[], state="readonly")

        ttk.Label(frm_cols, text="Frequency (Hz)").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.cb_freq.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(frm_cols, text="Z' (Real)").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.cb_zr.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(frm_cols, text="Z'' (Imag)").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.cb_zi.grid(row=2, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(frm_cols, text="Phase (deg)").grid(row=3, column=0, sticky="w", padx=6, pady=4)
        self.cb_ph.grid(row=3, column=1, sticky="ew", padx=6, pady=4)

        for cb in (self.cb_freq, self.cb_zr, self.cb_zi, self.cb_ph):
            cb.bind("<<ComboboxSelected>>", self._on_column_mapping_changed)

        frm_cols.columnconfigure(1, weight=1)

        # Level section
        frm_lvl = ttk.LabelFrame(p, text="Analysis Level")
        frm_lvl.pack(fill="x", padx=10, pady=6)
        for key, (label, _) in LEVEL_INFO.items():
            if key == "L5" and ae is None:
                continue
            ttk.Radiobutton(frm_lvl, text=label, variable=self.analysis_level, value=key).pack(anchor="w", padx=8, pady=2)

        # Fit section
        frm_fit = ttk.LabelFrame(p, text="Fit settings")
        frm_fit.pack(fill="x", padx=10, pady=6)

        ttk.Label(frm_fit, text="Target rel_max (stop)").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frm_fit, textvariable=self.target_rel_max, width=10).grid(row=0, column=1, sticky="w", padx=6, pady=4)

        ttk.Label(frm_fit, text="Restarts").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(frm_fit, textvariable=self.n_restarts, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=4)

        frm_fit.columnconfigure(1, weight=1)

        # What to show instead of 'Log'
        frm_info = ttk.LabelFrame(p, text="Results (replaces Log)")
        frm_info.pack(fill="x", padx=10, pady=6)
        self.summary_lbl = ttk.Label(frm_info, text="—", justify="left")
        self.summary_lbl.pack(fill="x", padx=8, pady=8)

    def _build_bottom_status(self):
        bottom = ttk.Frame(self)
        bottom.pack(fill="x")
        self.progress = ttk.Progressbar(bottom, mode="determinate", maximum=100)
        self.progress.pack(side="left", fill="x", expand=True, padx=8, pady=6)
        self.status = ttk.Label(bottom, text="Ready")
        self.status.pack(side="left", padx=8)

    def _show_tab(self, name):
        mapping = {"Data Editor": 0, "Nyquist": 1, "Bode": 2, "Parameters": 3, "Circuits": 4}
        idx = mapping.get(name, 0)
        self.notebook.select(idx)

    # ----------------------------
    # Working-data editor
    # ----------------------------
    def _editing_allowed(self):
        if self.df is None:
            messagebox.showwarning("No data", "Please open a data file first.")
            return False
        if self.analysis_running:
            messagebox.showwarning(
                "Analysis running",
                "Please wait until the current scan or fit is finished before editing the data.",
            )
            return False
        return True

    @staticmethod
    def _display_data_value(value):
        """Format a table value without silently changing its precision."""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.12g}"
        return str(value)

    def _cancel_active_cell_editor(self):
        editor = self._active_cell_editor
        self._active_cell_editor = None
        if editor is not None:
            try:
                editor.destroy()
            except Exception:
                pass

    def _refresh_data_table(self, selected_positions=None):
        self._cancel_active_cell_editor()
        for iid in self.data_tree.get_children():
            self.data_tree.delete(iid)

        if self.df is None:
            self.data_tree.configure(columns=())
            return

        data_column_ids = [f"c{i}" for i in range(len(self.df.columns))]
        tree_columns = ["source_row"] + data_column_ids
        self.data_tree.configure(columns=tree_columns, show="headings")
        self.data_tree.heading("source_row", text="Source row")
        self.data_tree.column("source_row", width=85, minwidth=70, anchor="center", stretch=False)

        for i, col_id in enumerate(data_column_ids):
            label = str(self.df.columns[i])
            width = min(240, max(105, 9 * len(label) + 30))
            self.data_tree.heading(col_id, text=label)
            self.data_tree.column(col_id, width=width, minwidth=80, anchor="center", stretch=True)

        invalid_positions = self._invalid_working_row_positions()
        for row_pos in range(len(self.df)):
            source_row = self._source_row_ids[row_pos] if row_pos < len(self._source_row_ids) else row_pos + 1
            values = [source_row] + [
                self._display_data_value(v) for v in self.df.iloc[row_pos, :].tolist()
            ]
            tags = ("invalid",) if row_pos in invalid_positions else ()
            self.data_tree.insert("", "end", iid=str(row_pos), values=values, tags=tags)

        if selected_positions:
            valid_iids = [str(p) for p in selected_positions if 0 <= int(p) < len(self.df)]
            if valid_iids:
                self.data_tree.selection_set(valid_iids)
                self.data_tree.see(valid_iids[0])

    def _push_data_history(self):
        if self.df is None:
            return
        self.edit_history.append((
            self.df.copy(deep=True),
            list(self._source_row_ids),
            bool(self.data_dirty),
        ))
        # A bounded history avoids excessive memory use for unusually large tables.
        if len(self.edit_history) > 30:
            self.edit_history.pop(0)

    def _update_data_dirty_flag(self):
        if self.df is None or self.original_df is None:
            self.data_dirty = False
            return
        original_rows = list(range(1, len(self.original_df) + 1))
        self.data_dirty = not (
            self.df.equals(self.original_df) and self._source_row_ids == original_rows
        )
        if self.file_path:
            suffix = "  [working copy edited]" if self.data_dirty else ""
            self.file_label.config(text=f"{os.path.basename(self.file_path)}{suffix}")

    def _record_data_edit(self, action, rows="", column="", old_value="", new_value=""):
        self.data_edit_log.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "source_rows": rows,
            "column": column,
            "old_value": old_value,
            "new_value": new_value,
        })

    def _mapped_numeric_column_positions(self):
        if self.df is None:
            return set()
        cols = list(self.df.columns)
        mapped = set()
        for name in (self.freq_col.get(), self.zreal_col.get(), self.zimag_col.get()):
            if name in cols:
                mapped.add(cols.index(name))
        if self.phase_col.get() != "<None>" and self.phase_col.get() in cols:
            mapped.add(cols.index(self.phase_col.get()))
        return mapped

    def _invalid_working_row_positions(self):
        """Return working-row positions unusable with the current column mapping."""
        if self.df is None:
            return set()
        cols = list(self.df.columns)
        required_names = [self.freq_col.get(), self.zreal_col.get(), self.zimag_col.get()]
        if self.phase_col.get() != "<None>":
            required_names.append(self.phase_col.get())
        if any(name not in cols for name in required_names):
            return set(range(len(self.df)))

        arrays = [
            pd.to_numeric(self.df.iloc[:, cols.index(name)], errors="coerce").to_numpy(dtype=float)
            for name in required_names
        ]
        valid = np.ones(len(self.df), dtype=bool)
        for arr in arrays:
            valid &= np.isfinite(arr)
        frequency = arrays[0]
        valid &= frequency > 0
        return set(np.flatnonzero(~valid).astype(int).tolist())

    def _on_column_mapping_changed(self, _event=None):
        selected = [int(iid) for iid in self.data_tree.selection()]
        self._refresh_data_table(selected_positions=selected)
        self._invalidate_analysis_after_data_change()

    def _begin_cell_edit(self, event):
        if not self._editing_allowed():
            return
        if self.data_tree.identify_region(event.x, event.y) != "cell":
            return
        row_iid = self.data_tree.identify_row(event.y)
        column_id = self.data_tree.identify_column(event.x)
        if not row_iid or not column_id:
            return

        # #1 is the immutable original/source row number.
        tree_col_number = int(column_id[1:])
        if tree_col_number <= 1:
            self.status.config(text="Source row numbers are read-only.")
            return
        row_pos = int(row_iid)
        col_pos = tree_col_number - 2
        if not (0 <= row_pos < len(self.df) and 0 <= col_pos < len(self.df.columns)):
            return

        bbox = self.data_tree.bbox(row_iid, column_id)
        if not bbox:
            return
        self._cancel_active_cell_editor()
        x, y, width, height = bbox
        old_value = self.df.iat[row_pos, col_pos]
        editor = ttk.Entry(self.data_tree)
        editor.insert(0, self._display_data_value(old_value))
        editor.select_range(0, "end")
        editor.place(x=x, y=y, width=width, height=height)
        editor.focus_set()
        self._active_cell_editor = editor
        finished = {"value": False}

        def cancel(_event=None):
            if finished["value"]:
                return
            finished["value"] = True
            self._cancel_active_cell_editor()

        def commit(_event=None):
            if finished["value"] or self._active_cell_editor is not editor:
                return
            raw_value = editor.get().strip()
            series = self.df.iloc[:, col_pos]
            must_be_numeric = (
                col_pos in self._mapped_numeric_column_positions()
                or pd.api.types.is_numeric_dtype(series.dtype)
            )
            if must_be_numeric:
                try:
                    new_value = float(raw_value)
                except Exception:
                    messagebox.showerror("Invalid value", "This column requires a numeric value.")
                    self.after_idle(editor.focus_set)
                    return
                if not np.isfinite(new_value):
                    messagebox.showerror(
                        "Invalid value",
                        "NaN and infinite values are not accepted. Edit the value or delete the row.",
                    )
                    self.after_idle(editor.focus_set)
                    return
            else:
                new_value = raw_value

            same_value = False
            try:
                same_value = bool(old_value == new_value)
            except Exception:
                same_value = False
            if same_value:
                cancel()
                return

            self._push_data_history()
            if (must_be_numeric and pd.api.types.is_integer_dtype(series.dtype)
                    and not float(new_value).is_integer()):
                self.df.iloc[:, col_pos] = pd.to_numeric(series, errors="coerce").astype(float)
            self.df.iat[row_pos, col_pos] = new_value
            source_row = self._source_row_ids[row_pos]
            self._record_data_edit(
                "edit_cell",
                rows=str(source_row),
                column=str(self.df.columns[col_pos]),
                old_value=self._display_data_value(old_value),
                new_value=self._display_data_value(new_value),
            )
            finished["value"] = True
            self._active_cell_editor = None
            try:
                editor.destroy()
            except Exception:
                pass
            self._after_working_data_change(selected_positions=[row_pos])
            self.status.config(
                text=f"Source row {source_row} updated in the working copy. Undo is available."
            )

        editor.bind("<Return>", commit)
        editor.bind("<KP_Enter>", commit)
        editor.bind("<Escape>", cancel)
        editor.bind("<FocusOut>", commit)

    def delete_selected_data_rows(self):
        if not self._editing_allowed():
            return
        selected = sorted({int(iid) for iid in self.data_tree.selection()})
        if not selected:
            messagebox.showinfo(
                "No rows selected",
                "Select row(s) in Data Editor, or pick a point on the Nyquist plot first.",
            )
            return
        if len(self.df) - len(selected) < 3:
            messagebox.showwarning("Too few data", "At least three data rows must remain.")
            return

        source_rows = [self._source_row_ids[p] for p in selected]
        selected_set = set(selected)
        self._push_data_history()
        keep_positions = [p for p in range(len(self.df)) if p not in selected_set]
        self.df = self.df.iloc[keep_positions, :].reset_index(drop=True)
        self._source_row_ids = [self._source_row_ids[p] for p in keep_positions]
        self._record_data_edit(
            "delete_rows",
            rows=", ".join(map(str, source_rows)),
        )
        self._after_working_data_change()
        self.status.config(
            text=f"Deleted {len(selected)} row(s) from the working copy. Undo is available."
        )

    def undo_data_edit(self):
        if not self._editing_allowed():
            return
        if not self.edit_history:
            messagebox.showinfo("Undo", "There is no data edit to undo.")
            return
        self.df, self._source_row_ids, previous_dirty = self.edit_history.pop()
        self.data_dirty = previous_dirty
        self._record_data_edit("undo")
        self._after_working_data_change()
        self.status.config(text="Last data edit was undone.")

    def restore_original_data(self):
        if not self._editing_allowed() or self.original_df is None:
            return
        if not self.data_dirty:
            messagebox.showinfo("Restore original", "The working data already match the source file.")
            return
        if not messagebox.askyesno(
            "Restore original",
            "Restore every row and value from the source file? The current state can still be recovered with Undo.",
        ):
            return
        self._push_data_history()
        self.df = self.original_df.copy(deep=True)
        self._source_row_ids = list(range(1, len(self.df) + 1))
        self._record_data_edit("restore_original")
        self._after_working_data_change()
        self.status.config(text="Original source data restored. Undo is available.")

    def _save_edit_log(self, folder, stem):
        if not self.data_edit_log:
            return None
        log_path = os.path.join(folder, f"{stem}_edit_log.csv")
        pd.DataFrame(self.data_edit_log).to_csv(log_path, index=False)
        return log_path

    def save_cleaned_data(self):
        if self.df is None:
            messagebox.showwarning("No data", "Please open a data file first.")
            return
        source_stem = os.path.splitext(os.path.basename(self.file_path or "eis_data"))[0]
        initial_dir = os.path.dirname(self.file_path) if self.file_path else os.getcwd()
        save_path = filedialog.asksaveasfilename(
            title="Save cleaned working data",
            initialdir=initial_dir,
            initialfile=f"{source_stem}_cleaned.csv",
            defaultextension=".csv",
            filetypes=[("CSV file", "*.csv"), ("Excel workbook", "*.xlsx")],
        )
        if not save_path:
            return
        try:
            ext = os.path.splitext(save_path)[1].lower()
            if ext == ".xlsx":
                self.df.to_excel(save_path, index=False)
            else:
                if ext != ".csv":
                    save_path += ".csv"
                self.df.to_csv(save_path, index=False)
            folder = os.path.dirname(save_path) or os.getcwd()
            stem = os.path.splitext(os.path.basename(save_path))[0]
            log_path = self._save_edit_log(folder, stem)
            extra = f"\nEdit log:\n{log_path}" if log_path else ""
            messagebox.showinfo(
                "Cleaned data saved",
                f"The source file was not changed.\n\nCleaned copy:\n{save_path}{extra}",
            )
            self.status.config(text="Cleaned working data saved; source file remains unchanged.")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def toggle_pick_data_point(self):
        if self.df is None:
            messagebox.showwarning("No data", "Please open a data file first.")
            return
        self._pick_data_mode = not self._pick_data_mode
        self.pick_data_btn.config(
            text="Stop picking" if self._pick_data_mode else "Pick point on Nyquist"
        )
        if self._pick_data_mode:
            self.notebook.select(self.tab_nyquist)
            self.status.config(
                text="Pick mode: click a Nyquist point, then press Delete to remove it (Ctrl-click selects more)."
            )
        else:
            self.status.config(text="Point-pick mode stopped.")

    def _on_nyquist_data_click(self, event):
        if not self._pick_data_mode or self.df is None or self.Z is None:
            return
        if event.inaxes is not self.ax_nyq or event.x is None or event.y is None or len(self.Z) == 0:
            return
        points = np.column_stack((self.Z.real, -self.Z.imag))
        display_points = self.ax_nyq.transData.transform(points)
        distances = np.hypot(display_points[:, 0] - event.x, display_points[:, 1] - event.y)
        vector_pos = int(np.argmin(distances))
        if not np.isfinite(distances[vector_pos]) or distances[vector_pos] > 25:
            self.status.config(text="No point selected; click closer to a Nyquist marker.")
            return
        row_pos = int(self._vector_row_positions[vector_pos])
        iid = str(row_pos)
        ctrl_pressed = bool(event.key and "control" in str(event.key).lower())
        if ctrl_pressed:
            self.data_tree.selection_add(iid)
        else:
            self.data_tree.selection_set(iid)
        self.data_tree.see(iid)
        self._highlight_selected_data_rows()
        try:
            self.canvas_nyq.get_tk_widget().focus_set()
        except Exception:
            pass
        source_row = self._source_row_ids[row_pos]
        self.status.config(
            text=f"Source row {source_row} selected. Press Delete, or open Data Editor to inspect it."
        )

    def _on_data_table_selection(self, _event=None):
        self._highlight_selected_data_rows()

    def _remove_selected_plot_artist(self):
        if self._selected_plot_artist is not None:
            try:
                self._selected_plot_artist.remove()
            except Exception:
                pass
            self._selected_plot_artist = None

    def _highlight_selected_data_rows(self):
        self._remove_selected_plot_artist()
        if self.Z is None or not len(self._vector_row_positions):
            return
        selected_rows = {int(iid) for iid in self.data_tree.selection()}
        vector_indices = [
            i for i, row_pos in enumerate(self._vector_row_positions)
            if int(row_pos) in selected_rows
        ]
        if vector_indices:
            self._selected_plot_artist = self.ax_nyq.scatter(
                self.Z.real[vector_indices],
                -self.Z.imag[vector_indices],
                s=90,
                marker="x",
                linewidths=2.2,
                color="crimson",
                label="Selected for review",
                zorder=10,
            )
            self.canvas_nyq.draw_idle()

    def _after_working_data_change(self, selected_positions=None):
        self._update_data_dirty_flag()
        self._refresh_data_table(selected_positions=selected_positions)
        self._invalidate_analysis_after_data_change()

    def _invalidate_analysis_after_data_change(self):
        self.all_results = []
        self.chosen_circuit = None
        self.chosen_fit = None
        self.chosen_error = None
        self.chosen_aic = None
        self.chosen_bic = None
        for tv in (self.param_tree, self.circuit_tree):
            for iid in tv.get_children():
                tv.delete(iid)
        self.result_box.delete("1.0", "end")
        self.summary_lbl.config(text="—")
        self.progress["value"] = 0
        self._draw_raw_data_preview()

    def _draw_raw_data_preview(self):
        self._remove_selected_plot_artist()
        try:
            self._prepare_vectors()
        except Exception as e:
            self.f = self.Z = self.Phase_raw = None
            self._vector_row_positions = np.array([], dtype=int)
            self.ax_nyq.clear()
            self.ax_mag.clear()
            self.ax_phase.clear()
            self.ax_nyq.set_title(f"Data preview unavailable: {e}")
            self.canvas_nyq.draw_idle()
            self.canvas_bode.draw_idle()
            return

        self.ax_nyq.clear()
        self.ax_nyq.grid(True, which="major", linestyle="--", alpha=0.5)
        self.ax_nyq.plot(self.Z.real, -self.Z.imag, "o-", ms=4, lw=1, label="Working data")
        self.ax_nyq.set_xlabel("Z' (Ohm)")
        self.ax_nyq.set_ylabel("-Z'' (Ohm)")
        self.ax_nyq.set_title("Nyquist — working-data preview")
        self.ax_nyq.legend()
        self.fig_nyq.tight_layout()
        self.canvas_nyq.draw_idle()

        phase = self.Phase_raw if self.Phase_raw is not None else np.angle(self.Z, deg=True)
        self.ax_mag.clear()
        self.ax_phase.clear()
        self.ax_mag.loglog(self.f, np.abs(self.Z), "o-", ms=3, lw=1)
        self.ax_mag.set_xlabel("Frequency (Hz)")
        self.ax_mag.set_ylabel("|Z| (Ohm)")
        self.ax_mag.set_title("Bode magnitude — working data")
        self.ax_mag.grid(True, which="major", linestyle="--", alpha=0.5)
        self.ax_phase.semilogx(self.f, phase, "o-", ms=3, lw=1)
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.set_ylabel("Phase (deg)")
        self.ax_phase.set_title("Bode phase — working data")
        self.ax_phase.grid(True, which="major", linestyle="--", alpha=0.5)
        self.fig_bode.tight_layout()
        self.canvas_bode.draw_idle()
        self._highlight_selected_data_rows()

    # ----------------------------
    # Data & workflow
    # ----------------------------
    def open_file(self):
        if self.analysis_running:
            messagebox.showwarning(
                "Analysis running",
                "Please wait until the current scan or fit is finished before opening another file.",
            )
            return
        fp = filedialog.askopenfilename(
            title="Select EIS data file (CSV/TXT/DAT/Excel)",
            filetypes=[
                ("Table files", "*.csv;*.txt;*.dat;*.tsv;*.xlsx;*.xls"),
                ("All files", "*.*"),
            ],
        )
        if not fp:
            return
        self.file_path = fp
        self.file_label.config(text=os.path.basename(fp))

        # Read table exactly like legacy_main()
        ext = os.path.splitext(fp)[1].lower()
        try:
            if ext in [".csv", ".txt", ".dat", ".tsv"]:
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    df = pd.read_csv(fp, sep=None, engine="python")
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(fp)
            else:
                try:
                    df = pd.read_csv(fp)
                except Exception:
                    df = pd.read_csv(fp, sep=None, engine="python")
        except Exception as e:
            messagebox.showerror("Read error", f"Failed to read file:\n{e}")
            return

        self.df = df.reset_index(drop=True).copy(deep=True)
        self.original_df = self.df.copy(deep=True)
        self._source_row_ids = list(range(1, len(self.df) + 1))
        self.edit_history = []
        self.data_edit_log = []
        self.data_dirty = False
        cols = list(df.columns)
        if not cols:
            messagebox.showerror("Error", "No columns found in file.")
            return

        # Default mapping (same logic as the old popup)
        self.freq_col.set(cols[0])
        self.zreal_col.set(cols[1] if len(cols) > 1 else cols[0])
        self.zimag_col.set(cols[2] if len(cols) > 2 else cols[0])
        self.phase_col.set("<None>")

        self.cb_freq.configure(values=cols)
        self.cb_zr.configure(values=cols)
        self.cb_zi.configure(values=cols)
        self.cb_ph.configure(values=["<None>"] + cols)

        # Output folder
        self.output_folder = build_output_run_folder(fp, analysis_name="Nyquist and Bode Analysis")

        self._clear_views()
        self._refresh_data_table()
        self._draw_raw_data_preview()
        self.notebook.select(self.tab_data)
        excluded = len(self.df) - len(self.f) if self.f is not None else len(self.df)
        if excluded:
            self.status.config(
                text=f"File loaded. {excluded} invalid/nonpositive row(s) are excluded from analysis; review Data Editor."
            )
        else:
            self.status.config(text="File loaded. Double-click cells to edit or select rows to delete.")

    def _clear_views(self):
        self._cancel_active_cell_editor()
        self._remove_selected_plot_artist()
        # clear plots
        self.ax_nyq.clear()
        self.ax_mag.clear()
        self.ax_phase.clear()
        self.canvas_nyq.draw()
        self.canvas_bode.draw()
        # clear trees
        for tv in (self.param_tree, self.circuit_tree):
            for iid in tv.get_children():
                tv.delete(iid)
        self.result_box.delete("1.0", "end")
        self.summary_lbl.config(text="—")
        self.progress["value"] = 0

    def _prepare_vectors(self):
        if self.df is None:
            raise RuntimeError("No data loaded.")
        cols = list(self.df.columns)
        def idx_of(name):
            return cols.index(name) if name in cols else None

        idx_freq = idx_of(self.freq_col.get())
        idx_real = idx_of(self.zreal_col.get())
        idx_imag = idx_of(self.zimag_col.get())
        idx_phase = None if self.phase_col.get() == "<None>" else idx_of(self.phase_col.get())

        if idx_freq is None or idx_real is None or idx_imag is None:
            raise RuntimeError("Please select valid columns.")

        f = pd.to_numeric(self.df.iloc[:, idx_freq], errors="coerce").to_numpy(dtype=float)
        Z_real = pd.to_numeric(self.df.iloc[:, idx_real], errors="coerce").to_numpy(dtype=float)
        Z_imag = pd.to_numeric(self.df.iloc[:, idx_imag], errors="coerce").to_numpy(dtype=float)
        Phase_raw = None
        if idx_phase is not None:
            Phase_raw = pd.to_numeric(
                self.df.iloc[:, idx_phase], errors="coerce"
            ).to_numpy(dtype=float)

        Z = Z_real + 1j * Z_imag
        mask = (
            np.isfinite(f)
            & np.isfinite(Z_real)
            & np.isfinite(Z_imag)
            & (f > 0)
        )
        if Phase_raw is not None:
            mask &= np.isfinite(Phase_raw)
        self._vector_row_positions = np.flatnonzero(mask).astype(int)
        f = f[mask]
        Z = Z[mask]
        if Phase_raw is not None:
            Phase_raw = Phase_raw[mask]

        if len(f) < 3:
            raise RuntimeError("At least three finite rows with positive frequency are required.")

        self.f, self.Z, self.Phase_raw = f, Z, Phase_raw

        # Bode phase loops estimate (same as legacy_main)
        try:
            phase_for_features = Phase_raw if Phase_raw is not None else np.angle(Z, deg=True)
            expected_loops, phase_info = estimate_time_constants_from_phase(f, phase_for_features)
            self.expected_loops = expected_loops
            self.phase_info = phase_info
        except Exception:
            self.expected_loops = None
            self.phase_info = {}

        # Shape mode
        delta_real = float(max(Z.real) - min(Z.real))
        delta_imag = float(max(-Z.imag) - min(-Z.imag))
        if delta_real == 0:
            delta_real = 1e-9
        aspect_ratio = delta_imag / delta_real
        self.shape_mode = "linear" if aspect_ratio > 3.0 else "arc"

    def _ui_update(self, *, stage, done, total, circuit, msg):
        pct = int(100 * done / max(total, 1))
        self.progress["value"] = pct
        self.status.config(text=f"{pct}%")
        self.update_idletasks()

    def _mark_analysis_idle(self):
        self.analysis_running = False

    def detect_loops_from_phase(self):
        """Re-detect expected number of loops from Bode phase peaks (keep peaks >= 30% of strongest peak)."""
        if self.f is None or self.Z is None:
            messagebox.showwarning("No data", "Please load data first.")
            return

        try:
            phase_for_features = self.Phase_raw if self.Phase_raw is not None else np.angle(self.Z, deg=True)
            expected_loops, phase_info = estimate_time_constants_from_phase(self.f, phase_for_features, second_peak_ratio=0.30)
            self.expected_loops = expected_loops
            self.phase_info = phase_info

            # Recompute loop_mismatch in existing results (if any) using phase-derived expected_loops
            if self.all_results:
                for r in self.all_results:
                    c = (r.get("circuit", "") or "")
                    n_loops = c.count("p(")
                    r["loop_mismatch"] = abs(int(n_loops) - int(self.expected_loops)) if self.expected_loops is not None else 0

                # Re-rank statistically; the new loop estimate remains an
                # advisory tie-break rather than a hard model-selection rule.
                self.all_results = rank_circuit_results(self.all_results)

            self._refresh_circuit_table()
            self.status.config(text="0%")
            # Jump to Circuits tab
            try:
                self.notebook.select(self.tab_circuits)
            except Exception:
                pass

        except Exception as e:
            messagebox.showerror("Detect loops error", str(e))

    def run_full_pipeline(self):
        if self.analysis_running:
            messagebox.showinfo("Please wait", "An analysis is already running.")
            return
        if self.file_path is None:
            messagebox.showwarning("No file", "Please open a file first.")
            return

        try:
            self._prepare_vectors()
        except Exception as e:
            messagebox.showerror("Input error", str(e))
            return

        self.status.config(text="Running...")
        self.progress["value"] = 0
        self.all_results = []
        self.chosen_circuit = None
        self.chosen_fit = None
        self.physics_features = {}
        self.lin_kk_result = {}
        self.selection_confidence = {}
        self.candidate_count = 0
        # Read Tk state on the UI thread; worker threads must not access Tk
        # variables directly.
        self.search_level = str(self.analysis_level.get() or "L1")
        self.search_allowed_elements = {
            name for name, variable in self.elem_vars.items() if variable.get()
        }
        self.search_allowed_elements.add("R")
        self.analysis_running = True

        # Run in a thread to keep UI responsive
        t = threading.Thread(target=self._run_pipeline_worker, daemon=True)
        t.start()

    def _run_pipeline_worker(self):
        try:
            started = time.perf_counter()
            level = self.search_level
            _, candidate_cap = LEVEL_INFO.get(level, LEVEL_INFO["L1"])
            allowed = set(self.search_allowed_elements)

            # 1) Extract conservative morphology before constructing models.
            features = analyze_eis_physics(self.f, self.Z, self.Phase_raw)
            self.physics_features = features
            self.expected_loops = int(features.get("apparent_loops", 1))
            self.phase_info = {
                "peak_freqs_hz": features.get("phase_peak_frequencies_hz", []),
                "method": "physics_guided_phase_and_nyquist",
                "advisory_only": True,
            }
            self.lin_kk_result = run_lin_kk_diagnostic(self.f, self.Z)

            # 2) A compact deterministic library replaces the old random
            # enumeration of thousands of arbitrary/nested topologies.
            learned = []
            if auto_library is not None and is_auto_learning_enabled("Researcher"):
                try:
                    learned = auto_library.get_suggestions(max_n=max(4, candidate_cap // 2))
                except Exception:
                    learned = []

            canonical = build_physics_guided_candidates(
                features,
                level=level,
                allowed_elements=allowed,
            )
            circuits = build_physics_guided_candidates(
                features,
                level=level,
                allowed_elements=allowed,
                learned_circuits=learned,
            )
            source_map = {c: "physics-guided" for c in canonical}
            for circuit in circuits:
                source_map.setdefault(circuit, "learned-physics-filtered")

            # 3) Level 5 remains an optional AutoEIS extension, not the core
            # engine.  Its proposals must pass the same physical gate.
            self.autoeis_used = False
            if level == "L5":
                if ae is None:
                    raise RuntimeError("Level5 not installed (Level 5 unavailable).")
                try:
                    imp_data = ae.utils.ImpedanceData(self.f, self.Z)
                    imp_data = imp_data.preprocess()
                    try:
                        external_candidates = ae.models.get_candidate_circuits(imp_data)
                    except Exception:
                        external_candidates = []
                    for external in external_candidates or []:
                        circuit = str(external)
                        diagnostic = physics_compatibility(circuit, features, allowed)
                        if diagnostic.get("hard_reject", True) or circuit in circuits:
                            continue
                        circuits.append(circuit)
                        source_map[circuit] = "AutoEIS-physics-filtered"
                        self.autoeis_used = True
                        if len(circuits) >= candidate_cap:
                            break
                except Exception:
                    self.autoeis_used = False

            circuits = list(dict.fromkeys(circuits))[:candidate_cap]
            if not circuits:
                raise RuntimeError(
                    "No circuit is compatible with the detected spectrum and enabled element families."
                )
            self.candidate_count = len(circuits)

            results = _evaluate_circuit_list_with_callback(
                circuits,
                self.f,
                self.Z,
                expected_loops=self.expected_loops,
                update_cb=lambda **kw: self.after(0, lambda: self._ui_update(**kw)),
                max_iter=600,
                physics_features=features,
                allowed_elements=allowed,
                source_map=source_map,
            )

            # 4) Refine only a small, diverse shortlist.  This is the largest
            # speed gain after eliminating unphysical enumeration.
            results_ok = [
                r for r in results
                if r.get("status") == "OK"
                and not r.get("hard_reject", False)
                and np.isfinite(r.get("error", np.inf))
            ]
            shortlist = []
            seen_shortlist = set()
            pools = [
                rank_circuit_results(results_ok)[:6],
                sorted(results_ok, key=lambda r: float(r.get("error", np.inf)))[:3],
            ]
            for pool in pools:
                for r in pool:
                    c = r.get("circuit")
                    if c and c not in seen_shortlist and len(shortlist) < 8:
                        seen_shortlist.add(c)
                        shortlist.append(r)

            for j, r in enumerate(shortlist, 1):
                c = r["circuit"]
                self.after(0, lambda j=j, n=len(shortlist), c=c: self.status.config(
                    text=f"Refining {j}/{n}: {c}"
                ))
                err_ref, model_ref, info_ref = _refine_circuit_fit_with_callback(
                    c,
                    self.f,
                    self.Z,
                    n_restarts=3 if level in ("L3", "L4", "L5") else 2,
                    target_rel_max=0.01,
                    max_iter=2500,
                    update_cb=None,
                    random_seed=DEFAULT_RANDOM_SEED,
                )
                _update_result_from_fit(r, c, err_ref, model_ref, info_ref)
                annotate_result_physics(
                    r,
                    features,
                    self.f,
                    self.Z,
                    allowed_elements=allowed,
                )

            # 5) Report statistical ambiguity instead of claiming that the
            # first numerically close circuit is a unique physical answer.
            self.selection_confidence = annotate_model_confidence(results)
            results_sorted = rank_circuit_results(results)
            self.all_results = results_sorted
            try:
                save_candidate_ranking(results_sorted, self.output_folder)
                save_physics_selection_audit(
                    self.output_folder,
                    features=features,
                    lin_kk=self.lin_kk_result,
                    confidence=self.selection_confidence,
                    results=results_sorted,
                    level=level,
                    allowed_elements=allowed,
                )
            except Exception as ranking_error:
                print(f"⚠️ Candidate ranking export failed: {ranking_error}")
            self.after(0, self._refresh_circuit_table)

            # Auto-select top circuit
            best_result = select_best_circuit_result(results_sorted)
            if best_result is not None:
                self.chosen_circuit = best_result["circuit"]
                confidence = self.selection_confidence.get("selection_status", "unavailable")
                message = (
                    "Physics-guided candidate ranking completed.\n"
                    f"Top candidate: {self.chosen_circuit}\n"
                    f"Selection status: {confidence} (advisory, not a unique mechanistic proof).\n"
                    f"Candidates fitted: {self.candidate_count}.\n\n"
                    "Review the diagnostics, then select a circuit and click 'Fit selected'.\n"
                )
                self.after(0, lambda m=message: self.result_box.insert("end", m))
                self.after(0, lambda: self._set_summary_text())
            else:
                self.after(0, lambda: self.result_box.insert("end", "No valid circuits found.\n"))

            elapsed = time.perf_counter() - started
            self.after(0, lambda t=elapsed: self.status.config(
                text=f"Physics-guided scan done in {t:.1f} s. Choose circuit and fit."
            ))
            self.after(0, lambda: self.notebook.select(self.tab_circuits))
            self.after(0, lambda: self.progress.config(value=0))
            self.after(0, self._mark_analysis_idle)

        except Exception as e:
            self.after(0, lambda msg=str(e): messagebox.showerror("Run error", msg))
            self.after(0, lambda: self.status.config(text="Error"))
            self.after(0, self._mark_analysis_idle)

    def _set_summary_text(self):
        if not self.all_results:
            self.summary_lbl.config(text="—")
            return
        r0 = select_best_circuit_result(self.all_results)
        if r0 is None:
            self.summary_lbl.config(text="No physically admissible fitted model.")
            return
        confidence = self.selection_confidence or {}
        lin_kk = self.lin_kk_result or {}
        diffusion = (self.physics_features or {}).get("diffusion_kind", "unknown")
        txt = (
            f"Physics engine: {PHYSICS_ENGINE_VERSION}\n"
            f"Resolved relaxations: {self.expected_loops} | diffusion: {diffusion}\n"
            f"Candidates fitted: {self.candidate_count}\n"
            f"Top circuit: {r0.get('circuit')}\n"
            f"Physical: {r0.get('physical_status')} | identifiability: {r0.get('identifiability_status')}\n"
            f"Selection: {confidence.get('selection_status', 'unavailable')} "
            f"(weight={float(r0.get('akaike_weight', 0.0)):.2f})\n"
            f"Lin-KK: {lin_kk.get('status', 'unavailable')} | bound hits: {r0.get('bound_hits', 0)}\n"
            f"Weighted RMS: {r0.get('error'):.3e} | rel_max: {r0.get('rel_max'):.4f}\n"
            "Automatic topology is advisory; inspect diagnostics before physical interpretation.\n"
        )
        self.summary_lbl.config(text=txt)

    def _refresh_circuit_table(self):
        # Clear
        for iid in self.circuit_tree.get_children():
            self.circuit_tree.delete(iid)

        flt = (self.filter_var.get() or "").strip().lower()
        only_cpe = bool(getattr(self, "only_cpe_var", tk.BooleanVar(value=False)).get())
        sort_mode = (getattr(self, "sort_mode_var", tk.StringVar(value="auto")).get() or "auto").strip()
        max_show = int(getattr(self, "max_show_var", tk.IntVar(value=30)).get() or 30)

        # Prepare list (filter)
        items = []
        for r in (self.all_results or []):
            c = r.get("circuit", "") or ""

            # Family-aware matching avoids treating the C in CPE as an ideal
            # capacitor and the W in Wo/Ws as a separate token.
            enabled = {
                el for el, variable in getattr(self, "elem_vars", {}).items()
                if variable.get()
            }
            enabled.add("R")
            skip = not circuit_element_families(c).issubset(enabled)
            if skip:
                continue

            if flt and flt not in c.lower():
                continue
            if only_cpe and ("cpe" not in c.lower()):
                continue

            items.append(r)

        # Sort
        if sort_mode == "auto":
            items.sort(key=circuit_selection_key)
        elif sort_mode == "physics_aicc":
            items.sort(key=lambda r: float(r.get("physics_adjusted_aicc", np.inf)))
        elif sort_mode == "aicc":
            items.sort(key=lambda r: float(r.get("aicc", r.get("aic", np.inf))))
        elif sort_mode == "bic":
            items.sort(key=lambda r: float(r.get("bic", np.inf)))
        elif sort_mode == "rel_max":
            items.sort(key=lambda r: float(r.get("rel_max", np.inf)))
        elif sort_mode == "bound_hits":
            items.sort(key=lambda r: int(r.get("bound_hits", 999999)))
        elif sort_mode == "loop_mismatch":
            items.sort(key=lambda r: int(r.get("loop_mismatch", 999999)))
        else:
            items.sort(key=lambda r: float(r.get("error", np.inf)))

        # Determine thresholds for coloring (based on displayed slice)
        slice_items = items[:max_show]
        finite_errors = [float(r.get("error", np.inf)) for r in slice_items
                         if np.isfinite(float(r.get("error", np.inf)))]
        finite_errors.sort()
        if finite_errors:
            q1 = finite_errors[int(0.2 * (len(finite_errors)-1))]
            q2 = finite_errors[int(0.7 * (len(finite_errors)-1))]
        else:
            q1 = q2 = np.inf

        # Insert rows
        for r in slice_items:
            c = r.get("circuit", "") or ""
            err = float(r.get("error", np.inf))
            rel_max = float(r.get("rel_max", np.inf))
            bic = float(r.get("bic", np.inf))
            aicc = float(r.get("aicc", r.get("aic", np.inf)))
            vals = (
                r.get("loops", ""),
                c,
                r.get("physical_status", ""),
                r.get("identifiability_status", ""),
                f"{float(r.get('akaike_weight', 0.0)):.3f}",
                "inf" if not np.isfinite(bic) else f"{bic:.2f}",
                "inf" if not np.isfinite(aicc) else f"{aicc:.2f}",
                r.get("bound_hits", ""),
                "inf" if not np.isfinite(err) else f"{err:.3e}",
                "inf" if not np.isfinite(rel_max) else f"{rel_max:.4f}",
                r.get("status", ""),
            )

            if (
                r.get("status") != "OK"
                or r.get("hard_reject", False)
                or r.get("physical_status") in ("weak", "inconsistent")
                or r.get("identifiability_status") in ("poor", "failed")
            ):
                tag = "bad"
            elif (
                r.get("physical_status") == "consistent"
                and r.get("identifiability_status") == "good"
                and int(r.get("bound_hits", 999999)) == 0
            ):
                tag = "good"
            else:
                tag = "ok"

            self.circuit_tree.insert("", "end", values=vals, tags=(tag,))

    def _get_selected_circuit(self):
        sel = self.circuit_tree.selection()
        if not sel:
            return self.chosen_circuit
        vals = self.circuit_tree.item(sel[0], "values")
        if vals and len(vals) >= 2:
            return vals[1]
        return self.chosen_circuit

    def fit_selected_circuit(self):
        if self.analysis_running:
            messagebox.showinfo("Please wait", "An analysis is already running.")
            return
        if self.f is None or self.Z is None:
            messagebox.showwarning("Not ready", "Run the scan first (Run button).")
            return
        c = self._get_selected_circuit()
        if not c:
            messagebox.showwarning("No selection", "Please select a circuit.")
            return
        self.chosen_circuit = c
        self.result_box.insert("end", f"\nFitting selected circuit:\n{c}\n")
        self.status.config(text="Refining fit...")
        self.progress["value"] = 0
        try:
            restarts = max(1, min(100, int(self.n_restarts.get())))
            target_rel_max = max(0.0, float(self.target_rel_max.get()))
        except Exception:
            messagebox.showwarning("Fit settings", "Restarts or target rel_max is invalid.")
            return

        seed_guess = None
        for result in self.all_results:
            if result.get("circuit") == c and result.get("model") is not None:
                seed_guess = np.asarray(result["model"].parameters_, dtype=float).copy()
                break
        self.analysis_running = True

        t = threading.Thread(
            target=self._fit_worker,
            args=(c, restarts, target_rel_max, seed_guess),
            daemon=True,
        )
        t.start()

    def _fit_worker(self, circuit_str, nrs, trm, seed_guess=None):
        try:
            def cb(**kw):
                self.after(0, lambda: self._ui_update(**kw))

            err, model, info = _refine_circuit_fit_with_callback(
                circuit_str,
                self.f,
                self.Z,
                n_restarts=nrs,
                target_rel_max=trm,
                max_iter=10000,
                update_cb=cb,
                initial_guess=seed_guess,
            )
            if model is None:
                raise RuntimeError("Fit failed.")

            self.chosen_error = float(err)
            self.chosen_fit = model
            if info is not None and len(info) >= 2:
                self.chosen_aic, self.chosen_bic = float(info[0]), float(info[1])
            else:
                self.chosen_aic, self.chosen_bic = np.inf, np.inf

            Z_fit = model.predict(self.f)

            # Refresh physical and identifiability diagnostics for the final
            # selected fit and keep the audit table synchronized.
            fitted_result = next(
                (r for r in self.all_results if r.get("circuit") == circuit_str),
                {"circuit": circuit_str, "loops": circuit_relaxation_count(circuit_str)},
            )
            _update_result_from_fit(fitted_result, circuit_str, err, model, info)
            annotate_result_physics(
                fitted_result,
                self.physics_features,
                self.f,
                self.Z,
                allowed_elements=self.search_allowed_elements,
            )
            if not any(existing is fitted_result for existing in self.all_results):
                self.all_results.append(fitted_result)
            self.selection_confidence = annotate_model_confidence(self.all_results)
            self.all_results = rank_circuit_results(self.all_results)

            # Update plots + param table in UI thread
            self.after(0, lambda zfit=Z_fit: self._update_plots(zfit))
            self.after(0, lambda: self._update_params_table(circuit_str, model.parameters_))
            self.after(0, self._refresh_circuit_table)

            # Summary text
            rel_max = float(info[2]) if (info is not None and len(info) >= 3) else np.inf
            chi_sq = float(info[3]) if (info is not None and len(info) >= 4) else np.inf
            summary = (
                f"Circuit: {circuit_str}\n"
                f"Physical: {fitted_result.get('physical_status')} | "
                f"identifiability: {fitted_result.get('identifiability_status')}\n"
                f"Bound hits: {fitted_result.get('bound_hits', 0)} | "
                f"residual structure: {float(fitted_result.get('residual_structure', np.inf)):.3f}\n"
                f"Weighted RMS: {float(err):.3e} | rel_max: {rel_max:.4f}\n"
                f"χ²: {chi_sq:.3e}\n"
                f"AICc: {float(self.chosen_aic):.2f} | BIC: {float(self.chosen_bic):.2f}\n"
                "Interpret the circuit together with the physical diagnostics.\n"
            )
            self.after(0, lambda text=summary: self.summary_lbl.config(text=text))

            self.after(0, lambda: self.status.config(text="Fit done. You can Export."))
            self.after(0, lambda: self.notebook.select(self.tab_circuits))
            self.after(0, lambda: self.progress.config(value=0))
            self.after(0, self._mark_analysis_idle)

        except Exception as e:
            self.after(0, lambda msg=str(e): messagebox.showerror("Fit error", msg))
            self.after(0, lambda: self.status.config(text="Error"))
            self.after(0, self._mark_analysis_idle)

    def _update_plots(self, Z_fit):
        # Nyquist
        self.ax_nyq.clear()
        self.ax_nyq.grid(True, which="major", linestyle="--", alpha=0.6)
        self.ax_nyq.plot(self.Z.real, -self.Z.imag, "o", ms=4, label="Exp")
        self.ax_nyq.plot(Z_fit.real, -Z_fit.imag, "-", lw=2, label="Fit")
        self.ax_nyq.set_xlabel("Z' (Ohm)")
        self.ax_nyq.set_ylabel("-Z'' (Ohm)")
        self.ax_nyq.set_title("Nyquist Plot")
        self.ax_nyq.legend()
        self.canvas_nyq.draw()

        # Bode
        f = self.f
        absZ = np.abs(self.Z)
        absZ_fit = np.abs(Z_fit)
        phase_exp = self.Phase_raw if self.Phase_raw is not None else np.angle(self.Z, deg=True)
        phase_fit = np.angle(Z_fit, deg=True)

        self.ax_mag.clear()
        self.ax_phase.clear()

        self.ax_mag.loglog(f, absZ, "o", ms=3, label="|Z| Exp")
        self.ax_mag.loglog(f, absZ_fit, "-", lw=2, label="|Z| Fit")
        self.ax_mag.set_xlabel("Frequency (Hz)")
        self.ax_mag.set_ylabel("|Z| (Ohm)")
        self.ax_mag.set_title("Bode Magnitude")
        self.ax_mag.minorticks_off()
        self.ax_mag.grid(True, which="major", linestyle="--", alpha=0.6)
        self.ax_mag.legend()

        self.ax_phase.semilogx(f, phase_exp, "o", ms=3, label="Phase Exp")
        self.ax_phase.semilogx(f, phase_fit, "-", lw=2, label="Phase Fit")
        self.ax_phase.set_xlabel("Frequency (Hz)")
        self.ax_phase.set_ylabel("Phase (deg)")
        self.ax_phase.set_title("Bode Phase")
        self.ax_phase.minorticks_off()
        self.ax_phase.grid(True, which="major", linestyle="--", alpha=0.6)
        self.ax_phase.legend()

        self.fig_bode.tight_layout()
        self.canvas_bode.draw()

    def _update_params_table(self, circuit_str, params):
        for iid in self.param_tree.get_children():
            self.param_tree.delete(iid)

        dfp = _make_param_table_df(circuit_str, params)
        for _, row in dfp.iterrows():
            self.param_tree.insert("", "end", values=(row["Element"], f"{row['Value']:.6g}", row["Unit"]))

    def export_current(self):
        if self.chosen_fit is None or self.chosen_circuit is None:
            messagebox.showwarning("Nothing to export", "Please fit a circuit first.")
            return
        try:
            Z_fit = self.chosen_fit.predict(self.f)
            pdf_path, fit_folder = save_fit_outputs(
                self.f,
                self.Z,
                Z_fit,
                self.Phase_raw,
                self.chosen_fit,
                self.chosen_circuit,
                self.chosen_error,
                self.chosen_aic,
                self.chosen_bic,
                self.analysis_level.get(),
                self.shape_mode,
                self.autoeis_used,
                self.file_path,
                self.output_folder,
                1,
                expected_loops=self.expected_loops,
                phase_info=self.phase_info,
            )
            audit_note = ""
            if self.data_dirty:
                cleaned_path = os.path.join(fit_folder, "working_data_used.csv")
                self.df.to_csv(cleaned_path, index=False)
                edit_log_path = self._save_edit_log(fit_folder, "working_data_used")
                audit_note = f"\n\nWorking data:\n{cleaned_path}"
                if edit_log_path:
                    audit_note += f"\n\nEdit log:\n{edit_log_path}"
            messagebox.showinfo(
                "Export complete",
                f"Exported to:\n{fit_folder}\n\nPDF:\n{pdf_path}{audit_note}",
            )
        except Exception as e:
            messagebox.showerror("Export error", str(e))

    def _run_legacy(self):
        proceed = messagebox.askyesno(
            "Legacy workflow",
            "The legacy workflow uses the older broad topology enumeration and can be much slower.\n\n"
            "Use the main Run button for the physics-guided engine. Continue with legacy mode?",
        )
        if not proceed:
            return
        try:
            legacy_main()
        except Exception as e:
            messagebox.showerror("Legacy error", str(e))


def main():
    """Default entry point (Corva UI)."""
    app = CorvaEISApp()
    app.mainloop()


if __name__ == "__main__":
    main()
