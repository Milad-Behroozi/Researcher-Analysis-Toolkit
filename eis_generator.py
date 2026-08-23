import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import os
import sys
import numpy as np
import pandas as pd
import re
import json
from datetime import datetime

from impedance.models.circuits import CustomCircuit


# Prevent Windows console encodings (for example cp1252) from terminating
# GUI workflows when a path or third-party message contains Unicode text.
for stream in (sys.stdout, sys.stderr):
    if stream is not None and hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, OSError, ValueError):
            pass


# -----------------------------
# default values for parameters
# -----------------------------
def default_for_param(name: str) -> float:
    if name.startswith("R"):
        return 10.0
    elif name.startswith("CPE") and name.endswith("_0"):
        return 1e-4   # Q
    elif name.startswith("CPE") and name.endswith("_1"):
        return 0.85   # n
    elif name.startswith("C"):
        return 1e-6   # F
    elif name.startswith(("W", "Wo", "Ws")):
        return 10.0
    else:
        return 1.0


# -----------------------------
# extract param names
# -----------------------------
def extract_param_names(circuit_str: str):
    tokens = re.findall(r'([A-Za-z]+[0-9]+)', circuit_str)
    param_names, seen = [], set()

    for tok in tokens:
        if tok.startswith("CPE"):
            for p in (f"{tok}_0", f"{tok}_1"):
                if p not in seen:
                    seen.add(p)
                    param_names.append(p)
        else:
            if tok not in seen:
                seen.add(tok)
                param_names.append(tok)

    return param_names


# -----------------------------
# making frequency vector
# -----------------------------
def make_frequency_vector(f_min, f_max, points):
    return np.logspace(np.log10(f_min), np.log10(f_max), points)


# -----------------------------
# Windows-safe filename component
# -----------------------------
def safe_filename_component(value: str, fallback: str) -> str:
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = value.strip().rstrip(". ")
    return value or fallback


# -----------------------------
# saving EIS data to CSV
# -----------------------------
def save_eis_csv(filename, freqs, Z):
    mag = np.abs(Z)
    phase_deg = np.degrees(np.arctan2(Z.imag, Z.real))

    df = pd.DataFrame({
        "Freq/Hz": freqs,
        "Z'/ohm": Z.real,
        'Z"/ohm': Z.imag,
        "Z/ohm": mag,
        "Phase/deg": phase_deg,
    })
    df.to_csv(filename, index=False)
    print(f"[OK] Saved CSV: {filename}")


# -----------------------------
# saving metadata to JSON
# -----------------------------
def save_metadata_json(filename, meta_dict):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(meta_dict, f, indent=2)
    print(f"[OK] Saved metadata JSON: {filename}")


# -----------------------------
# the main GUI class
# -----------------------------
class EISCSVGUI:
    def __init__(self, master):
        self.master = master
        master.title("Researcher Analysis Toolkit - EIS Synthetic Data Generator")
        # ----- Circuit -----
        frame_circuit = ttk.LabelFrame(master, text="Circuit")
        frame_circuit.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_circuit, text="Circuit string:").grid(row=0, column=0, padx=5, pady=3, sticky="w")
        self.circuit_var = tk.StringVar(value="R0-p(R1,CPE1)-C2-W1")
        ttk.Entry(frame_circuit, textvariable=self.circuit_var, width=45)\
            .grid(row=0, column=1, padx=5, pady=3, sticky="w")

        ttk.Button(frame_circuit, text="Detect Parameters",
                   command=self.detect_parameters).grid(row=0, column=2, padx=5, pady=3)

        # ----- Parameters panel -----
        frame_params = ttk.LabelFrame(master, text="Parameters")
        frame_params.pack(fill="both", expand=True, padx=10, pady=5)

        self.canvas = tk.Canvas(frame_params)
        self.scrollbar = ttk.Scrollbar(frame_params, orient="vertical", command=self.canvas.yview)
        self.inner_frame = ttk.Frame(self.canvas)

        self.inner_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.inner_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.param_entries = {}
        self.param_order = []

        # ----- Frequency & Noise -----
        frame_settings = ttk.LabelFrame(master, text="Frequency & Noise")
        frame_settings.pack(fill="x", padx=10, pady=5)

        ttk.Label(frame_settings, text="f_min (Hz):").grid(row=0, column=0, sticky="e")
        ttk.Label(frame_settings, text="f_max (Hz):").grid(row=0, column=2, sticky="e")
        ttk.Label(frame_settings, text="#points:").grid(row=0, column=4, sticky="e")

        self.fmin_var = tk.StringVar(value="0.01")
        self.fmax_var = tk.StringVar(value="1000000")
        self.points_var = tk.StringVar(value="50")

        ttk.Entry(frame_settings, textvariable=self.fmin_var, width=10).grid(row=0, column=1)
        ttk.Entry(frame_settings, textvariable=self.fmax_var, width=10).grid(row=0, column=3)
        ttk.Entry(frame_settings, textvariable=self.points_var, width=10).grid(row=0, column=5)

        ttk.Label(frame_settings, text="Noise (%) :").grid(row=1, column=0, sticky="e")
        self.noise_var = tk.StringVar(value="0")
        ttk.Entry(frame_settings, textvariable=self.noise_var, width=10).grid(row=1, column=1)

        ttk.Label(frame_settings, text="Random Seed (optional):").grid(row=1, column=2, sticky="e")
        self.seed_var = tk.StringVar(value="")
        ttk.Entry(frame_settings, textvariable=self.seed_var, width=10).grid(row=1, column=3)

        ttk.Label(frame_settings, text="Base output name:").grid(row=2, column=0, sticky="e")
        self.out_var = tk.StringVar(value="synthetic_eis")
        ttk.Entry(frame_settings, textvariable=self.out_var, width=25)\
            .grid(row=2, column=1, columnspan=3, sticky="w")

        ttk.Button(frame_settings, text="Browse folder...",
                   command=self.browse_output_folder).grid(row=2, column=4)

        self.output_folder = ""

        # ----- Generate button -----
        frame_buttons = ttk.Frame(master)
        frame_buttons.pack(fill="x", padx=10, pady=10)
        ttk.Button(frame_buttons, text="Generate CSV",
                   command=self.generate_csv).pack(side="right")

        self.detect_parameters()

    def browse_output_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.output_folder = folder

    # -------------------------
    # detect parameters (regex)
    # -------------------------
    def detect_parameters(self):
        circuit_str = self.circuit_var.get().strip()
        if not circuit_str:
            messagebox.showerror("Error", "Circuit string is empty.")
            return

        try:
            param_names = extract_param_names(circuit_str)
            if not param_names:
                raise ValueError("No parameters detected. Check circuit string.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to parse circuit:\n{e}")
            return

        for w in self.inner_frame.winfo_children():
            w.destroy()

        self.param_entries.clear()
        self.param_order = param_names

        ttk.Label(self.inner_frame, text="Parameter", font=("Segoe UI", 9, "bold"))\
            .grid(row=0, column=0, padx=5, pady=3)
        ttk.Label(self.inner_frame, text="Value", font=("Segoe UI", 9, "bold"))\
            .grid(row=0, column=1, padx=5, pady=3)

        for i, name in enumerate(param_names, start=1):
            ttk.Label(self.inner_frame, text=name).grid(row=i, column=0, sticky="w", padx=5)

            var = tk.StringVar(value=str(default_for_param(name)))
            ttk.Entry(self.inner_frame, textvariable=var, width=15)\
                .grid(row=i, column=1, sticky="w", padx=5)

            self.param_entries[name] = var

    # -------------------------
    # produce CSV (clean + noisy) + metadata
    # -------------------------
    def generate_csv(self):
        circuit_str = self.circuit_var.get().strip()
        if not circuit_str:
            messagebox.showerror("Error", "Circuit string is empty.")
            return

        # Frequency settings
        try:
            fmin = float(self.fmin_var.get())
            fmax = float(self.fmax_var.get())
            points = int(self.points_var.get())
            if fmin <= 0 or fmax <= fmin or points < 3:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Frequency settings are invalid.")
            return

        # noise
        try:
            noise_percent = float(self.noise_var.get())
            if noise_percent < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Noise percentage is invalid.")
            return

        # Seed
        seed = None
        seed_txt = self.seed_var.get().strip()
        if seed_txt:
            try:
                seed = int(seed_txt)
            except Exception:
                messagebox.showerror("Error", "Random seed must be integer (or empty).")
                return

        # parameter values
        param_values = []
        param_dict = {}
        try:
            for name in self.param_order:
                val = float(self.param_entries[name].get().strip())
                param_values.append(val)
                param_dict[name] = val
        except Exception as e:
            messagebox.showerror("Error", f"Parameter values invalid:\n{e}")
            return

        freqs = make_frequency_vector(fmin, fmax, points)

        # producing clean Z_clean
        try:
            if seed is not None:
                np.random.seed(seed)
            circuit = CustomCircuit(circuit=circuit_str, initial_guess=param_values)
            Z_clean = circuit.predict(freqs, use_initial=True)
        except Exception as e:
            messagebox.showerror("Error", f"Error generating clean EIS data:\n{e}")
            return

        # Base name includes a Windows-safe circuit description.
        base = safe_filename_component(
            self.out_var.get().strip(),
            "synthetic_eis",
        )
        circuit_safe = safe_filename_component(
            circuit_str.replace(" ", ""),
            "circuit",
        )
        base = f"{base}_[{circuit_safe}]"


        # -----------------------------
        # Default Researcher Output Path (AUTO)
        # Desktop/Researcher Output/Generate Synthetic Data/EIS
        # -----------------------------
        try:
            if not self.output_folder:
                home = os.path.expanduser("~")
                desktop = os.path.join(home, "Desktop")
                if not os.path.isdir(desktop):
                    desktop = os.path.join(
                        os.environ.get("USERPROFILE", home),
                        "Desktop",
                    )
                self.output_folder = os.path.join(
                    desktop,
                    "Researcher Output",
                    "Generate Synthetic Data",
                    "EIS",
                )
            os.makedirs(self.output_folder, exist_ok=True)

            # Auto run folder
            existing = [
                d
                for d in os.listdir(self.output_folder)
                if os.path.isdir(os.path.join(self.output_folder, d))
                and d.startswith("run_")
            ]
            nums = []
            for directory_name in existing:
                try:
                    nums.append(int(directory_name.split("_")[1]))
                except (IndexError, ValueError):
                    pass
            next_idx = max(nums) + 1 if nums else 1
            run_dir = os.path.join(
                self.output_folder,
                f"run_{next_idx:03d}",
            )
            os.makedirs(run_dir, exist_ok=True)

            base_path = os.path.join(run_dir, base)
            if base_path.lower().endswith(".csv"):
                base_path = base_path[:-4]

            # Store clean data.
            clean_csv = base_path + "_clean.csv"
            save_eis_csv(clean_csv, freqs, Z_clean)

            # Store noisy data when requested.
            noisy_csv = None
            if noise_percent > 0.0:
                nl = noise_percent / 100.0
                mag = np.abs(Z_clean) + 1e-12
                Z_noisy = (
                    Z_clean.real
                    + nl * mag * np.random.randn(len(Z_clean))
                ) + 1j * (
                    Z_clean.imag
                    + nl * mag * np.random.randn(len(Z_clean))
                )

                noise_str = str(noise_percent).rstrip("0").rstrip(".")
                noisy_csv = base_path + f"_noisy_%{noise_str}.csv"
                save_eis_csv(noisy_csv, freqs, Z_noisy)
            else:
                print("[INFO] Noise = 0; no noisy CSV generated.")

            # Store metadata only after all requested CSV files succeed.
            meta = {
                "circuit": circuit_str,
                "parameters": param_dict,
                "frequency": {
                    "f_min": fmin,
                    "f_max": fmax,
                    "points": points,
                    "distribution": "logarithmic, ascending",
                },
                "noise_percent": noise_percent,
                "seed": seed,
                "files": {
                    "clean_csv": os.path.basename(clean_csv),
                    "noisy_csv": (
                        os.path.basename(noisy_csv)
                        if noisy_csv
                        else None
                    ),
                },
                "generated_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
            }

            meta_file = base_path + "_meta.json"
            save_metadata_json(meta_file, meta)
        except Exception as exc:
            messagebox.showerror(
                "Save Error",
                "The generated data could not be saved:\n"
                f"{exc}",
            )
            return

        messagebox.showinfo("Done", "CSV file(s) and metadata generated successfully.")


if __name__ == "__main__":
    import ctypes

    root = tk.Tk()

    # -------- app id for taskbar --------
    if os.name == "nt":
        myappid = "researcher.analysis.toolkit.v1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

    # -------- Title --------
    root.title("Researcher Analysis Toolkit - EIS Generator")

    # -------- Icon --------
    ICON_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Researcherlogo.ico",
    )
    if os.path.exists(ICON_PATH):
        root.iconbitmap(ICON_PATH)

    app = EISCSVGUI(root)
    root.mainloop()
