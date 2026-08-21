import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.ndimage import percentile_filter
from fpdf import FPDF
from datetime import datetime
import shutil
import os
import ctypes


class TEAM_PEC_Analyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Research Analysis Toolkit - ABPE")
        self.root.geometry("1750x980")
        self.root.configure(bg="#FFFFFF")

        try:
            app_id = "ResearchAnalysisToolkit.ABPE.App"
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
        except Exception:
            pass

        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Researcherlogo.ico")
        try:
            self.root.iconbitmap(default=icon_path)
        except Exception:
            try:
                self.root.iconbitmap(icon_path)
            except Exception:
                pass

        self.df_light = None
        self.df_dark = None
        self.file_path_light = None
        self.file_path_dark = None

        self.processed_results = None
        self.last_mode_used = "Unknown"
        self.last_data_mode = "Unknown"
        self.last_dark_source = "Unknown"

        self.reference_map = {
            "Already vs RHE": None,
            "Already vs NHE": "NHE_ONLY",
            "Already vs SHE": "NHE_ONLY",

            "RHE": 0.000,
            "NHE": 0.000,
            "SHE": 0.000,

            "Ag/AgCl (sat. KCl)": 0.197,
            "Ag/AgCl (3M KCl)": 0.210,
            "Ag/AgCl (1M KCl)": 0.235,
            "Ag/AgCl (0.1M KCl)": 0.288,

            "SCE (sat. KCl)": 0.241,
            "Calomel (1M KCl)": 0.280,
            "Calomel (0.1M KCl)": 0.333,

            "Hg/HgO (1M KOH)": 0.098,
            "Hg/HgO (0.1M KOH)": 0.165,
            "Hg/HgO (alkaline, generic)": 0.140,

            "Hg/Hg2SO4 (sat. K2SO4)": 0.640,
            "Hg/Hg2SO4 (1M H2SO4)": 0.680,
            "MSE (Mercurous Sulfate Electrode)": 0.650,

            "Ag/Ag2SO4 (sat. K2SO4)": 0.680,
            "Ag/Ag2SO4 (0.5M H2SO4)": 0.700,

            "Cu/CuSO4 (sat. CuSO4)": 0.316,

            "Pb/PbSO4": -0.356,
            "Zn/ZnSO4": -0.763,

            "Pt pseudo-reference": "USER_DEFINED",
            "Ag pseudo-reference": "USER_DEFINED",
            "Custom offset vs NHE": "USER_DEFINED",
        }

        self.product_rows = []

        self.setup_styles()
        self.create_widgets()
        self.add_product_row("H2", "100", "1.23")
        self.add_product_row("CO", "0", "1.33")
        self.add_product_row("HCOO-", "0", "1.23")

    # ---------------- UI ----------------
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#FFFFFF")
        style.configure("Sidebar.TFrame", background="#F1F5F9", relief="flat")
        style.configure("TLabel", background="#FFFFFF")
        style.configure("TButton", font=("Segoe UI", 9))
        style.configure("TCombobox", padding=2)

    def create_widgets(self):
        # ===== LEFT SCROLLABLE SIDEBAR =====
        sidebar_container = ttk.Frame(self.root, width=560, style="Sidebar.TFrame")
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y)
        sidebar_container.pack_propagate(False)

        self.sidebar_canvas = tk.Canvas(
            sidebar_container,
            bg="#F1F5F9",
            highlightthickness=0,
            bd=0
        )
        self.sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sidebar_scrollbar = ttk.Scrollbar(
            sidebar_container,
            orient="vertical",
            command=self.sidebar_canvas.yview
        )
        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar = ttk.Frame(self.sidebar_canvas, style="Sidebar.TFrame")
        self.sidebar_window = self.sidebar_canvas.create_window((0, 0), window=sidebar, anchor="nw")

        def _update_sidebar_scrollregion(event=None):
            self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all"))

        def _resize_sidebar_width(event):
            self.sidebar_canvas.itemconfig(self.sidebar_window, width=event.width)

        sidebar.bind("<Configure>", _update_sidebar_scrollregion)
        self.sidebar_canvas.bind("<Configure>", _resize_sidebar_width)

        def _on_mousewheel(event):
            self.sidebar_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel(event):
            self.sidebar_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel(event):
            self.sidebar_canvas.unbind_all("<MouseWheel>")

        self.sidebar_canvas.bind("<Enter>", _bind_mousewheel)
        self.sidebar_canvas.bind("<Leave>", _unbind_mousewheel)

        # ---------------- 1. DATA SOURCE ----------------
        tk.Label(
            sidebar, text="1. DATA SOURCE", bg="#F1F5F9", fg="#475569",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(15, 4))

        # Light block
        light_box = tk.LabelFrame(sidebar, text="Light File", bg="#F1F5F9", fg="#0F172A",
                                  font=("Segoe UI", 9, "bold"), bd=1, relief="groove")
        light_box.pack(fill=tk.X, padx=10, pady=4)

        light_btn_frame = tk.Frame(light_box, bg="#F1F5F9")
        light_btn_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Button(
            light_btn_frame,
            text="Load Light Data (CSV / Excel)",
            command=self.load_light_file
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            light_btn_frame,
            text="↻",
            command=self.clear_light_file,
            bg="#E2E8F0",
            fg="#0F172A",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            width=3
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.lbl_light_status = tk.Label(
            light_box, text="No light file loaded", bg="#F1F5F9", fg="#EF4444",
            font=("Segoe UI", 9, "italic")
        )
        self.lbl_light_status.pack(anchor=tk.W, padx=8, pady=(0, 6))

        tk.Label(light_box, text="Potential Column:", bg="#F1F5F9", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, padx=8)
        self.light_potential_col = tk.StringVar()
        self.cmb_light_potential = ttk.Combobox(light_box, textvariable=self.light_potential_col, state="readonly")
        self.cmb_light_potential.pack(fill=tk.X, padx=8, pady=(2, 4))

        tk.Label(light_box, text="Current Column:", bg="#F1F5F9", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, padx=8)
        self.light_current_col = tk.StringVar()
        self.cmb_light_current = ttk.Combobox(light_box, textvariable=self.light_current_col, state="readonly")
        self.cmb_light_current.pack(fill=tk.X, padx=8, pady=(2, 8))

        # Dark block
        dark_box = tk.LabelFrame(sidebar, text="Dark File", bg="#F1F5F9", fg="#0F172A",
                                 font=("Segoe UI", 9, "bold"), bd=1, relief="groove")
        dark_box.pack(fill=tk.X, padx=10, pady=4)

        dark_btn_frame = tk.Frame(dark_box, bg="#F1F5F9")
        dark_btn_frame.pack(fill=tk.X, padx=8, pady=(8, 4))

        ttk.Button(
            dark_btn_frame,
            text="Load Dark Data (CSV / Excel)",
            command=self.load_dark_file
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            dark_btn_frame,
            text="↻",
            command=self.clear_dark_file,
            bg="#E2E8F0",
            fg="#0F172A",
            relief="flat",
            font=("Segoe UI", 11, "bold"),
            width=3
        ).pack(side=tk.LEFT, padx=(6, 0))

        self.lbl_dark_status = tk.Label(
            dark_box, text="No dark file loaded", bg="#F1F5F9", fg="#EF4444",
            font=("Segoe UI", 9, "italic")
        )
        self.lbl_dark_status.pack(anchor=tk.W, padx=8, pady=(0, 6))

        tk.Label(dark_box, text="Potential Column:", bg="#F1F5F9", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, padx=8)
        self.dark_potential_col = tk.StringVar()
        self.cmb_dark_potential = ttk.Combobox(dark_box, textvariable=self.dark_potential_col, state="readonly")
        self.cmb_dark_potential.pack(fill=tk.X, padx=8, pady=(2, 4))

        tk.Label(dark_box, text="Current Column:", bg="#F1F5F9", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, padx=8)
        self.dark_current_col = tk.StringVar()
        self.cmb_dark_current = ttk.Combobox(dark_box, textvariable=self.dark_current_col, state="readonly")
        self.cmb_dark_current.pack(fill=tk.X, padx=8, pady=(2, 8))

        # ---------------- 2. CONFIGURATION ----------------
        tk.Label(
            sidebar, text="2. CONFIGURATION", bg="#F1F5F9", fg="#475569",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(15, 4))

        tk.Label(sidebar, text="■ Electrode Type:", bg="#F1F5F9", fg="#64748B",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(4, 0))

        self.electrode_type = tk.StringVar(value="Photoanode")
        tk.Radiobutton(sidebar, text="Photoanode (n-type)", variable=self.electrode_type,
                       value="Photoanode", bg="#F1F5F9", bd=0, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=25)
        tk.Radiobutton(sidebar, text="Photocathode (p-type)", variable=self.electrode_type,
                       value="Photocathode", bg="#F1F5F9", bd=0, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=25)

        tk.Label(sidebar, text="■ Illumination Mode:", bg="#F1F5F9", fg="#64748B",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(8, 0))

        self.exp_type = tk.StringVar(value="Auto")
        tk.Radiobutton(sidebar, text="Auto Detect", variable=self.exp_type,
                       value="Auto", bg="#F1F5F9", bd=0, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=25)
        tk.Radiobutton(sidebar, text="Continuous Light", variable=self.exp_type,
                       value="Continuous", bg="#F1F5F9", bd=0, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=25)
        tk.Radiobutton(sidebar, text="Chopped Light", variable=self.exp_type,
                       value="Chopped", bg="#F1F5F9", bd=0, font=("Segoe UI", 9)).pack(anchor=tk.W, padx=25)

        tk.Label(sidebar, text="■ Potential Input Scale:", bg="#F1F5F9", fg="#64748B",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(8, 0))

        self.ref_electrode = tk.StringVar(value="Already vs RHE")
        ref_combo = ttk.Combobox(
            sidebar,
            textvariable=self.ref_electrode,
            values=list(self.reference_map.keys()),
            state="readonly"
        )
        ref_combo.pack(fill=tk.X, padx=20, pady=4)

        tk.Label(sidebar, text="■ Custom Offset vs NHE (V):", bg="#F1F5F9", fg="#64748B",
                 font=("Segoe UI", 9, "bold")).pack(anchor=tk.W, padx=15, pady=(8, 0))
        self.custom_ref_offset = tk.StringVar(value="0.000")
        ttk.Entry(sidebar, textvariable=self.custom_ref_offset).pack(fill=tk.X, padx=20, pady=4)

        # ---------------- 3. PARAMETERS ----------------
        tk.Label(
            sidebar, text="3. PARAMETERS", bg="#F1F5F9", fg="#475569",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(15, 4))

        p_frame = tk.Frame(sidebar, bg="#F1F5F9")
        p_frame.pack(fill=tk.X, padx=10)

        self.ent_ph = self.create_p_row(p_frame, "pH:", "14.0", 0)
        self.ent_area = self.create_p_row(p_frame, "Area (cm²):", "1.0", 1)
        self.ent_pin = self.create_p_row(p_frame, "Pin (mW/cm²):", "100.0", 2)

        # ---------------- 4. FE PRODUCTS ----------------
        tk.Label(
            sidebar, text="4. FARADAIC EFFICIENCY & PRODUCTS", bg="#F1F5F9", fg="#475569",
            font=("Segoe UI", 10, "bold")
        ).pack(anchor=tk.W, padx=10, pady=(15, 4))

        header_frame = tk.Frame(sidebar, bg="#F1F5F9")
        header_frame.pack(fill=tk.X, padx=10)

        tk.Label(header_frame, text="Product", width=16, bg="#E2E8F0", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, padx=1, pady=1)
        tk.Label(header_frame, text="FE (%)", width=10, bg="#E2E8F0", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, padx=1, pady=1)
        tk.Label(header_frame, text="Erev (V)", width=10, bg="#E2E8F0", font=("Segoe UI", 9, "bold")).grid(row=0, column=2, padx=1, pady=1)

        self.products_frame = tk.Frame(sidebar, bg="#F1F5F9")
        self.products_frame.pack(fill=tk.X, padx=10, pady=4)

        product_btn_frame = tk.Frame(sidebar, bg="#F1F5F9")
        product_btn_frame.pack(fill=tk.X, padx=10, pady=4)

        tk.Button(product_btn_frame, text="+ Add Product", bg="#0EA5E9", fg="white",
                  relief="flat", command=self.add_product_row).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        tk.Button(product_btn_frame, text="- Remove Last", bg="#EF4444", fg="white",
                  relief="flat", command=self.remove_product_row).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        # ---------------- 5. ACTIONS ----------------
        self.btn_run = tk.Button(
            sidebar, text="GENERATE ANALYSIS", bg="#0F172A", fg="white",
            font=("Segoe UI", 11, "bold"), relief="flat", pady=12, command=self.run_engine
        )
        self.btn_run.pack(fill=tk.X, padx=10, pady=(20, 8))

        self.btn_export_all = tk.Button(
            sidebar, text="EXPORT ALL RESULTS", bg="#4F46E5", fg="white",
            font=("Segoe UI", 10, "bold"), relief="flat", pady=12, command=self.export_all
        )
        self.btn_export_all.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_note = tk.Label(
            sidebar,
            text="Outputs will be saved to Desktop\\Researcher Output\\ABPE\\run_00X",
            bg="#F1F5F9",
            fg="#475569",
            font=("Segoe UI", 8, "italic"),
            wraplength=500,
            justify="left"
        )
        self.lbl_note.pack(anchor=tk.W, padx=10, pady=(8, 10))

        # ---------------- MAIN VIEW ----------------
        self.main_view = ttk.Frame(self.root)
        self.main_view.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 10), facecolor="#FFFFFF")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_view)

        toolbar = NavigationToolbar2Tk(self.canvas, self.main_view)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def create_p_row(self, master, text, default, row):
        tk.Label(master, text=text, bg="#F1F5F9", font=("Segoe UI", 9)).grid(row=row, column=0, sticky="w", pady=2)
        ent = ttk.Entry(master)
        ent.insert(0, default)
        ent.grid(row=row, column=1, sticky="ew", pady=2, padx=5)
        return ent

    def add_product_row(self, name="", fe="", erev=""):
        row_idx = len(self.product_rows)

        row_frame = tk.Frame(self.products_frame, bg="#F1F5F9")
        row_frame.grid(row=row_idx, column=0, sticky="ew", pady=1)

        ent_name = ttk.Entry(row_frame, width=16)
        ent_name.insert(0, name)
        ent_name.grid(row=0, column=0, padx=1)

        ent_fe = ttk.Entry(row_frame, width=12)
        ent_fe.insert(0, fe)
        ent_fe.grid(row=0, column=1, padx=1)

        ent_erev = ttk.Entry(row_frame, width=12)
        ent_erev.insert(0, erev)
        ent_erev.grid(row=0, column=2, padx=1)

        self.product_rows.append((row_frame, ent_name, ent_fe, ent_erev))

    def remove_product_row(self):
        if not self.product_rows:
            return
        row_frame, *_ = self.product_rows.pop()
        row_frame.destroy()

    # ---------------- File clear ----------------
    def clear_light_file(self):
        self.df_light = None
        self.file_path_light = None
        self.light_potential_col.set("")
        self.light_current_col.set("")
        self.cmb_light_potential["values"] = []
        self.cmb_light_current["values"] = []
        self.lbl_light_status.config(text="No light file loaded", fg="#EF4444")

    def clear_dark_file(self):
        self.df_dark = None
        self.file_path_dark = None
        self.dark_potential_col.set("")
        self.dark_current_col.set("")
        self.cmb_dark_potential["values"] = []
        self.cmb_dark_current["values"] = []
        self.lbl_dark_status.config(text="No dark file loaded", fg="#EF4444")

    # ---------------- File loading ----------------
    def load_table_file(self, which="light"):
        path = filedialog.askopenfilename(filetypes=[("Data Files", "*.csv *.xlsx *.xls")])
        if not path:
            return

        try:
            if path.lower().endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)

            cols = list(df.columns)
            cols_str = [str(c) for c in cols]

            if which == "light":
                self.df_light = df
                self.file_path_light = path
                self.lbl_light_status.config(text=f"● {os.path.basename(path)} loaded", fg="#10B981")
                self.cmb_light_potential["values"] = cols_str
                self.cmb_light_current["values"] = cols_str

                pot_guess = self.find_best_column(cols, ["potential", "voltage", "bias", "ewe", "potential/v", "v"])
                cur_guess = self.find_best_column(cols, ["current", "photocurrent", "current/a", "i", "j"])

                if pot_guess is None and len(cols_str) > 0:
                    pot_guess = cols_str[0]
                if cur_guess is None and len(cols_str) > 1:
                    cur_guess = cols_str[1]

                if pot_guess is not None:
                    self.light_potential_col.set(str(pot_guess))
                if cur_guess is not None:
                    self.light_current_col.set(str(cur_guess))

            else:
                self.df_dark = df
                self.file_path_dark = path
                self.lbl_dark_status.config(text=f"● {os.path.basename(path)} loaded", fg="#10B981")
                self.cmb_dark_potential["values"] = cols_str
                self.cmb_dark_current["values"] = cols_str

                pot_guess = self.find_best_column(cols, ["potential", "voltage", "bias", "ewe", "potential/v", "v"])
                cur_guess = self.find_best_column(cols, ["current", "photocurrent", "current/a", "i", "j"])

                if pot_guess is None and len(cols_str) > 0:
                    pot_guess = cols_str[0]
                if cur_guess is None and len(cols_str) > 1:
                    cur_guess = cols_str[1]

                if pot_guess is not None:
                    self.dark_potential_col.set(str(pot_guess))
                if cur_guess is not None:
                    self.dark_current_col.set(str(cur_guess))

        except Exception as e:
            messagebox.showerror("Load Error", f"Could not load file:\n{e}")

    def load_light_file(self):
        self.load_table_file("light")

    def load_dark_file(self):
        self.load_table_file("dark")

    # ---------------- Helpers ----------------
    def find_best_column(self, columns, keywords):
        lower_cols = {str(c).lower(): c for c in columns}
        for key in keywords:
            for lc, original in lower_cols.items():
                if key in lc:
                    return original
        return None

    def get_selected_series(self, df, pot_name, cur_name):
        if df is None:
            return None, None

        if not pot_name or not cur_name:
            raise ValueError("Please select both potential and current columns.")

        if pot_name not in df.columns or cur_name not in df.columns:
            raise ValueError("Selected columns were not found in the file.")

        pot = pd.to_numeric(df[pot_name], errors="coerce").values
        cur = pd.to_numeric(df[cur_name], errors="coerce").values

        valid = np.isfinite(pot) & np.isfinite(cur)
        pot = pot[valid]
        cur = cur[valid]

        if len(pot) == 0:
            raise ValueError("Selected columns do not contain valid numeric data.")

        return pot, cur

    def to_rhe(self, potential_raw, ph):
        ref_name = self.ref_electrode.get()
        ref_value = self.reference_map.get(ref_name, None)

        if ref_name == "Already vs RHE":
            return potential_raw.copy()

        if ref_value == "NHE_ONLY":
            return potential_raw + (0.0591 * ph)

        if ref_name == "RHE":
            return potential_raw.copy()

        if ref_value == "USER_DEFINED":
            try:
                offset = float(self.custom_ref_offset.get())
            except Exception:
                raise ValueError("Please enter a valid custom reference offset vs NHE.")
            return potential_raw + offset + (0.0591 * ph)

        if ref_value is None:
            return potential_raw.copy()

        return potential_raw + ref_value + (0.0591 * ph)

    def get_products(self):
        products = []
        for _, ent_name, ent_fe, ent_erev in self.product_rows:
            name = ent_name.get().strip()
            fe_txt = ent_fe.get().strip()
            erev_txt = ent_erev.get().strip()

            if not name and not fe_txt and not erev_txt:
                continue

            if not name:
                raise ValueError("One product row has empty product name.")
            if fe_txt == "":
                raise ValueError(f"FE missing for product: {name}")
            if erev_txt == "":
                raise ValueError(f"Erev missing for product: {name}")

            fe = float(fe_txt)
            erev = float(erev_txt)
            products.append({"name": name, "fe_percent": fe, "erev": erev})

        if not products:
            raise ValueError("Please define at least one product.")

        total_fe = sum(p["fe_percent"] for p in products)
        return products, total_fe

    def detect_mode(self, current):
        if len(current) < 20:
            return "Continuous"

        diff = np.diff(current)
        mad = np.median(np.abs(diff - np.median(diff))) + 1e-12
        jump_score = np.sum(np.abs(diff) > 6 * mad)

        if jump_score > max(10, len(current) * 0.05):
            return "Chopped"
        return "Continuous"

    def extract_light_dark_chopped(self, j_ma, is_anode):
        w = max(25, len(j_ma) // 18)

        if is_anode:
            jl_raw = percentile_filter(j_ma, 98, size=w)
            jd_raw = percentile_filter(j_ma, 2, size=w)
        else:
            jl_raw = percentile_filter(j_ma, 2, size=w)
            jd_raw = percentile_filter(j_ma, 98, size=w)

        jl = pd.Series(jl_raw).rolling(w, center=True, min_periods=1).mean().values
        jd = pd.Series(jd_raw).rolling(w, center=True, min_periods=1).mean().values
        return jl, jd

    def compute_signed_photocurrent(self, jl, jd, is_anode):
        return (jl - jd) if is_anode else (jd - jl)

    def build_weighted_driving_force(self, v_rhe, products):
        total_term = np.zeros_like(v_rhe, dtype=float)
        for p in products:
            fe_fraction = p["fe_percent"] / 100.0
            erev = p["erev"]
            total_term += fe_fraction * (erev - v_rhe)
        return total_term

    def interpolate_dark_to_light(self, v_light, v_dark, j_dark):
        if len(v_dark) < 2:
            return np.full_like(v_light, np.nanmedian(j_dark) if len(j_dark) > 0 else 0.0, dtype=float)
        order = np.argsort(v_dark)
        v_dark_sorted = v_dark[order]
        j_dark_sorted = j_dark[order]
        return np.interp(v_light, v_dark_sorted, j_dark_sorted,
                         left=j_dark_sorted[0], right=j_dark_sorted[-1])

    def dynamic_annotation(self, ax, x_data, y_data):
        if len(y_data) == 0:
            return
        if np.nanmax(y_data) <= 0:
            return

        idx = int(np.nanargmax(y_data))
        x = x_data[idx]
        y = y_data[idx]

        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        dx = (x_max - x_min) * 0.14
        dy = (y_max - y_min) * 0.10

        if x > (x_min + x_max) / 2:
            text_x = x - dx
            ha = "right"
        else:
            text_x = x + dx
            ha = "left"

        if y > (y_min + y_max) / 2:
            text_y = y - dy
            va = "top"
        else:
            text_y = y + dy
            va = "bottom"

        ax.annotate(
            f"Max: {y:.3f}%",
            xy=(x, y),
            xytext=(text_x, text_y),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            fontsize=10,
            color="red",
            fontweight="bold",
            ha=ha,
            va=va,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.85)
        )

    # ---------------- Core analysis ----------------
    def run_engine(self):
        if self.df_light is None and self.df_dark is None:
            messagebox.showwarning("Warning", "Please load at least one file (Light or Dark).")
            return

        try:
            ph = float(self.ent_ph.get())
            area = float(self.ent_area.get())
            pin = float(self.ent_pin.get())

            if area <= 0:
                raise ValueError("Area must be > 0.")
            if pin <= 0:
                raise ValueError("Pin must be > 0.")

            products, total_fe = self.get_products()
            is_anode = self.electrode_type.get() == "Photoanode"

            light_loaded = self.df_light is not None
            dark_loaded = self.df_dark is not None

            if light_loaded:
                v_light_raw, i_light_raw = self.get_selected_series(
                    self.df_light,
                    self.light_potential_col.get(),
                    self.light_current_col.get()
                )
                v_light_rhe = self.to_rhe(v_light_raw, ph)
                j_light_total = (i_light_raw * 1000.0) / area
            else:
                v_light_raw, i_light_raw, v_light_rhe, j_light_total = None, None, None, None

            if dark_loaded:
                v_dark_raw, i_dark_raw = self.get_selected_series(
                    self.df_dark,
                    self.dark_potential_col.get(),
                    self.dark_current_col.get()
                )
                v_dark_rhe = self.to_rhe(v_dark_raw, ph)
                j_dark_total = (i_dark_raw * 1000.0) / area
            else:
                v_dark_raw, i_dark_raw, v_dark_rhe, j_dark_total = None, None, None, None

            selected_mode = self.exp_type.get()

            if light_loaded and dark_loaded:
                self.last_data_mode = "Separate Light + Separate Dark"

                if selected_mode == "Auto":
                    mode = self.detect_mode(j_light_total)
                else:
                    mode = selected_mode
                self.last_mode_used = mode

                v_common = v_light_rhe.copy()
                jl = j_light_total.copy()
                jd = self.interpolate_dark_to_light(v_common, v_dark_rhe, j_dark_total)

                j_total_for_plot = jl.copy()
                j_ph_signed = self.compute_signed_photocurrent(jl, jd, is_anode)
                dark_source = "measured_separate_file"

            elif light_loaded and not dark_loaded:
                self.last_data_mode = "Light only"

                if selected_mode == "Auto":
                    mode = self.detect_mode(j_light_total)
                else:
                    mode = selected_mode
                self.last_mode_used = mode

                v_common = v_light_rhe.copy()

                if mode == "Chopped":
                    jl, jd = self.extract_light_dark_chopped(j_light_total, is_anode)
                    dark_source = "estimated_from_chopped_light_file"
                else:
                    jl = j_light_total.copy()
                    jd = np.zeros_like(jl)
                    dark_source = "assumed_zero"

                j_total_for_plot = j_light_total.copy()
                j_ph_signed = self.compute_signed_photocurrent(jl, jd, is_anode)

                if dark_source == "assumed_zero":
                    messagebox.showwarning(
                        "Dark Current Warning",
                        "Only Light file was loaded and no Dark file was provided.\n"
                        "For Continuous mode, Dark current was assumed to be zero.\n"
                        "ABPE may be approximate."
                    )

            else:
                self.last_data_mode = "Dark only"
                self.last_mode_used = "Continuous"
                v_common = v_dark_rhe.copy()

                jl = np.zeros_like(j_dark_total)
                jd = j_dark_total.copy()
                j_total_for_plot = j_dark_total.copy()
                j_ph_signed = np.zeros_like(jd)
                dark_source = "dark_only"

                messagebox.showwarning(
                    "Dark Only Mode",
                    "Only Dark file was loaded.\n"
                    "Photocurrent cannot be extracted rigorously without Light data.\n"
                    "The program will show the Dark curve and efficiency will remain zero."
                )

            self.last_dark_source = dark_source

            j_ph_useful = np.clip(j_ph_signed, 0, None)

            weighted_driving = self.build_weighted_driving_force(v_common, products)
            weighted_driving_useful = np.clip(weighted_driving, 0, None)

            eff = (j_ph_useful * weighted_driving_useful / pin) * 100.0
            eff = np.clip(eff, 0, None)

            out = {
                "Potential_RHE": v_common,
                "J_total_or_primary_mA_cm2": j_total_for_plot,
                "J_light_mA_cm2": jl,
                "J_dark_mA_cm2": jd,
                "J_ph_signed_mA_cm2": j_ph_signed,
                "J_ph_useful_mA_cm2": j_ph_useful,
                "Weighted_Driving_Force_V": weighted_driving,
                "Weighted_Driving_Useful_V": weighted_driving_useful,
                "ABPE_FE_weighted_percent": eff
            }

            for p in products:
                out[f"FE_{p['name']}_percent"] = np.full_like(v_common, p["fe_percent"], dtype=float)
                out[f"Erev_{p['name']}_V"] = np.full_like(v_common, p["erev"], dtype=float)
                out[f"Contribution_{p['name']}_V"] = np.clip(
                    (p["fe_percent"] / 100.0) * (p["erev"] - v_common), 0, None
                )

            self.processed_results = pd.DataFrame(out)

            meta_text = (
                f"Mode: {self.last_mode_used} | "
                f"Data: {self.last_data_mode} | "
                f"Electrode: {'n-type photoanode' if is_anode else 'p-type photocathode'} | "
                f"Potential scale: {self.ref_electrode.get()} | "
                f"Dark source: {dark_source} | "
                f"Total FE: {total_fe:.2f}%"
            )

            self.plot_results(v_common, j_total_for_plot, jl, jd, eff, meta_text)

            if total_fe > 100.0001:
                messagebox.showwarning(
                    "FE Warning",
                    f"Total FE = {total_fe:.2f}% which is greater than 100%.\n"
                    f"Calculation was still done. Please verify FE inputs."
                )

            if np.all(weighted_driving_useful <= 0):
                messagebox.showwarning(
                    "Driving Force Warning",
                    "All FE-weighted driving-force terms are non-positive.\n"
                    "This will make ABPE become zero.\n"
                    "Please check potential scale / reference conversion / Erev values."
                )

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ---------------- Plotting ----------------
    def plot_results(self, v, j_total, jl, jd, eff, meta_text=""):
        self.ax1.clear()
        self.ax2.clear()

        self.ax1.plot(v, j_total, color="#CBD5E1", alpha=0.75, lw=1.8, label="Raw / Primary Current")
        self.ax1.plot(v, jl, label="Light State", color="#F59E0B", lw=2.0)

        if np.any(np.abs(jd) > 1e-15):
            self.ax1.plot(v, jd, label="Dark State", color="#1E293B", lw=2.0)
        else:
            self.ax1.plot(v, jd, label="Dark State (assumed zero)", color="#1E293B", lw=1.5, ls="--")

        self.ax1.set_title("Processed J-V / Light-Dark Analysis", fontweight="bold")
        self.ax1.set_ylabel("J (mA/cm²)", fontweight="bold")
        self.ax1.grid(True, alpha=0.25)
        self.ax1.legend()

        self.ax2.plot(v, eff, color="#10B981", lw=2.5, label="FE-weighted ABPE")
        self.ax2.fill_between(v, eff, color="#10B981", alpha=0.15)

        self.ax2.set_title("Applied Bias Photon-to-Current Efficiency (FE-weighted)", fontweight="bold")
        self.ax2.set_xlabel("Potential vs RHE (V)", fontweight="bold")
        self.ax2.set_ylabel("Efficiency (%)", fontweight="bold")
        self.ax2.grid(True, alpha=0.25)
        self.ax2.legend()

        self.dynamic_annotation(self.ax2, v, eff)

        self.fig.suptitle(meta_text, fontsize=9.5, y=0.985)
        self.fig.tight_layout(rect=[0, 0, 1, 0.97])
        self.canvas.draw()

    # ---------------- Output paths ----------------
    def get_output_run_dir(self):
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        base_dir = os.path.join(desktop, "Researcher Output", "ABPE")
        os.makedirs(base_dir, exist_ok=True)

        existing = []
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if os.path.isdir(full) and name.lower().startswith("run_"):
                try:
                    existing.append(int(name.split("_")[1]))
                except Exception:
                    pass

        next_idx = 1 if not existing else max(existing) + 1
        run_dir = os.path.join(base_dir, f"run_{next_idx:03d}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def create_equation_image(self, run_dir):
        eq_path = os.path.join(run_dir, "equation.png")

        fig = plt.figure(figsize=(12, 3.4), facecolor="white")

    # Line 1
        fig.text(
            0.5, 0.72,
            r"$J_{\mathrm{ph}} = J_{\mathrm{light}} - J_{\mathrm{dark}}$" + "   (n-type photoanode)",
            ha="center", va="center", fontsize=16
        )

    # Line 2
        fig.text(
            0.5, 0.46,
            r"$J_{\mathrm{ph}} = J_{\mathrm{dark}} - J_{\mathrm{light}}$" + "   (p-type photocathode)",
            ha="center", va="center", fontsize=16
        )

    # Main equation
        fig.text(
            0.5, 0.16,
            r"$\eta(\%) = \frac{J_{\mathrm{ph,useful}} \times \sum_i \left(FE_i (E_{\mathrm{rev},i} - V_{\mathrm{RHE}})\right)}{P_{\mathrm{in}}} \times 100$",
            ha="center", va="center", fontsize=17
        )

        plt.axis("off")
        plt.savefig(eq_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return eq_path

    # ---------------- HTML report ----------------
    def build_html_report(self, run_dir, fig_filename, max_eff):
        now_txt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        electrode_txt = self.electrode_type.get()
        mode_txt = self.last_mode_used
        data_mode_txt = self.last_data_mode
        ref_txt = self.ref_electrode.get()
        ph_txt = self.ent_ph.get()
        area_txt = self.ent_area.get()
        pin_txt = self.ent_pin.get()

        product_rows_html = ""
        total_fe = 0.0

        for _, ent_name, ent_fe, ent_erev in self.product_rows:
            name = ent_name.get().strip()
            fe = ent_fe.get().strip()
            erev = ent_erev.get().strip()

            if not name and not fe and not erev:
                continue

            try:
                total_fe += float(fe)
            except Exception:
                pass

            product_rows_html += f"""
            <tr>
                <td>{name}</td>
                <td>{fe}</td>
                <td>{erev}</td>
            </tr>
            """

        if product_rows_html.strip() == "":
            product_rows_html = """
            <tr>
                <td colspan="3">No products defined</td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ABPE Analysis Report</title>
<style>
    body {{
        font-family: Arial, sans-serif;
        margin: 30px;
        color: #111827;
        background: #ffffff;
    }}
    h1 {{
        color: #0f172a;
        border-bottom: 2px solid #cbd5e1;
        padding-bottom: 8px;
    }}
    h2 {{
        color: #1e293b;
        margin-top: 28px;
    }}
    .meta {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px;
        border-radius: 10px;
        margin-bottom: 20px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
    }}
    th, td {{
        border: 1px solid #cbd5e1;
        padding: 8px 10px;
        text-align: center;
    }}
    th {{
        background: #e2e8f0;
    }}
    .highlight {{
        font-size: 18px;
        font-weight: bold;
        color: #059669;
    }}
    img {{
        max-width: 100%;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        margin-top: 12px;
    }}
    .eq-img {{
        max-width: 95%;
        display: block;
        margin: 16px auto;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        background: #ffffff;
        padding: 10px;
    }}
</style>
</head>
<body>

<h1>Research Analysis Toolkit - ABPE</h1>

<div class="meta">
    <p><strong>Generated:</strong> {now_txt}</p>
    <p><strong>Electrode Type:</strong> {electrode_txt}</p>
    <p><strong>Illumination Mode:</strong> {mode_txt}</p>
    <p><strong>Data Mode:</strong> {data_mode_txt}</p>
    <p><strong>Potential Input Scale:</strong> {ref_txt}</p>
    <p><strong>pH:</strong> {ph_txt}</p>
    <p><strong>Area (cm²):</strong> {area_txt}</p>
    <p><strong>Pin (mW/cm²):</strong> {pin_txt}</p>
    <p><strong>Maximum Efficiency:</strong> <span class="highlight">{max_eff:.4f} %</span></p>
</div>

<h2>Products and FE Inputs</h2>
<table>
    <thead>
        <tr>
            <th>Product</th>
            <th>FE (%)</th>
            <th>Erev (V)</th>
        </tr>
    </thead>
    <tbody>
        {product_rows_html}
        <tr>
            <td><strong>Total</strong></td>
            <td><strong>{total_fe:.2f}</strong></td>
            <td>-</td>
        </tr>
    </tbody>
</table>

<h2>Methodology</h2>
<p>Only useful positive photocurrent contributes to the final efficiency.</p>
<img src="equation.png" alt="ABPE equations" class="eq-img">

<h2>Analysis Figure</h2>
<img src="{fig_filename}" alt="ABPE analysis figure">

</body>
</html>
"""
        html_path = os.path.join(run_dir, "report.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        return html_path

    # ---------------- Export ----------------
    def export_all(self):
        if self.processed_results is None:
            messagebox.showwarning("Warning", "No processed results to export.")
            return

        try:
            run_dir = self.get_output_run_dir()

            processed_path = os.path.join(run_dir, "processed_data.xlsx")
            light_excel_path = os.path.join(run_dir, "light_loaded_data.xlsx")
            dark_excel_path = os.path.join(run_dir, "dark_loaded_data.xlsx")
            fig_path = os.path.join(run_dir, "analysis_figure.png")
            pdf_path = os.path.join(run_dir, "report.pdf")
            eq_path = self.create_equation_image(run_dir)

            self.processed_results.to_excel(processed_path, index=False)

            if self.df_light is not None:
                self.df_light.to_excel(light_excel_path, index=False)

            if self.df_dark is not None:
                self.df_dark.to_excel(dark_excel_path, index=False)

            self.fig.savefig(fig_path, dpi=300, bbox_inches="tight")

            copied_files = []

            if self.file_path_light and os.path.isfile(self.file_path_light):
                raw_name = os.path.basename(self.file_path_light)
                dest = os.path.join(run_dir, f"raw_light_{raw_name}")
                try:
                    shutil.copy2(self.file_path_light, dest)
                    copied_files.append(os.path.basename(dest))
                except Exception:
                    pass

            if self.file_path_dark and os.path.isfile(self.file_path_dark):
                raw_name = os.path.basename(self.file_path_dark)
                dest = os.path.join(run_dir, f"raw_dark_{raw_name}")
                try:
                    shutil.copy2(self.file_path_dark, dest)
                    copied_files.append(os.path.basename(dest))
                except Exception:
                    pass

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)

            pdf.add_page()
            pdf.set_font("Arial", "B", 20)
            pdf.cell(0, 12, "Research Analysis Toolkit", ln=True, align="C")
            pdf.set_font("Arial", "I", 13)
            pdf.cell(0, 8, "Photoelectrochemical ABPE Report", ln=True, align="C")
            pdf.ln(10)

            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(
                0, 7,
                "This report includes FE-weighted ABPE calculation with support for "
                "separate Light/Dark files, potential-scale conversion, and n-type / p-type logic."
            )
            pdf.ln(4)

            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Run Information", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 7, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
            pdf.cell(0, 7, f"Electrode Type: {self.electrode_type.get()}", ln=True)
            pdf.cell(0, 7, f"Illumination Mode: {self.last_mode_used}", ln=True)
            pdf.cell(0, 7, f"Data Mode: {self.last_data_mode}", ln=True)
            pdf.cell(0, 7, f"Dark Source: {self.last_dark_source}", ln=True)
            pdf.cell(0, 7, f"Potential Input Scale: {self.ref_electrode.get()}", ln=True)
            pdf.cell(0, 7, f"pH: {self.ent_ph.get()}", ln=True)
            pdf.cell(0, 7, f"Area (cm^2): {self.ent_area.get()}", ln=True)
            pdf.cell(0, 7, f"Pin (mW/cm^2): {self.ent_pin.get()}", ln=True)

            pdf.ln(4)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Products and FE", ln=True)

            pdf.set_font("Arial", "", 11)
            total_fe = 0.0
            for _, ent_name, ent_fe, ent_erev in self.product_rows:
                name = ent_name.get().strip()
                fe = ent_fe.get().strip()
                erev = ent_erev.get().strip()
                if not name and not fe and not erev:
                    continue
                pdf.cell(0, 7, f"{name} | FE = {fe}% | Erev = {erev} V", ln=True)
                try:
                    total_fe += float(fe)
                except Exception:
                    pass

            pdf.cell(0, 7, f"Total FE = {total_fe:.2f}%", ln=True)

            max_eff = float(self.processed_results["ABPE_FE_weighted_percent"].max())
            pdf.ln(4)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Maximum Efficiency", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 7, f"Maximum FE-weighted efficiency: {max_eff:.4f} %", ln=True)

            pdf.ln(4)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Equations", ln=True)
            pdf.image(eq_path, x=15, w=180)

            pdf.add_page()
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, "Analysis Figure", ln=True)
            pdf.image(fig_path, x=10, y=25, w=190)

            pdf.output(pdf_path)

            self.build_html_report(run_dir, "analysis_figure.png", max_eff)

            msg_lines = [
                "All files saved successfully.",
                "",
                f"Folder: {run_dir}",
                "",
                "Saved files:",
                "- processed_data.xlsx",
                "- analysis_figure.png",
                "- equation.png",
                "- report.pdf",
                "- report.html",
            ]

            if self.df_light is not None:
                msg_lines.append("- light_loaded_data.xlsx")
            if self.df_dark is not None:
                msg_lines.append("- dark_loaded_data.xlsx")
            for x in copied_files:
                msg_lines.append(f"- {x}")

            messagebox.showinfo("Success", "\n".join(msg_lines))

        except Exception as e:
            messagebox.showerror("Export Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = TEAM_PEC_Analyzer(root)
    root.mainloop()
