import os
import sys
import shutil
import ctypes
import subprocess
import hashlib
import json
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt6.QtGui import QIcon, QAction, QPixmap, QColor
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QHBoxLayout, QFrame, QMessageBox, QProgressBar,
    QListWidget, QListWidgetItem, QSplitter, QTabWidget,
    QGraphicsDropShadowEffect
)

# =========================================================
# APP CONFIG
# =========================================================
APP_NAME = "Researcher Analysis Toolkit – v1.0.0"
APP_SUBTITLE = "Scientific Software Package"
ORG_NAME = "TEAM Clean Energy Research Center"
APP_ID = "TEAM.ResearcherAnalysisToolkit.Launcher"
SUPPORTED_PYTHON = (3, 11)

def get_application_dir() -> Path:
    """Return the folder containing the installed application or source files."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


BASE_DIR = get_application_dir()
LOCAL_APPDATA = Path(os.environ.get("LOCALAPPDATA", str(BASE_DIR)))
APP_DATA_DIR = LOCAL_APPDATA / "ResearcherToolkit"
STATE_FILE = APP_DATA_DIR / "env_state.json"

VENV_DIR = APP_DATA_DIR / "venv"
REQ_FILE = BASE_DIR / "requirements.txt"
PYTHON_INSTALLER = BASE_DIR / "python-3.11.9-amd64.exe"
LOG_FILE = APP_DATA_DIR / "install_deps.log"
DEPS_MARKER = VENV_DIR / "deps_installed.ok"
ARTICLE_URL = "https://doi.org/10.1039/D6RA08618G"
GUIDE_PATH = BASE_DIR / "USER_GUIDE.md"

MODULES = {
    "(Photo)Electrochemical Analysis": {
        "Tafel Analysis": "Tafel_Analysis.py",
        "Linear Sweep Voltammetry (LSV)": "LSV_Analysis.py",
        "Electrochemical Impedance Spectroscopy(EIS)": {
            "Manual EIS": "manual_eis.py",
            "Auto EIS": "eis_auto.py",
            "EIS Plot Generator": "eis_generator.py",
        },
        "Cyclic Voltammetry (CV)": "CV_Analysis.py",
        "Mott-Schottky (MS)": "MottSchottky_Analysis.py",
        "Electrochemical Potential Converter": "electrochemical_potential_converter.py",
        "Applied Bias Photon-to-current Efficiency (ABPE)": "abpe.py",
        "Incident Photon-to-current Efficiency (IPCE)": "ipce.py",
        "Faradaic Efficiency (FE)": "fe.py",
        "Solar-to-hydrogen Efficiency (STH)": "sth.py",
    },
    "Optical Absorption and Band Structure": {
        "X-ray Photoelectron Spectroscopy (XPS)": "xps.py",
        "Time-Resolved Photoluminescence (TRPL)": "TRPL_Analysis.py",
        "Tauc Plot": "tauc.py",
    },
    "Paper Standard": {
        "Standard Paper Color Pro": "standard_paper_color_pro.py",
    }
}

# =========================================================
# HELPERS
# =========================================================
def ensure_appdata():
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_icon_path():
    candidates = [
        BASE_DIR / "ResearcherLogo.ico",
        BASE_DIR / "researcherlogo.ico",
        BASE_DIR / "logo.ico",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def get_logo_png_path():
    candidates = [
        BASE_DIR / "ResearcherLogo.png",
        BASE_DIR / "researcherlogo.png",
        BASE_DIR / "logo.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def set_windows_app_id():
    if os.name == "nt":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
        except Exception:
            pass


def find_system_python():
    """Locate a supported Python 3.11 interpreter.

    The explicit version check prevents an unrelated Python installation from
    creating an environment that is incompatible with the pinned scientific
    stack. A frozen launcher must never identify itself as Python.
    """
    candidates = []

    # When running from source, prefer the base interpreter rather than a
    # virtual-environment executable.
    if not getattr(sys, "frozen", False):
        base_python = Path(getattr(sys, "_base_executable", sys.executable))
        if base_python.exists():
            candidates.append(str(base_python))

    local_appdata = os.environ.get("LOCALAPPDATA")
    if local_appdata:
        candidates.append(str(Path(local_appdata) / "Programs" / "Python" / "Python311" / "python.exe"))

    program_files = os.environ.get("ProgramFiles")
    if program_files:
        candidates.append(str(Path(program_files) / "Python311" / "python.exe"))

    py_launcher = shutil.which("py")
    if py_launcher:
        try:
            probe = subprocess.run(
                [py_launcher, "-3.11", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip():
                candidates.append(probe.stdout.strip())
        except Exception:
            pass

    python_path = shutil.which("python")
    if python_path:
        candidates.append(python_path)

    seen = set()
    for candidate in candidates:
        normalized = os.path.normcase(os.path.abspath(candidate))
        if normalized in seen or not Path(candidate).exists():
            continue
        seen.add(normalized)
        try:
            probe = subprocess.run(
                [candidate, "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if probe.returncode == 0 and probe.stdout.strip() == "3.11":
                return candidate
        except Exception:
            continue

    return None


def get_venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def python_version(executable) -> tuple[int, int] | None:
    """Return an interpreter's major/minor version without importing packages."""
    try:
        probe = subprocess.run(
            [str(executable), "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        parts = probe.stdout.strip().split()
        if probe.returncode == 0 and len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None


def is_python_installed():
    return find_system_python() is not None


def is_venv_ready():
    venv_python = get_venv_python()
    venv_cfg = VENV_DIR / "pyvenv.cfg"
    return (
        venv_python.exists()
        and venv_cfg.exists()
        and python_version(venv_python) == SUPPORTED_PYTHON
    )


def are_deps_installed():
    return DEPS_MARKER.exists()


def open_path(path: Path):
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


def run_subprocess_live(command, log_callback=None, cwd=None):
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(cwd) if cwd else None,
            shell=False,
        )
    except Exception as e:
        if log_callback:
            log_callback(f"[ERROR] Failed to start process: {e}")
        return 1

    if process.stdout:
        for line in process.stdout:
            if log_callback:
                log_callback(line.rstrip())

    process.wait()
    return process.returncode


def add_soft_shadow(widget, blur=14, dx=0, dy=3, alpha=12):
    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur)
    shadow.setOffset(dx, dy)
    shadow.setColor(QColor(15, 39, 69, alpha))
    widget.setGraphicsEffect(shadow)


def file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_env_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_env_state(state: dict):
    ensure_appdata()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# =========================================================
# WORKERS
# =========================================================
class InitWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    done_signal = pyqtSignal(bool, str)

    def __init__(self, force_repair=False):
        super().__init__()
        self.force_repair = force_repair

    def log(self, text):
        self.log_signal.emit(text)

    def run(self):
        try:
            ensure_appdata()
            self.progress_signal.emit(5)
            self.log("=" * 70)
            self.log("Initializing environment...")
            self.log(f"Base directory    : {BASE_DIR}")
            self.log(f"App data directory: {APP_DATA_DIR}")
            self.log(f"Repair mode       : {'Yes' if self.force_repair else 'No'}")

            system_python = find_system_python()
            self.progress_signal.emit(15)

            if not system_python:
                self.done_signal.emit(
                    False,
                    "Python runtime was not found. Please reinstall the software."
                )
                return

            self.log("Python runtime detected.")

            # Repair: remove old environment first
            if self.force_repair:
                if VENV_DIR.exists():
                    self.log("Removing previous virtual environment...")
                    shutil.rmtree(VENV_DIR, ignore_errors=True)

                if STATE_FILE.exists():
                    try:
                        STATE_FILE.unlink()
                    except Exception:
                        pass

                if DEPS_MARKER.exists():
                    try:
                        DEPS_MARKER.unlink()
                    except Exception:
                        pass

            # Recreate environments produced by another Python release. This
            # prevents binary-extension conflicts such as python311.dll being
            # loaded by Python 3.13 after an upgrade or an older installation.
            if VENV_DIR.exists() and not is_venv_ready():
                self.log("Removing incompatible virtual environment...")
                shutil.rmtree(VENV_DIR, ignore_errors=True)

            # Create venv if missing
            self.progress_signal.emit(35)
            if not is_venv_ready():
                self.log("Creating virtual environment...")
                rc = run_subprocess_live(
                    [system_python, "-m", "venv", str(VENV_DIR)],
                    log_callback=self.log,
                    cwd=BASE_DIR,
                )
                if rc != 0:
                    self.done_signal.emit(False, "Failed to create virtual environment.")
                    return
                self.log("Virtual environment created successfully.")
            else:
                self.log("Virtual environment already exists.")

            venv_python = get_venv_python()

            # Install requirements only when needed
            self.progress_signal.emit(60)

            if not REQ_FILE.exists():
                self.done_signal.emit(False, "requirements.txt not found.")
                return

            current_req_hash = file_sha256(REQ_FILE)
            state = load_env_state()
            saved_req_hash = state.get("requirements_hash", "")

            needs_install = False

            if self.force_repair:
                needs_install = True
                self.log("Repair mode enabled. Dependencies will be reinstalled.")
            elif not DEPS_MARKER.exists():
                needs_install = True
                self.log("First-time dependency installation required.")
            elif saved_req_hash != current_req_hash:
                needs_install = True
                self.log("requirements.txt has changed. Reinstalling dependencies...")
            else:
                self.log("Dependencies already installed. Skipping...")

            if needs_install:
                with open(LOG_FILE, "w", encoding="utf-8") as f:
                    f.write("===== Researcher dependency install log =====\n")
                    f.write(f"Base: {BASE_DIR}\n")
                    f.write(f"Venv: {VENV_DIR}\n")
                    f.write(f"Req : {REQ_FILE}\n")
                    f.write(f"Hash: {current_req_hash}\n")
                    f.write("=" * 60 + "\n")

                self.log("Installing required libraries...")

                # capture output into log file too
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    process = subprocess.Popen(
                        [str(venv_python), "-m", "pip", "install", "-r", str(REQ_FILE)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                        cwd=str(BASE_DIR),
                        shell=False,
                    )

                    if process.stdout:
                        for line in process.stdout:
                            line = line.rstrip()
                            self.log(line)
                            f.write(line + "\n")

                    process.wait()
                    rc = process.returncode

                if rc != 0:
                    self.done_signal.emit(False, "Failed to install required libraries.")
                    return

                DEPS_MARKER.write_text("ok", encoding="utf-8")
                save_env_state({
                    "requirements_hash": current_req_hash
                })
                self.log("Dependencies installed successfully.")

            self.progress_signal.emit(100)
            self.log("Environment is ready.")
            self.done_signal.emit(True, "Environment initialized successfully.")

        except Exception as e:
            self.done_signal.emit(False, f"Unexpected error: {e}")


class ScriptRunnerWorker(QThread):
    log_signal = pyqtSignal(str)
    done_signal = pyqtSignal(int, str)

    def __init__(self, script_path):
        super().__init__()
        self.script_path = Path(script_path)

    def log(self, text):
        self.log_signal.emit(text)

    def run(self):
        try:
            if not self.script_path.exists():
                self.done_signal.emit(1, "Selected module file was not found.")
                return

            venv_python = get_venv_python()
            if not is_venv_ready():
                self.done_signal.emit(1, "Virtual environment is not ready.")
                return

            self.log("=" * 70)
            self.log("Starting module execution...")

            rc = run_subprocess_live(
                [str(venv_python), str(self.script_path)],
                log_callback=self.log,
                cwd=BASE_DIR,
            )

            if rc == 0:
                self.done_signal.emit(0, "Module finished successfully.")
            else:
                self.done_signal.emit(rc, f"Module exited with code {rc}.")
        except Exception as e:
            self.done_signal.emit(1, f"Execution error: {e}")


# =========================================================
# UI COMPONENTS
# =========================================================
class StatusCard(QFrame):
    def __init__(self, title, value="Unknown"):
        super().__init__()
        self.setObjectName("StatusCard")

        self.title_label = QLabel(title)
        self.title_label.setObjectName("StatusCardTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("StatusCardValue")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)

    def set_value(self, text):
        self.value_label.setText(text)


class ModuleListPane(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("ModulePane")

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 12, 12, 12)
        self.main_layout.setSpacing(10)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("ModulePaneTitle")
        self.main_layout.addWidget(self.title_label)

        self.main_list = QListWidget()
        self.main_list.setObjectName("ModuleList")

        self.subgroup_frame = QFrame()
        self.subgroup_frame.setObjectName("SubgroupFrame")
        subgroup_layout = QVBoxLayout(self.subgroup_frame)
        subgroup_layout.setContentsMargins(10, 10, 10, 10)
        subgroup_layout.setSpacing(8)

        self.subgroup_title = QLabel("Submodules")
        self.subgroup_title.setObjectName("SubgroupTitle")

        self.sub_list = QListWidget()
        self.sub_list.setObjectName("ModuleList")

        subgroup_layout.addWidget(self.subgroup_title)
        subgroup_layout.addWidget(self.sub_list)

        self.subgroup_frame.setVisible(False)

        self.main_layout.addWidget(self.main_list)
        self.main_layout.addWidget(self.subgroup_frame)

    def set_subgroup_visible(self, visible: bool):
        self.subgroup_frame.setVisible(visible)


class HeaderBar(QFrame):
    def __init__(self, logo_png_path, icon_path):
        super().__init__()
        self.setObjectName("HeaderFrame")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.setSpacing(14)

        logo_wrap = QFrame()
        logo_wrap.setObjectName("HeaderLogoWrap")
        logo_wrap.setFixedSize(52, 52)

        logo_layout = QVBoxLayout(logo_wrap)
        logo_layout.setContentsMargins(0, 0, 0, 0)

        icon_label = QLabel()
        icon_label.setFixedSize(36, 36)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("background: transparent; border: none;")

        pixmap = QPixmap()
        if logo_png_path and logo_png_path.exists():
            pixmap = QPixmap(str(logo_png_path))
        elif icon_path and icon_path.exists():
            pixmap = QIcon(str(icon_path)).pixmap(QSize(256, 256))

        if not pixmap.isNull():
            pixmap = pixmap.scaled(
                36, 36,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            icon_label.setPixmap(pixmap)

        logo_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)

        text_col = QVBoxLayout()
        text_col.setSpacing(1)

        title = QLabel(APP_NAME)
        title.setObjectName("HeaderTitle")

        subtitle = QLabel(APP_SUBTITLE)
        subtitle.setObjectName("HeaderSubtitle")

        text_col.addWidget(title)
        text_col.addWidget(subtitle)

        layout.addWidget(logo_wrap, alignment=Qt.AlignmentFlag.AlignVCenter)
        layout.addLayout(text_col, stretch=1)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)
        buttons_layout.setContentsMargins(0, 0, 0, 0)

        article_btn = QPushButton("Article")
        article_btn.setObjectName("HeaderActionButton")
        article_btn.setFixedHeight(34)
        article_btn.setEnabled(bool(ARTICLE_URL))
        article_btn.setToolTip(
            "Open the associated article" if ARTICLE_URL
            else "The article link will be activated after publication."
        )
        if ARTICLE_URL:
            article_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(ARTICLE_URL)))

        guide_btn = QPushButton("Guide")
        guide_btn.setObjectName("HeaderActionButton")
        guide_btn.setFixedHeight(34)
        guide_btn.setEnabled(GUIDE_PATH.exists())
        guide_btn.setToolTip("Open the local user guide")
        if GUIDE_PATH.exists():
            guide_btn.clicked.connect(
                lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(GUIDE_PATH)))
            )

        buttons_layout.addWidget(article_btn)
        buttons_layout.addWidget(guide_btn)

        layout.addLayout(buttons_layout)


# =========================================================
# MAIN WINDOW
# =========================================================
class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.selected_script = None
        self.selected_name = None
        self.init_worker = None
        self.runner_worker = None
        self.last_log_lines = []

        self.icon_path = get_icon_path()
        self.logo_png_path = get_logo_png_path()

        self.tab_panes = {}

        self.setup_window()
        self.build_ui()
        self.build_menu()
        self.apply_styles()
        self.populate_tabs()
        self.refresh_status()

    def setup_window(self):
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 900)
        self.setMinimumSize(1260, 780)

        if self.icon_path:
            icon = QIcon(str(self.icon_path))
            self.setWindowIcon(icon)
            QApplication.instance().setWindowIcon(icon)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        self.header = HeaderBar(self.logo_png_path, self.icon_path)
        root.addWidget(self.header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.center_panel = self.build_center_panel()
        self.right_panel = self.build_right_panel()

        splitter.addWidget(self.center_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([1050, 350])

        root.addWidget(splitter, stretch=1)
        root.addWidget(self.build_bottom_bar())

        add_soft_shadow(self.header, blur=14, dy=3, alpha=12)
        add_soft_shadow(self.center_panel, blur=14, dy=3, alpha=12)
        add_soft_shadow(self.right_panel, blur=14, dy=3, alpha=12)

    def build_center_panel(self):
        panel = QFrame()
        panel.setObjectName("Panel")

        outer = QVBoxLayout(panel)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.setObjectName("MainTabs")
        self.tabs.currentChanged.connect(self.on_tab_changed)
        outer.addWidget(self.tabs)

        return panel

    def build_right_panel(self):
        panel = QFrame()
        panel.setObjectName("Panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("System")
        title.setObjectName("PanelTitle")

        self.python_card = StatusCard("Python", "Checking...")
        self.venv_card = StatusCard("Virtual Environment", "Checking...")
        self.deps_card = StatusCard("Dependencies", "Checking...")
        self.selected_card = StatusCard("Selected Module", "None")
        self.file_card = StatusCard("Selected File Status", "None")

        log_title = QLabel("Live Log")
        log_title.setObjectName("SubPanelTitle")

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setObjectName("LogBox")

        layout.addWidget(title)
        layout.addWidget(self.python_card)
        layout.addWidget(self.venv_card)
        layout.addWidget(self.deps_card)
        layout.addWidget(self.selected_card)
        layout.addWidget(self.file_card)
        layout.addWidget(log_title)
        layout.addWidget(self.log_box, stretch=1)

        return panel

    def build_bottom_bar(self):
        frame = QFrame()
        frame.setObjectName("BottomBar")

        layout = QHBoxLayout(frame)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setFixedHeight(22)

        self.btn_refresh = QPushButton("Refresh")
        self.btn_init = QPushButton("Initialize Environment")
        self.btn_repair = QPushButton("Repair")
        self.btn_open_log = QPushButton("Install Log")
        self.btn_run = QPushButton("Run")
        self.btn_exit = QPushButton("Exit")

        self.btn_run.setObjectName("RunButton")
        self.btn_exit.setObjectName("ExitButton")
        self.btn_run.setEnabled(False)

        self.btn_refresh.clicked.connect(self.refresh_status)
        self.btn_init.clicked.connect(self.initialize_environment)
        self.btn_repair.clicked.connect(self.repair_environment)
        self.btn_open_log.clicked.connect(self.open_install_log)
        self.btn_run.clicked.connect(self.run_selected_module)
        self.btn_exit.clicked.connect(self.close)

        layout.addWidget(self.progress, stretch=1)
        layout.addWidget(self.btn_refresh)
        layout.addWidget(self.btn_init)
        layout.addWidget(self.btn_repair)
        layout.addWidget(self.btn_open_log)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.btn_exit)

        add_soft_shadow(frame, blur=12, dy=3, alpha=10)
        return frame

    def build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        tools_menu = menubar.addMenu("Tools")
        help_menu = menubar.addMenu("Help")

        act_open_log = QAction("Open Install Log", self)
        act_open_log.triggered.connect(self.open_install_log)
        file_menu.addAction(act_open_log)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        act_refresh = QAction("Refresh Status", self)
        act_refresh.triggered.connect(self.refresh_status)
        tools_menu.addAction(act_refresh)

        act_init = QAction("Initialize Environment", self)
        act_init.triggered.connect(self.initialize_environment)
        tools_menu.addAction(act_init)

        act_repair = QAction("Repair Environment", self)
        act_repair.triggered.connect(self.repair_environment)
        tools_menu.addAction(act_repair)

        act_about = QAction("About", self)
        act_about.triggered.connect(self.show_about)
        help_menu.addAction(act_about)

    def populate_tabs(self):
        self.tabs.clear()
        self.tab_panes.clear()

        for category_name, category_data in MODULES.items():
            pane = ModuleListPane(category_name)
            self.tab_panes[category_name] = pane

            for module_name, module_value in category_data.items():
                item = QListWidgetItem(module_name)
                if isinstance(module_value, dict):
                    item.setData(Qt.ItemDataRole.UserRole, None)
                    item.setData(Qt.ItemDataRole.UserRole + 1, module_value)
                else:
                    full_path = BASE_DIR / module_value
                    item.setData(Qt.ItemDataRole.UserRole, str(full_path))
                    item.setData(Qt.ItemDataRole.UserRole + 1, None)

                pane.main_list.addItem(item)

            pane.main_list.itemClicked.connect(
                lambda item, cat=category_name: self.on_main_module_clicked(cat, item)
            )
            pane.main_list.itemDoubleClicked.connect(
                lambda item, cat=category_name: self.on_main_module_double_clicked(cat, item)
            )

            pane.sub_list.itemClicked.connect(
                lambda item, cat=category_name: self.on_submodule_clicked(cat, item)
            )
            pane.sub_list.itemDoubleClicked.connect(
                lambda item, cat=category_name: self.on_submodule_double_clicked(cat, item)
            )

            self.tabs.addTab(pane, category_name)

        if self.tabs.count() > 0:
            self.tabs.setCurrentIndex(0)
            self.on_tab_changed(0)

    def current_pane(self):
        return self.tabs.currentWidget()

    def on_tab_changed(self, index):
        self.selected_script = None
        self.selected_name = None
        self.btn_run.setEnabled(False)
        self.selected_card.set_value("None")
        self.file_card.set_value("None")

        pane = self.current_pane()
        if pane:
            pane.sub_list.clear()
            pane.set_subgroup_visible(False)

    def on_main_module_clicked(self, category_name, item):
        pane = self.tab_panes[category_name]
        subgroup = item.data(Qt.ItemDataRole.UserRole + 1)
        script = item.data(Qt.ItemDataRole.UserRole)

        if subgroup:
            pane.sub_list.clear()
            for sub_name, sub_file in subgroup.items():
                child_item = QListWidgetItem(sub_name)
                full_path = BASE_DIR / sub_file
                child_item.setData(Qt.ItemDataRole.UserRole, str(full_path))
                pane.sub_list.addItem(child_item)

            pane.set_subgroup_visible(True)
            self.selected_script = None
            self.selected_name = item.text()
            self.selected_card.set_value(f"{item.text()} (submenu)")
            self.file_card.set_value("Group only")
            self.btn_run.setEnabled(False)
        else:
            pane.set_subgroup_visible(False)
            self.select_script(item.text(), script)

    def on_main_module_double_clicked(self, category_name, item):
        script = item.data(Qt.ItemDataRole.UserRole)
        subgroup = item.data(Qt.ItemDataRole.UserRole + 1)
        if script and not subgroup:
            self.select_script(item.text(), script)
            self.run_selected_module()

    def on_submodule_clicked(self, category_name, item):
        script = item.data(Qt.ItemDataRole.UserRole)
        self.select_script(item.text(), script)

    def on_submodule_double_clicked(self, category_name, item):
        script = item.data(Qt.ItemDataRole.UserRole)
        self.select_script(item.text(), script)
        self.run_selected_module()

    def select_script(self, name, script):
        self.selected_script = script
        self.selected_name = name
        exists = Path(script).exists() if script else False

        self.selected_card.set_value(name)
        self.file_card.set_value("Available" if exists else "Missing")
        self.btn_run.setEnabled(exists)

    def append_log(self, text):
        self.log_box.append(text)
        self.last_log_lines.append(text)
        self.last_log_lines = self.last_log_lines[-12:]

    def refresh_status(self):
        ensure_appdata()

        py = find_system_python()
        self.python_card.set_value("Installed" if py else "Not Found")
        if is_venv_ready():
            self.venv_card.set_value("Ready (Python 3.11)")
        elif VENV_DIR.exists():
            self.venv_card.set_value("Incompatible")
        else:
            self.venv_card.set_value("Missing")
        self.deps_card.set_value("Installed" if are_deps_installed() else "Not Installed")

        if self.selected_script:
            exists = Path(self.selected_script).exists()
            self.file_card.set_value("Available" if exists else "Missing")

        self.append_log("Status refreshed.")

    def initialize_environment(self):
        if not is_python_installed():
            self.ask_install_python()
            return
        self.start_init_worker(force_repair=False)

    def repair_environment(self):
        if not is_python_installed():
            self.ask_install_python()
            return

        reply = QMessageBox.question(
            self,
            "Repair Environment",
            "This will recreate the virtual environment and reinstall dependencies.\n\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.start_init_worker(force_repair=True)

    def start_init_worker(self, force_repair=False):
        self.progress.setValue(0)
        self.btn_init.setEnabled(False)
        self.btn_repair.setEnabled(False)
        self.btn_run.setEnabled(False)

        self.append_log("=" * 70)
        self.append_log("Starting environment workflow...")

        self.init_worker = InitWorker(force_repair=force_repair)
        self.init_worker.log_signal.connect(self.append_log)
        self.init_worker.progress_signal.connect(self.progress.setValue)
        self.init_worker.done_signal.connect(self.on_init_finished)
        self.init_worker.start()

    def on_init_finished(self, success, message):
        self.btn_init.setEnabled(True)
        self.btn_repair.setEnabled(True)

        if self.selected_script and Path(self.selected_script).exists():
            self.btn_run.setEnabled(True)

        self.refresh_status()
        self.append_log(message)

        if success:
            QMessageBox.information(self, "Initialization", message)
        else:
            QMessageBox.warning(self, "Initialization Failed", message)

    def ask_install_python(self):
        if not PYTHON_INSTALLER.exists():
            QMessageBox.critical(
                self,
                "Python Not Found",
                "Python was not found in PATH and the Python installer file was not found beside launcher.py.",
            )
            self.append_log("Python not found and installer file is missing.")
            return

        reply = QMessageBox.question(
            self,
            "Python Required",
            "Python is not installed or not found in PATH.\n\nDo you want to install it now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.append_log("Launching Python installer...")
            try:
                subprocess.Popen(
                    [
                        str(PYTHON_INSTALLER),
                        "/passive",
                        "InstallAllUsers=0",
                        "PrependPath=1",
                        "Include_pip=1",
                        "Include_launcher=1",
                        "Include_test=0",
                    ],
                    cwd=str(BASE_DIR),
                )
                QMessageBox.information(
                    self,
                    "Installer Started",
                    "Python installer has been launched.\nAfter installation, reopen the launcher or press Refresh.",
                )
            except Exception as e:
                QMessageBox.critical(self, "Installer Error", f"Failed to launch installer:\n{e}")

    def run_selected_module(self):
        if not self.selected_script:
            QMessageBox.warning(self, "No Module Selected", "Please select a module first.")
            return

        script_path = Path(self.selected_script)
        if not script_path.exists():
            QMessageBox.critical(self, "Missing File", f"File not found:\n{script_path.name}")
            self.append_log(f"[ERROR] Missing module file: {script_path}")
            return

        if not is_venv_ready():
            QMessageBox.warning(
                self,
                "Environment Not Ready",
                "The virtual environment is not ready.\nPlease click 'Initialize' first.",
            )
            return

        self.btn_run.setEnabled(False)
        self.btn_init.setEnabled(False)
        self.btn_repair.setEnabled(False)

        self.append_log("=" * 70)
        self.append_log("Preparing to execute selected module...")
        self.runner_worker = ScriptRunnerWorker(script_path)
        self.runner_worker.log_signal.connect(self.append_log)
        self.runner_worker.done_signal.connect(self.on_run_finished)
        self.runner_worker.start()

    def on_run_finished(self, return_code, message):
        self.btn_init.setEnabled(True)
        self.btn_repair.setEnabled(True)
        self.btn_run.setEnabled(bool(self.selected_script and Path(self.selected_script).exists()))

        self.append_log(message)

        if return_code == 0:
            QMessageBox.information(self, "Execution Complete", message)
        else:
            error_preview = "\n".join(self.last_log_lines[-8:])
            QMessageBox.critical(
                self,
                "Module Execution Failed",
                f"{message}\n\nRecent log output:\n{error_preview}",
            )

    def open_install_log(self):
        if not LOG_FILE.exists():
            QMessageBox.information(self, "Install Log", "Install log file does not exist yet.")
            return
        open_path(LOG_FILE)

    def show_about(self):
        QMessageBox.information(
            self,
            "About",
            f"{APP_NAME}\n\nA scientific software platform developed based on the associated research article, providing an integrated environment for reproducible electrochemical and optical analysis, computational tools, and structured scientific workflows.",
        )

    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #f6f8fc;
                color: #17233a;
                font-family: "Segoe UI";
                font-size: 10pt;
            }

            QMenuBar {
                background: #ffffff;
                border-bottom: 1px solid #e3e9f2;
            }

            QMenuBar::item {
                padding: 6px 12px;
                background: transparent;
                border-radius: 6px;
            }

            QMenuBar::item:selected {
                background: #eef4ff;
            }

            QMenu {
                background: #ffffff;
                border: 1px solid #d9e2ef;
                padding: 6px;
            }

            QMenu::item {
                padding: 8px 18px;
                border-radius: 6px;
            }

            QMenu::item:selected {
                background: #eef4ff;
            }

            #HeaderFrame {
                background: #ffffff;
                border: 1px solid #dfe7f1;
                border-radius: 16px;
            }

            #HeaderLogoWrap {
                background: #f7f9fc;
                border: 1px solid #e3e9f2;
                border-radius: 12px;
            }

            #HeaderTitle {
                color: #0f2745;
                font-size: 18pt;
                font-weight: 800;
                background: transparent;
                border: none;
            }

            #HeaderSubtitle {
                color: #64768c;
                font-size: 10pt;
                font-weight: 500;
                background: transparent;
                border: none;
            }

            #HeaderBadge {
                background: #f4f7fb;
                color: #51657f;
                border: 1px solid #e3e9f2;
                border-radius: 10px;
                padding: 7px 12px;
                font-size: 9pt;
                font-weight: 700;
            }

            #Panel {
                background: #ffffff;
                border: 1px solid #dfe7f1;
                border-radius: 16px;
            }

            #PanelTitle {
                font-size: 13.5pt;
                font-weight: 800;
                color: #132744;
            }

            #SubPanelTitle {
                font-size: 10.5pt;
                font-weight: 700;
                color: #274466;
            }

            #InfoLabel {
                background: #f8fbff;
                border: 1px solid #dfe8f5;
                border-radius: 12px;
                padding: 12px;
                color: #61758f;
            }

            QTabWidget#MainTabs::pane {
                border: 1px solid #e1e8f2;
                border-radius: 12px;
                top: -1px;
                background: #fbfcff;
            }

            QTabBar::tab {
                background: #f4f7fb;
                color: #536981;
                border: 1px solid #e1e8f2;
                border-bottom: none;
                padding: 10px 16px;
                min-width: 150px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                margin-right: 4px;
                font-weight: 700;
            }

            QTabBar::tab:selected {
                background: #ffffff;
                color: #113764;
            }

            QTabBar::tab:hover:!selected {
                background: #eef4ff;
            }

            #ModulePane {
                background: transparent;
                border: none;
            }

            #ModulePaneTitle {
                font-size: 12.5pt;
                font-weight: 800;
                color: #15365d;
            }

            QListWidget#ModuleList {
                background: #fbfcff;
                border: 1px solid #e1e8f2;
                border-radius: 12px;
                padding: 6px;
            }

            QListWidget#ModuleList::item {
                padding: 12px 12px;
                border-radius: 10px;
                margin: 3px 0px;
            }

            QListWidget#ModuleList::item:selected {
                background: #e8f0ff;
                color: #0f3f7a;
                font-weight: 700;
            }

            #SubgroupFrame {
                background: #f8fbff;
                border: 1px solid #dfe8f5;
                border-radius: 12px;
            }

            #SubgroupTitle {
                color: #2a4b70;
                font-size: 10.5pt;
                font-weight: 700;
            }

            #SearchBox {
                background: #fbfdff;
                border: 1px solid #d8e2ef;
                border-radius: 10px;
                padding: 9px 12px;
                min-width: 250px;
            }

            #StatusCard {
                background: #fbfcff;
                border: 1px solid #e1e8f2;
                border-radius: 12px;
            }

            #StatusCardTitle {
                font-size: 9.2pt;
                font-weight: 700;
                color: #718197;
            }

            #StatusCardValue {
                font-size: 10.5pt;
                font-weight: 700;
                color: #10233d;
            }

            #LogBox {
                background: #0d1624;
                color: #e8eef8;
                border: 1px solid #243650;
                border-radius: 12px;
                padding: 8px;
                font-family: Consolas, "Courier New";
                font-size: 9.2pt;
            }

            #BottomBar {
                background: #ffffff;
                border: 1px solid #dfe7f1;
                border-radius: 14px;
            }

            QPushButton {
                background: #eef4ff;
                color: #17385e;
                border: 1px solid #d9e5f8;
                border-radius: 10px;
                padding: 8px 13px;
                font-weight: 700;
                min-height: 36px;
            }

            QPushButton:hover {
                background: #e2eeff;
            }

            QPushButton:pressed {
                background: #d6e7ff;
            }

            QPushButton:disabled {
                background: #eef2f7;
                color: #96a2b3;
                border: 1px solid #d9e0ea;
            }

            QPushButton#RunButton {
                background: #1b66d1;
                color: white;
                border: 1px solid #1b66d1;
            }

            QPushButton#RunButton:hover {
                background: #1459ba;
            }

            QPushButton#ExitButton {
                background: #fff5f5;
                color: #a33a3a;
                border: 1px solid #f0d4d4;
            }

            QPushButton#ExitButton:hover {
                background: #ffeceb;
            }

            QProgressBar {
                background: #f2f5fa;
                border: 1px solid #d2dbe8;
                border-radius: 9px;
                text-align: center;
                min-height: 18px;
            }

            QProgressBar::chunk {
                background: #62a5ff;
                border-radius: 7px;
            }

            #HeaderLinksBox {
                background: #f8fbff;
                border: 1px solid #dfe7f1;
                border-radius: 12px;
            }

            QPushButton#HeaderLinkButton {
                background: #ffffff;
                color: #14385d;
                border: 1px solid #d7e3f2;
                border-radius: 10px;
                padding: 7px 12px;
                font-weight: 700;
                min-height: 30px;
            }

            QPushButton#HeaderLinkButton:hover {
                background: #eef5ff;
            }

            QPushButton#HeaderLinkButton:pressed {
                background: #dfeeff;
            }

            QPushButton#HeaderActionButton {
                background: #f4f7fb;
                color: #17385e;
                border: 1px solid #d8e2ef;
                border-radius: 10px;
                padding: 6px 14px;
                font-weight: 700;
                min-width: 90px;
            }

            QPushButton#HeaderActionButton:hover {
                background: #e8f0ff;
            }

            QPushButton#HeaderActionButton:pressed {
                background: #dbe9ff;
            }
        """)


# =========================================================
# MAIN
# =========================================================
def main():
    set_windows_app_id()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    app.setStyle("Fusion")

    icon_path = get_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(str(icon_path)))

    window = LauncherWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
