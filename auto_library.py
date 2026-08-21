import json
import os
import time
from typing import Dict, List, Any, Optional

# ============================================
# Researcher Auto-Library (InnoSetup-safe)
# - Stores learned circuits + metrics persistently
# - Default location: %LOCALAPPDATA%\Researcher\
# - Reads config.json for "auto_learning" toggle
# ============================================

APP_NAME = "Researcher"
DEFAULT_FILENAME = "auto_library.json"
DEFAULT_CONFIG_FILENAME = "config.json"

def get_app_dir(app_name: str = APP_NAME) -> str:
    """
    Return a writable per-user directory for storing persistent data.
    On Windows prefer LOCALAPPDATA / APPDATA; fallback to user home.
    """
    base = os.getenv("LOCALAPPDATA") or os.getenv("APPDATA")
    if not base:
        base = os.path.expanduser("~")
    d = os.path.join(base, app_name)
    os.makedirs(d, exist_ok=True)
    return d

def _path_in_appdir(filename: str, app_name: str = APP_NAME) -> str:
    return os.path.join(get_app_dir(app_name), filename)

def debug_log(msg: str, app_name: str = APP_NAME) -> None:
    """Write debug log for installed builds (prints may be hidden)."""
    try:
        p = _path_in_appdir("auto_library_debug.log", app_name)
        with open(p, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

def is_auto_learning_enabled(app_name: str = APP_NAME,
                            config_filename: str = DEFAULT_CONFIG_FILENAME) -> bool:
    """
    Returns True if config.json contains {"auto_learning": true}.
    If config does not exist, default is True (convenient for portable/dev).
    """
    try:
        cfg_path = _path_in_appdir(config_filename, app_name)
        if not os.path.exists(cfg_path):
            return True
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return bool(cfg.get("auto_learning", True))
    except Exception:
        return True

def load_library(filename: str = DEFAULT_FILENAME, app_name: str = APP_NAME) -> List[Dict[str, Any]]:
    path = _path_in_appdir(filename, app_name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        debug_log(f"load_library error: {e}", app_name)
        return []

def save_library(items: List[Dict[str, Any]], filename: str = DEFAULT_FILENAME, app_name: str = APP_NAME) -> None:
    path = _path_in_appdir(filename, app_name)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    except Exception as e:
        debug_log(f"save_library error: {e} | path={path}", app_name)

def _norm_circuit(c: str) -> str:
    return "".join(str(c).split())

def maybe_store(
    circuit_str: str,
    metrics: Dict[str, Any],
    tags: Optional[Dict[str, Any]] = None,
    filename: str = DEFAULT_FILENAME,
    app_name: str = APP_NAME,
    *,
    min_quality_mean_rel_err: float = 0.03,
    min_quality_rel_max: float = 0.05,
) -> bool:
    """
    Store/update a circuit in the auto-library if it meets minimum quality.
    Returns True if stored/updated, False otherwise.
    """
    if not circuit_str:
        debug_log("maybe_store: empty circuit_str", app_name)
        return False

    if not is_auto_learning_enabled(app_name):
        debug_log("maybe_store: auto_learning disabled", app_name)
        return False

    tags = tags or {}

    mean_rel_err = float(metrics.get("mean_rel_err", float("inf")))
    rel_max = float(metrics.get("rel_max", float("inf")))

    debug_log(f"maybe_store called | circuit={circuit_str}", app_name)
    debug_log(f"metrics | mean_rel_err={mean_rel_err} rel_max={rel_max} "
              f"th=({min_quality_mean_rel_err},{min_quality_rel_max})", app_name)
    debug_log(f"lib_path={_path_in_appdir(filename, app_name)}", app_name)

    # Quality gate
    if not (mean_rel_err <= min_quality_mean_rel_err or rel_max <= min_quality_rel_max):
        debug_log("REJECTED by quality gate", app_name)
        return False

    items = load_library(filename, app_name)
    now = time.time()
    key = _norm_circuit(circuit_str)

    # Update existing entry if present
    for it in items:
        c = it.get("circuit_str") or it.get("circuit") or ""
        if _norm_circuit(c) == key:
            it["updated_at"] = now
            it["success_count"] = int(it.get("success_count", 0)) + 1
            it["metrics"] = dict(metrics)
            it["tags"] = dict(tags)
            save_library(items, filename, app_name)
            debug_log("UPDATED existing circuit", app_name)
            return True

    # Else append new
    items.append({
        "circuit_str": circuit_str,
        "metrics": dict(metrics),
        "tags": dict(tags),
        "created_at": now,
        "updated_at": now,
        "success_count": 1,
    })
    save_library(items, filename, app_name)
    debug_log("ADDED new circuit", app_name)
    return True

def get_suggestions(max_n: int = 60, filename: str = DEFAULT_FILENAME, app_name: str = APP_NAME) -> List[str]:
    """
    Return circuit strings sorted by best historical mean_rel_err (then rel_max, then error).
    Used as an internal library boost.
    """
    items = load_library(filename, app_name)

    best: Dict[str, Dict[str, Any]] = {}
    for it in items:
        c = it.get("circuit_str") or it.get("circuit") or ""
        if not c:
            continue
        key = _norm_circuit(c)
        m = it.get("metrics", {}) if isinstance(it.get("metrics", {}), dict) else {}
        score = (
            float(m.get("mean_rel_err", float("inf"))),
            float(m.get("rel_max", float("inf"))),
            float(m.get("error", float("inf"))),
        )
        if key not in best or score < best[key]["_score"]:
            best[key] = {"circuit": c, "_score": score}

    ordered = sorted(best.values(), key=lambda x: x["_score"])
    return [x["circuit"] for x in ordered[:max_n]]
