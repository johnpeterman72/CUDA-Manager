"""
PyTorch/CUDA Version Manager for SwarmUI's ComfyUI Backend
Manages packages in the embedded Python at dlbackend/comfy/python_embeded/
Run via manage-pytorch.bat
"""

import http.server
import json
import subprocess
import sys
import threading
import os
from urllib.parse import urlparse, parse_qs
from collections import deque
import re

PORT = 9090
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.join(SCRIPT_DIR, "dlbackend", "comfy", "python_embeded")
COMFYUI_DIR = os.path.join(SCRIPT_DIR, "dlbackend", "comfy", "ComfyUI")

# --- State ---
install_lock = threading.Lock()
install_running = False
install_log = deque(maxlen=5000)
install_done = threading.Event()

# Packages that are tied to the CUDA/PyTorch index
CUDA_PACKAGES = [
    {"id": "torch",       "name": "PyTorch",      "pip": "torch",       "required": True},
    {"id": "torchvision", "name": "TorchVision",   "pip": "torchvision", "required": True},
    {"id": "torchaudio",  "name": "TorchAudio",    "pip": "torchaudio",  "required": True},
    {"id": "torchsde",    "name": "TorchSDE",      "pip": "torchsde",    "required": True},
    {"id": "xformers",    "name": "xFormers",       "pip": "xformers",    "required": False},
]

# CUDA Toolkit versions available via winget (Nvidia.CUDA)
CUDA_TOOLKIT_VERSIONS = [
    {"label": "Latest (winget default)", "version": None,     "cu_tag": None},
    {"label": "CUDA 12.8  →  cu128",     "version": "12.8.0", "cu_tag": "cu128"},
    {"label": "CUDA 12.6  →  cu126",     "version": "12.6.3", "cu_tag": "cu126"},
    {"label": "CUDA 12.4  →  cu124",     "version": "12.4.1", "cu_tag": "cu124"},
    {"label": "CUDA 11.8  →  cu118",     "version": "11.8.0", "cu_tag": "cu118"},
]


def get_environment_info():
    """Return paths and environment details for the embedded Python."""
    python_exe = sys.executable
    # Prefer the embedded site-packages over user-level paths
    embedded_sp = os.path.join(os.path.dirname(python_exe), "Lib", "site-packages")
    if os.path.isdir(embedded_sp):
        site_packages = embedded_sp
    else:
        site_packages = None
        for p in sys.path:
            if "site-packages" in p and os.path.isdir(p):
                site_packages = p
                break
    return {
        "python_exe": python_exe,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "python_dir": os.path.dirname(python_exe),
        "site_packages": site_packages or "unknown",
        "comfyui_dir": COMFYUI_DIR if os.path.isdir(COMFYUI_DIR) else "not found",
        "script_dir": SCRIPT_DIR,
    }


def get_package_status():
    """Get install status for each CUDA-related package."""
    packages = []
    for pkg in CUDA_PACKAGES:
        entry = {
            "id": pkg["id"],
            "name": pkg["name"],
            "pip_name": pkg["pip"],
            "required": pkg["required"],
            "installed": False,
            "version": None,
        }
        try:
            mod = __import__(pkg["id"])
            entry["installed"] = True
            entry["version"] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
        packages.append(entry)
    # Extra CUDA info from torch
    cuda_info = {"cuda_version": None, "cudnn": None, "cuda_available": False}
    try:
        import torch
        cuda_info["cuda_version"] = torch.version.cuda
        cuda_info["cuda_available"] = torch.cuda.is_available()
        if torch.backends.cudnn.is_available():
            cuda_info["cudnn"] = str(torch.backends.cudnn.version())
    except Exception:
        pass
    return {"packages": packages, "cuda": cuda_info}


def get_gpu_info():
    """Get NVIDIA GPU info."""
    gpus = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                })
            return {"available": True, "devices": gpus, "driver": ""}
    except Exception:
        pass
    # Fallback: nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            driver = ""
            for i, line in enumerate(result.stdout.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    gpus.append({
                        "index": i,
                        "name": parts[0],
                        "total_memory_gb": round(float(parts[1]) / 1024, 2),
                        "compute_capability": "",
                    })
                    driver = parts[2]
            return {"available": True, "devices": gpus, "driver": driver}
    except Exception:
        pass
    return {"available": False, "devices": [], "driver": ""}


def get_cuda_toolkit_info():
    """Detect NVIDIA CUDA Toolkit installation via nvcc / CUDA_PATH."""
    info = {"installed": False, "version": None, "nvcc_path": None, "cuda_path": None}
    cuda_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
    if cuda_path:
        info["cuda_path"] = cuda_path
        nvcc = os.path.join(cuda_path, "bin", "nvcc.exe")
        if os.path.isfile(nvcc):
            info["nvcc_path"] = nvcc
    nvcc_cmd = info["nvcc_path"] or "nvcc"
    try:
        result = subprocess.run(
            [nvcc_cmd, "--version"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            m = re.search(r"release (\d+\.\d+)", result.stdout)
            if m:
                info["installed"] = True
                info["version"] = m.group(1)
    except Exception:
        pass
    return info


def build_pip_command(channel, cuda, packages, versions=None, force=False):
    """Build the pip install command for selected packages."""
    versions = versions or {}
    python = sys.executable
    cmd = [python, "-s", "-m", "pip", "install", "--upgrade"]
    if force:
        cmd.append("--force-reinstall")
    if channel == "nightly":
        cmd.append("--pre")
    for pkg_id in packages:
        ver = versions.get(pkg_id)
        if ver:
            cmd.append(f"{pkg_id}=={ver}")
        else:
            cmd.append(pkg_id)
    # Index URL
    if cuda == "cpu":
        index = "https://download.pytorch.org/whl/cpu"
        if channel == "nightly":
            index = "https://download.pytorch.org/whl/nightly/cpu"
    else:
        base = "https://download.pytorch.org/whl"
        if channel == "nightly":
            base += "/nightly"
        index = f"{base}/{cuda}"
    cmd.extend(["--index-url", index])
    return cmd


def run_install(channel, cuda, packages, versions=None, force=False):
    """Run pip install in background thread."""
    global install_running
    install_log.clear()
    install_done.clear()
    cmd = build_pip_command(channel, cuda, packages, versions, force=force)
    cmd_display = " ".join(cmd)
    # Show a cleaner display with relative python path
    display = cmd_display.replace(sys.executable, "python")
    install_log.append(f"$ {display}\n")
    install_log.append(f"Environment: {os.path.dirname(sys.executable)}\n")
    install_log.append("--- Starting installation ---\n\n")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace"
        )
        for line in process.stdout:
            install_log.append(line)
        process.wait()
        if process.returncode == 0:
            install_log.append("\n--- Installation completed successfully! ---\n")
            install_log.append("Click 'Refresh Status' to see updated versions.\n")
        else:
            install_log.append(f"\n--- Installation failed (exit code {process.returncode}) ---\n")
    except Exception as e:
        install_log.append(f"\n--- Error: {e} ---\n")
    finally:
        install_running = False
        install_done.set()


def run_uninstall(packages):
    """Uninstall selected packages."""
    global install_running
    install_log.clear()
    install_done.clear()
    cmd = [sys.executable, "-s", "-m", "pip", "uninstall", "-y"] + packages
    display = " ".join(cmd).replace(sys.executable, "python")
    install_log.append(f"$ {display}\n")
    install_log.append("--- Starting uninstall ---\n\n")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace"
        )
        for line in process.stdout:
            install_log.append(line)
        process.wait()
        if process.returncode == 0:
            install_log.append("\n--- Uninstall completed successfully! ---\n")
        else:
            install_log.append(f"\n--- Uninstall failed (exit code {process.returncode}) ---\n")
    except Exception as e:
        install_log.append(f"\n--- Error: {e} ---\n")
    finally:
        install_running = False
        install_done.set()


def run_cuda_toolkit_install(version):
    """Install NVIDIA CUDA Toolkit via winget in a background thread."""
    global install_running
    install_log.clear()
    install_done.clear()
    cmd = ["winget", "install", "Nvidia.CUDA",
           "--accept-package-agreements", "--accept-source-agreements", "--silent"]
    if version:
        cmd.extend(["--version", version])
    display = " ".join(cmd)
    install_log.append(f"$ {display}\n")
    install_log.append("NOTE: A UAC (administrator) prompt may appear — approve it to continue.\n")
    install_log.append(f"Installing NVIDIA CUDA Toolkit {version or '(latest)'}...\n\n")
    try:
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, encoding="utf-8", errors="replace"
        )
        for line in process.stdout:
            install_log.append(line)
        process.wait()
        if process.returncode == 0:
            install_log.append(f"\n--- CUDA Toolkit installed successfully! ---\n")
            install_log.append("A system restart may be required for changes to take effect.\n")
        else:
            install_log.append(f"\n--- Installation failed (exit code {process.returncode}) ---\n")
            install_log.append("Try running the command from an elevated (admin) command prompt.\n")
            install_log.append("To list available versions: winget search Nvidia.CUDA\n")
    except FileNotFoundError:
        install_log.append("\n--- Error: 'winget' not found ---\n")
        install_log.append("winget ships with Windows 10 1709+ / Windows 11.\n")
        install_log.append("Download CUDA Toolkit manually: https://developer.nvidia.com/cuda-downloads\n")
    except Exception as e:
        install_log.append(f"\n--- Error: {e} ---\n")
    finally:
        install_running = False
        install_done.set()


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PyTorch / CUDA Manager - SwarmUI</title>
<style>
:root {
    --bg: #1a1a2e;
    --surface: #16213e;
    --surface2: #0f3460;
    --accent: #e94560;
    --accent2: #533483;
    --text: #eee;
    --text-dim: #aaa;
    --success: #00d26a;
    --warning: #f5a623;
    --danger: #e94560;
    --border: #2a2a4a;
    --radius: 8px;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
}
.container { max-width: 1000px; margin: 0 auto; padding: 24px 16px; }
h1 {
    text-align: center; margin-bottom: 4px; font-size: 1.8rem;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.subtitle { text-align: center; color: var(--text-dim); margin-bottom: 24px; font-size: 0.9rem; }
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 20px; margin-bottom: 16px;
}
.card h2 {
    font-size: 1.1rem; margin-bottom: 16px; padding-bottom: 8px;
    border-bottom: 1px solid var(--border); display: flex; align-items: center; gap: 8px;
}
.card h2 .icon { font-size: 1.3rem; }

/* Environment info */
.env-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
    font-size: 0.85rem; background: var(--bg); border-radius: 6px; padding: 12px;
}
.env-row { display: contents; }
.env-label { color: var(--text-dim); }
.env-value { font-family: 'Cascadia Code', 'Fira Code', monospace; word-break: break-all; }

/* GPU info */
.gpu-bar {
    display: flex; gap: 12px; flex-wrap: wrap; margin-top: 12px;
}
.gpu-chip {
    background: var(--bg); border-radius: 6px; padding: 10px 14px;
    display: flex; flex-direction: column; gap: 2px; flex: 1; min-width: 200px;
}
.gpu-name { font-weight: 600; font-size: 0.95rem; }
.gpu-details { font-size: 0.8rem; color: var(--text-dim); }

/* Package table */
.pkg-table { width: 100%; border-collapse: collapse; margin-top: 8px; }
.pkg-table th {
    text-align: left; font-size: 0.75rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.5px; padding: 8px 12px;
    border-bottom: 1px solid var(--border);
}
.pkg-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); vertical-align: middle; }
.pkg-table tr:last-child td { border-bottom: none; }
.pkg-name { font-weight: 600; }
.pkg-pip { font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.85rem; color: var(--text-dim); }
.pkg-version {
    font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.9rem;
}

/* Badges */
.badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 0.75rem; font-weight: 600; white-space: nowrap;
}
.badge-installed { background: rgba(0,210,106,0.15); color: var(--success); border: 1px solid rgba(0,210,106,0.3); }
.badge-missing { background: rgba(233,69,96,0.15); color: var(--danger); border: 1px solid rgba(233,69,96,0.3); }
.badge-optional { background: rgba(245,166,35,0.15); color: var(--warning); border: 1px solid rgba(245,166,35,0.3); }
.badge-required { background: rgba(83,52,131,0.15); color: #a78bfa; border: 1px solid rgba(83,52,131,0.3); }

/* Install form */
.form-section { margin-bottom: 16px; }
.form-section > label {
    display: block; font-size: 0.8rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;
}
.radio-group { display: flex; gap: 8px; flex-wrap: wrap; }
.radio-btn { display: block; cursor: pointer; }
.radio-btn input { display: none; }
.radio-btn span {
    display: inline-block; padding: 8px 16px; border: 1px solid var(--border);
    border-radius: 6px; font-size: 0.9rem; transition: all 0.15s; background: var(--bg);
}
.radio-btn input:checked + span {
    background: var(--accent); border-color: var(--accent); color: white; font-weight: 600;
}
.radio-btn:hover span { border-color: var(--accent); }

/* Package checkboxes */
.pkg-select-grid { display: flex; gap: 8px; flex-wrap: wrap; }
.pkg-check { cursor: pointer; display: block; }
.pkg-check input { display: none; }
.pkg-check .pkg-check-inner {
    display: flex; align-items: center; gap: 8px; padding: 10px 14px;
    border: 1px solid var(--border); border-radius: 6px; background: var(--bg);
    transition: all 0.15s; min-width: 150px;
}
.pkg-check input:checked + .pkg-check-inner {
    border-color: var(--success); background: rgba(0,210,106,0.08);
}
.pkg-check:hover .pkg-check-inner { border-color: var(--accent); }
.pkg-check-box {
    width: 18px; height: 18px; border: 2px solid var(--border); border-radius: 4px;
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    transition: all 0.15s; font-size: 12px;
}
.pkg-check input:checked + .pkg-check-inner .pkg-check-box {
    background: var(--success); border-color: var(--success); color: #000;
}
.pkg-check-label { display: flex; flex-direction: column; }
.pkg-check-name { font-weight: 600; font-size: 0.9rem; }
.pkg-check-status { font-size: 0.7rem; }
.pkg-check-status.installed { color: var(--success); }
.pkg-check-status.missing { color: var(--danger); }

.version-input-row { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.text-input {
    background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    padding: 8px 12px; color: var(--text); font-size: 0.9rem; width: 180px;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
}
.text-input::placeholder { color: #555; }
.text-input:focus { outline: none; border-color: var(--accent); }
.hint { font-size: 0.8rem; color: var(--text-dim); }

/* Command preview */
.command-preview {
    background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
    padding: 12px; font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.8rem; color: var(--success); word-break: break-all; margin-bottom: 16px;
    min-height: 40px; line-height: 1.5;
}
.command-preview::before { content: "$ "; color: var(--text-dim); }
.command-preview.empty { color: var(--text-dim); }
.command-preview.empty::before { content: ""; }

/* Buttons */
.btn-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.btn {
    padding: 10px 24px; border: none; border-radius: 6px;
    font-size: 0.95rem; font-weight: 600; cursor: pointer; transition: all 0.15s;
}
.btn:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-primary { background: var(--accent); color: white; }
.btn-primary:hover:not(:disabled) { background: #ff5a75; transform: translateY(-1px); }
.btn-secondary { background: var(--surface2); color: var(--text); border: 1px solid var(--border); }
.btn-secondary:hover:not(:disabled) { background: var(--accent2); }
.btn-danger { background: transparent; color: var(--danger); border: 1px solid var(--danger); }
.btn-danger:hover:not(:disabled) { background: rgba(233,69,96,0.15); }
.btn-sm { padding: 6px 14px; font-size: 0.8rem; }

/* Presets */
.presets { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.preset-btn {
    padding: 6px 14px; background: var(--bg); border: 1px solid var(--border);
    border-radius: 16px; color: var(--text-dim); font-size: 0.8rem;
    cursor: pointer; transition: all 0.15s;
}
.preset-btn:hover { border-color: var(--accent); color: var(--text); }

/* Log */
.log-header { display: flex; justify-content: space-between; align-items: center; }
.log-area {
    background: #0d1117; border: 1px solid var(--border); border-radius: 6px;
    padding: 12px; height: 300px; overflow-y: auto;
    font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.8rem; line-height: 1.5; white-space: pre-wrap;
    word-break: break-all; color: #c9d1d9;
}

.spinner {
    display: inline-block; width: 16px; height: 16px;
    border: 2px solid var(--border); border-top: 2px solid var(--accent);
    border-radius: 50%; animation: spin 0.8s linear infinite;
    vertical-align: middle; margin-right: 6px;
}
@keyframes spin { to { transform: rotate(360deg); } }

.status-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75rem; font-weight: 600; }
.status-running { background: var(--warning); color: #000; }
.status-done { background: var(--success); color: #000; }
.status-error { background: var(--accent); color: white; }

.loading { color: var(--text-dim); font-style: italic; }
.sep { height: 1px; background: var(--border); margin: 16px 0; }
</style>
</head>
<body>
<div class="container">
    <h1>PyTorch / CUDA Manager</h1>
    <p class="subtitle">SwarmUI &mdash; ComfyUI Backend Embedded Python Environment</p>

    <!-- Environment -->
    <div class="card">
        <h2><span class="icon">&#128187;</span> Environment</h2>
        <div class="env-grid" id="env-grid">
            <span class="loading">Loading...</span>
        </div>
        <div id="gpu-bar" class="gpu-bar"></div>
    </div>

    <!-- Package Status -->
    <div class="card">
        <h2>
            <span class="icon">&#128230;</span> Package Status
            <span style="flex:1"></span>
            <button class="btn btn-secondary btn-sm" onclick="refreshStatus()">Refresh Status</button>
        </h2>
        <table class="pkg-table" id="pkg-table">
            <thead><tr><th>Package</th><th>pip name</th><th>Status</th><th>Version</th><th>Required</th></tr></thead>
            <tbody id="pkg-tbody"><tr><td colspan="5" class="loading">Loading...</td></tr></tbody>
        </table>
        <div id="cuda-summary" style="margin-top:12px"></div>
    </div>

    <!-- CUDA Toolkit -->
    <div class="card">
        <h2><span class="icon">&#9889;</span> NVIDIA CUDA Toolkit</h2>
        <div id="cuda-toolkit-info"><span class="loading">Loading...</span></div>
        <div class="sep"></div>
        <div class="form-section">
            <label>Install / Upgrade Toolkit Version</label>
            <div class="radio-group" id="toolkit-version-group"><span class="loading">Loading...</span></div>
        </div>
        <p style="font-size:0.8rem;color:var(--warning);margin-bottom:14px">
            &#9888;&nbsp; Requires administrator rights &mdash; a UAC prompt will appear.<br>
            Close SwarmUI before installing. Use <code style="font-family:monospace">winget search Nvidia.CUDA</code> to verify exact version strings.
        </p>
        <div class="btn-row">
            <button class="btn btn-primary" id="toolkit-install-btn" onclick="startToolkitInstall()">Install Selected Toolkit</button>
            <button class="btn btn-secondary btn-sm" onclick="checkToolkitVersions()">List Available Versions</button>
            <span id="toolkit-status"></span>
        </div>
    </div>

    <!-- Install / Upgrade -->
    <div class="card">
        <h2><span class="icon">&#11015;</span> Install / Upgrade / Switch</h2>

        <div class="form-section">
            <label>Quick Presets</label>
            <div class="presets">
                <button class="preset-btn" onclick="applyPreset('stable','cu128',['torch','torchvision','torchaudio'])">Stable + CUDA 12.8 (all)</button>
                <button class="preset-btn" onclick="applyPreset('stable','cu130',['torch','torchvision','torchaudio'])">Stable + CUDA 13.0 (all)</button>
                <button class="preset-btn" onclick="applyPreset('nightly','cu128',['torch','torchvision','torchaudio'])">Nightly + CUDA 12.8 (all)</button>
                <button class="preset-btn" onclick="applyPreset('stable','cpu',['torch','torchvision','torchaudio'])">CPU Only (all)</button>
            </div>
        </div>

        <div class="form-section">
            <label>Packages to Install / Upgrade</label>
            <div class="pkg-select-grid" id="pkg-select-grid"></div>
        </div>

        <div class="form-section">
            <label>Channel</label>
            <div class="radio-group">
                <label class="radio-btn"><input type="radio" name="channel" value="stable" checked onchange="updatePreview()"><span>Stable</span></label>
                <label class="radio-btn"><input type="radio" name="channel" value="nightly" onchange="updatePreview()"><span>Nightly</span></label>
            </div>
        </div>

        <div class="form-section">
            <label>CUDA / Compute Platform</label>
            <div class="radio-group">
                <label class="radio-btn"><input type="radio" name="cuda" value="cu126" onchange="updatePreview()"><span>CUDA 12.6</span></label>
                <label class="radio-btn"><input type="radio" name="cuda" value="cu128" checked onchange="updatePreview()"><span>CUDA 12.8</span></label>
                <label class="radio-btn"><input type="radio" name="cuda" value="cu130" onchange="updatePreview()"><span>CUDA 13.0</span></label>
                <label class="radio-btn"><input type="radio" name="cuda" value="cpu" onchange="updatePreview()"><span>CPU Only</span></label>
            </div>
        </div>

        <div class="form-section">
            <label>Pin Version (optional &mdash; blank = latest)</label>
            <div class="version-input-row">
                <input type="text" class="text-input" id="version-input" placeholder="e.g. 2.10.0" oninput="updatePreview()">
                <span class="hint">Applies to torch only. Other packages auto-resolve compatible versions.</span>
            </div>
        </div>

        <div class="form-section">
            <label class="pkg-check" style="display:inline-block">
                <input type="checkbox" id="force-reinstall" onchange="updatePreview()">
                <div class="pkg-check-inner">
                    <div class="pkg-check-box">&#10003;</div>
                    <div class="pkg-check-label">
                        <span class="pkg-check-name">Force Reinstall</span>
                        <span class="pkg-check-status" style="color:var(--text-dim)">Required when switching CUDA versions on already-installed packages</span>
                    </div>
                </div>
            </label>
        </div>

        <div class="command-preview" id="command-preview"></div>

        <div class="btn-row">
            <button class="btn btn-primary" id="install-btn" onclick="startInstall()">Install Selected</button>
            <button class="btn btn-danger btn-sm" id="uninstall-btn" onclick="startUninstall()">Uninstall Selected</button>
            <span id="install-status"></span>
        </div>
    </div>

    <!-- Log -->
    <div class="card">
        <div class="log-header">
            <h2 style="border:none;margin:0;padding:0"><span class="icon">&#128196;</span> Output Log</h2>
            <div style="display:flex;gap:6px">
                <button class="btn btn-secondary btn-sm" id="copy-log-btn" onclick="copyLog()">Copy Log</button>
                <button class="btn btn-secondary btn-sm" onclick="clearLog()">Clear</button>
            </div>
        </div>
        <div class="log-area" id="log-area">Ready. Select packages above and click Install.</div>
    </div>
</div>

<script>
// --- State ---
let installing = false;
let logPollInterval = null;
let lastLogIndex = 0;
let packageData = []; // filled from /api/packages
let currentOperation = null; // 'pip' | 'toolkit' | 'toolkit-check'

// --- Helpers ---
function getSelected(name) {
    const el = document.querySelector(`input[name="${name}"]:checked`);
    return el ? el.value : null;
}
function setSelected(name, value) {
    const el = document.querySelector(`input[name="${name}"][value="${value}"]`);
    if (el) el.checked = true;
}
function getCheckedPackages() {
    return Array.from(document.querySelectorAll('#pkg-select-grid .pkg-check input:checked')).map(el => el.value);
}

// --- Render environment ---
function renderEnv(env) {
    const grid = document.getElementById('env-grid');
    const rows = [
        ['Python Executable', env.python_exe],
        ['Python Version', env.python_version],
        ['Site-Packages', env.site_packages],
        ['ComfyUI Directory', env.comfyui_dir],
    ];
    grid.innerHTML = rows.map(([l, v]) =>
        `<span class="env-label">${l}</span><span class="env-value">${v}</span>`
    ).join('');
}

function renderGpu(gpu) {
    const el = document.getElementById('gpu-bar');
    if (!gpu.available || gpu.devices.length === 0) {
        el.innerHTML = '<div class="gpu-chip"><div class="gpu-name">No NVIDIA GPU detected</div></div>';
        return;
    }
    el.innerHTML = gpu.devices.map(d =>
        `<div class="gpu-chip">
            <span class="gpu-name">${d.name}</span>
            <span class="gpu-details">VRAM: ${d.total_memory_gb} GB${d.compute_capability ? ' | CC ' + d.compute_capability : ''}${d.driver ? ' | Driver ' + d.driver : ''}</span>
        </div>`
    ).join('');
}

// --- Render package status ---
function renderPackageTable(data) {
    packageData = data.packages;
    const tbody = document.getElementById('pkg-tbody');
    tbody.innerHTML = data.packages.map(p => {
        const statusBadge = p.installed
            ? '<span class="badge badge-installed">Installed</span>'
            : '<span class="badge badge-missing">Not Installed</span>';
        const version = p.installed ? `<span class="pkg-version">${p.version}</span>` : '<span class="pkg-version" style="color:var(--text-dim)">&mdash;</span>';
        const reqBadge = p.required
            ? '<span class="badge badge-required">Required</span>'
            : '<span class="badge badge-optional">Optional</span>';
        return `<tr>
            <td class="pkg-name">${p.name}</td>
            <td class="pkg-pip">${p.pip_name}</td>
            <td>${statusBadge}</td>
            <td>${version}</td>
            <td>${reqBadge}</td>
        </tr>`;
    }).join('');

    // CUDA summary
    const cs = document.getElementById('cuda-summary');
    if (data.cuda.cuda_version) {
        cs.innerHTML = `<span style="font-size:0.85rem;color:var(--text-dim)">
            CUDA toolkit (via PyTorch): <strong style="color:var(--text)">${data.cuda.cuda_version}</strong>
            &nbsp;|&nbsp; cuDNN: <strong style="color:var(--text)">${data.cuda.cudnn || 'N/A'}</strong>
            &nbsp;|&nbsp; CUDA available: <strong style="color:${data.cuda.cuda_available ? 'var(--success)' : 'var(--danger)'}">${data.cuda.cuda_available ? 'Yes' : 'No'}</strong>
        </span>`;
    } else {
        cs.innerHTML = '<span style="font-size:0.85rem;color:var(--text-dim)">PyTorch not installed &mdash; no CUDA info available</span>';
    }

    // Build package checkboxes for the install form
    renderPackageCheckboxes(data.packages);
}

function renderPackageCheckboxes(packages) {
    const grid = document.getElementById('pkg-select-grid');
    grid.innerHTML = packages.map(p => {
        const statusClass = p.installed ? 'installed' : 'missing';
        const statusText = p.installed ? `v${p.version}` : 'Not installed';
        const checked = (!p.installed && p.required) ? 'checked' : '';
        return `<label class="pkg-check">
            <input type="checkbox" value="${p.pip_name}" ${checked} onchange="updatePreview()">
            <div class="pkg-check-inner">
                <div class="pkg-check-box">&#10003;</div>
                <div class="pkg-check-label">
                    <span class="pkg-check-name">${p.name}</span>
                    <span class="pkg-check-status ${statusClass}">${statusText}</span>
                </div>
            </div>
        </label>`;
    }).join('');
    updatePreview();
}

// --- Preview ---
function updatePreview() {
    const channel = getSelected('channel');
    const cuda = getSelected('cuda');
    const version = document.getElementById('version-input').value.trim();
    const packages = getCheckedPackages();
    const preview = document.getElementById('command-preview');

    if (packages.length === 0) {
        preview.textContent = 'Select at least one package above';
        preview.classList.add('empty');
        return;
    }
    preview.classList.remove('empty');

    const force = document.getElementById('force-reinstall').checked;

    let cmd = 'python -s -m pip install --upgrade';
    if (force) cmd += ' --force-reinstall';
    if (channel === 'nightly') cmd += ' --pre';

    for (const pkg of packages) {
        if (pkg === 'torch' && version) {
            cmd += ` torch==${version}`;
        } else {
            cmd += ` ${pkg}`;
        }
    }

    let indexUrl;
    if (cuda === 'cpu') {
        indexUrl = channel === 'nightly'
            ? 'https://download.pytorch.org/whl/nightly/cpu'
            : 'https://download.pytorch.org/whl/cpu';
    } else {
        const base = channel === 'nightly'
            ? 'https://download.pytorch.org/whl/nightly'
            : 'https://download.pytorch.org/whl';
        indexUrl = `${base}/${cuda}`;
    }
    cmd += ` --index-url ${indexUrl}`;
    preview.textContent = cmd;
}

// --- Presets ---
function applyPreset(channel, cuda, pkgs) {
    setSelected('channel', channel);
    setSelected('cuda', cuda);
    document.getElementById('version-input').value = '';
    // Check only the specified packages
    document.querySelectorAll('.pkg-check input').forEach(el => {
        el.checked = pkgs.includes(el.value);
    });
    updatePreview();
}

// --- Fetch data ---
async function fetchAll() {
    try {
        const [envRes, gpuRes, pkgRes] = await Promise.all([
            fetch('/api/environment'),
            fetch('/api/gpu'),
            fetch('/api/packages'),
        ]);
        renderEnv(await envRes.json());
        renderGpu(await gpuRes.json());
        renderPackageTable(await pkgRes.json());
    } catch (e) {
        document.getElementById('env-grid').innerHTML = '<span class="loading">Failed to load</span>';
    }
    fetchCudaToolkit();
}

// --- CUDA Toolkit ---
async function fetchCudaToolkit() {
    try {
        const res = await fetch('/api/cuda-toolkit');
        renderCudaToolkit(await res.json());
    } catch (e) {
        document.getElementById('cuda-toolkit-info').innerHTML =
            '<span style="color:var(--text-dim)">Failed to load toolkit info.</span>';
    }
}

function renderCudaToolkit(data) {
    const info = document.getElementById('cuda-toolkit-info');
    if (data.installed) {
        const pathRow = data.cuda_path
            ? `<span class="env-label">Install Path</span><span class="env-value">${data.cuda_path}</span>`
            : '';
        const nvccRow = data.nvcc_path
            ? `<span class="env-label">nvcc</span><span class="env-value">${data.nvcc_path}</span>`
            : '';
        info.innerHTML = `<div class="env-grid">
            <span class="env-label">Toolkit Version</span>
            <span class="env-value" style="color:var(--success);font-weight:600">${data.version} &nbsp;<span class="badge badge-installed">Detected</span></span>
            ${pathRow}${nvccRow}
        </div>`;
    } else {
        info.innerHTML = '<span style="color:var(--warning)">&#9888; CUDA Toolkit not detected &mdash; nvcc not found in PATH or CUDA_PATH</span>';
    }

    const group = document.getElementById('toolkit-version-group');
    group.innerHTML = data.versions.map((v, i) =>
        `<label class="radio-btn">
            <input type="radio" name="toolkit-version" value="${v.version ?? ''}" ${i === 0 ? 'checked' : ''}>
            <span>${v.label}</span>
        </label>`
    ).join('');
}

async function startToolkitInstall() {
    if (installing) { alert('Another operation is already running.'); return; }
    const el = document.querySelector('input[name="toolkit-version"]:checked');
    if (!el) { alert('Select a version.'); return; }
    const version = el.value || null;
    const label = el.closest('label').querySelector('span').textContent;
    if (!confirm(`Install ${label}?\n\nA UAC (administrator) prompt will appear.\nMake sure SwarmUI is closed first.`)) return;

    installing = true;
    currentOperation = 'toolkit';
    lastLogIndex = 0;
    setButtonsDisabled(true);
    document.getElementById('toolkit-status').innerHTML =
        '<span class="spinner"></span> <span class="status-badge status-running">Installing Toolkit...</span>';
    document.getElementById('log-area').textContent = '';

    try {
        const res = await fetch('/api/cuda-toolkit/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ version }),
        });
        const data = await res.json();
        if (data.error) {
            document.getElementById('log-area').textContent = 'Error: ' + data.error;
            finishToolkitInstall(false);
            return;
        }
        startLogPolling();
    } catch (e) {
        document.getElementById('log-area').textContent = 'Error: ' + e.message;
        finishToolkitInstall(false);
    }
}

async function checkToolkitVersions() {
    if (installing) { alert('Another operation is already running.'); return; }
    installing = true;
    currentOperation = 'toolkit-check';
    lastLogIndex = 0;
    setButtonsDisabled(true);
    document.getElementById('log-area').textContent = '';
    try {
        const res = await fetch('/api/cuda-toolkit/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'search' }),
        });
        const data = await res.json();
        if (data.error) {
            document.getElementById('log-area').textContent = 'Error: ' + data.error;
            finishToolkitInstall(false);
            return;
        }
        startLogPolling();
    } catch (e) {
        document.getElementById('log-area').textContent = 'Error: ' + e.message;
        finishToolkitInstall(false);
    }
}

function finishToolkitInstall(success) {
    installing = false;
    setButtonsDisabled(false);
    const s = document.getElementById('toolkit-status');
    if (currentOperation === 'toolkit') {
        s.innerHTML = success
            ? '<span class="status-badge status-done">Done! Restart may be required.</span>'
            : '<span class="status-badge status-error">Failed &mdash; see log</span>';
        if (success) fetchCudaToolkit();
    } else {
        s.innerHTML = '';
    }
    setTimeout(() => { s.innerHTML = ''; }, 12000);
}

async function refreshStatus() {
    try {
        const res = await fetch('/api/packages');
        renderPackageTable(await res.json());
    } catch (e) {}
}

// --- Install ---
async function startInstall() {
    if (installing) return;
    const channel = getSelected('channel');
    const cuda = getSelected('cuda');
    const version = document.getElementById('version-input').value.trim();
    const force = document.getElementById('force-reinstall').checked;
    const packages = getCheckedPackages();
    if (!packages.length) { alert('Select at least one package.'); return; }

    installing = true;
    currentOperation = 'pip';
    lastLogIndex = 0;
    setButtonsDisabled(true);
    document.getElementById('install-status').innerHTML = '<span class="spinner"></span> <span class="status-badge status-running">Installing...</span>';
    document.getElementById('log-area').textContent = '';

    try {
        const res = await fetch('/api/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                action: 'install',
                channel, cuda,
                packages,
                version: version || null,
                force,
            })
        });
        const data = await res.json();
        if (data.error) {
            document.getElementById('log-area').textContent = 'Error: ' + data.error;
            finishInstall(false);
            return;
        }
        startLogPolling();
    } catch (e) {
        document.getElementById('log-area').textContent = 'Error: ' + e.message;
        finishInstall(false);
    }
}

async function startUninstall() {
    if (installing) return;
    const packages = getCheckedPackages();
    if (!packages.length) { alert('Select at least one package.'); return; }
    if (!confirm(`Uninstall: ${packages.join(', ')}?`)) return;

    installing = true;
    currentOperation = 'pip';
    lastLogIndex = 0;
    setButtonsDisabled(true);
    document.getElementById('install-status').innerHTML = '<span class="spinner"></span> <span class="status-badge status-running">Uninstalling...</span>';
    document.getElementById('log-area').textContent = '';

    try {
        const res = await fetch('/api/install', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'uninstall', packages })
        });
        const data = await res.json();
        if (data.error) {
            document.getElementById('log-area').textContent = 'Error: ' + data.error;
            finishInstall(false);
            return;
        }
        startLogPolling();
    } catch (e) {
        document.getElementById('log-area').textContent = 'Error: ' + e.message;
        finishInstall(false);
    }
}

function setButtonsDisabled(v) {
    document.getElementById('install-btn').disabled = v;
    document.getElementById('uninstall-btn').disabled = v;
    const tb = document.getElementById('toolkit-install-btn');
    if (tb) tb.disabled = v;
}

function startLogPolling() {
    if (logPollInterval) clearInterval(logPollInterval);
    logPollInterval = setInterval(pollLog, 500);
}

async function pollLog() {
    try {
        const res = await fetch(`/api/install/log?from=${lastLogIndex}`);
        const data = await res.json();
        if (data.lines && data.lines.length > 0) {
            const logArea = document.getElementById('log-area');
            for (const line of data.lines) logArea.textContent += line;
            lastLogIndex = data.next_index;
            logArea.scrollTop = logArea.scrollHeight;
        }
        if (data.done) {
            clearInterval(logPollInterval);
            logPollInterval = null;
            const allText = document.getElementById('log-area').textContent;
            const success = allText.includes('successfully');
            if (currentOperation === 'toolkit' || currentOperation === 'toolkit-check') {
                finishToolkitInstall(success);
            } else {
                finishInstall(success);
                if (success) refreshStatus();
            }
        }
    } catch (e) {}
}

function finishInstall(success) {
    installing = false;
    setButtonsDisabled(false);
    const s = document.getElementById('install-status');
    s.innerHTML = success
        ? '<span class="status-badge status-done">Done!</span>'
        : '<span class="status-badge status-error">Failed</span>';
    setTimeout(() => { s.innerHTML = ''; }, 8000);
}

function copyLog() {
    const text = document.getElementById('log-area').textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copy-log-btn');
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy Log'; }, 2000);
    });
}

function clearLog() {
    document.getElementById('log-area').textContent = 'Ready.';
}

// --- Init ---
fetchAll();
</script>
</body>
</html>
"""


class PyTorchManagerHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}" if args else "")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/":
            self.send_html(HTML_PAGE)
        elif path == "/api/environment":
            self.send_json(get_environment_info())
        elif path == "/api/gpu":
            self.send_json(get_gpu_info())
        elif path == "/api/packages":
            self.send_json(get_package_status())
        elif path == "/api/cuda-toolkit":
            self.send_json({**get_cuda_toolkit_info(), "versions": CUDA_TOOLKIT_VERSIONS})
        elif path == "/api/install/log":
            from_index = int(params.get("from", [0])[0])
            lines = list(install_log)
            new_lines = lines[from_index:]
            done = not install_running and install_done.is_set()
            self.send_json({
                "lines": new_lines,
                "next_index": len(lines),
                "done": done,
            })
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/install":
            global install_running
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON"}, 400)
                return

            action = data.get("action", "install")
            packages = data.get("packages", [])
            if not packages:
                self.send_json({"error": "No packages selected"}, 400)
                return
            # Validate package names (only allow known pip names)
            allowed = {p["pip"] for p in CUDA_PACKAGES}
            for pkg in packages:
                if pkg not in allowed:
                    self.send_json({"error": f"Unknown package: {pkg}"}, 400)
                    return

            if not install_lock.acquire(blocking=False):
                self.send_json({"error": "An operation is already running"}, 409)
                return

            install_running = True
            if action == "uninstall":
                thread = threading.Thread(target=self._do_uninstall, args=(packages,), daemon=True)
            else:
                channel = data.get("channel", "stable")
                cuda = data.get("cuda", "cu128")
                version = data.get("version")
                force = data.get("force", False)
                if channel not in ("stable", "nightly"):
                    install_running = False
                    install_lock.release()
                    self.send_json({"error": "Invalid channel"}, 400)
                    return
                if cuda not in ("cu126", "cu128", "cu130", "cpu"):
                    install_running = False
                    install_lock.release()
                    self.send_json({"error": "Invalid CUDA option"}, 400)
                    return
                versions = {}
                if version:
                    versions["torch"] = version
                thread = threading.Thread(
                    target=self._do_install,
                    args=(channel, cuda, packages, versions, force), daemon=True
                )
            thread.start()
            self.send_json({"status": "started"})
        elif parsed.path == "/api/cuda-toolkit/install":
            global install_running
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self.send_json({"error": "Invalid JSON"}, 400)
                return

            action = data.get("action", "install")

            if not install_lock.acquire(blocking=False):
                self.send_json({"error": "An operation is already running"}, 409)
                return
            install_running = True

            if action == "search":
                thread = threading.Thread(target=self._do_toolkit_search, daemon=True)
            else:
                version = data.get("version") or None
                allowed = {v["version"] for v in CUDA_TOOLKIT_VERSIONS}
                if version is not None and version not in allowed:
                    install_running = False
                    install_lock.release()
                    self.send_json({"error": f"Unknown version: {version}"}, 400)
                    return
                thread = threading.Thread(
                    target=self._do_toolkit_install, args=(version,), daemon=True
                )
            thread.start()
            self.send_json({"status": "started"})
        else:
            self.send_response(404)
            self.end_headers()

    @staticmethod
    def _do_install(channel, cuda, packages, versions, force=False):
        try:
            run_install(channel, cuda, packages, versions, force=force)
        finally:
            install_lock.release()

    @staticmethod
    def _do_uninstall(packages):
        try:
            run_uninstall(packages)
        finally:
            install_lock.release()

    @staticmethod
    def _do_toolkit_install(version):
        try:
            run_cuda_toolkit_install(version)
        finally:
            install_lock.release()

    @staticmethod
    def _do_toolkit_search():
        global install_running
        install_log.clear()
        install_done.clear()
        install_log.append("$ winget search Nvidia.CUDA\n\n")
        try:
            result = subprocess.run(
                ["winget", "search", "Nvidia.CUDA"],
                capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace"
            )
            for line in (result.stdout or result.stderr or "No output.").splitlines(keepends=True):
                install_log.append(line)
            install_log.append("\n--- Done ---\n")
        except FileNotFoundError:
            install_log.append("Error: winget not found.\n")
        except Exception as e:
            install_log.append(f"Error: {e}\n")
        finally:
            install_running = False
            install_done.set()
            install_lock.release()


def main():
    # Verify we're running from the right Python
    exe = sys.executable
    if "python_embeded" not in exe.replace("\\", "/").lower():
        print(f"WARNING: This script should be run with the embedded Python.")
        print(f"  Current: {exe}")
        print(f"  Expected: dlbackend/comfy/python_embeded/python.exe")
        print()

    server = http.server.HTTPServer(("0.0.0.0", PORT), PyTorchManagerHandler)
    print(f"PyTorch/CUDA Manager for SwarmUI")
    print(f"Environment: {os.path.dirname(exe)}")
    print(f"Running at:  http://localhost:{PORT}")
    print("Press Ctrl+C to stop.\n")
    try:
        import webbrowser
        webbrowser.open(f"http://localhost:{PORT}")
    except Exception:
        pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
