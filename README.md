# CUDA Manager for SwarmUI / ComfyUI

A web-based PyTorch and CUDA version manager for SwarmUI's embedded ComfyUI Python environment.

## What It Does

Provides a local web UI (http://localhost:9090) to install, upgrade, switch, or uninstall PyTorch and CUDA packages inside SwarmUI's bundled Python (`dlbackend/comfy/python_embeded/`). No command-line knowledge required.

**Managed packages:**
- PyTorch
- TorchVision
- TorchAudio
- TorchSDE
- xFormers (optional)

**Features:**
- Detects installed versions and CUDA availability
- Shows GPU info (name, VRAM, compute capability)
- Quick presets for common configurations (Stable/Nightly + CUDA 12.6/12.8/13.0/CPU)
- Pin specific torch versions
- Force reinstall option (for switching CUDA versions)
- Live install log output

## Prerequisites

- [SwarmUI](https://github.com/mcmonkeyprojects/SwarmUI) installed with the ComfyUI backend
- The embedded Python at `dlbackend/comfy/python_embeded/` must exist (run SwarmUI at least once to generate it)

## Installation

1. Copy `manage-pytorch.bat` and `pytorch_manager.py` into your SwarmUI root directory (same folder as `launch-windows.bat`).

```
SwarmUI/
├── manage-pytorch.bat      ← add this
├── launch-windows.bat
└── dlbackend/
    └── comfy/
        └── python_embeded/
            └── pytorch_manager.py      ← add this
```

2. Double-click `manage-pytorch.bat` to launch.

A browser window will open at **http://localhost:9090** automatically.

## Usage

1. **Environment card** — shows the Python path and ComfyUI directory being managed.
2. **Package Status card** — shows current installed versions and CUDA info.
3. **Install / Upgrade / Switch card** — select packages, choose a channel (Stable/Nightly), pick a CUDA version, then click **Install Selected**.
   - Use **Quick Presets** for one-click common setups.
   - Check **Force Reinstall** when switching CUDA versions on already-installed packages.
4. **Output Log** — shows live pip output during installs.

## Common Scenarios

| Goal | Settings |
|------|----------|
| Install latest stable PyTorch + CUDA 12.8 | Preset: "Stable + CUDA 12.8 (all)" |
| Switch from CUDA 12.6 to 12.8 | Select all packages → CUDA 12.8 → Force Reinstall ✓ |
| Install nightly for latest features | Channel: Nightly → select packages |
| CPU-only (no GPU) | Preset: "CPU Only (all)" |

## Notes

- SwarmUI must **not** be running during package changes (it holds locks on some Python files).
- After installing, click **Refresh Status** to confirm the new versions.
- xFormers is optional and only compatible with certain PyTorch/CUDA combinations.

## License

MIT
