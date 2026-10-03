"""`lipflow doctor`: check everything the app needs before you hit the hotkey."""
from __future__ import annotations

import os
import sys

from .vsr import MODELS
from .paths import WHO


def doctor() -> int:
    ok = True

    def line(good, what, fix=""):
        nonlocal ok
        ok &= bool(good)
        print(f"  {'✓' if good else '✗'} {what}" + ("" if good else f"\n      → {fix}"))

    print("Lipflow doctor\n")
    for rel, size in [("vsr/model.pth", 900e6), ("lm/model.pth", 200e6), ("face_landmarker.task", 3e6)]:
        path = os.path.join(MODELS, rel)
        line(os.path.exists(path) and os.path.getsize(path) > size, f"model file {rel}",
             "run .\\setup.ps1" if sys.platform == "win32" else "run ./setup.sh")

    import torch
    dev = "cuda (NVIDIA GPU)" if torch.cuda.is_available() else \
        "mps (Apple GPU)" if torch.backends.mps.is_available() else "cpu"
    line(True, f"torch {torch.__version__}, encoder on {dev}")

    if sys.platform == "win32":
        _windows_checks(line)
    elif sys.platform == "darwin":
        _mac_checks(line)
    else:
        _linux_checks(line)

    from .cleanup import Cleaner
    c = Cleaner()
    line(True, f"cleanup backend: {c.describe()}"
         + ("  (set ANTHROPIC_API_KEY or run Ollama for much better accuracy)" if c.backend == "basic" else ""))
    print("\nAll good." if ok else "\nFix the ✗ items above, then run `lipflow`.")
    return 0 if ok else 1


def _mac_checks(line):
    import Quartz
    line(Quartz.CGPreflightListenEventAccess(), "Input Monitoring (for the push-to-talk key)",
         f"System Settings → Privacy & Security → Input Monitoring → enable {WHO}, then restart it")
    line(Quartz.CGPreflightPostEventAccess(), "Accessibility (to paste at your cursor)",
         f"System Settings → Privacy & Security → Accessibility → enable {WHO}")

    from AVFoundation import AVCaptureDevice, AVMediaTypeVideo
    status = AVCaptureDevice.authorizationStatusForMediaType_(AVMediaTypeVideo)
    names = {0: "not asked yet (you'll be prompted on first use)", 1: "restricted", 2: "denied", 3: "granted"}
    line(status in (0, 3), f"Camera: {names.get(status, status)}",
         f"System Settings → Privacy & Security → Camera → enable {WHO}")


def _windows_checks(line):
    """Windows grants keyboard hooks and pasting to every desktop app; only the camera can be off."""
    import cv2
    from .camera import resolve_camera
    from .dictation import load_settings
    idx = resolve_camera(load_settings().get("camera", "auto"))
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    line(ok, f"Camera {idx} opens and sends frames",
         "Settings → Privacy & security → Camera → turn on \"Let desktop apps access your camera\", "
         "and close other apps using the camera")


def _linux_checks(line):
    """Linux checks: video device access, display session, clipboard tools."""
    import cv2
    import shutil
    from .camera import resolve_camera, list_cameras
    from .dictation import load_settings
    cams = list_cameras()
    idx = resolve_camera(load_settings().get("camera", "auto"))
    cap = cv2.VideoCapture(idx)
    ok = cap.isOpened() and cap.read()[0]
    cap.release()
    line(ok, f"Camera {idx} opens and sends frames" + (f" ({cams[0]['name']})" if cams else ""),
         "check /dev/video* permissions (e.g. video group: sudo usermod -aG video $USER) "
         "and close other apps using the webcam")

    disp = os.environ.get("WAYLAND_DISPLAY") or os.environ.get("DISPLAY")
    line(bool(disp), f"Display session available ({disp or 'none detected'})",
         "ensure DISPLAY or WAYLAND_DISPLAY environment variable is set")

    has_cb = bool(shutil.which("wl-copy") or shutil.which("xclip") or shutil.which("xsel"))
    line(has_cb, "Clipboard utility installed (wl-copy or xclip)",
         "install wl-clipboard or xclip (e.g. sudo dnf install wl-clipboard xclip / sudo apt install xclip)")

