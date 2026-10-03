"""Insert text at the cursor on Linux: clipboard + Ctrl+V, then restore previous clipboard."""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time

_uinput_dev = None
_uinput_lock = threading.Lock()


def get_uinput():
    """Return a shared UInput virtual keyboard device, or None if unavailable."""
    global _uinput_dev
    with _uinput_lock:
        if _uinput_dev is None:
            try:
                from evdev import UInput, ecodes as e
                cap = {
                    e.EV_KEY: [
                        e.KEY_LEFTCTRL, e.KEY_RIGHTCTRL, e.KEY_LEFTSHIFT, e.KEY_RIGHTSHIFT,
                        e.KEY_V, e.KEY_C, e.KEY_A, e.KEY_ENTER, e.KEY_INSERT
                    ]
                }
                _uinput_dev = UInput(cap, name="Lipflow Virtual Keyboard", bustype=e.BUS_USB)
                time.sleep(0.1)
            except Exception:
                _uinput_dev = False
        return _uinput_dev if _uinput_dev is not False else None


def close_uinput():
    """Close the virtual keyboard device on app shutdown."""
    global _uinput_dev
    with _uinput_lock:
        if _uinput_dev and _uinput_dev is not False:
            try:
                _uinput_dev.close()
            except Exception:
                pass
            _uinput_dev = None


def _run_cmd(cmd: list[str], input_data: "str | None" = None) -> "str | None":
    try:
        proc = subprocess.run(
            cmd,
            input=input_data.encode("utf-8") if input_data is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=0.5,
            check=True,
        )
        return proc.stdout.decode("utf-8", errors="replace")
    except Exception:
        return None


def get_text() -> "str | None":
    """Read text from system clipboard using wl-paste, klipper, xclip, or pyperclip."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-paste"):
        res = _run_cmd(["wl-paste", "--no-newline"])
        if res is not None:
            return res
    if shutil.which("qdbus"):
        res = _run_cmd(["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"])
        if res is not None:
            return res.rstrip("\r\n")
    if shutil.which("xclip"):
        res = _run_cmd(["xclip", "-selection", "clipboard", "-o"])
        if res is not None:
            return res
    try:
        import pyperclip
        return pyperclip.paste()
    except Exception:
        return None


def set_text(text: str):
    """Write text to system clipboard using wl-copy, klipper, xclip, and pyperclip."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        _run_cmd(["wl-copy"], input_data=text)
    if shutil.which("qdbus"):
        _run_cmd(["qdbus", "org.kde.klipper", "/klipper", "setClipboardContents", text])
    if shutil.which("xclip"):
        _run_cmd(["xclip", "-selection", "clipboard"], input_data=text)
    try:
        import pyperclip
        pyperclip.copy(text)
    except Exception:
        pass


def _press_ctrl_v():
    """Simulate Ctrl+V keystroke to paste text."""
    ui = get_uinput()
    if ui is not None:
        try:
            from evdev import ecodes as e
            ui.write(e.EV_KEY, e.KEY_LEFTCTRL, 1)
            ui.syn()
            ui.write(e.EV_KEY, e.KEY_V, 1)
            ui.syn()
            time.sleep(0.04)
            ui.write(e.EV_KEY, e.KEY_V, 0)
            ui.syn()
            ui.write(e.EV_KEY, e.KEY_LEFTCTRL, 0)
            ui.syn()
            return
        except Exception:
            pass

    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wtype"):
        if _run_cmd(["wtype", "-M", "ctrl", "-k", "v", "-m", "ctrl"]) is not None:
            return
    if shutil.which("xdotool"):
        if _run_cmd(["xdotool", "key", "--clearmodifiers", "ctrl+v"]) is not None:
            return
    try:
        from pynput.keyboard import Controller, Key
        kb = Controller()
        with kb.pressed(Key.ctrl):
            kb.press('v')
            kb.release('v')
    except Exception as e:
        print(f"[lipflow] warning: simulated keystroke failed ({e}); use --copy-only instead")


def paste_text(text: str, restore_after: float = 1.2):
    """Copy text, press Ctrl+V, then restore old clipboard."""
    if not text:
        return
    saved = get_text()
    set_text(text)
    time.sleep(0.05)
    _press_ctrl_v()

    def restore():
        time.sleep(restore_after)
        if saved is not None:
            try:
                # Only restore if clipboard hasn't changed to something else
                current = get_text()
                if current == text:
                    set_text(saved)
            except Exception:
                pass

    threading.Thread(target=restore, name="lipflow-restore-clipboard", daemon=True).start()


def copy_text(text: str):
    """Copy text to clipboard without pasting."""
    set_text(text)
