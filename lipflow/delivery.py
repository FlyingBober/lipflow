"""Shared decision and focus checks for all front ends (Linux, Windows, macOS)."""
from __future__ import annotations

import shutil
import sys
from .cleanup import basic_cleanup
from .confidence import Quality, assess


def quality_for(rec, rois):
    import numpy as np
    pixels = getattr(rec, "mouth_pixels", [])
    valid_pixels = [p for p in pixels if p > 0]
    return Quality(
        rec.face_ratio,
        float(np.mean(rois)) if rois is not None and len(rois) else 128.0,
        float(np.std(rois)) if rois is not None and len(rois) else 30.0,
        float(np.median(valid_pixels)) if valid_pixels else 50.0,
    )


def choose_result(hypotheses, greedy, quality, cleaner, ctx, context, policy="review", min_margin=0.5, input_mode="silent"):
    lang = getattr(cleaner, "language", "en")
    decision = assess(hypotheses, greedy, quality, policy, min_margin, language=lang, input_mode=input_mode)
    if decision.action == "retry":
        return decision, None, []
    candidates = [h.text for h in hypotheses]
    result = cleaner.process(candidates, context=context, names=ctx.names if ctx else None)
    choices = []

    labels = {
        "ru": ("Очистка / Предложение", "Сырое распознавание", "Альтернатива"),
        "zh": ("Cleanup / 纠错建议", "Raw / 原始识别", "Alternative / 其他候选"),
        "en": ("Cleanup", "Raw", "Alternative"),
    }
    lbl_cleanup, lbl_raw, lbl_alt = labels.get(lang, labels["en"])

    for label, text in [
        (lbl_cleanup, result.proposed),
        (lbl_raw, basic_cleanup(result.raw)),
        *[(lbl_alt, basic_cleanup(c)) for c in candidates[1:]],
    ]:
        if text and text not in [t for _, t in choices]:
            choices.append((label, text))
        if len(choices) == 3:
            break
    return decision, result, choices


def target_is_current(ctx) -> bool:
    if ctx is None or not ctx.target:
        return True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL("user32")
        u.GetForegroundWindow.restype = wintypes.HWND
        return u.GetForegroundWindow() == ctx.target
    elif sys.platform == "darwin":
        from AppKit import NSWorkspace
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None or app.processIdentifier() != ctx.target:
            return False
        if ctx.element is not None:
            from .context import _ax
            from ApplicationServices import AXUIElementCreateApplication, CFEqual
            focused = _ax(AXUIElementCreateApplication(ctx.target), "AXFocusedUIElement")
            return focused is not None and bool(CFEqual(focused, ctx.element))
        return True
    else:  # Linux
        return True


def restore_target(ctx) -> bool:
    """Restore focus to the previous active window."""
    if ctx is None or not ctx.target:
        return True
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes
        u = ctypes.WinDLL("user32")
        u.SetForegroundWindow.argtypes = [wintypes.HWND]
        return bool(u.SetForegroundWindow(ctx.target))
    elif sys.platform == "darwin":
        from AppKit import NSRunningApplication, NSApplicationActivateIgnoringOtherApps
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(ctx.target)
        return bool(app and app.activateWithOptions_(NSApplicationActivateIgnoringOtherApps))
    else:  # Linux
        try:
            import subprocess
            if shutil.which("xdotool"):
                subprocess.run(["xdotool", "windowactivate", str(ctx.target)], check=False)
        except Exception:
            pass
        return True
