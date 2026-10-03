"""What you're typing into: app, window title, and the text around the cursor.

Read once, locally, through the Accessibility API (on Windows, just the window title) when you
press the push-to-talk key (the app you're dictating into is frontmost at that moment). Only names
and terms are pulled out, to help with the words lip reading gets wrong most often — who you're
writing to, the thread you're in.
Nothing is saved to disk or sent anywhere.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field

_STOP = set("""a an the and or but if of to in on at for with from by as is are was were be been am i
you he she it we they me my your our their this that these those do does did have has had not no so
just can will would could should there here what when where who how why all about up out then than
too very really also like get got go going new re fwd inbox search compose sent drafts home today
monday tuesday wednesday thursday friday saturday sunday january february march april may june july
august september october november december untitled message messages chat thread channel reply""".split())


@dataclass
class Context:
    app: str = ""
    title: str = ""
    near_text: str = ""
    names: list[str] = field(default_factory=list)
    element: object = None   # the focused text field (for learning from corrections); memory only
    value: str = ""          # its full contents before the paste; memory only

    def describe(self) -> str:
        bits = [self.app] + ([f'"{self.title[:60]}"'] if self.title else [])
        return " · ".join(b for b in bits if b)


def _ax(el, attr):
    from ApplicationServices import AXUIElementCopyAttributeValue
    err, val = AXUIElementCopyAttributeValue(el, attr, None)
    return val if err == 0 else None


def extract_names(*texts: str, limit: int = 30) -> list[str]:
    """Capitalised words that aren't sentence starts or common words: names, products, places."""
    seen, out = set(), []
    for t in texts:
        for sent in re.split(r"[.!?\n|•·—\-–:]+", t or ""):
            toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", sent)
            for i, w in enumerate(toks):
                if not w[0].isupper() or w.isupper() and len(w) > 4:
                    continue
                if w.lower() in _STOP or len(w) < 3:
                    continue
                if i == 0 and w.lower() in _STOP:
                    continue
                if w.lower() not in seen:
                    seen.add(w.lower())
                    out.append(w)
    return out[:limit]


def _capture_windows(ctx: Context) -> Context:
    """Windows: the foreground window's title and program name. Text near the cursor would need
    UI Automation, so names come from the title only (e.g. a chat or document name)."""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.WinDLL("user32"), ctypes.WinDLL("kernel32")
    user32.GetForegroundWindow.restype = wintypes.HWND
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ctx
    buf = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(hwnd, buf, 512)
    ctx.title = buf.value
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    kernel32.OpenProcess.restype = wintypes.HANDLE
    proc = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
    if proc:
        size = wintypes.DWORD(1024)
        path = ctypes.create_unicode_buffer(1024)
        if kernel32.QueryFullProcessImageNameW(proc, 0, path, ctypes.byref(size)):
            ctx.app = os.path.splitext(os.path.basename(path.value))[0]
        kernel32.CloseHandle(proc)
    ctx.names = extract_names(ctx.title)
    return ctx


def _capture_linux(ctx: Context) -> Context:
    """Linux: active window title from xdotool or python-xlib."""
    import subprocess
    try:
        win_id = subprocess.check_output(["xdotool", "getactivewindow"], stderr=subprocess.DEVNULL, timeout=0.2).decode().strip()
        if win_id:
            title = subprocess.check_output(["xdotool", "getwindowname", win_id], stderr=subprocess.DEVNULL, timeout=0.2).decode().strip()
            ctx.title = title
            try:
                pid = subprocess.check_output(["xdotool", "getwindowpid", win_id], stderr=subprocess.DEVNULL, timeout=0.2).decode().strip()
                if pid:
                    comm_path = f"/proc/{pid}/comm"
                    if os.path.exists(comm_path):
                        ctx.app = open(comm_path).read().strip()
            except Exception:
                pass
            ctx.names = extract_names(ctx.title)
            return ctx
    except Exception:
        pass
    try:
        from Xlib import X, display
        d = display.Display()
        root = d.screen().root
        net_active_atom = d.intern_atom("_NET_ACTIVE_WINDOW")
        raw = root.get_full_property(net_active_atom, X.AnyPropertyType)
        if raw and raw.value:
            win_id = raw.value[0]
            win = d.create_resource_object('window', win_id)
            wm_name_atom = d.intern_atom("_NET_WM_NAME")
            title_prop = win.get_full_property(wm_name_atom, 0)
            if title_prop and title_prop.value:
                ctx.title = title_prop.value.decode("utf-8", errors="replace")
            else:
                ctx.title = win.get_wm_name() or ""
            ctx.names = extract_names(ctx.title)
    except Exception:
        pass
    return ctx


def capture(max_chars: int = 600) -> Context:
    """Snapshot of the frontmost app. Never raises: context is a bonus, not a requirement."""
    ctx = Context()
    try:
        if sys.platform == "win32":
            return _capture_windows(ctx)
        if sys.platform.startswith("linux"):
            return _capture_linux(ctx)
        from AppKit import NSWorkspace
        from ApplicationServices import AXUIElementCreateApplication
        app = NSWorkspace.sharedWorkspace().frontmostApplication()
        if app is None:
            return ctx
        ctx.app = str(app.localizedName() or "")
        ax_app = AXUIElementCreateApplication(app.processIdentifier())
        win = _ax(ax_app, "AXFocusedWindow")
        if win is not None:
            ctx.title = str(_ax(win, "AXTitle") or "")
        focused = _ax(ax_app, "AXFocusedUIElement")
        if focused is not None:
            val = _ax(focused, "AXValue")
            if isinstance(val, str):
                ctx.element, ctx.value = focused, val
                ctx.near_text = val[-max_chars:]
            if not ctx.near_text:
                ph = _ax(focused, "AXPlaceholderValue")  # e.g. Slack's "Message Miguel"
                if isinstance(ph, str):
                    ctx.near_text = ph
        ctx.names = extract_names(ctx.title, ctx.near_text)
    except Exception as e:  # permissions, sandboxed apps, odd elements
        print(f"[lipflow] context unavailable: {e.__class__.__name__}")
    return ctx
