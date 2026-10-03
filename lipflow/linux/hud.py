"""The floating HUD pill above the taskbar on Linux: status, live words, and mouth view."""
from __future__ import annotations

import sys
import tkinter as tk

import numpy as np

W, H = 420, 86
VIDEO_W, VIDEO_H = 112, 70
BG, FG, DIM = "#1d1b20", "#ffffff", "#b9b4bf"
ACCENT, GREEN, AMBER, RED = "#ff5473", "#4dd98c", "#ffb840", "#ff6b5e"
FONT = "Sans"
STATES = {"listening": ACCENT, "reading": AMBER, "done": GREEN, "error": RED}


def rounded_rect(c: tk.Canvas, x0, y0, x1, y1, r, **kw):
    pts = [
        x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1, x1 - r, y1, x0 + r, y1,
        x0, y1, x0, y1 - r, x0, y0 + r, x0, y0
    ]
    return c.create_polygon(pts, smooth=True, **kw)


def photo(bgr: np.ndarray) -> tk.PhotoImage:
    """BGR array -> tk PhotoImage via PPM format (native to Tk, no PIL ImageTk hook required)."""
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    h, w, _ = rgb.shape
    header = f"P6 {w} {h} 255\n".encode("ascii")
    return tk.PhotoImage(data=header + rgb.tobytes())


def get_monitors():
    """Detect connected monitors (x, y, width, height, primary) via xrandr."""
    import re
    import subprocess
    monitors = []
    try:
        out = subprocess.check_output(["xrandr", "--current"], stderr=subprocess.DEVNULL, timeout=0.5).decode()
        for line in out.splitlines():
            m = re.search(r" connected (?:primary )?(\d+)x(\d+)\+(\d+)\+(\d+)", line)
            if m:
                w, h, x, y = map(int, m.groups())
                monitors.append((x, y, w, h, "primary" in line))
    except Exception:
        pass
    return monitors


def get_current_monitor(root, monitors):
    """Return (x, y, width, height) of the monitor where pointer is, or primary, or full screen."""
    if not monitors:
        return 0, 0, root.winfo_screenwidth(), root.winfo_screenheight()
    try:
        px, py = root.winfo_pointerx(), root.winfo_pointery()
        for x, y, w, h, _ in monitors:
            if x <= px < x + w and y <= py < y + h:
                return x, y, w, h
    except Exception:
        pass
    for x, y, w, h, prim in monitors:
        if prim:
            return x, y, w, h
    return monitors[0][:4]


class HUD:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.attributes("-topmost", True)
        self.win.configure(bg=BG)

        self._monitors = []
        self._monitors_at = 0.0
        self._reposition()

        c = self.c = tk.Canvas(self.win, width=W, height=H, bg=BG, highlightthickness=0)
        c.pack()
        rounded_rect(c, 2, 2, W - 2, H - 2, 26, fill=BG, outline="#3a3640")
        self.dot = c.create_oval(22, 22, 32, 32, fill=AMBER, outline="")
        self.video_item = c.create_image(18 + VIDEO_W // 2, H // 2, state="hidden")
        self.title = c.create_text(42, 27, anchor="w", fill=DIM, font=(FONT, 9, "bold"), text="")
        self.body = c.create_text(22, 54, anchor="w", fill=FG, font=(FONT, 12), text="", width=W - 44)
        self._img = None
        self._hide_job = None
        self.mode = ""
        self.body_text = ""
        self._visible = True
        self.hide()

    def show(self, mode: str, title: str, body: str = "", hide_after: "float | None" = None):
        self._cancel_hide()
        self.mode = mode
        self.c.itemconfigure(self.dot, fill=STATES.get(mode, AMBER))
        self.c.itemconfigure(self.title, text=title.upper())
        if mode != "listening":
            self._clear_video()
            self._layout(video=False)
        self.set_text(body)
        self._show_window()
        if hide_after:
            self._hide_job = self.root.after(int(hide_after * 1000), self.hide)

    def set_text(self, body: str):
        self.body_text = body
        live = self.mode in ("listening", "reading")
        limit = 46 if self.c.itemcget(self.video_item, "state") == "hidden" else 34
        if len(body) > limit:
            body = "…" + body[-limit:] if live else body[:limit] + "…"
        self.c.itemconfigure(self.body, text=body)

    def set_frame(self, video_bgr, level: float = 0.0):
        if video_bgr is None or self.mode != "listening":
            return
        self._img = photo(video_bgr)
        self.c.itemconfigure(self.video_item, image=self._img, state="normal")
        self._layout(video=True)

    def hide(self):
        self._cancel_hide()
        self.mode = ""
        self._clear_video()
        if not self._visible:
            return
        self._visible = False
        self.win.withdraw()

    def _layout(self, video: bool):
        x = 18 + VIDEO_W + 14 if video else 22
        self.c.coords(self.dot, x, 22, x + 10, 32)
        self.c.coords(self.title, x + 20, 27)
        self.c.coords(self.body, x, 54)
        self.c.itemconfigure(self.body, width=W - x - 22)

    def _clear_video(self):
        self.c.itemconfigure(self.video_item, state="hidden", image="")
        self._img = None

    def _reposition(self):
        import time
        now = time.time()
        if now - self._monitors_at > 5.0 or not self._monitors:
            self._monitors = get_monitors()
            self._monitors_at = now
        mx, my, mw, mh = get_current_monitor(self.root, self._monitors)
        x = mx + (mw - W) // 2
        y = my + mh - H - 96
        self.win.geometry(f"{W}x{H}+{x}+{y}")

    def _show_window(self):
        self._reposition()
        if self._visible:
            return
        self._visible = True
        self.win.deiconify()

    def _cancel_hide(self):
        if self._hide_job is not None:
            self.root.after_cancel(self._hide_job)
            self._hide_job = None


def tray_image(listening: bool = False, size: int = 64):
    """Pink rounded square with a white mouth; solid lips while listening."""
    from PIL import Image, ImageDraw
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([s * 0.04, s * 0.04, s * 0.96, s * 0.96], radius=s * 0.22,
                        fill=(255, 84, 115, 255) if not listening else (214, 41, 97, 255))
    box = [s * 0.2, s * 0.34, s * 0.8, s * 0.68]
    if listening:
        d.ellipse(box, fill="white")
        d.line([s * 0.24, s * 0.51, s * 0.76, s * 0.51], fill=(214, 41, 97, 255), width=int(s * 0.05))
    else:
        d.ellipse(box, outline="white", width=int(s * 0.07))
        d.line([s * 0.24, s * 0.51, s * 0.76, s * 0.51], fill="white", width=int(s * 0.05))
    return img.resize((size, size), Image.LANCZOS)
