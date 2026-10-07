"""Candidate review dialog for Linux using Tkinter.

Floating topmost window allowing the user to select between candidates using 1, 2, 3 or cancel with Esc.
"""
from __future__ import annotations

import tkinter as tk

BG = "#1d1b20"
FG = "#ffffff"
DIM = "#b9b4bf"
BTN_BG = "#2c2830"
ACCENT = "#ff5473"
FONT_MAIN = ("Sans", 11)
FONT_TITLE = ("Sans", 12, "bold")


class Review:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.win: tk.Toplevel | None = None
        self.callback = None

    def show(self, choices: list[tuple[str, str]], reason: str, callback):
        self.dismiss()
        self.callback = callback

        self.win = tk.Toplevel(self.root)
        self.win.title("Lipflow · Выбор результата")
        self.win.configure(bg=BG)
        self.win.attributes("-topmost", True)
        self.win.minsize(560, 240)

        # Header / reason label
        header = tk.Label(
            self.win,
            text=reason,
            bg=BG,
            fg=DIM,
            font=FONT_MAIN,
            wraplength=520,
            justify="left",
        )
        header.pack(padx=20, pady=(15, 10), anchor="w")

        # Candidate buttons
        for i, (title, text) in enumerate(choices):
            btn_text = f"[{i + 1}]  {title}:  «{text}»"
            btn = tk.Button(
                self.win,
                text=btn_text,
                bg=BTN_BG,
                fg=FG,
                activebackground=ACCENT,
                activeforeground=FG,
                font=FONT_MAIN,
                anchor="w",
                padx=12,
                pady=8,
                relief="flat",
                cursor="hand2",
                command=lambda t=text: self.finish(t),
            )
            btn.pack(fill="x", padx=20, pady=4)
            # Bind number key on the review window
            self.win.bind(str(i + 1), lambda event, t=text: self.finish(t))

        # Cancel button
        cancel_btn = tk.Button(
            self.win,
            text="Отмена (Esc)",
            bg=BG,
            fg=DIM,
            activebackground=BTN_BG,
            activeforeground=FG,
            font=FONT_MAIN,
            relief="flat",
            cursor="hand2",
            command=lambda: self.finish(None),
        )
        cancel_btn.pack(pady=(10, 15))
        self.win.bind("<Escape>", lambda event: self.finish(None))
        self.win.protocol("WM_DELETE_WINDOW", lambda: self.finish(None))

        # Center on screen
        self.win.update_idletasks()
        w = max(560, self.win.winfo_reqwidth())
        h = max(240, self.win.winfo_reqheight())
        sw = self.win.winfo_screenwidth()
        sh = self.win.winfo_screenheight()
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.win.geometry(f"{w}x{h}+{x}+{y}")

        self.win.lift()
        self.win.focus_force()

    def finish(self, text: str | None):
        cb = self.callback
        self.dismiss()
        if cb:
            if text is not None:
                self.root.after(100, lambda: cb(text))
            else:
                cb(None)

    def dismiss(self):
        self.callback = None
        if self.win is not None:
            try:
                self.win.destroy()
            except Exception:
                pass
            self.win = None
