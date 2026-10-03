"""Lipflow for Linux: a system tray icon, floating HUD, and background inference.

Threads: tk owns the main thread (overlay, setup window); pynput / IPC reports the key;
pystray runs the tray menu on its own thread; one model thread reads lips. Everything that touches
tk goes through ui(), which queues it for the main thread.
"""
from __future__ import annotations

import fcntl
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

from ..camera import Camera, Recording, list_cameras, mouth_view
from ..cleanup import Cleaner
from ..dictation import (
    HISTORY, JOIN_WINDOW, MAX_SECONDS, PREVIEW_EVERY, TAIL_SECONDS, clip_problem, keep_clip, load_settings,
    log_history, rois_for, save_settings, train_on_face,
)
from ..paths import HOME
from ..vsr import LipReader
from .hotkey import DEFAULT_KEY, KEYS, PushToTalk
from .hud import HUD, tray_image
from .paste import copy_text, paste_text

LOG = os.path.join(HOME, "Lipflow.log")
AUTOSTART_DIR = os.path.expanduser("~/.config/autostart")
AUTOSTART_FILE = os.path.join(AUTOSTART_DIR, "lipflow.desktop")
_LOCK_FILE = None


@dataclass
class Options:
    key: str = DEFAULT_KEY
    beam: int = 4
    backend: str = "auto"
    camera: "int | str" = "auto"
    paste: bool = True
    live_preview: bool = True
    onboard: bool = False


def key_label(key: str) -> str:
    if key.lower().startswith("f") and key[1:].isdigit():
        return key.upper()
    return key.replace("_", " ").title().replace("Control", "Ctrl")


class Lipflow:
    def __init__(self, opts: Options):
        self.opts = opts
        self.settings = load_settings()
        if opts.key == DEFAULT_KEY and self.settings.get("key") in KEYS:
            opts.key = self.settings["key"]
        self.root = tk.Tk()
        self.root.withdraw()
        self._q: "queue.Queue" = queue.Queue()
        self.reader: LipReader | None = None
        self.cleaner = Cleaner(opts.backend)
        self.jobs: "queue.Queue" = queue.Queue()
        self.session = 0
        self.preview_busy = False
        self.last_output = ""
        self.last_paste_at = 0.0
        self.context: list[str] = []
        self.hands_free = False
        self.pending_stop = None
        from ..mic import Mic
        self.mic = Mic()
        self.av_reader = None
        self._ui_busy = False
        self.onboarding = None
        self.onboarding_text = ""
        self.setup = None
        self.loading = True
        self.state_text = "Loading model…"
        self.ctx = None
        cam = opts.camera if opts.camera != "auto" else self.settings.get("camera", "auto")
        self.camera = Camera(cam, on_frame=self.on_frame)

    @property
    def key_name(self) -> str:
        return key_label(self.opts.key)

    def ui(self, fn, *args, **kw):
        """Run fn on the tk thread safely from any background thread."""
        self._q.put((fn, args, kw))

    def _pump(self):
        try:
            while True:
                fn, args, kw = self._q.get_nowait()
                try:
                    fn(*args, **kw)
                except Exception:
                    import traceback
                    traceback.print_exc()
        except queue.Empty:
            pass
        self.root.after(15, self._pump)

    def start(self):
        self.hud = HUD(self.root)
        self._install_key()
        self._build_tray()
        from .paste import get_uinput
        threading.Thread(target=get_uinput, name="lipflow-uinput-init", daemon=True).start()
        self.hud.show("reading", "Lipflow", "Loading the lip-reading model…")
        threading.Thread(target=self._worker, name="lipflow-model", daemon=True).start()
        self.jobs.put(("load",))
        self.root.after(15, self._pump)

    def _install_key(self):
        self.ptt = PushToTalk(
            self.opts.key,
            lambda hands_free: self.ui(self.on_start, hands_free),
            lambda: self.ui(self.on_stop),
            lambda silent=False: self.ui(self.on_cancel, silent),
        )
        self.ptt.install()

    def _build_tray(self):
        import pystray
        from pystray import Menu, MenuItem as Item

        def toggle(name, default=True, then=None):
            def act(icon, item):
                self.settings[name] = not self.settings.get(name, default)
                save_settings(self.settings)
                if then:
                    then()
            return Item(name_labels[name], act, checked=lambda item: self.settings.get(name, default))

        name_labels = {
            "whisper": "Whisper mode (lips + a soft whisper)",
            "use_context": "Use active window title for names",
            "save_clips": "Keep my last 100 clips (local)",
        }

        cams = list_cameras()
        cam_choices = ["auto"] + [c["index"] for c in cams] if cams else ["auto", 0, 1]

        def cam_title(v):
            if v == "auto":
                return "Automatic"
            for c in cams:
                if c["index"] == v:
                    return f"{c['name']} ({v})"
            return f"Camera {v}"

        def pick_camera(value):
            return Item(
                cam_title(value),
                lambda icon, item: self.ui(self._pick_camera, value),
                checked=lambda item: self.settings.get("camera", "auto") == value,
                radio=True,
            )

        def pick_key(name):
            return Item(
                key_label(name),
                lambda icon, item: self.ui(self._pick_key, name),
                checked=lambda item: self.opts.key == name,
                radio=True,
            )

        menu = Menu(
            Item(lambda item: self.state_text, None, enabled=False),
            Item(lambda item: f"Hold {self.key_name} to dictate, double-tap for hands-free", None, enabled=False),
            Item(lambda item: f"Cleanup: {self.cleaner.describe()}", None, enabled=False),
            Menu.SEPARATOR,
            Item("Copy last dictation", lambda icon, item: self.ui(self._copy_last)),
            Item("Practice & train more…", lambda icon, item: self.ui(self.show_setup, "practice")),
            Item("Run setup again…", lambda icon, item: self.ui(self.show_setup)),
            Menu.SEPARATOR,
            Item("Camera", Menu(*[pick_camera(v) for v in cam_choices])),
            Item("Push-to-talk key", Menu(*[pick_key(k) for k in KEYS])),
            toggle("whisper", False, then=lambda: self.ui(self._whisper_changed)),
            toggle("use_context"),
            toggle("save_clips"),
            Item("Start with Linux", lambda icon, item: self._toggle_autostart(),
                 checked=lambda item: self._autostart_enabled()),
            Menu.SEPARATOR,
            Item("Open history", lambda icon, item: self._open_file(HISTORY)),
            Item("Edit custom words…", lambda icon, item: self._edit_words()),
            Item("Open log", lambda icon, item: self._open_file(LOG)),
            Menu.SEPARATOR,
            Item("Quit Lipflow", lambda icon, item: self.ui(self.quit)),
        )
        self.icon = pystray.Icon("Lipflow", tray_image(False), "Lipflow", menu)
        threading.Thread(target=self.icon.run, name="lipflow-tray", daemon=True).start()

    def _set_icon(self, listening: bool):
        try:
            self.icon.icon = tray_image(listening)
        except Exception:
            pass

    def _set_state(self, text: str):
        self.state_text = text
        try:
            self.icon.update_menu()
        except Exception:
            pass

    def _copy_last(self):
        if self.last_output:
            copy_text(self.last_output)

    def _pick_camera(self, value):
        self.settings["camera"] = value
        save_settings(self.settings)
        self.camera.set_source(value)
        self.icon.update_menu()
        print(f"[lipflow] camera: {value}")

    def _pick_key(self, name):
        self.ptt.stop()
        self.opts.key = name
        self.settings["key"] = name
        save_settings(self.settings)
        self._install_key()
        self.icon.update_menu()
        self.hud.show("done", "Push-to-talk key", f"Hold {self.key_name} to dictate", 2.0)

    def _whisper_changed(self):
        if self.settings.get("whisper") and self.av_reader is None and not self.loading:
            self.jobs.put(("whisper",))

    @staticmethod
    def _open_file(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a", encoding="utf-8").close()
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", path])
        else:
            editor = os.environ.get("EDITOR", "nano")
            subprocess.Popen([editor, path])

    def _edit_words(self):
        from .. import vocab
        vocab.load()
        self._open_file(vocab.PATH)

    @staticmethod
    def _autostart_enabled() -> bool:
        return os.path.exists(AUTOSTART_FILE)

    def _toggle_autostart(self):
        os.makedirs(AUTOSTART_DIR, exist_ok=True)
        if self._autostart_enabled():
            try:
                os.remove(AUTOSTART_FILE)
            except OSError:
                pass
        else:
            desktop_entry = (
                "[Desktop Entry]\n"
                "Type=Application\n"
                "Name=Lipflow\n"
                "Comment=Silent dictation by lip reading\n"
                "Exec=lipflow run\n"
                "Terminal=false\n"
                "Categories=Utility;Accessibility;\n"
            )
            with open(AUTOSTART_FILE, "w", encoding="utf-8") as f:
                f.write(desktop_entry)

    def show_setup(self, start_at: str = "welcome"):
        from .setup import Setup
        if self.loading and start_at == "practice":
            self.hud.show("error", "Still loading", "Try again in a moment", 1.5)
            return
        if self.setup is None:
            self.setup = Setup(self)
        self.setup.show(start_at)

    def quit(self):
        self.camera.close()
        self.ptt.stop()
        from .paste import close_uinput
        close_uinput()
        try:
            self.icon.stop()
        except Exception:
            pass
        self.root.destroy()

    def on_start(self, hands_free: bool):
        if self.loading:
            self.hud.show("error", "Still loading", "The model is almost ready…", hide_after=1.5)
            return
        if self.pending_stop is not None:
            self._finish_stop(self.pending_stop)
        if hands_free and self.camera.recording is not None:
            self.hands_free = True
            self.hud.show("listening", "Hands-free · tap to finish", self.hud.body_text)
            return
        self.session += 1
        self.hands_free = hands_free
        from ..context import Context, capture
        self.ctx = capture() if self.settings.get("use_context", True) else Context()
        rec = self.camera.start_recording()
        if self.whisper_on:
            self.mic.start()
        self._set_icon(True)
        title = "Hands-free · tap to finish" if hands_free else "Listening"
        self.hud.show("listening", title, "" if self.camera.ready.is_set() else "Starting camera…")
        threading.Thread(target=self._preview_loop, args=(self.session, rec), daemon=True).start()

    def on_stop(self):
        self.hands_free = False
        if self.camera.recording is None:
            return
        self.session += 1
        self.pending_stop = self.session
        self._set_icon(False)
        self.hud.show("reading", "Reading your lips", self.hud.body_text)
        token = self.session
        self.root.after(int(TAIL_SECONDS * 1000), lambda: self._finish_stop(token))

    def _finish_stop(self, token):
        if self.pending_stop != token:
            return
        self.pending_stop = None
        rec = self.camera.stop_recording()
        audio = self.mic.stop() if self.whisper_on else []
        if rec is not None:
            rec.audio = audio
            self.jobs.put(("final", rec))

    def on_cancel(self, silent: bool = False):
        self.pending_stop = None
        self.mic.stop()
        self._set_icon(False)
        self.session += 1
        self.hands_free = False
        self.camera.stop_recording()
        if silent:
            self.hud.hide()
        else:
            self.hud.show("error", "Cancelled", "", hide_after=0.8)

    def on_frame(self, frame, obs, recording):
        rec = self.camera.recording
        if recording and rec is not None and rec.duration > MAX_SECONDS:
            self.ui(self.on_stop)
            return
        if self._ui_busy or (not recording and self.onboarding is None):
            return
        setup = mouth_view(frame, obs, 208, 130) if self.onboarding is not None else None
        pill = mouth_view(frame, obs, 112, 70) if recording else None
        self._ui_busy = True
        self.ui(self._show_frame, setup, pill)

    def _show_frame(self, setup, pill):
        try:
            if setup is not None and self.onboarding is not None:
                self.onboarding.set_frame(setup)
            if pill is not None:
                self.hud.set_frame(pill)
        finally:
            self._ui_busy = False

    def _preview_loop(self, session: int, rec: Recording):
        if not self.opts.live_preview:
            return
        waited = 0.0
        while self.session == session:
            time.sleep(PREVIEW_EVERY)
            waited += PREVIEW_EVERY
            if not rec.ts and (self.camera.error or waited > 6):
                msg = self.camera.error or "The camera isn't sending frames"
                print(f"[lipflow] camera problem: {msg}")
                self.ui(self.hud.show, "error", "Camera problem", msg, 6.0)
                return
            if self.session != session or self.preview_busy or len(rec.ts) < 15:
                continue
            self.preview_busy = True
            self.jobs.put(("preview", session, rec))

    def _worker(self):
        while True:
            job = self.jobs.get()
            try:
                if job[0] == "load":
                    self._load()
                elif job[0] == "preview":
                    self._preview(*job[1:])
                elif job[0] == "final":
                    self._final(job[1])
                elif job[0] == "train":
                    self._train(job[1])
                elif job[0] == "whisper":
                    self._load_whisper()
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.ui(self.hud.show, "error", "Something went wrong", str(e)[:80], 3.0)
            finally:
                if job[0] == "preview":
                    self.preview_busy = False

    def _load(self):
        t = time.time()
        self.reader = LipReader(beam_size=self.opts.beam)
        self.reader.warmup()
        self.loading = False
        print(f"[lipflow] model ready in {time.time() - t:.1f}s "
              f"(encoder on {self.reader.enc_device}, cleanup: {self.cleaner.describe()})")
        self.ui(self._set_state, "Ready")
        if self.settings.get("whisper"):
            self.jobs.put(("whisper",))
        if self.opts.onboard or not self.settings.get("onboarded"):
            self.ui(self.hud.hide)
            self.ui(self.show_setup)
        else:
            self.ui(self.hud.show, "done", "Lipflow is ready", f"Hold {self.key_name} and mouth your words", 2.5)

    @property
    def whisper_on(self) -> bool:
        return bool(self.settings.get("whisper")) and self.av_reader is not None and self.onboarding is None

    def _load_whisper(self):
        from .. import av
        if not av.available():
            self.ui(self.hud.show, "reading", "Whisper mode", "Downloading the audio-visual model (1.8 GB)…")
            try:
                av.download(lambda pct: self.ui(self.hud.set_text, f"Downloading the audio-visual model… {pct:.0f}%"))
            except Exception as e:
                self.ui(self.hud.show, "error", "Whisper mode", f"Download failed: {e}"[:80], 5.0)
                return
        self.ui(self.hud.show, "reading", "Whisper mode", "Loading…")
        self.av_reader = av.AVReader(beam_size=self.opts.beam)
        self.av_reader.warmup_av()
        print("[lipflow] whisper mode ready (lips + audio)")
        self.ui(self.hud.show, "done", "Whisper mode on", "Whisper or speak softly while you mouth the words", 3.0)

    def _av_candidates(self, rec, rois):
        from ..mic import segment
        if not self.whisper_on or not getattr(rec, "audio", None):
            return None
        ts, _, _ = rec.snapshot()
        wave = segment(rec.audio, ts[0], rois.shape[0])
        if wave is None:
            return None
        return self.av_reader.beam_search(self.av_reader.encode_av(rois, wave), nbest=5)

    def _preview(self, session: int, rec: Recording):
        if self.session != session:
            return
        rois = rois_for(rec)
        if rois is None:
            self.ui(self.hud.set_text, "Can't see your face…")
            return
        text = self.reader.greedy(self.reader.encode(rois))
        if self.session == session and text:
            self.ui(self.hud.set_text, text.lower())

    def _final(self, rec: Recording):
        t0 = time.time()
        ob = self.onboarding
        problem = clip_problem(rec)
        if problem and ob is not None:
            print(f"[lipflow] practice clip rejected ({rec.duration:.1f}s, {len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            self.ui(ob.clip_done, False, f"{problem[0]}. {problem[1]}.")
            self.ui(self.hud.hide)
            return
        if problem:
            print(f"[lipflow] skipped {rec.duration:.1f}s clip ({len(rec.ts)} frames, face in "
                  f"{rec.face_ratio:.0%}): {problem[0]}")
            self.ui(self.hud.show, "error", problem[0], problem[1], 2.2)
            return
        rois = rois_for(rec)
        enc = self.reader.encode(rois)
        t_enc = time.time() - t0
        if ob is not None:
            raw = self.reader.greedy(enc)
            print(f"[lipflow] practice clip saved ({rec.duration:.1f}s): {raw!r}")
            self.ui(ob.clip_done, True, "", rois, self.onboarding_text, raw)
            self.ui(self.hud.hide)
            return
        candidates = self._av_candidates(rec, rois)
        if not candidates or not candidates[0]:
            if candidates is not None:
                print("[lipflow] lips + audio read nothing, using lips only")
            candidates = self.reader.beam_search(enc, nbest=5)
        t_beam = time.time() - t0 - t_enc
        if not candidates or not candidates[0]:
            print(f"[lipflow] {rec.duration:.1f}s clip: nothing read")
            self.ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        self.ui(self.hud.set_text, candidates[0].lower())
        ctx = self.ctx
        text = self.cleaner(candidates, context=" ".join(self.context[-3:]), names=ctx.names if ctx else None)
        t_all = time.time() - t0
        print(f"[lipflow] {rec.duration:.1f}s clip → raw: {candidates[0]!r}\n"
              f"          → typed: {text!r}  (encode {t_enc:.2f}s, beam {t_beam:.2f}s, total {t_all:.2f}s)")
        if not text:
            self.ui(self.hud.show, "error", "Couldn't read that", "Try again, a little slower", 2.2)
            return
        out = text
        if self.last_paste_at and time.time() - self.last_paste_at < JOIN_WINDOW:
            out = " " + text
        self.last_output = text
        self.last_paste_at = time.time()
        self.context.append(text)
        log_history(rec, candidates, text, t_all, self.cleaner.describe())
        keep_clip(rois, candidates, text, self.settings)
        self.ui(paste_text if self.opts.paste else copy_text, out if self.opts.paste else text)
        self.ui(self.hud.show, "done", "Pasted" if self.opts.paste else "Copied", text, 2.4)

    def _train(self, ob):
        self.loading = True
        self.ui(self._set_state, "Training on your face…")
        r = train_on_face(self.opts.beam, ob.report)
        if r["after"] is not None:
            self.reader = LipReader(beam_size=self.opts.beam)
            self.reader.warmup()
            self.settings["training"] = {
                "before": r["before"], "after": r["after"], "kept": r["kept"],
                "clips": r["clips"], "at": time.time(),
            }
            save_settings(self.settings)
        self.loading = False
        self.ui(self._set_state, "Ready")
        ob.finished(r["before"], r["after"], r["kept"], r["note"])


def _log_to_file():
    os.makedirs(HOME, exist_ok=True)
    try:
        if os.path.getsize(LOG) > 5_000_000:
            os.replace(LOG, LOG + ".old")
    except OSError:
        pass
    f = open(LOG, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = f
    print(f"\n[lipflow] started {time.strftime('%Y-%m-%d %H:%M:%S')}")


def _already_running() -> bool:
    global _LOCK_FILE
    lock_path = os.path.join(HOME, "lipflow.lock")
    os.makedirs(HOME, exist_ok=True)
    try:
        _LOCK_FILE = open(lock_path, "w")
        fcntl.flock(_LOCK_FILE, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return False
    except (BlockingIOError, OSError):
        return True


def run(opts: Options):
    if sys.stdout is None or os.environ.get("LIPFLOW_APP"):
        _log_to_file()
    if _already_running():
        print("[lipflow] already running; look for the mouth icon in your system tray.")
        return
    lf = Lipflow(opts)
    lf.start()
    signal.signal(signal.SIGINT, lambda *a: lf.ui(lf.quit))
    print(f"[lipflow] hold {lf.key_name} and mouth your words · double-tap for hands-free · Esc cancels · Ctrl-C quits")
    lf.root.mainloop()
    os._exit(0)
