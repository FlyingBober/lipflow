"""Lipflow tray app (Windows and Linux). Hold a key, mouth the words, let go — the text appears at your cursor.

Threads: tk owns the main thread (overlay, setup window); pynput / evdev hook reports the key;
pystray runs the tray menu on its own thread; one model thread reads lips. Everything that touches
tk goes through ui(), which queues it for the main thread. Linux paste is wl-copy/wtype, or
xclip/xdotool on X11 (lipflow/linux/paste.py).
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from dataclasses import dataclass

from ..camera import Camera, Recording, mouth_view
from ..cleanup import CHOICES, Cleaner, unavailable
from ..dictation import (
    HISTORY, JOIN_WINDOW, MAX_SECONDS, PREVIEW_EVERY, TAIL_SECONDS, clip_problem, keep_clip, load_settings,
    log_history, rois_for, save_settings, train_on_face,
)
from ..paths import HOME
from ..vsr import LipReader
from .hotkey import DEFAULT_KEY, KEYS, PushToTalk
from .hud import HUD, tray_image

if sys.platform == "linux":
    from ..linux.paste import copy_text, paste_text
else:
    from .paste import copy_text, paste_text

LOG = os.path.join(HOME, "Lipflow.log")
CAMERAS = 4  # Windows can't name cameras through OpenCV: offer the first few by number
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
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
    language: str | None = None
    cleanup_mode: str | None = None
    confidence_policy: str | None = None
    min_margin: float = 0.5
    input_mode: str | None = None
    whisper_model: str = "small"


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
        opts.language = opts.language or self.settings.get("language", "en")
        opts.cleanup_mode = opts.cleanup_mode or self.settings.get("cleanup_mode", "faithful")
        if opts.onboard:
            opts.input_mode = "silent"
        else:
            opts.input_mode = opts.input_mode or self.settings.get(
                "input_mode", "whisper" if (self.settings.get("whisper") or opts.language in ("ru", "zh")) else "silent"
            )
        opts.whisper_model = getattr(opts, "whisper_model", None) or self.settings.get("whisper_model", "small")
        self.cleanup_choice = opts.backend if opts.backend != "auto" else self.settings.get("cleanup", "auto")
        if self.cleanup_choice == "auto" and opts.input_mode == "whisper":
            self.cleanup_choice = "basic"
        opts.confidence_policy = opts.confidence_policy or self.settings.get(
            "confidence_policy", "auto" if opts.input_mode == "whisper" else "review"
        )
        self.root = tk.Tk()
        self.root.withdraw()
        self._q: "queue.Queue" = queue.Queue()
        self.reader: LipReader | None = None
        self.whisper_asr = None
        self.review = None
        self.review_pending = False
        self.cleaner = Cleaner(self.cleanup_choice, opts.cleanup_mode, opts.language)
        self.jobs: "queue.Queue" = queue.Queue()
        self.session = 0          # bumps on every start/cancel so stale previews are dropped
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
        self.state_text = "Загрузка модели…" if opts.language == "ru" else "Loading model…"
        self.ctx = None
        cam = opts.camera if opts.camera != "auto" else self.settings.get("camera", "auto")
        self.camera = Camera(cam, on_frame=self.on_frame)

    @property
    def key_name(self) -> str:
        return key_label(self.opts.key)

    # -- threading -------------------------------------------------------------------------
    def ui(self, fn, *args, **kw):
        """Run fn on the tk thread (safe from any thread)."""
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

    # -- setup -----------------------------------------------------------------------------
    def start(self):
        self.hud = HUD(self.root)
        from .review import Review
        self.review = Review(self.root)
        self._install_key()
        self._build_tray()
        start_title = "Lipflow"
        start_msg = "Загрузка модели…" if self.opts.language == "ru" else "Loading the lip-reading model…"
        self.hud.show("reading", start_title, start_msg)
        threading.Thread(target=self._worker, name="lipflow-model", daemon=True).start()
        self.jobs.put(("load",))
        self.root.after(15, self._pump)

    def _install_key(self):
        # The hook thread only queues: all state changes happen on the tk thread.
        self.ptt = PushToTalk(self.opts.key,
                              lambda hands_free: self.ui(self.on_start, hands_free),
                              lambda: self.ui(self.on_stop),
                              lambda silent=False: self.ui(self.on_cancel, silent))
        self.ptt.install()

    def _build_tray(self):
        import pystray
        from pystray import Menu, MenuItem as Item

        is_ru = self.opts.language == "ru"

        def toggle(name, default=True, then=None):
            def act(icon, item):
                self.settings[name] = not self.settings.get(name, default)
                save_settings(self.settings)
                if then:
                    then()
            return Item(name_labels[name], act, checked=lambda item: self.settings.get(name, default))

        name_labels = {
            "use_context": "Использовать заголовок окна для имен" if is_ru else "Use the window title for names",
            "save_clips": "Хранить последние 100 клипов (локально)" if is_ru else "Keep my last 100 clips (local)",
        }

        def pick_lang(code):
            return Item("English" if code == "en" else "Русский",
                        lambda icon, item: self.ui(self._pick_language, code),
                        checked=lambda item: self.opts.language == code, radio=True)

        def pick_mode(m):
            label = ("Без звука (веб-камера)" if m == "silent" else "Whisper (тихая речь)") if self.opts.language == "ru" else (
                "Silent (webcam only)" if m == "silent" else "Whisper (quiet speech)"
            )
            return Item(label, lambda icon, item: self.ui(self._pick_input_mode, m),
                        checked=lambda item: self.opts.input_mode == m, radio=True)

        def pick_camera(value):
            return Item("Автоматически" if value == "auto" and is_ru else ("Automatic" if value == "auto" else f"Camera {value + 1}"),
                        lambda icon, item: self.ui(self._pick_camera, value),
                        checked=lambda item: self.settings.get("camera", "auto") == value, radio=True)

        from ..camera import list_cameras
        cams = list_cameras()
        if cams:
            def pick_cam(c):
                name = c["name"]
                idx = c["index"]
                label = f"{name} ({idx})" if any(o["name"] == name and o["index"] != idx for o in cams) else name
                return Item(label,
                            lambda icon, item, v=idx: self.ui(self._pick_camera, v),
                            checked=lambda item, v=idx: self.settings.get("camera", "auto") == v, radio=True)

            cam_menu = Menu(
                Item("Автоматически" if is_ru else "Automatic", lambda icon, item: self.ui(self._pick_camera, "auto"),
                     checked=lambda item: self.settings.get("camera", "auto") == "auto", radio=True),
                Menu.SEPARATOR,
                *[pick_cam(c) for c in cams],
            )
        else:
            cam_menu = Menu(*[pick_camera(v) for v in ["auto", *range(CAMERAS)]])

        def pick_cleanup(name):
            why = unavailable(name)
            return Item(CHOICES[name] + (f" ({why})" if why else ""),
                        lambda icon, item: self.ui(self._pick_cleanup, name),
                        checked=lambda item: self.cleanup_choice == name, radio=True, enabled=why is None)

        def pick_key(name):
            return Item(key_label(name), lambda icon, item: self.ui(self._pick_key, name),
                        checked=lambda item: self.opts.key == name, radio=True)

        autostart = []
        if sys.platform == "win32":
            autostart = [
                Item("Start with Windows", lambda icon, item: self._toggle_autostart(),
                     checked=lambda item: self._autostart_enabled()),
                Menu.SEPARATOR,
            ]
        elif sys.platform == "linux":
            autostart = [
                Item("Автозапуск при входе" if is_ru else "Start with system",
                     lambda icon, item: self._toggle_autostart_linux(),
                     checked=lambda item: os.path.exists(AUTOSTART_FILE)),
                Menu.SEPARATOR,
            ]

        header_hint = (
            f"Удерживайте {self.key_name} для диктовки, дважды нажмите для hands-free"
            if is_ru else
            f"Hold {self.key_name} to dictate, double-tap for hands-free"
        )
        cleanup_label = f"Очистка: {self.cleaner.describe()}" if is_ru else f"Cleanup: {self.cleaner.describe()}"
        copy_last_label = "Скопировать последнюю диктовку" if is_ru else "Copy last dictation"
        practice_label = "Тренировка и дообучение…" if is_ru else "Practice && train more…"
        run_setup_label = "Запустить настройку заново…" if is_ru else "Run setup again…"
        cam_label = "Камера" if is_ru else "Camera"
        clean_menu_label = "Очистка текста" if is_ru else "Cleanup"
        key_menu_label = "Клавиша push-to-talk" if is_ru else "Push-to-talk key"
        lang_menu_label = "Язык (Language)"
        mode_menu_label = "Режим ввода" if is_ru else "Input mode"
        history_label = "История диктовки" if is_ru else "Open history"
        words_label = "Пользовательские слова…" if is_ru else "Edit custom words…"
        log_label = "Открыть лог" if is_ru else "Open log"
        quit_label = "Выход из Lipflow" if is_ru else "Quit Lipflow"

        menu = Menu(
            Item(lambda item: self.state_text, None, enabled=False),
            Item(lambda item: header_hint, None, enabled=False),
            Item(lambda item: cleanup_label, None, enabled=False),
            Menu.SEPARATOR,
            Item(copy_last_label, lambda icon, item: self.ui(self._copy_last)),
            Item(practice_label, lambda icon, item: self.ui(self.show_setup, "practice")),
            Item(run_setup_label, lambda icon, item: self.ui(self.show_setup)),
            Menu.SEPARATOR,
            Item(lang_menu_label, Menu(*[pick_lang(l) for l in ["en", "ru"]])),
            Item(mode_menu_label, Menu(*[pick_mode(m) for m in ["silent", "whisper"]])),
            Item(cam_label, cam_menu),
            Item(clean_menu_label, Menu(*[pick_cleanup(b) for b in CHOICES])),
            Item(key_menu_label, Menu(*[pick_key(k) for k in KEYS])),
            toggle("use_context"),
            toggle("save_clips"),
            *autostart,
            Item(history_label, lambda icon, item: self._notepad(HISTORY)),
            Item(words_label, lambda icon, item: self._edit_words()),
            Item(log_label, lambda icon, item: self._notepad(LOG)),
            Menu.SEPARATOR,
            Item(quit_label, lambda icon, item: self.ui(self.quit)),
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

    # -- menu actions (tk thread unless noted) ---------------------------------------------------
    def _pick_language(self, code):
        if self.opts.language == code:
            return
        self.opts.language = self.settings["language"] = code
        save_settings(self.settings)
        self.cleaner = Cleaner(self.cleanup_choice, self.opts.cleanup_mode, self.opts.language)
        self.icon.update_menu()
        self.jobs.put(("load",))

    def _pick_input_mode(self, mode):
        if self.opts.input_mode == mode:
            return
        self.opts.input_mode = self.settings["input_mode"] = mode
        save_settings(self.settings)
        self.icon.update_menu()
        self.jobs.put(("load",))

    def _copy_last(self):
        if self.last_output:
            copy_text(self.last_output)

    def _pick_camera(self, value):
        self.settings["camera"] = value
        save_settings(self.settings)
        self.camera.set_source(value)
        self.icon.update_menu()
        print(f"[lipflow] camera: {value}")

    def _pick_cleanup(self, name):
        self.cleanup_choice = self.settings["cleanup"] = name
        save_settings(self.settings)
        self.icon.update_menu()
        self.jobs.put(("cleanup", name))  # on the model thread, between dictations

    def _pick_key(self, name):
        self.ptt.stop()
        self.opts.key = name
        self.settings["key"] = name
        save_settings(self.settings)
        self._install_key()
        self.icon.update_menu()
        title = "Клавиша push-to-talk" if self.opts.language == "ru" else "Push-to-talk key"
        body = f"Удерживайте {self.key_name} для диктовки" if self.opts.language == "ru" else f"Hold {self.key_name} to dictate"
        self.hud.show("done", title, body, 2.0)

    @staticmethod
    def _notepad(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "a", encoding="utf-8").close()
        if sys.platform == "win32":
            subprocess.Popen(["notepad.exe", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def _edit_words(self):
        from .. import vocab
        vocab.load()
        self._notepad(vocab.PATH)

    @staticmethod
    def _autostart_command() -> str:
        exe = sys.executable
        if exe.lower().endswith("python.exe"):  # no console window at login
            exe = exe[:-len("python.exe")] + "pythonw.exe"
        return f'"{exe}" -X utf8 -m lipflow'

    def _autostart_enabled(self) -> bool:
        if sys.platform != "win32":
            return False
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
                winreg.QueryValueEx(k, "Lipflow")
                return True
        except OSError:
            return False

    def _toggle_autostart(self):  # tray thread; registry only
        if sys.platform != "win32":
            return
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            if self._autostart_enabled():
                winreg.DeleteValue(k, "Lipflow")
            else:
                winreg.SetValueEx(k, "Lipflow", 0, winreg.REG_SZ, self._autostart_command())

    def _toggle_autostart_linux(self):
        if os.path.exists(AUTOSTART_FILE):
            try:
                os.remove(AUTOSTART_FILE)
            except OSError:
                pass
        else:
            os.makedirs(AUTOSTART_DIR, exist_ok=True)
            desktop_entry = (
                "[Desktop Entry]\n"
                "Type=Application\n"
                "Name=Lipflow\n"
                "Comment=Silent dictation by lip reading\n"
                f"Exec={sys.executable} -m lipflow\n"
                "Terminal=false\n"
                "Categories=Utility;\n"
            )
            with open(AUTOSTART_FILE, "w", encoding="utf-8") as f:
                f.write(desktop_entry)

    def show_setup(self, start_at: str = "welcome"):
        from .setup import Setup
        if self.loading and start_at == "practice":
            err_title = "Загрузка" if self.opts.language == "ru" else "Still loading"
            err_body = "Попробуйте еще раз через мгновение" if self.opts.language == "ru" else "Try again in a moment"
            self.hud.show("error", err_title, err_body, 1.5)
            return
        if self.setup is None:
            self.setup = Setup(self)
        self.setup.show(start_at)

    def quit(self):
        self.camera.close()
        self.ptt.stop()
        try:
            self.icon.stop()
        except Exception:
            pass
        self.root.destroy()

    # -- push-to-talk (tk thread) -------------------------------------------------------------
    def on_start(self, hands_free: bool):
        if self.loading:
            err_title = "Загрузка" if self.opts.language == "ru" else "Still loading"
            err_body = "Модель почти готова…" if self.opts.language == "ru" else "The model is almost ready…"
            self.hud.show("error", err_title, err_body, hide_after=1.5)
            return
        if self.pending_stop is not None:  # pressed again during the tail: finish the last one now
            self._finish_stop(self.pending_stop)
        if hands_free and self.camera.recording is not None:
            self.hands_free = True  # the second tap of a double-tap: keep the recording going
            hf_msg = "Hands-free · нажмите для завершения" if self.opts.language == "ru" else "Hands-free · tap to finish"
            self.hud.show("listening", hf_msg, self.hud.body_text)
            return
        self.session += 1
        self.hands_free = hands_free
        from ..context import Context, capture
        # the app you're typing into is in front right now
        self.ctx = capture() if self.settings.get("use_context", True) else Context()
        rec = self.camera.start_recording()
        if self.whisper_on:
            self.mic.start()
        self._set_icon(True)
        title = ("Hands-free · нажмите для завершения" if hands_free else "Слушаю") if self.opts.language == "ru" else (
            "Hands-free · tap to finish" if hands_free else "Listening"
        )
        cam_wait = "Запуск камеры…" if self.opts.language == "ru" else "Starting camera…"
        self.hud.show("listening", title, "" if self.camera.ready.is_set() else cam_wait)
        threading.Thread(target=self._preview_loop, args=(self.session, rec), daemon=True).start()

    def on_stop(self):
        self.hands_free = False
        if self.camera.recording is None:
            return
        self.session += 1
        self.pending_stop = self.session
        self._set_icon(False)
        read_title = "Чтение по губам" if self.opts.language == "ru" else "Reading your lips"
        self.hud.show("reading", read_title, self.hud.body_text)
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
            cancel_msg = "Отменено" if self.opts.language == "ru" else "Cancelled"
            self.hud.show("error", cancel_msg, "", hide_after=0.8)

    # -- camera thread ------------------------------------------------------------------------
    def on_frame(self, frame, obs, recording):
        """At most one video frame waits for the tk thread at a time, so frames never pile up in
        front of the key handling."""
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

    # -- model thread -------------------------------------------------------------------------
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
                err_title = "Ошибка камеры" if self.opts.language == "ru" else "Camera problem"
                self.ui(self.hud.show, "error", err_title, msg, 6.0)
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
                elif job[0] == "cleanup":
                    self.cleaner = Cleaner(job[1], self.opts.cleanup_mode, self.opts.language)
                    print(f"[lipflow] cleanup: {self.cleaner.describe()}")
                    self.ui(self.icon.update_menu)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.ui(self.hud.show, "error", "Something went wrong", str(e)[:80], 3.0)
            finally:
                if job[0] == "preview":
                    self.preview_busy = False

    def _load(self):
        t = time.time()
        # Silent VSR mode relies on an English visual encoder — for Russian it produces
        # random output unless fine-tuned. Switch to Whisper automatically unless we're in onboarding.
        if self.opts.input_mode == "silent" and self.opts.language == "ru" and not self.opts.onboard:
            print("[lipflow] silent mode не поддерживается для русского языка — переключаюсь на Whisper")
            self.opts.input_mode = "whisper"
            self.settings["input_mode"] = "whisper"
            save_settings(self.settings)

        if self.opts.input_mode == "whisper":
            self._load_whisper()
        else:
            try:
                self.reader = LipReader(beam_size=self.opts.beam, language=self.opts.language)
                self.reader.warmup()
            except Exception as e:
                print(f"[lipflow] visual model not available for {self.opts.language}: {e}")
                print("[lipflow] falling back to whisper mode")
                self.opts.input_mode = "whisper"
                self.settings["input_mode"] = "whisper"
                save_settings(self.settings)
                self._load_whisper()
        self.loading = False
        print(f"[lipflow] model ready in {time.time() - t:.1f}s "
              f"(mode: {self.opts.input_mode}, lang: {self.opts.language}, cleanup: {self.cleaner.describe()})")
        self.ui(self._set_state, "Ready")
        if self.opts.onboard or not self.settings.get("onboarded"):
            self.ui(self.hud.hide)
            self.ui(self.show_setup)
        else:
            ready_msg = "Удерживайте клавишу и говорите" if self.opts.language == "ru" else f"Hold {self.key_name} and mouth your words"
            ready_title = "Lipflow готов" if self.opts.language == "ru" else "Lipflow is ready"
            self.ui(self.hud.show, "done", ready_title, ready_msg, 2.5)

    @property
    def whisper_on(self) -> bool:
        return self.opts.input_mode == "whisper" or bool(self.settings.get("whisper"))

    def _load_whisper(self):
        from ..whisper import WhisperASR
        self.ui(self._set_state, f"Loading Whisper ({self.opts.language})…")
        print(f"[lipflow] loading Whisper model '{self.opts.whisper_model}' for {self.opts.language}...")
        self.whisper_asr = WhisperASR(model=self.opts.whisper_model, language=self.opts.language)
        self.whisper_asr.load()
        # Also try to load reader for live preview if weights exist
        if self.reader is None:
            try:
                self.reader = LipReader(beam_size=4, language=self.opts.language)
            except Exception:
                try:
                    self.reader = LipReader(beam_size=4, language="en")
                except Exception:
                    self.reader = None
        print(f"[lipflow] whisper mode ready ({self.opts.language})")
        self.ui(self._set_state, "Ready")

    def _preview(self, session: int, rec: Recording):
        if self.session != session:
            return
        rois = rois_for(rec)
        if rois is None:
            self.ui(self.hud.set_text, "Не видно лица…" if self.opts.language == "ru" else "Can't see your face…")
            return
        if self.reader is not None:
            text = self.reader.greedy(self.reader.encode(rois))
            if self.session == session and text:
                self.ui(self.hud.set_text, text.lower())
        elif self.session == session:
            self.ui(self.hud.set_text, "Говорите…" if self.opts.language == "ru" else "Listening…")

    def _final(self, rec: Recording):
        t0 = time.time()
        rec._t0 = t0
        ob = self.onboarding  # the setup window can be closed meanwhile on the tk thread
        problem = clip_problem(rec, input_mode=self.opts.input_mode)
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
        from ..delivery import choose_result, quality_for
        quality = quality_for(rec, rois) if rois is not None else None

        if ob is not None and rois is not None:
            # Don't run the model during onboarding — predictions are meaningless before
            # the user-specific adapter is trained. The correct text is self.onboarding_text.
            print(f"[lipflow] practice clip saved ({rec.duration:.1f}s, {len(rois)} frames)")
            self.ui(ob.clip_done, True, "", rois, self.onboarding_text, "")
            self.ui(self.hud.hide)
            return

        hyps = []
        greedy = ""

        if self.opts.input_mode == "whisper":
            from ..mic import segment
            ts, _, _ = rec.snapshot()
            wave = segment(rec.audio, ts[0], rois.shape[0]) if (rois is not None and getattr(rec, "audio", None)) else None
            hyps = self.whisper_asr.hypotheses(wave) if self.whisper_asr else []
            greedy = hyps[0].text if hyps else ""
        else:
            if self.reader is None or rois is None:
                err_title = "Нет модели VSR" if self.opts.language == "ru" else "No VSR model"
                err_msg = "Используйте режим whisper для русского" if self.opts.language == "ru" else "Use whisper mode for Russian"
                self.ui(self.hud.show, "error", err_title, err_msg, 3.0)
                return
            enc = self.reader.encode(rois)
            hyps = self.reader.hypotheses(enc, nbest=5)
            greedy = self.reader.greedy(enc)

        if not hyps or not hyps[0].text.strip():
            print(f"[lipflow] {rec.duration:.1f}s clip: nothing read")
            msg = "Ничего не распознано" if self.opts.language == "ru" else "Couldn't read that"
            sub = "Попробуйте еще раз" if self.opts.language == "ru" else "Try again, a little slower"
            self.ui(self.hud.show, "error", msg, sub, 2.2)
            return

        self.ui(self.hud.set_text, hyps[0].text)
        decision, clean_res, choices = choose_result(
            hyps,
            greedy,
            quality,
            self.cleaner,
            self.ctx,
            context=" ".join(self.context[-3:]),
            policy=self.opts.confidence_policy,
            min_margin=self.opts.min_margin,
            input_mode=self.opts.input_mode,
        )

        candidates = [h.text for h in hyps]
        raw_text = candidates[0]
        cleaned_text = clean_res.text if clean_res else raw_text
        t_all = time.time() - t0
        reason_info = f", reason: {decision.reason}" if decision.action != "auto" else ""
        print(f"[lipflow] {rec.duration:.1f}s clip → raw: {raw_text!r}\n"
              f"          → typed: {cleaned_text!r}  (process {t_all:.2f}s, decision: {decision.action}{reason_info})")

        if decision.action == "retry":
            err_title = "Повторите" if self.opts.language == "ru" else "Retry"
            self.ui(self.hud.show, "error", err_title, decision.reason, 2.5)
            return

        if decision.action == "review" and choices:
            self.review_pending = True
            self.ui(
                self.review.show,
                choices,
                decision.reason,
                lambda chosen: self.ui(self._deliver_choice, chosen, rec, rois, candidates),
            )
            return

        self._deliver_choice(cleaned_text, rec, rois, candidates)

    def _deliver_choice(self, text, rec, rois, candidates):
        self.review_pending = False
        if not text:
            msg = "Отменено" if self.opts.language == "ru" else "Cancelled"
            self.ui(self.hud.show, "error", msg, "", 1.0)
            return
        t_all = time.time() - getattr(rec, "_t0", time.time())
        out = text
        if self.last_paste_at and time.time() - self.last_paste_at < JOIN_WINDOW:
            out = " " + text
        self.last_output = text
        self.last_paste_at = time.time()
        self.context.append(text)
        log_history(rec, candidates, text, t_all, self.cleaner.describe())
        keep_clip(rois, candidates, text, self.settings)
        done_title = "Вставлено" if self.opts.paste else "Скопировано"
        if self.opts.language != "ru":
            done_title = "Pasted" if self.opts.paste else "Copied"
        self.ui(paste_text if self.opts.paste else copy_text, out if self.opts.paste else text)
        self.ui(self.hud.show, "done", done_title, text, 2.4)

    def _train(self, ob):
        self.loading = True
        train_msg = "Обучение модели под ваше лицо…" if self.opts.language == "ru" else "Training on your face…"
        self.ui(self._set_state, train_msg)
        r = train_on_face(self.opts.beam, ob.report, language=self.opts.language)
        if r["after"] is not None:
            self.reader = LipReader(beam_size=self.opts.beam, language=self.opts.language)
            self.reader.warmup()
            self.settings["training"] = {
                "before": r["before"], "after": r["after"], "kept": r["kept"],
                "clips": r["clips"], "at": time.time(),
            }
            save_settings(self.settings)
        self.loading = False
        self.ui(self._set_state, "Готов" if self.opts.language == "ru" else "Ready")
        ob.finished(r["before"], r["after"], r["kept"], r["note"])


def _log_to_file():
    """pythonw has no console: send prints and tracebacks to %APPDATA%\\Lipflow\\Lipflow.log."""
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
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateMutexW.restype = ctypes.c_void_p
        _already_running.handle = kernel32.CreateMutexW(None, False, "Local\\LipflowTray")  # held until exit
        return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS
    elif sys.platform == "linux":
        import fcntl
        lock_path = os.path.join(HOME, "lipflow.lock")
        os.makedirs(HOME, exist_ok=True)
        try:
            _LOCK_FILE = open(lock_path, "w")
            fcntl.flock(_LOCK_FILE, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except (BlockingIOError, OSError):
            return True
    return False


def run(opts: Options):
    if sys.stdout is None or os.environ.get("LIPFLOW_APP"):
        _log_to_file()
    if _already_running():
        if sys.platform == "win32":
            import ctypes
            ctypes.WinDLL("user32").MessageBoxW(None, "Lipflow is already running. Look for the pink mouth icon "
                                                      "in the system tray (you may need to click ^ to see it).",
                                                "Lipflow", 0x40)
        else:
            print("[lipflow] already running; look for the mouth icon in your system tray.")
        return
    lf = Lipflow(opts)
    lf.start()
    import signal
    signal.signal(signal.SIGINT, lambda *a: lf.ui(lf.quit))  # tk would swallow Ctrl-C in its callbacks
    print(f"[lipflow] hold {lf.key_name} and mouth your words · double-tap for hands-free · Esc cancels · Ctrl-C quits")
    lf.root.mainloop()
    os._exit(0)  # the hook, tray and model threads don't need a clean shutdown
