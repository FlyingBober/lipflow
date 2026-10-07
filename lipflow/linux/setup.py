"""First-run setup on Linux: welcome -> import Wispr Flow -> mouth ~24 sentences -> train on your face."""
from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import ttk

from ..practice import N_SENTENCES, practice_sentences, save_clip, saved_clips
from .hud import ACCENT, AMBER, BG, DIM, FG, FONT, GREEN, photo

WW, WH = 600, 560


class Setup:
    def __init__(self, app):
        self.app = app
        opts = getattr(app, "opts", None)
        self.lang = (getattr(opts, "language", "en") if opts else "en") or "en"
        self.win = tk.Toplevel(app.root)
        self.win.title("Настройка Lipflow" if self.lang == "ru" else "Lipflow setup")
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        from .hud import get_current_monitor, get_monitors
        mx, my, mw, mh = get_current_monitor(app.root, get_monitors())
        self.win.geometry(f"{WW}x{WH}+{mx + (mw - WW) // 2}+{my + (mh - WH) // 3}")
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.page = None
        self.video = None
        self._img = None

    def _new_page(self, title: str, subtitle: str, step: int):
        if self.page is not None:
            self.page.destroy()
        self.video = None
        p = self.page = tk.Frame(self.win, bg=BG)
        p.pack(fill="both", expand=True, padx=40, pady=28)
        steps = tk.Frame(p, bg=BG)
        steps.pack(side="bottom", pady=(10, 0))
        step_labels = ["Слова", "Практика", "Обучение"] if self.lang == "ru" else ["Your words", "Practice", "Train"]
        for k, lab in enumerate(step_labels):
            color = ACCENT if k == step else GREEN if k < step else "#6b6670"
            tk.Label(steps, text=("✓ " if k < step else "") + lab, fg=color, bg=BG,
                     font=(FONT, 9, "bold")).pack(side="left", padx=18)
        self.page_title = tk.Label(p, text=title, fg=FG, bg=BG, font=(FONT, 20, "bold"))
        self.page_title.pack(pady=(24, 8))
        self.page_sub = tk.Label(p, text=subtitle, fg=DIM, bg=BG, font=(FONT, 11), wraplength=WW - 100,
                                 justify="center")
        self.page_sub.pack()
        return p

    def _button(self, parent, text, command, primary=False):
        return tk.Button(parent, text=text, command=command, relief="flat", cursor="hand2",
                         bg=ACCENT if primary else "#34303a", fg="white", activebackground="#d62961",
                         activeforeground="white", font=(FONT, 11, "bold" if primary else "normal"),
                         padx=22, pady=8, bd=0)

    def show(self, start_at: str = "welcome"):
        self.win.deiconify()
        self.win.lift()
        self.win.focus_force()
        if start_at == "practice":
            self.goPractice()
        else:
            self.welcome()

    def close(self):
        self.app.onboarding = None
        self.app.camera.track_always = False
        self.win.withdraw()

    def welcome(self):
        key = self.app.key_name
        if self.lang == "ru":
            p = self._new_page("Lipflow читает по губам",
                               f"Удерживайте {key}, произносите слова беззвучно одними губами, отпустите клавишу — "
                               "и текст появится там, где вы печатаете. Настройка займет около 10 минут: система изучит "
                               "вашу речь и движения губ.", -1)
            tk.Label(p, text="Все данные остаются на этом компьютере: видео с камеры, ваши тренировочные клипы "
                             "и обученные модели. Ничего никуда не отправляется.",
                     fg=DIM, bg=BG, font=(FONT, 10), wraplength=WW - 120, justify="center").pack(pady=26)
            self._button(p, "Начать", self.goWords, primary=True).pack(pady=12)
        else:
            p = self._new_page("Lipflow reads your lips",
                               f"Hold {key}, mouth what you want to say without making a sound, let go, and the "
                               "text appears wherever you're typing. Setup takes about 10 minutes: it learns your "
                               "words and your face.", -1)
            tk.Label(p, text="Everything stays on this PC: the camera video, your practice clips and the models "
                             "trained on them. Nothing is uploaded.",
                     fg=DIM, bg=BG, font=(FONT, 10), wraplength=WW - 120, justify="center").pack(pady=26)
            self._button(p, "Get started", self.goWords, primary=True).pack(pady=12)

    def goWords(self):
        from ..personal import PHRASES, WISPR_DIR
        if self.lang == "ru":
            p = self._new_page("Обучение вашим словам",
                               "Для русского языка практика использует сбалансированные фразы, "
                               "покрывающие все основные звуки и формы губ.", 0)
            self.words_status = tk.Label(p, text="Готово к записи тренировочных предложений на русском языке.",
                                         fg=FG, bg=BG, font=(FONT, 11), wraplength=WW - 120)
            self.words_status.pack(pady=24)
            row = tk.Frame(p, bg=BG)
            row.pack(pady=8)
            self._button(row, "Продолжить", self.goPractice, primary=True).pack(side="left", padx=6)
        else:
            p = self._new_page("Teach it your words",
                               "Most of what you'll mouth is stuff you already say. If you use Wispr Flow, Lipflow "
                               "can learn your phrasing and names from its history, read locally, never uploaded.", 0)
            self.words_status = tk.Label(p, text="", fg=FG, bg=BG, font=(FONT, 11), wraplength=WW - 120)
            self.words_status.pack(pady=24)
            row = tk.Frame(p, bg=BG)
            row.pack(pady=8)
            if os.path.exists(PHRASES):
                n = sum(1 for _ in open(PHRASES, encoding="utf-8"))
                self.words_status.configure(text=f"Already imported: {n:,} of your phrases.")
            elif os.path.isdir(WISPR_DIR):
                self._button(row, "Import from Wispr Flow", self.importWispr).pack(side="left", padx=6)
            else:
                self.words_status.configure(text="Wispr Flow isn't installed here, so practice uses standard "
                                                 "sentences. That works fine.")
            self._button(row, "Continue", self.goPractice, primary=True).pack(side="left", padx=6)

    def importWispr(self):
        self.words_status.configure(text="Reading your Wispr Flow history…")

        def work():
            from ..personal import import_wispr
            try:
                s = import_wispr()
                msg = f"Imported {s['phrases']:,} phrases and {len(s['new_names'])} names."
            except Exception as e:
                msg = f"Couldn't import: {e}"
            self.app.ui(self.words_status.configure, text=msg)
        threading.Thread(target=work, daemon=True).start()

    def goPractice(self):
        done = {c["text"] for c in saved_clips(language=self.lang)}
        self.sentences = [x for x in practice_sentences(N_SENTENCES * 2, language=self.lang) if x not in done][:N_SENTENCES]
        self.i = 0
        self.round_clips = []
        prior = len(done)
        if self.lang == "ru":
            sub = (f"Удерживайте {self.app.key_name}, беззвучно произнесите фразу одними губами в обычном темпе, затем "
                   "отпустите. Смотрите в камеру при хорошем освещении."
                   + (f" У вас уже записано {prior} клипов; новые добавятся к ним." if prior else ""))
            p = self._new_page("Произнесите каждую фразу", sub, 1)
        else:
            sub = (f"Hold {self.app.key_name}, silently mouth the sentence at your normal pace, then let "
                   "go. Face the camera with your mouth in good light."
                   + (f" You have {prior} clips from before; these add to them." if prior else ""))
            p = self._new_page("Mouth each sentence", sub, 1)
        self.count = tk.Label(p, text="", fg=ACCENT, bg=BG, font=(FONT, 9, "bold"))
        self.count.pack(pady=(18, 2))
        self.prompt = tk.Label(p, text="", fg=FG, bg=BG, font=(FONT, 17), wraplength=WW - 100)
        self.prompt.pack(pady=6)
        holder = tk.Frame(p, bg="#000000", width=208, height=130)
        holder.pack_propagate(False)
        holder.pack(pady=8)
        self.video = tk.Label(holder, bg="#000000")
        self.video.pack(fill="both", expand=True)
        self.feedback = tk.Label(p, text="", fg=DIM, bg=BG, font=(FONT, 10))
        self.feedback.pack()
        row = tk.Frame(p, bg=BG)
        row.pack(pady=10)
        self._button(row, "Повторить" if self.lang == "ru" else "Redo last", self.redo).pack(side="left", padx=6)
        self._button(row, "Пропустить" if self.lang == "ru" else "Skip sentence", self.skip).pack(side="left", padx=6)
        self.app.camera.track_always = True
        self.app.camera.ensure_open()
        self.app.onboarding = self
        self._show_sentence()

    def _show_sentence(self):
        count_lbl = f"ПРЕДЛОЖЕНИЕ {self.i + 1} ИЗ {N_SENTENCES}" if self.lang == "ru" else f"SENTENCE {self.i + 1} OF {N_SENTENCES}"
        self.count.configure(text=count_lbl)
        s = self.sentences[self.i % len(self.sentences)]
        self.prompt.configure(text=s)
        self.app.onboarding_text = s

    def set_frame(self, video_bgr):
        if self.app.onboarding is self and video_bgr is not None and self.video is not None:
            self._img = photo(video_bgr)
            self.video.configure(image=self._img)

    def clip_done(self, rec_ok: bool, message: str, rois=None, text=None, raw=None):
        if not rec_ok:
            self.feedback.configure(fg=AMBER, text=message)
            return
        self.round_clips.append(save_clip(rois, text, raw, language=self.lang))
        if raw and raw.strip():
            msg = f"Сохранено. Распознано: \"{raw.lower()}\"" if self.lang == "ru" else f"Saved. The model read: \"{raw.lower()}\""
        else:
            msg = "Сохранено." if self.lang == "ru" else "Saved."
        self.feedback.configure(fg=DIM, text=msg)
        self.i += 1
        if self.i >= N_SENTENCES:
            self.app.onboarding = None
            return self.goTrain()
        self._show_sentence()

    def redo(self):
        if self.round_clips and self.i > 0:
            os.remove(self.round_clips.pop())
            self.i -= 1
            msg = "Удален последний клип. Попробуйте еще раз." if self.lang == "ru" else "Removed the last clip. Try it again."
            self.feedback.configure(fg=DIM, text=msg)
            self._show_sentence()

    def skip(self):
        self.sentences.append(self.sentences.pop(self.i % len(self.sentences)))
        self._show_sentence()

    def goTrain(self):
        self.app.onboarding = None
        self.app.camera.track_always = False
        gpu = self.app.reader is not None and getattr(self.app.reader, "enc_device", None) is not None and self.app.reader.enc_device.type == "cuda"
        if self.lang == "ru":
            title = "Обучение модели под ваше лицо"
            sub = (("Обучение выполняется на видеокарте" if gpu else
                    "Обучение выполняется на процессоре, это займет некоторое время")
                   + ". Вы можете продолжать работу; Lipflow приостановлен до завершения.")
            status = "Запуск…"
            btn_text = "Начать использование Lipflow"
        else:
            title = "Learning your face"
            sub = (("Training runs on your graphics card" if gpu else
                    "Training runs on the processor, so it can take a while")
                   + ". You can keep working; Lipflow is paused until it's done.")
            status = "Starting…"
            btn_text = "Start using Lipflow"
        p = self._new_page(title, sub, 2)
        self.progress = ttk.Progressbar(p, length=WW - 160, maximum=100)
        self.progress.pack(pady=(40, 12))
        self.train_status = tk.Label(p, text=status, fg=DIM, bg=BG, font=(FONT, 11), wraplength=WW - 120)
        self.train_status.pack()
        self.result = tk.Label(p, text="", fg=FG, bg=BG, font=(FONT, 13, "bold"), wraplength=WW - 120)
        self.result.pack(pady=18)
        self.done_btn = self._button(p, btn_text, self.finish, primary=True)
        self.app.jobs.put(("train", self))

    def report(self, pct: float, text: str):
        self.app.ui(self.progress.configure, value=pct)
        self.app.ui(self.train_status.configure, text=text)

    def finished(self, before: float, after: "float | None", kept: bool, note: str):
        def ui():
            self.progress.configure(value=100)
            if self.lang == "ru":
                self.page_title.configure(text="Все готово!" if (after is None or kept) else "Настройка завершена")
                self.page_sub.configure(text=f"Удерживайте {self.app.key_name} в любой программе и произносите слова. "
                                             "Вы можете переобучить модель в любое время через меню в трее.")
                if after is None:
                    self.result.configure(text=note)
                else:
                    self.result.configure(
                        fg=GREEN if kept else AMBER,
                        text=f"Обучение успешно завершено!\n{note}".strip())
            else:
                self.page_title.configure(text="You're all set" if (after is None or kept) else "Setup finished")
                self.page_sub.configure(text=f"Hold {self.app.key_name} anywhere and mouth your words. You can "
                                             "retrain any time from the tray menu: more practice makes it better.")
                if after is None:
                    self.result.configure(text=note)
                else:
                    self.result.configure(
                        fg=GREEN if kept else AMBER,
                        text=f"Words read correctly on sentences it didn't train on: {1 - before:.0%} → {1 - after:.0%}"
                             + ("" if kept else "\nNo improvement, so the standard model stays."))
            self.train_status.configure(text=note if after is not None else "")
            self.done_btn.pack(pady=8)
        self.app.ui(ui)

    def finish(self):
        from ..dictation import load_settings, save_settings
        s = load_settings()
        s["onboarded"] = True
        save_settings(s)
        self.app.settings["onboarded"] = True
        self.close()
        ready_title = "Lipflow готов" if self.lang == "ru" else "Lipflow is ready"
        ready_body = f"Удерживайте {self.app.key_name} и произносите слова" if self.lang == "ru" else f"Hold {self.app.key_name} anywhere and mouth your words"
        self.app.hud.show("done", ready_title, ready_body, 3.0)
