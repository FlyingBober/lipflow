"""lipflow [run] | lipflow file VIDEO | lipflow doctor"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("GLOG_minloglevel", "2")  # quiet MediaPipe
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# OpenCV would ask for camera access from its capture thread, which silently fails; the app
# asks on the main thread instead (app.request_camera).
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")


def _app():
    """(Options, run) for this OS's front end: menu bar on macOS, system tray on Windows and Linux."""
    if sys.platform == "win32":
        from .win.app import Options, run
    elif sys.platform == "darwin":
        from .app import Options, run
    else:
        from .linux.app import Options, run
    return Options, run


def main(argv=None):
    if sys.platform == "win32":
        for stream in (sys.stdout, sys.stderr):  # ✓ and → in a cp1252 console or a pipe
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        from .win.hotkey import DEFAULT_KEY, KEYS
    elif sys.platform == "darwin":
        from .hotkey import KEYS
        DEFAULT_KEY = "right_option"
    else:
        from .linux.hotkey import DEFAULT_KEY, KEYS

    p = argparse.ArgumentParser(prog="lipflow", description="Silent dictation by lip reading.")
    sub = p.add_subparsers(dest="cmd")

    r = sub.add_parser("run", help="start the dictation app in the menu bar / system tray (default)")
    r.add_argument("--key", default=DEFAULT_KEY, choices=list(KEYS), help="push-to-talk key")
    r.add_argument("--beam", type=int, default=4, help="beam size (higher = slower, about the same accuracy)")
    r.add_argument("--cleanup", default="auto", choices=["auto", "claude", "local", "ollama", "basic"])
    r.add_argument("--camera", default="auto",
                   help="'auto' (the built-in camera), a camera number, part of a camera's name (Mac), "
                        "or a video file")
    r.add_argument("--copy-only", action="store_true", help="copy to the clipboard instead of pasting")
    r.add_argument("--no-preview", action="store_true", help="don't show live words while you talk")
    r.add_argument("--language", choices=["en", "ru", "zh"], default=None, help="recognition language (en, ru, zh)")
    r.add_argument("--cleanup-mode", choices=["faithful", "polish"], default=None, help="faithful or polish cleanup")
    r.add_argument("--confidence-policy", choices=["review", "auto"], default=None,
                   help="review by default; auto uses uncalibrated heuristics")
    r.add_argument("--min-margin", type=float, default=0.5, help="length-normalized score gap for opt-in auto routing")
    r.add_argument("--input-mode", choices=["silent", "whisper"], default=None,
                   help="silent: webcam only; whisper: quiet-speech ASR with visual gating")
    r.add_argument("--whisper-model", default="small",
                   help="Whisper model name for whisper mode (small, base, tiny, medium, large-v3-turbo; default: small)")

    f = sub.add_parser("file", help="lip-read a video file")
    f.add_argument("video")
    f.add_argument("--start", type=float, default=0.0)
    f.add_argument("--end", type=float, default=None)
    f.add_argument("--beam", type=int, default=10)
    f.add_argument("--cleanup", default="auto", choices=["auto", "claude", "local", "ollama", "basic", "none"])
    f.add_argument("--mouth-roi", action="store_true", help="video already contains aligned 96x96 mouth crops")
    f.add_argument("--language", choices=["en", "ru", "zh"], default="en")
    f.add_argument("--cleanup-mode", choices=["faithful", "polish"], default="faithful")

    sub.add_parser("doctor", help="check permissions, camera and model files")
    o = sub.add_parser("onboard", help="open the setup window (permissions, Wispr import, train on your face)")
    o.add_argument("--language", choices=["en", "ru", "zh"], default=None, help="language (en, ru, zh)")
    sub.add_parser("train-lm", help="fine-tune the language model on your imported phrases")
    w = sub.add_parser("import-wispr", help="learn your phrasing from your Wispr Flow history (stays local)")
    w.add_argument("--from-text", help="import a plain-text file of your writing instead (one phrase per line)")
    sub.add_parser("trigger-start", help="send push-to-talk key down via IPC (for Wayland custom shortcuts)")
    sub.add_parser("trigger-stop", help="send push-to-talk key up via IPC")
    sub.add_parser("trigger-toggle", help="toggle push-to-talk recording via IPC")
    sub.add_parser("trigger-cancel", help="cancel active recording via IPC")

    valid_cmds = {"run", "file", "doctor", "import-wispr", "onboard", "train-lm",
                  "trigger-start", "trigger-stop", "trigger-toggle", "trigger-cancel", "-h", "--help"}
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in valid_cmds:
        argv.insert(0, "run")
    args = p.parse_args(argv)
    cmd = args.cmd

    if cmd.startswith("trigger-"):
        action = cmd.replace("trigger-", "")
        if sys.platform.startswith("linux"):
            from .linux.hotkey import send_command
            ok = send_command(action)
            if not ok:
                print(f"[lipflow] could not send {action} to running Lipflow (is 'lipflow run' active?)")
                sys.exit(1)
            sys.exit(0)
        else:
            print("[lipflow] trigger commands are supported on Linux")
            sys.exit(1)

    if cmd == "file":
        from .offline import transcribe_file
        from .vsr import LipReader
        mouth_roi = getattr(args, "mouth_roi", False)
        raw = transcribe_file(args.video, LipReader(beam_size=args.beam, language=args.language),
                              args.start, args.end, mouth_roi=mouth_roi)
        print("raw:  ", raw)
        if args.cleanup != "none":
            from .cleanup import Cleaner
            c = Cleaner(args.cleanup, args.cleanup_mode, args.language)
            result = c.process([raw])
            print(f"text:  {result.text}   [{c.describe()}]")
            if result.needs_review:
                print("review:", result.proposed, "\nreasons:", "; ".join(result.warnings))
    elif cmd == "import-wispr":
        from .personal import PHRASES, import_wispr, save_phrases
        from .vocab import PATH as WORDS
        if args.from_text:
            stats = save_phrases(open(args.from_text, encoding="utf-8").read().splitlines()) | {"source": args.from_text}
        else:
            stats = import_wispr()
        print(f"Imported {stats['phrases']:,} phrases ({stats['words']:,} words) from {stats['source']}")
        print(f"  saved to {PHRASES}")
        if stats["new_names"]:
            print(f"  added {len(stats['new_names'])} names/terms to {WORDS}. Review them: "
                  "Lipflow menu → Edit custom words")
        print("Restart Lipflow to use them.")
    elif cmd == "train-lm":
        from .train_lm import train
        r = train()
        print(f"Your held-out phrases: perplexity {r['before']['yours']:.1f} → {r['after']['yours']:.1f}; "
              f"general text {r['before']['general']:.1f} → {r['after']['general'] or r['before']['general']:.1f}"
              f" ({'saved' if r['saved'] else 'not better, not saved'})")
    elif cmd == "onboard":
        Options, run = _app()
        run(Options(onboard=True, language=args.language))
    elif cmd == "doctor":
        from .doctor import doctor
        sys.exit(doctor())
    else:
        Options, run = _app()
        camera = int(args.camera) if args.camera.isdigit() else args.camera
        run(Options(key=args.key, beam=args.beam, backend=args.cleanup, camera=camera,
                    paste=not args.copy_only, live_preview=not args.no_preview,
                    language=args.language, cleanup_mode=args.cleanup_mode,
                    confidence_policy=args.confidence_policy, min_margin=args.min_margin,
                    input_mode=args.input_mode, whisper_model=args.whisper_model))


if __name__ == "__main__":
    main()
