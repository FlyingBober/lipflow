"""Linux front end: clipboard, window context, overlay, tray icon, setup window and IPC triggers.
Runs on Linux only.
"""
import importlib
import os
import sys
import types

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only")


def test_data_lives_in_xdg_data_home(monkeypatch):
    import lipflow.paths as paths
    monkeypatch.delenv("LIPFLOW_HOME", raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", "/custom/data")
    try:
        assert importlib.reload(paths).HOME == "/custom/data/lipflow"
    finally:
        importlib.reload(paths)


def test_clipboard_round_trip_with_unicode():
    from lipflow.linux import paste
    before = paste.get_text()
    test_str = "lipflow linux · naïve ✓"
    paste.set_text(test_str)
    assert paste.get_text() == test_str
    if before is not None:
        paste.set_text(before)


def test_context_capture_never_raises():
    from lipflow.context import capture
    ctx = capture()
    assert isinstance(ctx.title, str) and isinstance(ctx.app, str)


def test_camera_backend_and_names():
    from lipflow.camera import list_cameras, resolve_camera
    cams = list_cameras()
    assert isinstance(cams, list)
    resolved = resolve_camera("auto")
    assert isinstance(resolved, int)


@pytest.fixture
def tk_root():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    yield root
    root.destroy()


def test_overlay_shows_text_and_video(tk_root):
    from lipflow.linux.hud import HUD
    hud = HUD(tk_root)
    hud.show("listening", "Listening", "")
    hud.set_frame(np.zeros((70, 112, 3), np.uint8))
    hud.set_text("the birch canoe slid on the smooth planks")
    tk_root.update()
    assert hud.body_text.startswith("the birch") and hud._visible
    hud.show("done", "Pasted", "Hello.", hide_after=0.01)
    hud.hide()
    tk_root.update()
    assert not hud._visible


def test_tray_icons():
    from lipflow.linux.hud import tray_image
    assert tray_image(False).size == (64, 64) and tray_image(True).getpixel((32, 32))[3] == 255


def test_setup_window_pages(tk_root, tmp_path, monkeypatch):
    import queue
    monkeypatch.setattr("lipflow.practice.CLIPS", str(tmp_path))
    from lipflow.linux.setup import Setup
    camera = types.SimpleNamespace(track_always=False, ensure_open=lambda: None)
    app = types.SimpleNamespace(
        root=tk_root, key_name="Right Ctrl", camera=camera, onboarding=None,
        onboarding_text="", reader=None, jobs=queue.Queue(), settings={},
        ui=lambda fn, *a, **k: fn(*a, **k),
        hud=types.SimpleNamespace(show=lambda *a, **k: None),
    )
    s = Setup(app)
    s.show()
    s.goWords()
    s.goPractice()
    assert app.onboarding is s and app.onboarding_text
    s.set_frame(np.zeros((130, 208, 3), np.uint8))
    s.clip_done(False, "Too short. Hold the key.")
    s.clip_done(True, "", np.zeros((30, 88, 88), np.uint8), app.onboarding_text, "THE BIRCH")
    assert s.i == 1 and len(list(tmp_path.glob("*.npz"))) == 1
    s.redo()
    assert s.i == 0 and not list(tmp_path.glob("*.npz"))
    s.goTrain()
    assert app.jobs.get_nowait()[0] == "train"
    s.report(50, "Training")
    s.finished(0.4, 0.3, True, "")
    tk_root.update()
    s.close()


def test_linux_app_imports():
    import lipflow.linux.app as A
    assert A.key_label("right_control") == "Right Ctrl"
    assert A.key_label("right_alt") == "Right Alt"
    assert A.key_label("f9") == "F9"
    assert A.key_label("f12") == "F12"


def test_linux_hotkey_toggle():
    from lipflow.linux.hotkey import PushToTalk
    started, stopped = [], []
    ptt = PushToTalk(
        "f9",
        lambda hands_free=False: started.append(hands_free),
        lambda: stopped.append(True),
        lambda silent=False: None,
    )
    # First toggle starts hands-free recording
    ptt.toggle()
    assert len(started) == 1 and started[0] is True
    assert ptt.active and ptt.hands_free

    # Second toggle stops recording
    ptt._last_toggle = 0.0  # bypass debounce for test
    ptt.toggle()
    assert len(stopped) == 1
    assert not ptt.active and not ptt.hands_free


def test_press_ctrl_v_never_raises():
    from lipflow.linux.paste import _press_ctrl_v, close_uinput
    try:
        _press_ctrl_v()
    finally:
        close_uinput()


def test_linux_right_control_and_layout_switching():
    from pynput.keyboard import Key, KeyCode
    from lipflow.linux.hotkey import PushToTalk

    log = []
    ptt = PushToTalk(
        "right_control",
        lambda hands_free=False: log.append(("start", hands_free)),
        lambda: log.append(("stop",)),
        lambda silent=False: log.append(("cancel", silent)),
    )

    # 1. Standard pynput Key.ctrl_r
    ptt.press(Key.ctrl_r)
    assert log == [("start", False)]
    ptt.down_at -= 1.0  # hold
    ptt.release(Key.ctrl_r)
    assert log == [("start", False), ("stop",)]

    # 2. X11 raw keysym 65508 (0xffe4: Control_R)
    log.clear()
    ptt.press(KeyCode.from_vk(65508))
    assert log == [("start", False)]
    ptt.down_at -= 1.0
    ptt.release(KeyCode.from_vk(65508))
    assert log == [("start", False), ("stop",)]

    # 3. Russian layout switcher keysym 65032 (0xfe08: ISO_Next_Group) on keycode 105
    log.clear()
    ptt.press(KeyCode.from_vk(65032))
    assert log == [("start", False)]
    ptt.down_at -= 1.0
    ptt.release(KeyCode.from_vk(65032))
    assert log == [("start", False), ("stop",)]



