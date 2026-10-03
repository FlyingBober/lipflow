"""Global push-to-talk key on Linux via pynput and a local Unix domain socket. Timing lives in ptt.py."""
from __future__ import annotations

import os
import socket
import threading
import time

from ..paths import HOME
from ..ptt import PushToTalkState

try:
    from pynput import keyboard
    K = keyboard.Key
    KEYS = {
        "right_control": (K.ctrl_r,),
        "right_alt": (K.alt_r, K.alt_gr),
        "left_alt": (K.alt_l,),
        "right_shift": (K.shift_r,),
        "f9": (K.f9,),
        "f8": (K.f8,),
        "f10": (K.f10,),
        "f12": (K.f12,),
    }
    MODIFIERS = {
        K.ctrl, K.ctrl_l, K.ctrl_r,
        K.alt, K.alt_l, K.alt_r, K.alt_gr,
        K.shift, K.shift_l, K.shift_r,
        K.cmd, K.cmd_l, K.cmd_r,
    }
except Exception:
    keyboard = None
    K = None
    KEYS = {
        "right_control": ("ctrl_r",),
        "right_alt": ("alt_r", "alt_gr"),
        "left_alt": ("alt_l",),
        "right_shift": ("shift_r",),
        "f9": ("f9",),
        "f8": ("f8",),
        "f10": ("f10",),
        "f12": ("f12",),
    }
    MODIFIERS = set()
DEFAULT_KEY = "right_control"
SOCKET_PATH = os.path.join(HOME, "lipflow.sock")


def send_command(cmd: str) -> bool:
    """Send an IPC command ('start', 'stop', 'toggle', 'cancel') to a running Lipflow instance."""
    if not os.path.exists(SOCKET_PATH):
        return False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.connect(SOCKET_PATH)
            s.sendall(cmd.strip().encode("utf-8"))
            return True
    except OSError:
        return False


class PushToTalk(PushToTalkState):
    def __init__(self, key: str, on_start, on_stop, on_cancel):
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; choose from {', '.join(KEYS)}")
        super().__init__(on_start, on_stop, on_cancel)
        self.keys = KEYS[key]
        self._listener = None
        self._server_sock = None
        self._server_thread = None
        self._running = False

    def install(self):
        if keyboard is not None:
            try:
                self._listener = keyboard.Listener(on_press=self.press, on_release=self.release)
                self._listener.daemon = True
                self._listener.start()
            except Exception as e:
                print(f"[lipflow] warning: keyboard listener failed ({e}). Use IPC triggers on Wayland.")
        else:
            print("[lipflow] warning: keyboard listener unavailable (no display). Use IPC triggers.")

        self._start_ipc_server()

    def _start_ipc_server(self):
        os.makedirs(HOME, exist_ok=True)
        if os.path.exists(SOCKET_PATH):
            try:
                os.remove(SOCKET_PATH)
            except OSError:
                pass
        try:
            self._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server_sock.bind(SOCKET_PATH)
            self._server_sock.listen(5)
            self._running = True
            self._server_thread = threading.Thread(target=self._ipc_loop, name="lipflow-ipc", daemon=True)
            self._server_thread.start()
        except Exception as e:
            print(f"[lipflow] warning: could not bind IPC socket at {SOCKET_PATH}: {e}")

    def toggle(self):
        now = time.time()
        if now - getattr(self, "_last_toggle", 0.0) < 0.35:
            return  # debounce rapid repeats or key bounce
        self._last_toggle = now
        if self.active or self.hands_free:
            self.active = self.hands_free = self.down = False
            self.on_stop()
        else:
            self.active = self.hands_free = True
            self.down = False
            self.on_start(hands_free=True)

    def _ipc_loop(self):
        while self._running:
            try:
                conn, _ = self._server_sock.accept()
                with conn:
                    data = conn.recv(64).decode("utf-8").strip()
                    if data == "start":
                        self.key_down()
                    elif data == "stop":
                        self.key_up()
                    elif data == "toggle":
                        self.toggle()
                    elif data == "cancel":
                        self.other_key(True)
            except Exception:
                if not self._running:
                    break

    def stop(self):
        self._running = False
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:
                pass
        if self._server_sock is not None:
            try:
                self._server_sock.close()
            except Exception:
                pass
            if os.path.exists(SOCKET_PATH):
                try:
                    os.remove(SOCKET_PATH)
                except OSError:
                    pass

    def press(self, key):
        if key in self.keys:
            self.key_down()
        elif key not in MODIFIERS:
            self.other_key(K is not None and key == K.esc)

    def release(self, key):
        if key in self.keys:
            self.key_up()
