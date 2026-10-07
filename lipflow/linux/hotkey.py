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
    KC = keyboard.KeyCode
    KEYS = {
        "right_control": (K.ctrl_r, KC.from_vk(65508), KC.from_vk(65032)),
        "left_control": (K.ctrl_l, K.ctrl, KC.from_vk(65507)),
        "right_alt": (K.alt_r, K.alt_gr, KC.from_vk(65514), KC.from_vk(65406)),
        "left_alt": (K.alt_l, KC.from_vk(65513)),
        "right_shift": (K.shift_r, KC.from_vk(65506)),
        "f9": (K.f9, KC.from_vk(65478)),
        "f8": (K.f8, KC.from_vk(65477)),
        "f10": (K.f10, KC.from_vk(65479)),
        "f12": (K.f12, KC.from_vk(65481)),
    }
    MODIFIERS = {
        K.ctrl, K.ctrl_l, K.ctrl_r,
        K.alt, K.alt_l, K.alt_r, K.alt_gr,
        K.shift, K.shift_l, K.shift_r,
        K.cmd, K.cmd_l, K.cmd_r,
    }
    MODIFIER_VKS = {
        65507, 65508,  # ctrl_l, ctrl_r
        65513, 65514, 65406,  # alt_l, alt_r, alt_gr
        65505, 65506,  # shift_l, shift_r
        65511, 65512,  # super_l, super_r
        65032,  # ISO_Next_Group (Russian layout toggle / AltGr)
    }
except Exception:
    keyboard = None
    K = None
    KC = None
    KEYS = {
        "right_control": ("ctrl_r",),
        "left_control": ("ctrl_l", "ctrl"),
        "right_alt": ("alt_r", "alt_gr"),
        "left_alt": ("alt_l",),
        "right_shift": ("shift_r",),
        "f9": ("f9",),
        "f8": ("f8",),
        "f10": ("f10",),
        "f12": ("f12",),
    }
    MODIFIERS = set()
    MODIFIER_VKS = set()
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


EVDEV_KEYS = {
    "right_control": (97,),   # KEY_RIGHTCTRL
    "left_control": (29,),    # KEY_LEFTCTRL
    "right_alt": (100,),      # KEY_RIGHTALT
    "left_alt": (56,),        # KEY_LEFTALT
    "right_shift": (54,),     # KEY_RIGHTSHIFT
    "f9": (67,),              # KEY_F9
    "f8": (66,),              # KEY_F8
    "f10": (68,),             # KEY_F10
    "f12": (88,),             # KEY_F12
}


class PushToTalk(PushToTalkState):
    def __init__(self, key: str, on_start, on_stop, on_cancel):
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; choose from {', '.join(KEYS)}")
        super().__init__(on_start, on_stop, on_cancel)
        self.key_name = key
        self.keys = KEYS[key]
        self._listener = None
        self._evdev_thread = None
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

        self._start_evdev_listener()
        self._start_ipc_server()

    def _start_evdev_listener(self):
        try:
            import evdev
            import select
            from evdev import ecodes
            devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
            keyboards = [
                d for d in devices
                if ecodes.EV_KEY in d.capabilities()
                and ecodes.KEY_A in d.capabilities().get(ecodes.EV_KEY, [])
            ]
            if not keyboards:
                if os.environ.get("WAYLAND_DISPLAY") or os.environ.get("XDG_SESSION_TYPE") == "wayland":
                    print("[lipflow] Note: on Wayland, reading global keys (like Right Ctrl) across all windows "
                          "requires: sudo usermod -aG input $USER (or use KDE Shortcuts with F9/commands).")
                return
            target_codes = set(EVDEV_KEYS.get(self.key_name, ()))
            if not target_codes:
                return

            self._running = True

            def loop():
                while self._running:
                    r, _, _ = select.select(keyboards, [], [], 0.5)
                    for dev in r:
                        try:
                            for ev in dev.read():
                                if ev.type == ecodes.EV_KEY and ev.code in target_codes:
                                    if ev.value == 1:
                                        self.key_down()
                                    elif ev.value == 0:
                                        self.key_up()
                        except Exception:
                            pass

            self._evdev_thread = threading.Thread(target=loop, name="lipflow-evdev", daemon=True)
            self._evdev_thread.start()
            print(f"[lipflow] kernel evdev active for {self.key_name} on {len(keyboards)} keyboard(s)")
        except Exception:
            pass

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

    def _matches_key(self, key) -> bool:
        if key in self.keys:
            return True
        vk = getattr(key, "vk", None)
        if vk is None and hasattr(key, "value"):
            vk = getattr(key.value, "vk", None)
        for target in self.keys:
            if target == key:
                return True
            target_vk = getattr(target, "vk", None)
            if target_vk is None and hasattr(target, "value"):
                target_vk = getattr(target.value, "vk", None)
            if target_vk is not None and vk is not None and target_vk == vk:
                return True
        return False

    def _is_modifier(self, key) -> bool:
        if key in MODIFIERS:
            return True
        vk = getattr(key, "vk", None)
        if vk is None and hasattr(key, "value"):
            vk = getattr(key.value, "vk", None)
        return vk in MODIFIER_VKS

    def press(self, key):
        if self._matches_key(key):
            self.key_down()
        elif not self._is_modifier(key):
            self.other_key(K is not None and key == K.esc)

    def release(self, key):
        if self._matches_key(key):
            self.key_up()
