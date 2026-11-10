"""Global push-to-talk key. Timing lives in ptt.py.

Windows and X11 use a pynput hook. Wayland reads /dev/input instead: pynput only sees XWayland,
so a native Wayland window would swallow the hotkey. Windows skips keystrokes Lipflow injects
itself (the Ctrl+V paste). On Wayland those come from a virtual device, which the evdev hook ignores.
"""
from __future__ import annotations

import sys
import time

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
    # Like macOS, only a non-modifier key turns a held push-to-talk key into a shortcut. AltGr also sends
    # a Left Ctrl of its own, which must not cancel the dictation.
    MODIFIERS = {K.ctrl, K.ctrl_l, K.ctrl_r, K.alt, K.alt_l, K.alt_r, K.alt_gr, K.shift, K.shift_l, K.shift_r,
                 K.cmd, K.cmd_l, K.cmd_r}
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

from ..ptt import PushToTalkState
DEFAULT_KEY = "right_control"
LLKHF_INJECTED = 0x10
MASK_VK = 0xE8  # unassigned: tapping it while Alt is held stops Alt's release from opening app menus


class PushToTalk(PushToTalkState):
    def __init__(self, key: str, on_start, on_stop, on_cancel):
        if key not in KEYS:
            raise ValueError(f"unknown key {key!r}; choose from {', '.join(KEYS)}")
        super().__init__(on_start, on_stop, on_cancel)
        self.keys = KEYS[key]
        self._listener = None
        self._evdev = None
        self._ipc = None
        self._last_toggle = 0.0

    def toggle(self):
        """Toggle recording between active and inactive with debounce."""
        now = time.time()
        if now - self._last_toggle < 0.35:
            return
        self._last_toggle = now
        if self.active or self.hands_free:
            self.active = self.hands_free = self.down = False
            self.on_stop()
        else:
            self.active = self.hands_free = True
            self.down = False
            self.on_start(hands_free=True)

    def install(self):
        if sys.platform == "linux":
            from ..linux.ipc import IPCServer
            self._ipc = IPCServer(
                on_start=self.key_down,
                on_stop=self.key_up,
                on_toggle=self.toggle,
                on_cancel=lambda: self.other_key(is_esc=True),
            )
            self._ipc.start()

            from ..linux.paste import wayland_session
            if wayland_session():
                from ..linux.hotkey import EvdevHook
                hook = EvdevHook(self.press, self.release)
                if hook.start():
                    self._evdev = hook
                    return
                print("[lipflow] can't read /dev/input; Wayland hotkey falls back to the X11 hook. "
                      "Add your user to the input group and log in again, or use IPC triggers.")
        if keyboard is None:
            if sys.platform == "linux":
                print("[lipflow] no display server available; keyboard listener disabled. Use IPC triggers.")
            return
        kwargs = {"on_press": self.press, "on_release": self.release}
        if sys.platform == "win32":
            def filt(msg, data):
                # Runs before on_press/on_release; returning False hides the event from them only.
                return not (data.flags & LLKHF_INJECTED)
            kwargs["win32_event_filter"] = filt
        self._listener = keyboard.Listener(**kwargs)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._ipc is not None:
            self._ipc.stop()
        if self._evdev is not None:
            self._evdev.stop()
        if self._listener is not None:
            self._listener.stop()

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

    # Called on the listener thread with pynput Key / KeyCode objects.
    def press(self, key):
        if self._matches_key(key):
            if not self.down and K is not None and any(k in (K.alt_l, K.alt_r, K.alt_gr) for k in self.keys):
                self._mask_alt()
            self.key_down()
        elif not self._is_modifier(key):
            self.other_key(K is not None and key == K.esc)

    @staticmethod
    def _mask_alt():
        try:
            kc = keyboard.KeyCode.from_vk(MASK_VK)
            c = keyboard.Controller()
            c.press(kc)
            c.release(kc)
        except Exception:
            pass  # XTest does not exist on a Wayland session; the hotkey still works

    def release(self, key):
        if self._matches_key(key):
            self.key_up()
