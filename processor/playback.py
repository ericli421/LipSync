"""Fallback for the demo: play a prerecorded stream recording to every listener.

The presenter picks a saved MP4 (from recordings/) on the dashboard. Every /listen
page then shows that video, with its sound, starting from the same position;
pages that join later start where it currently is. Turning it off returns them to
the live feed. Files are served read-only from recordings/ by /media/<path>.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path


class Playback:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lock = threading.Lock()
        self.file = None       # path relative to root, POSIX style
        self.started = None    # time.monotonic() when it started from 0 s

    def files(self, limit=20):
        """Newest recordings first; rescanned at most every 2 s (status is sent often)."""
        now = time.monotonic()
        cached = getattr(self, "_cache", None)
        if cached and now - cached[0] < 2.0:
            return cached[1][:limit]
        found = []
        if self.root.exists():
            found = sorted(self.root.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        names = [p.relative_to(self.root).as_posix() for p in found[:20]]
        self._cache = (now, names)
        return names[:limit]

    def resolve(self, relative):
        """The file under root for a relative path, or None (rejects anything outside root)."""
        if not isinstance(relative, str) or not relative.endswith(".mp4"):
            return None
        path = (self.root / relative).resolve()
        if self.root not in path.parents or not path.is_file():
            return None
        return path

    def update(self, message):
        enabled = message.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError("set_playback needs enabled: true or false")
        with self.lock:
            if not enabled:
                self.file, self.started = None, None
                return
            choice = message.get("file") or (self.files(1) or [None])[0]
            if choice is None:
                raise ValueError("no recordings to play: record the stream first")
            if self.resolve(choice) is None:
                raise ValueError(f"recording not found: {choice}")
            self.file, self.started = choice, time.monotonic()

    def status(self):
        with self.lock:
            active = self.file is not None
            return {"active": active, "file": self.file,
                    "position_s": round(time.monotonic() - self.started, 2) if active else None,
                    "files": self.files()}
