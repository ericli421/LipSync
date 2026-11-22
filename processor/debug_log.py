"""Bounded server debug log, shown in the dashboard and the phone app."""
from collections import deque
from datetime import datetime
import logging
import threading

LEVELS = ("info", "warn", "error")
logger = logging.getLogger("lipsync")


class DebugLog:
    def __init__(self, size=200):
        self.lines = deque(maxlen=size)
        self.lock = threading.Lock()
        self.seq = 0
        self.last_key_time = {}

    def log(self, level, source, message, every_s=None):
        """Add a line; with every_s, repeat the same source+level at most that often."""
        now = datetime.now()
        key = (source, level)
        with self.lock:
            if every_s is not None:
                last = self.last_key_time.get(key)
                if last and (now - last).total_seconds() < every_s:
                    return
                self.last_key_time[key] = now
            self.seq += 1
            line = {"seq": self.seq, "time": now.strftime("%H:%M:%S"), "level": level,
                    "source": source, "message": str(message)[:300]}
            self.lines.append(line)
        getattr(logger, {"warn": "warning"}.get(level, level))("[%s] %s", source, message)
        return line

    def recent(self, count=40):
        with self.lock:
            return list(self.lines)[-count:]
