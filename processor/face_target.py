"""Follow one chosen face across frames, so lip motion always comes from the same person.

The tracker used to pick "the largest face near the centre" independently in every
frame. With two people in view at similar sizes that choice can jump between them,
mixing both mouths into one openness trace, so the lip match flips between voices.
FaceTarget keeps the same person (nearest to their last position), holds through
short detection dropouts, and only re-acquires after `lost_after_s`. A viewer can
pick the person explicitly (dashboard click, or "next face" from the phone).
"""
from __future__ import annotations

import threading


def face_box(landmarks):
    """Normalized (cx, cy, width, height) of a MediaPipe landmark list."""
    xs = [p.x for p in landmarks]
    ys = [p.y for p in landmarks]
    return ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, max(xs) - min(xs), max(ys) - min(ys))


class FaceTarget:
    def __init__(self, max_jump=0.2, lost_after_s=1.0, region=(0.5, 0.6)):
        self.max_jump = max_jump          # normalized distance a face may move between frames
        self.lost_after_s = lost_after_s
        self.region = region              # automatic choice: largest face whose centre is in this box
        self.lock = threading.Lock()
        self.center = None                # last known target position (normalized)
        self.last_seen = None             # seconds (capture clock) when the target was last matched
        self.request = None               # pending ("point", x, y) or ("next",) from a viewer
        self.mode = "auto"                # auto | selected
        self.faces = 0
        self.switches = 0                 # times the target moved to a different person

    # ---- viewer requests ----
    def select_point(self, x, y):
        with self.lock:
            self.request = ("point", float(x), float(y))

    def select_next(self):
        with self.lock:
            self.request = ("next",)

    def clear(self):
        with self.lock:
            self.request, self.mode, self.center, self.last_seen = None, "auto", None, None

    # ---- per frame ----
    def choose(self, faces, now_s):
        """faces: MediaPipe landmark lists. Returns the target's landmarks or None."""
        with self.lock:
            boxes = [face_box(f) for f in faces]
            self.faces = len(faces)
            request, self.request = self.request, None
            index = None
            if request and boxes:
                index = self._requested(request, boxes)
                if index is not None:
                    self.mode = "selected"
                    self._set(boxes[index], now_s, switched=True)
                    return faces[index]
            if self.center is not None:
                index = self._nearest(boxes)
                if index is not None:
                    self._set(boxes[index], now_s)
                    return faces[index]
                if self.last_seen is not None and now_s - self.last_seen < self.lost_after_s:
                    return None  # briefly lost: keep waiting for the same person
                if self.mode == "selected":
                    return None  # a chosen person stays chosen until they reappear or are cleared
            index = self._automatic(boxes)
            if index is None:
                return None
            self._set(boxes[index], now_s, switched=self.center is not None)
            return faces[index]

    def _set(self, box, now_s, switched=False):
        self.switches += switched
        self.center, self.last_seen = (box[0], box[1]), now_s

    def _nearest(self, boxes):
        best, best_d = None, self.max_jump
        for i, (cx, cy, _, _) in enumerate(boxes):
            d = ((cx - self.center[0]) ** 2 + (cy - self.center[1]) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = i, d
        return best

    def _automatic(self, boxes):
        if not boxes:
            return None
        rw, rh = self.region
        inside = [i for i, (cx, cy, _, _) in enumerate(boxes)
                  if abs(cx - 0.5) <= rw / 2 and abs(cy - 0.5) <= rh / 2]
        pool = inside or range(len(boxes))
        if inside:
            return max(pool, key=lambda i: boxes[i][2] * boxes[i][3])
        return min(pool, key=lambda i: (boxes[i][0] - 0.5) ** 2 + (boxes[i][1] - 0.5) ** 2)

    def _requested(self, request, boxes):
        if request[0] == "point":
            _, x, y = request
            return min(range(len(boxes)), key=lambda i: (boxes[i][0] - x) ** 2 + (boxes[i][1] - y) ** 2)
        order = sorted(range(len(boxes)), key=lambda i: boxes[i][0])  # left to right
        if self.center is None:
            return order[0]
        current = self._nearest(boxes)
        if current is None or current not in order:
            return order[0]
        return order[(order.index(current) + 1) % len(order)]

    def status(self):
        with self.lock:
            return {"mode": self.mode, "faces": self.faces, "switches": self.switches,
                    "center": None if self.center is None else [round(v, 3) for v in self.center]}
