"""
Among faces inside a region around the frame center, pick the largest one and locate its lips.
 
Setup:
    pip install opencv-python mediapipe
    Download the model file (one time) into this folder:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
 
Run:
    python center_face_lips.py        (press q to quit)
"""
 
import os
import time
import cv2
import numpy as np
import mediapipe as mp
 
# Resolve relative to this file so imports from other folders (e.g. processor/) work.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")
CAMERA_INDEX = 0          # change if your device isn't the default camera
MAX_FACES = 5
 
# Size of the center region, as a fraction of the frame width and height.
# A face counts as "in the region" if its center falls inside this box.
REGION_WIDTH = 0.5
REGION_HEIGHT = 0.6
 
# MediaPipe face-mesh landmark indices for the lips
OUTER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
              409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
              415, 310, 311, 312, 13, 82, 81, 80, 191]
 
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode
 
 
def to_pixels(landmarks, idxs, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs],
                    dtype=np.int32)
 
 
def center_region(w, h):
    """Pixel box (x1, y1, x2, y2) of the region around the frame center."""
    rw, rh = w * REGION_WIDTH, h * REGION_HEIGHT
    x1, y1 = int((w - rw) / 2), int((h - rh) / 2)
    return x1, y1, int(x1 + rw), int(y1 + rh)

def pick_center_face(faces, w, h):
    """Return the landmark list of the face whose center is nearest the frame center."""
    cx, cy = w / 2, h / 2
    best, best_dist = None, float("inf")
    for lm in faces:
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        fx, fy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        d = (fx - cx) ** 2 + (fy - cy) ** 2
        if d < best_dist:
            best, best_dist = lm, d
    return best

 
def pick_largest_face_in_region(faces, w, h):
    """Among faces whose center lies inside the center region, return the largest one."""
    x1, y1, x2, y2 = center_region(w, h)
    best, best_area = pick_center_face(faces,w,h), 0.0
    for lm in faces:
        xs = [p.x * w for p in lm]
        ys = [p.y * h for p in lm]
        fx, fy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        if not (x1 <= fx <= x2 and y1 <= fy <= y2):
            continue
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best, best_area = lm, area
    return best
 
 
def create_landmarker():
    """Create a VIDEO-mode FaceLandmarker. Use one per video stream (timestamps must increase)."""
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=MAX_FACES,
        # Defaults (0.5) missed most frames of real phone video in the first tests.
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return FaceLandmarker.create_from_options(options)


def process_frame(landmarker, frame, timestamp_ms):
    """Detect the target face's lips in a BGR frame and draw annotations onto it (in place).

    Returns (frame, info) where info is a JSON-serialisable dict:
        face_found: bool
        openness:   inner-lip gap / mouth width (0 when no face)
        mouth_box:  [x, y, w, h] in pixels, or None
    """
    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(image, timestamp_ms)

    info = {"face_found": False, "openness": 0.0, "mouth_box": None}

    face = None
    if result.face_landmarks:
        face = pick_largest_face_in_region(result.face_landmarks, w, h)
    if face is not None:
        outer = to_pixels(face, OUTER_LIPS, w, h)
        inner = to_pixels(face, INNER_LIPS, w, h)

        # Lip bounding box (useful if you want to crop the mouth region)
        x, y, bw, bh = cv2.boundingRect(outer)

        # Simple mouth-openness measure: inner lip gap / mouth width
        gap = np.linalg.norm(
            np.subtract(to_pixels(face, [13], w, h)[0], to_pixels(face, [14], w, h)[0]))
        width = np.linalg.norm(
            np.subtract(to_pixels(face, [61], w, h)[0], to_pixels(face, [291], w, h)[0]))
        openness = gap / width if width else 0.0

        cv2.polylines(frame, [outer], True, (0, 0, 255), 2)
        cv2.polylines(frame, [inner], True, (0, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 1)
        cv2.putText(frame, f"mouth open: {openness:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        info = {
            "face_found": True,
            "openness": float(openness),
            "mouth_box": [int(x), int(y), int(bw), int(bh)],
        }

    rx1, ry1, rx2, ry2 = center_region(w, h)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
    return frame, info


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")
 
    start = time.monotonic()
    with create_landmarker() as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw = frame.copy()
            timestamp_ms = int((time.monotonic() - start) * 1000)
            frame, info = process_frame(landmarker, frame, timestamp_ms)

            if info["mouth_box"]:
                x, y, bw, bh = info["mouth_box"]
                mouth_crop = raw[y:y + bh, x:x + bw]
                if mouth_crop.size:
                    cv2.imshow("mouth", cv2.resize(mouth_crop, (160, 80)))

            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
