"""
Find the face closest to the center of the camera frame and locate its lips.
 
Setup:
    pip install opencv-python mediapipe
    Download the model file (one time) into this folder:
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
 
Run:
    python center_face_lips.py        (press q to quit)
"""
 
import time
import cv2
import numpy as np
import mediapipe as mp
 
MODEL_PATH = "face_landmarker.task"
CAMERA_INDEX = 0          # change if your device isn't the default camera
MAX_FACES = 5
 
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
 
 
def main():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_faces=MAX_FACES,
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {CAMERA_INDEX}")
 
    start = time.monotonic()
    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
 
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.monotonic() - start) * 1000)
            result = landmarker.detect_for_video(image, timestamp_ms)
 
            if result.face_landmarks:
                face = pick_center_face(result.face_landmarks, w, h)
                outer = to_pixels(face, OUTER_LIPS, w, h)
                inner = to_pixels(face, INNER_LIPS, w, h)
 
                # Lip bounding box (useful if you want to crop the mouth region)
                x, y, bw, bh = cv2.boundingRect(outer)
                mouth_crop = frame[y:y + bh, x:x + bw]
 
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
                if mouth_crop.size:
                    cv2.imshow("mouth", cv2.resize(mouth_crop, (160, 80)))
 
            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255),
                           cv2.MARKER_CROSS, 20, 1)
            cv2.imshow("camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 