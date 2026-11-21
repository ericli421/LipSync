"""
av_sync.py - Reference pipeline for synchronized audio + video capture.
 
What it does
  1. Captures audio (sounddevice) and video (OpenCV) in separate threads.
  2. Timestamps every audio block and video frame at CAPTURE time on one
     shared clock (time.monotonic), not at arrival time.
  3. Corrects the fixed latency offset between the two paths (measured with
     the clap/flash calibration below).
  4. Corrects audio clock drift by estimating the sound card's real sample rate.
  5. Pairs each video frame with the audio window around it and hands the
     pair to your transcription / AV model.
 
Install:  pip install numpy sounddevice opencv-python
Run:      python av_sync.py --calibrate      # clap while a light flashes, ~10 s
          python av_sync.py --offset-ms 42   # run with the measured offset
"""
 
import argparse, cv2, collections, threading, time

import numpy as np
import sounddevice as sd
 
SAMPLE_RATE = 16_000       # nominal; the real rate is estimated (drift)
AUDIO_BLOCK = 320          # 20 ms blocks
CHANNELS = 1               # use one multichannel device for mic arrays
FPS = 30
AUDIO_WINDOW_S = 0.20      # audio context per video frame (+/- 100 ms)
AUDIO_BUFFER_S = 10.0      # how much audio history to keep
 
 
def now() -> float:
    """The single shared clock every timestamp is expressed in."""
    return time.monotonic()
 
 
# --------------------------------------------------------------------------
# Drift estimation: fit "samples received" against "shared clock time".
# The slope is the sound card's true sample rate on our clock.
# --------------------------------------------------------------------------
class DriftEstimator:
    def __init__(self, nominal_rate: float, history: int = 500):
        self.nominal = nominal_rate
        self.points = collections.deque(maxlen=history)   # (time, sample_index)
 
    def add(self, t: float, sample_index: int) -> None:
        self.points.append((t, sample_index))
 
    def rate(self) -> float:
        if len(self.points) < 50:
            return self.nominal
        t, n = np.array(self.points).T
        slope, _ = np.polyfit(t - t[0], n, 1)
        return slope
 
    def ppm(self) -> float:
        return (self.rate() / self.nominal - 1.0) * 1e6
 
 
# --------------------------------------------------------------------------
# Audio capture
# --------------------------------------------------------------------------
class AudioCapture:
    """Timestamps audio from the sample counter, anchored to the shared clock.
 
    Each block's capture time = t0 + first_sample / estimated_rate.
    Counting samples (instead of reading the clock per callback) removes
    callback jitter; the estimated rate removes long-term drift.
    """
 
    def __init__(self, device=None):
        self.device = device
        self.lock = threading.Lock()
        self.buffer = collections.deque()   # (sample_index, block ndarray)
        self.samples_seen = 0
        self.t0 = None                      # shared-clock time of sample 0
        self.drift = DriftEstimator(SAMPLE_RATE)
        self.stream = None
 
    def _callback(self, indata, frames, time_info, status):
        if status:
            print("audio status:", status)   # under/overruns show up here
        # inputBufferAdcTime is when the ADC captured the first sample, on
        # PortAudio's clock. Convert it to our clock via the current offset.
        if time_info.inputBufferAdcTime > 0:
            pa_to_mono = now() - time_info.currentTime
            adc_time = time_info.inputBufferAdcTime + pa_to_mono
        else:                                # some backends report 0
            adc_time = now() - frames / SAMPLE_RATE
 
        with self.lock:
            if self.t0 is None:
                self.t0 = adc_time
            self.drift.add(adc_time, self.samples_seen)
            self.buffer.append((self.samples_seen, indata[:, 0].copy()))
            self.samples_seen += frames
            max_samples = int(AUDIO_BUFFER_S * SAMPLE_RATE)
            while self.buffer and self.samples_seen - self.buffer[0][0] > max_samples:
                self.buffer.popleft()
 
    def start(self):
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=AUDIO_BLOCK, channels=CHANNELS,
            dtype="float32", device=self.device, callback=self._callback,
            latency="low")
        self.stream.start()
 
    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
 
    def sample_to_time(self, n: float) -> float:
        return self.t0 + n / self.drift.rate()
 
    def time_to_sample(self, t: float) -> int:
        return int(round((t - self.t0) * self.drift.rate()))
 
    def window(self, center_t: float, width_s: float):
        """Audio samples covering [center - width/2, center + width/2].
        Returns None if that span isn't fully buffered yet."""
        with self.lock:
            if self.t0 is None or not self.buffer:
                return None
            start = self.time_to_sample(center_t - width_s / 2)
            end = self.time_to_sample(center_t + width_s / 2)
            first = self.buffer[0][0]
            if start < first or end > self.samples_seen:
                return None
            audio = np.concatenate([b for _, b in self.buffer])
        return audio[start - first:end - first]
 
 
# --------------------------------------------------------------------------
# Video capture
# --------------------------------------------------------------------------
class VideoCapture:
    """Timestamps each frame right after grab(), minus the camera's latency.
 
    grab() returns when the frame is available to the host; the exposure
    happened earlier by the camera's pipeline latency. That difference is
    folded into the calibrated offset.
    """
 
    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # avoid stale queued frames
        self.frames = collections.deque(maxlen=FPS * 5)   # (t, frame)
        self.running = False
        self.dropped = 0
        self.last_t = None
 
    def _loop(self):
        while self.running:
            if not self.cap.grab():
                continue
            t = now()
            ok, frame = self.cap.retrieve()
            if not ok:
                continue
            if self.last_t is not None and t - self.last_t > 1.5 / FPS:
                self.dropped += int((t - self.last_t) * FPS) - 1
            self.last_t = t
            self.frames.append((t, frame))
 
    def start(self):
        if not self.cap.isOpened():
            raise RuntimeError("camera not available")
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
 
    def stop(self):
        self.running = False
        self.cap.release()
 
 
# --------------------------------------------------------------------------
# Aligner: pairs frames with audio once the audio for that frame has arrived
# --------------------------------------------------------------------------
class Aligner:
    def __init__(self, audio: AudioCapture, video: VideoCapture, offset_s: float):
        """offset_s = video_latency - audio_latency (positive: video is late).
        Subtracted from frame timestamps to get true exposure time."""
        self.audio, self.video, self.offset = audio, video, offset_s
 
    def pairs(self):
        """Yields (frame_time, frame, audio_window). Waits for audio to catch up."""
        while True:
            if not self.video.frames:
                time.sleep(0.005)
                continue
            t_host, frame = self.video.frames[0]
            t_true = t_host - self.offset
            audio = self.audio.window(t_true, AUDIO_WINDOW_S)
            if audio is None:
                # Audio not buffered that far yet; or frame too old - drop it.
                if self.audio.t0 and t_true < self.audio.sample_to_time(
                        self.audio.buffer[0][0] if self.audio.buffer else 0):
                    self.video.frames.popleft()
                time.sleep(0.005)
                continue
            self.video.frames.popleft()
            yield t_true, frame, audio
 
 
# --------------------------------------------------------------------------
# Calibration: flash an LED / screen and clap (or beep) at the same moment.
# Finds each event in its stream and reports the offset.
# --------------------------------------------------------------------------
def detect_flashes(frames, thresh=40.0):
    times, prev = [], None
    for t, f in frames:
        b = float(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).mean())
        if prev is not None and b - prev > thresh:
            times.append(t)
        prev = b
    return times
 
 
def detect_claps(audio, t0, rate, hop=160, thresh_ratio=8.0):
    energy = np.array([np.mean(audio[i:i + hop] ** 2)
                       for i in range(0, len(audio) - hop, hop)])
    floor = np.median(energy) + 1e-10
    times, last = [], -1.0
    for i, e in enumerate(energy):
        t = t0 + i * hop / rate
        if e > thresh_ratio * floor and t - last > 0.5:
            times.append(t)
            last = t
    return times
 
 
def calibrate(duration=10.0):
    audio, video = AudioCapture(), VideoCapture()
    audio.start()
    video.start()
    print(f"Recording {duration:.0f}s: clap at the same moment as a bright flash, "
          "a few times, ~2 s apart.")
    time.sleep(duration)
    video.stop()
    audio.stop()
 
    frames = list(video.frames)
    samples = np.concatenate([b for _, b in audio.buffer])
    first = audio.buffer[0][0]
    flashes = detect_flashes(frames)
    claps = detect_claps(samples, audio.sample_to_time(first), audio.drift.rate())
 
    offsets = []
    for fl in flashes:
        nearest = min(claps, key=lambda c: abs(c - fl), default=None)
        if nearest is not None and abs(nearest - fl) < 0.5:
            offsets.append(fl - nearest)
    if not offsets:
        print("No matched flash/clap events found; try a brighter flash or louder clap.")
        return
    off_ms = np.median(offsets) * 1000
    print(f"Matched {len(offsets)} events. Offset (video late by): {off_ms:.1f} ms "
          f"(spread {np.std(offsets) * 1000:.1f} ms). Audio drift: {audio.drift.ppm():.1f} ppm")
    print(f"Run with: python av_sync.py --offset-ms {off_ms:.0f}")
 
 
# --------------------------------------------------------------------------
# Main loop: plug your model in here
# --------------------------------------------------------------------------
def transcribe(frame_time, frame, audio_window):
    """Placeholder: feed into your ASR / audio-visual speech model."""
    pass
 
 
def run(offset_ms: float):
    audio, video = AudioCapture(), VideoCapture()
    audio.start()
    video.start()
    aligner = Aligner(audio, video, offset_ms / 1000.0)
    last_report = now()
    try:
        for t, frame, window in aligner.pairs():
            transcribe(t, frame, window)
            if now() - last_report > 5:
                print(f"drift {audio.drift.ppm():+.1f} ppm | "
                      f"dropped frames {video.dropped} | "
                      f"audio rate {audio.drift.rate():.2f} Hz")
                last_report = now()
    except KeyboardInterrupt:
        pass
    finally:
        video.stop()
        audio.stop()
 
 
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--offset-ms", type=float, default=0.0)
    args = p.parse_args()
    calibrate() if args.calibrate else run(args.offset_ms)
 