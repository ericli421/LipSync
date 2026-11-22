"""Versioned local capture protocol and bounded subscriber queues."""
import asyncio
import base64
from collections import deque
from dataclasses import dataclass, field
import uuid

from processor.audio_chain import ChainSettings

MODE = "live_filters"

MAX_MESSAGE_BYTES = 750_000


def integer(message, key, minimum=0):
    value = message.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{key} must be an integer >= {minimum}")
    return value


def payload(message, key, limit):
    encoded = message.get(key)
    if not isinstance(encoded, str) or len(encoded) > (limit + 2) // 3 * 4:
        raise ValueError(f"invalid {key}")
    try:
        data = base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError(f"invalid {key}") from exc
    if not data or len(data) > limit:
        raise ValueError(f"invalid {key} size")
    return data


class Outbox:
    """At most 200 ms of PCM, one video frame, the latest pipeline status, and bounded controls."""
    def __init__(self):
        self.audio = deque()
        self.controls = deque(maxlen=8)
        self.video = None
        self.pipeline = None
        self.ready = asyncio.Event()

    def reset(self, status):
        self.audio.clear()
        self.controls.clear()
        self.video = None
        self.pipeline = None
        self.put(status)

    def put(self, message):
        kind = message["type"]
        if kind == "audio":
            if len(self.audio) >= 10:
                self.audio.clear()
                self.controls.append({"type": "discontinuity", "session_id": message["session_id"]})
            self.audio.append(message)
        elif kind == "video":
            self.video = message
        elif kind == "pipeline":  # meters: only the newest matters
            self.pipeline = message
        else:
            self.controls.append(message)
        self.ready.set()

    async def get(self):
        while True:
            if self.controls:
                return self.controls.popleft()
            if self.audio:
                return self.audio.popleft()
            if self.pipeline is not None:
                result, self.pipeline = self.pipeline, None
                return result
            if self.video is not None:
                result, self.video = self.video, None
                return result
            self.ready.clear()
            await self.ready.wait()


@dataclass
class Session:
    device_name: str
    sample_rate: int
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    last_audio_seq: int = -1
    next_sample: int = 0
    last_video_seq: int = -1
    last_video_time: int = -1

    @classmethod
    def start(cls, message):
        if message.get("type") != "start" or message.get("protocol_version") != 1:
            raise ValueError("expected start with protocol_version 1")
        rate = integer(message, "sample_rate", 1)
        if rate not in (16000, 44100, 48000) or message.get("channels") != 1 or message.get("format") != "pcm_s16le":
            raise ValueError("expected mono PCM16 at 16000, 44100, or 48000 Hz")
        name = message.get("device_name")
        if not isinstance(name, str) or not 1 <= len(name) <= 100:
            raise ValueError("invalid device_name")
        return cls(name, rate)

    def status(self):
        return {"type": "session", "protocol_version": 1, "session_id": self.id,
                "device_name": self.device_name, "sample_rate": self.sample_rate,
                "channels": 1, "format": "pcm_s16le", "mode": MODE}

    def validate(self, message):
        if message.get("session_id") != self.id:
            raise ValueError("wrong session_id")
        kind = message.get("type")
        seq = integer(message, "seq")
        timestamp = integer(message, "capture_time_us")
        if message.get("timing") not in ("device", "estimated"):
            raise ValueError("timing must be device or estimated")
        if kind == "audio":
            first = integer(message, "first_sample")
            pcm = payload(message, "pcm_base64", self.sample_rate // 50 * 2)
            if len(pcm) != self.sample_rate // 50 * 2:
                raise ValueError("audio must contain exactly 20 ms of PCM16")
            if seq != self.last_audio_seq + 1 or first != self.next_sample:
                raise ValueError("audio sequence/sample gap; restart capture")
            self.last_audio_seq = seq
            self.next_sample += len(pcm) // 2
            return pcm
        if kind == "video":
            if seq <= self.last_video_seq or timestamp <= self.last_video_time:
                raise ValueError("video sequence/timestamp must increase")
            jpeg = payload(message, "jpeg_base64", 512_000)
            self.last_video_seq, self.last_video_time = seq, timestamp
            return jpeg
        raise ValueError("unsupported message type")


class Hub:
    def __init__(self):
        self.session = None
        self.owner = None
        self.viewers = set()
        self.settings = ChainSettings()
        self.chain = None

    def idle_pipeline(self):
        version, layers, (low, high) = self.settings.snapshot()
        return {"type": "pipeline", "layers": layers,
                "band": {"low_hz": low, "high_hz": high, "effective_high_hz": high},
                "in_db": None, "out_db": None, "process_ms": None, "error": None}

    def status(self):
        return self.session.status() if self.session else {"type": "waiting", "mode": MODE}

    def publish(self, message, include_phone=False):
        for viewer in self.viewers:
            viewer.put(message)
        if include_phone and self.owner:
            self.owner.put(message)

    def begin(self, session, outbox):
        if self.session is not None:
            raise ValueError("another phone is already capturing")
        self.session, self.owner = session, outbox
        for target in self.viewers | {outbox}:
            target.reset(session.status())

    def end(self, session):
        if self.session is session:
            self.session, self.owner = None, None
            for target in self.viewers:
                target.reset(self.status())
