# Local Android demonstration

The native APK captures a phone camera and microphone. A laptop runs FastAPI for
mouth tracking and raw audio relay, and Next.js provides the demonstration view.
There is no authentication, database, Supabase configuration, or cloud service.
Only run this unauthenticated demonstration on your trusted local network.

Returned audio passes through the **live filter chain** (`processor/audio_chain.py`).
Its first layer is a human-voice band gate: a streaming 4th-order Butterworth
band-pass, 80 Hz–8 kHz by default, that treats rumble and hiss as background.
Measured at 48 kHz: −0.6 dB at 100 Hz and 6 kHz, −3 dB at the corners, −24 dB at
40 Hz, −19 dB at 12 kHz; 0.09 ms CPU per 20 ms block. At 16 kHz capture only the
low cut applies. Voice separation, voice selection, and captions are not connected.
The earlier offline experiments remain available separately.

## Start the laptop

After first-time setup below, double-click **Start-LipSync.cmd** in the repository
root. It starts both services, opens the dashboard, and displays the local address
to enter on the phone. Keep the launcher window open and press Enter there to stop
both services. Logs are saved under ignored `.tools/startup/`. If either port is
already occupied, the launcher asks you to close the existing server first.
The local dashboard uses `http://127.0.0.1:3000/dashboard` to avoid slow IPv6
`localhost` resolution on Windows. Startup checks allow time for the first page
compilation and verify both pages and the judge media connection. Each run saves
a `launcher-<timestamp>.log`, including startup failures; previous service logs
are copied into `.tools/startup/history/` before being replaced. For diagnostics
without automatically opening a browser, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts/start-local.ps1 -NoBrowser
```

The launcher refreshes Python or dashboard dependencies when their requirements
or package lock changes after a pull; those launches may take longer and need
internet access.

From the repository root, use Python 3.11 and Node.js compatible with Next 16:

```powershell
py -3.11 -m venv .venv-local
.\.venv-local\Scripts\python.exe -m pip install -r processor/requirements.txt
.\.venv-local\Scripts\python.exe -m uvicorn processor.server:app --host 0.0.0.0 --port 8000 --ws-max-size 750000 --ws-max-queue 4
```

In another terminal:

```powershell
npm ci
npm run dev
```

Open http://localhost:3000/dashboard on the laptop. Phone is the default source;
it does not request access to the laptop camera. The root `.env.example` documents
the optional processor address override. Browser-camera debugging defaults to
`ws://127.0.0.1:8000/ws`.
Phone dashboards use the same-origin `/live/viewer` media proxy, so another
laptop can open `http://<processor-PC-IP>:3000/dashboard` to see the phone feed
and control shared filters. For judges, share the read-only
`http://<processor-PC-IP>:3000/listen` instead; see [the listening guide](judge-listening.md).
Only frontend port 3000 is needed by viewers; the phone still uses port 8000.
Leave `NEXT_PUBLIC_PROCESSOR_URL` unset. Scheduled Web Audio supports monitoring
on ordinary LAN HTTP without browser flags or certificate installation.
Use Browser camera for the original video-only debugging mode.

For face tracking, place `face_landmarker.task` in the repository root. If it is
missing, audio and unannotated video continue with a visible warning. Download:

```powershell
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task' -OutFile face_landmarker.task
```

The backend must use **one worker process**, since the active phone and viewers
are kept in memory. Session data and media are not persisted.

## Build and connect the APK

Open `Lipsync/` in Android Studio and sync dependencies. The checked-in SDK/Gradle
versions are retained. Configure your SDK using Android Studio's local.properties;
that file and generated build output are ignored.

```powershell
cd Lipsync
.\gradlew.bat :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
```

The installable APK is `Lipsync/app/build/outputs/apk/debug/app-debug.apk`.
Install it from Android Studio or use:

```powershell
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

Connect phone and laptop to the same Wi-Fi. Use `ipconfig` to find the laptop's
LAN IPv4 address. Enter `192.168.x.x:8000` (with your actual address) in the APK,
then tap Start capture and allow camera and microphone permissions. `localhost`
in the APK means the phone, not the laptop. An emulator uses `10.0.2.2:8000`.

If the connection fails, check that FastAPI is listening on `0.0.0.0`, Windows
Firewall permits the Python process on your private network, and Wi-Fi client
isolation is disabled. Do not expose this server to the internet. The debug APK
allows plain HTTP/WebSockets for local use; release builds require TLS.

Keep the target face reasonably large in the frame; faces in a wide shot or a
small embedded debate panel may not be detected at the 640-pixel capture size.

Use headphones. Monitoring is initially off on both the APK and dashboard;
enable it on whichever endpoint you want to hear. Phone monitoring returns audio
through the backend, rather than directly looping the microphone locally. Capture
stops when the APK backgrounds, rotates/recreates, loses connection, or is stopped.
The active input device is displayed; built-in microphone preference depends on
the phone honoring the routing request. Output uses Android's selected media route.

## Protocol and extension points

- `/` reports health, protocol version 1, and `live_filters` mode.
- `/ws/capture` accepts one phone. Send a `start` JSON message with
  `protocol_version: 1`, `device_name`, `sample_rate`, `channels: 1`, and
  `format: "pcm_s16le"`. Supported rates are 48000, 16000, and 44100 Hz.
- The server replies with `type: "session"`, a new `session_id`, device/format
  information, and mode. A second simultaneous phone is rejected explicitly.
- Audio messages use `type: "audio"`, `session_id`, consecutive `seq` starting at
  zero, `first_sample` starting at zero, `capture_time_us`, `timing`, and
  `pcm_base64`. Each block contains exactly 20 ms of little-endian mono PCM16.
  Audio is filtered (same length and metadata) and returned to the phone and
  every viewer. With every layer off it is returned bit-exact.
- Video messages use `type: "video"`, `session_id`, increasing `seq`,
  `capture_time_us`, `timing`, and `jpeg_base64`. The APK rotates JPEGs upright,
  scales their longest edge to at most 640 pixels, and targets 25 FPS. Only one
  frame is in flight until `video_ack`; actual processed FPS is displayed.
- `/ws/viewer` sends the current `session` or `waiting` state, raw `audio`, and
  annotated `video` messages with `info` (face_found, openness, mouth_box, optional
  warning). Viewers never acquire the laptop's camera or microphone.
- Viewers receive `type: "pipeline"` status (layer switches, band, input/output
  dB, processing ms per block) on connect, after every change, and every 100 ms
  while a phone streams. Viewers control the chain with
  `{"type": "set_layer", "layer": "voice_band", "enabled": false}` and
  `{"type": "set_band", "low_hz": 300, "high_hz": 3400}`; invalid requests get
  `control_error`. Settings are shared by all viewers and kept across sessions.
- `/ws` retains the browser debug protocol: binary JPEG in, metadata JSON followed
  by annotated binary JPEG out. This mode does not capture audio.

Audio timestamps use AudioRecord's BOOTTIME hardware timestamp when available,
otherwise an estimated device clock anchor plus sample count. Video currently
uses an estimated device timestamp taken at the analysis callback; the original
CameraX `sensor_time_ns` is also preserved as an opaque decimal string. Its sensor
clock domain is not assumed to match BOOTTIME. These timestamps are useful metadata,
**not verified audio/video synchronization** for lip matching.

Audio relay, vision processing, and subscriber playback are separate. A later
separator can replace raw audio relay while retaining the capture/session format.
A glasses client can implement the same protocol; precise capture synchronization
must be established before using it for live lip matching.

Each subscriber retains at most ten 20 ms audio blocks and one pending video
frame. Overflow drops stale audio and emits `discontinuity`; clients reset their
playback buffers. Slow socket writes time out. New sessions clear old queues and
turn monitoring off in the web view. The APK also caps its outgoing socket queue
and stops rather than letting network delay grow indefinitely.

## Verification

```powershell
.\.venv-local\Scripts\python.exe -m pip install -r processor/requirements-dev.txt
.\.venv-local\Scripts\python.exe -m unittest discover -s processor/tests -v
node --test tests/pcm-player.test.mjs
npm run lint
npx tsc --noEmit
npm run build
```

On a physical phone, verify permission denial/retry, front/rear orientation,
web preview, microphone level, headphone playback on both endpoints, background
cleanup, network loss/reconnect, and second-phone rejection. Confirm that stopping
or reconnecting never plays audio from the previous session. No physical phone was
attached during the initial implementation; these hardware checks remain required.
