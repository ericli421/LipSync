# Listening with judges on the same network

Double-click **Start-LipSync.cmd**, connect the phone as usual, and open the
presenter's dashboard. The launcher checks the judge page and its media proxy
before reporting readiness, then prints **Share with judges** for each LAN adapter.
Share `http://<PC-LAN-IP>:3000/listen` (for example,
`http://192.168.30.48:3000/listen`). Use the adapter on the judges' network.
The address can change when the PC changes networks; restart the launcher then.

Each judge opens that link, puts on headphones, and clicks **Listen**. No browser
flags, certificates, account, or laptop camera/microphone permission are needed.
This is intended for up to five simultaneous judge laptops on a trusted LAN.
The page automatically reconnects after a network/server interruption. After
switching tabs, reconnecting, or restarting phone capture, click Listen again.
Mute and volume affect only that judge's laptop.

The presenter chooses Original, Voice 1, Voice 2, or Focus in the dashboard or
phone. Everyone receives the same final processed PCM, including the speech-band
filter and auto volume if enabled. The listening page shows the shared output
and filter status but cannot change them. Falcon remains offline. The annotated
camera preview is live, capped at eight FPS for judges; separated audio has the
existing model delay and is **not synchronized to that preview**. Playback adds
an initial 200 ms jitter buffer, rising to at most 500 ms after underruns, plus
network/device delay. This buffer improves network playback; it cannot repair
chopping or leakage already present in model output.

## Network setup and troubleshooting

Both the page and judge media travel through port **3000**, with same-origin
`/live/listen` forwarding to the backend's read-only `/ws/listen`. The presenter
dashboard uses `/live/viewer`. Keep `NEXT_PUBLIC_PROCESSOR_URL` unset for this
setup. `PROCESSOR_HTTP_URL` is a server-only upstream override, defaulting to
`http://127.0.0.1:8000`; rebuild production after changing it. The phone still
connects to port **8000** over Wi-Fi, or uses the existing USB/ADB forwarding setup.

If another laptop cannot open the page, allow inbound Node TCP 3000 for the local
subnet. In **Administrator PowerShell**, a scoped example is:

```powershell
New-NetFirewallRule -DisplayName 'LipSync judge listening' -Direction Inbound -Action Allow -Protocol TCP -LocalPort 3000 -RemoteAddress LocalSubnet -Profile Private,Public -Program (Get-Command node.exe).Source
```

To remove that rule later:

```powershell
Remove-NetFirewallRule -DisplayName 'LipSync judge listening'
```

Do not disable Windows Firewall. Venue/guest Wi-Fi may isolate devices even on
the same SSID; use a network that permits device-to-device connections. Use the
PC's LAN IP, not `localhost`. The launcher explicitly binds Next to `0.0.0.0`;
manual development startup is `npm run dev -- --hostname 0.0.0.0`. Next dev permits
its scripts only for this PC's assigned LAN IPv4 addresses, evaluated at startup.

If the page opens but stays disconnected, inspect `.tools/startup/` and run:

```powershell
.\.venv-local\Scripts\python.exe scripts/check_listener.py
```

This checks the actual WebSocket upgrade through Next, rather than just the HTML
page or backend health. A waiting status means the relay works and the phone is
not capturing. If connected with video but no sound, click Listen, check personal
volume and OS output, and check that the presenter has selected a working output.
The player uses scheduled Web Audio rather than AudioWorklet, so ordinary LAN
HTTP playback is supported. Browser camera debugging still requires a secure
context for camera capture.

The listener endpoint rejects application messages and omits device names,
recording paths, logs, and private diagnostics. This is a read-only audience view,
not an authentication boundary for the whole app: the existing dashboard remains
unauthenticated. Share only on a trusted LAN; no public hosting or router port
forwarding is configured.

## Before the demonstration

1. Start the PC and phone; confirm camera and audio on the dashboard.
2. Open the printed judge link on a **second physical laptop**. Confirm camera
   movement, click Listen, and confirm speech through headphones.
3. Switch presenter output between Original and Focus/Voice 1/Voice 2. Confirm
   all judges follow, while one judge's mute/volume leaves everyone else alone.
4. Test five laptops together. Check for smooth playback and usable video. Try
   a phone stop/start and a browser reconnect; old audio must not replay.
5. Keep the launcher open and prevent host sleep for the demonstration. Stop it
   with Enter afterward. Network/browser delivery tests do not establish speech
   isolation quality, physical phone synchronization, or model throughput.

For a generated five-client proxy smoke test, with both services running and
**no phone capturing**, run:

```powershell
.\.venv-local\Scripts\python.exe scripts/verify_listener_stream.py --seconds 35
```

This sends quiet synthetic audio and a test card, compares each listener's PCM
with the final phone return, verifies video and read-only status, and tests two
sessions. It refuses to interrupt an active capture. It is a transport test,
not a five-device listening or separation-quality test.

Automated regression checks:

```powershell
.\.venv-local\Scripts\python.exe -m unittest discover -s processor/tests -v
node --test tests/*.test.mjs
npm run lint
npx tsc --noEmit
npm run build
```
