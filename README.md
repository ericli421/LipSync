<div align="center">

# <!--🏆 --> LipSync

**A step towards improving accessibility to all**

Built in 36 hours at **CodeJam16** · <!-- 🥇 {prize_or_track} -->

[🎥 Demo Video (to be added)](https://www.youtube.com/watch?v=dQw4w9WgXcQ) · [🚀 Live App (to be added)](https://www.youtube.com/watch?v=dQw4w9WgXcQ) · [📊 Pitch Deck (to be added)](https://www.youtube.com/watch?v=dQw4w9WgXcQ)

</div>

## 💡 Inspiration

Hearing aids are currently the primary accessibility measure offered to people with hearing disabilities to allow them to hear better.
While they work wonderfully, they are unfortunately still not fully adequate to being able to cover the full capacities of the human ear

As of right now, One of the biggest issues with hearing aids is that they amplify all sounds captured. This is fine in most situations, but
is a huge problem in crowded environments. Our human ear has a natural ability to filter out "noise" and focus mentally onto the voices that
we are trying to listen to. However, a person with hearing disabilities loses that natural ability when they use hearing aids, and therefore
struggle to hear well in noisy environments (which tends to happen in the real world).

We were inspired by this issue to create a fix to this problem. Leveraging the leaps in Machine Learning in recent years and computer vision
principles, we attempted to build a system that emulates the natural noise filtering system our brain performs.

## 🛠️ What It Does

1- Captures the raw audio
2- Passes a human frequency noise filtering
3- Seperates the different audio sources using **SepFormer**
4- Using video tracking we follow the user's gaze to focus on a speaker
5- Using lip timing we identify the speaker's audio source and lower other sounds


## 🏗️ How We Built It

Three programs talking over Wi-Fi via WebSockets:

- **Android app (Kotlin, Jetpack Compose):** CameraX captures frames, AudioRecord captures raw mono 16-bit audio at 48 kHz with the phone's own noise processing turned off, and OkHttp streams both to the laptop. 
- Processing server (Python, FastAPI + Uvicorn):
  - **SciPy** Butterworth voice-band filter
  - **SpeechBrain / PyTorch** running **SepFormer** on the GPU (or ReSepFormer on CPU)
  - **MediaPipe** face mesh for lip tracking
  - **ECAPA-TDNN** voiceprints to keep each voice attached to the same person
  - **A spectral (STFT) mask** to boost one voice and duck the others
- **🌐 Web dashboard (Next.js, React, TypeScript):** live video, meters, layer toggles, a Raw ↔ LipSync A/B switch, and a listen-only demo page for anyone on the network
- **Real-time design:**
  - The AI model runs on its own thread over a sliding 2 s window, updated every 0.25 s, so the live audio never waits on it
  - Every queue has a size limit and drops old data rather than letting the delay grow
- No training: every model is pretrained and open source. Our work is the system that makes them run together, live and pick up the right voice.

## 🧗 Challenges We Ran Into

- **The phone can't tell direction.** Our first idea was beamforming with two phones to reduce the amount of information going to the program. However their microphones aren't synchronized, and the clock drift ruins direction finding. 
- **Speed:** on a laptop CPU, the best separation model took **7.5 s to process 2 s of audio**. We had to move to a GPU (≈150 ms per step) and keep a lighter model as a CPU fallback this is also where our optimisation came handy.
- **Discerning voices from a stream** separating a stream of overlapping voices into a multiple singular voices. This proved to be harder than expected and we had to process the information better and try different models in order to make it work.
- **The model doesn't know who's who.** Each step, the separated voices can come out in a different order. We had to line up every new result with the previous one and add voiceprint matching so voices don't swap.
- **Lips are ambiguous.** When someone isn't talking or their face is hidden, lip timing can't decide. Our lip picker still makes no decision a large part of the time (59% in our latest recording), and we chose "no decision" over a wrong one.
- **Network hiccups:** Wi-Fi delivers audio in bursts. We built bounded queues, sequence checks and late-step skipping so a hiccup becomes a short glitch instead of a growing delay.
- **Latency vs quality:** a longer window separates better but adds delay. We settled on 2 s windows with 0.25 s updates, about **0.75 s** total delay.

## 🎉 Accomplishments We're Proud Of

- A **full live pipeline**, phone → laptop GPU → headphones and browsers, built in one hackathon
- Research-grade voice separation running **in real time**: ≈150 ms per update (95th percentile) against a 250 ms budget on an RTX A3000
- An original idea that works without special hardware: **using lip timing to choose which separated voice to keep**
- A live **A/B switch** so anyone can hear the difference instantly
- Solid engineering: 28 server tests, 51 offline tests, phone protocol tests and type checks, all passing
- Something that can eventually make a real impact

## 📚 What We Learned

- **Real-time is a different discipline:** late data is worse than lost data, and the audio path must never wait for the AI
- How AI speech separation works (SepFormer, sliding windows, why the voices come out in a random order) and its limits
- Why phone microphones can't do directional hearing, and why two phones can't be synchronized well enough
- Designing a client–server system with WebSockets, threads and async code in Python
- **Measure everything:** our assumptions about speed were often wrong until we timed them

## 🔮 What's Next

- [ ] **Smart glasses / headsets:** pointing a phone at people is awkward; glasses with a camera and microphones know where you're looking
- [ ] **On-device models:** smaller lip-aware models (e.g. Falcon, 0.5 M parameters) so it runs without a laptop
- [ ] **Lower latency, better accuracy:** smaller hops, better lip matching, and a measured quality score
- [ ] **Live translation** of the isolated voice

## ⚡ Run It Locally

> Requires Windows, Python 3, Node.js, and an Android phone on the same Wi-Fi. An NVIDIA GPU is recommended for live separation.

```bash
git clone https://github.com/ericli421/LipSync.git
cd LipSync
git checkout separation-test-mode
```

1. **First-time setup:** double-click `Setup-Models.cmd` (downloads the pinned models)
2. **Start everything:** double-click `Start-LipSync.cmd` (Python server on port 8000 + website on port 3000)
3. **Phone:** build `Lipsync/` in Android Studio (or `adb install` the APK), then enter your laptop's IP, e.g. `192.168.x.x:8000`
4. **Dashboard:** open `http://localhost:3000/dashboard` on the laptop

Full guide: `docs/local-android-demo.md`
