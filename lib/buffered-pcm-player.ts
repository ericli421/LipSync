/** Scheduled Web Audio playback: works on LAN HTTP without AudioWorklet. */
export class BufferedPcmPlayer {
  private readonly gain: GainNode;
  private readonly gate: GainNode;
  private sources = new Set<AudioBufferSourceNode>();
  private pending: Float32Array[] = [];
  private pendingSamples = 0;
  private nextTime = 0;
  private rate = 48000;
  private started = false;
  private closed = false;
  private session: string | null = null;
  private sequence = -1;
  private target = 0.2;
  private underruns = 0;
  private resets = 0;

  constructor(private readonly context: AudioContext = new AudioContext()) {
    this.gain = context.createGain();
    this.gate = context.createGain();
    this.gate.connect(this.gain);
    this.gain.connect(context.destination);
  }

  async resume() {
    if (this.closed) throw new Error("Audio player has been closed");
    await this.context.resume();
    if (this.context.state !== "running") throw new Error("Audio is suspended. Click Listen again to enable playback.");
  }

  setSession(id: string) {
    this.reset();
    this.session = id;
    this.sequence = -1;
    this.target = 0.2;
  }

  setVolume(value: number) {
    if (!Number.isFinite(value)) return;
    this.gain.gain.setTargetAtTime(Math.min(1, Math.max(0, value)), this.context.currentTime, 0.01);
  }

  push(samples: Float32Array, rate: number, seq: number, session: string) {
    if (this.closed || session !== this.session || !Number.isSafeInteger(seq) || seq <= this.sequence) return false;
    if (![16000, 44100, 48000].includes(rate) || samples.length !== rate / 50 || !samples.every(Number.isFinite)) {
      throw new Error("Invalid microphone audio received");
    }
    if ((this.sequence >= 0 && seq !== this.sequence + 1) || rate !== this.rate) this.reset();
    this.sequence = seq;
    this.rate = rate;
    const now = this.context.currentTime;
    if (this.started && now > this.nextTime + 0.005) {
      this.underruns++;
      this.target = Math.min(0.5, this.target + 0.05);
      this.reset();
    }
    const queued = Math.max(0, this.nextTime - now) + (this.pendingSamples + samples.length) / rate;
    if (queued > 0.6) this.reset();
    this.pending.push(samples.slice());
    this.pendingSamples += samples.length;
    if (!this.started) {
      if (this.pendingSamples / rate + 1e-7 < this.target) return true;
      this.started = true;
      this.nextTime = now + 0.02;
      this.gate.gain.setValueAtTime(0, this.nextTime);
      this.gate.gain.linearRampToValueAtTime(1, this.nextTime + 0.005);
    }
    const count = Math.round(rate * 0.1);
    while (this.pendingSamples >= count) {
      const output = new Float32Array(count);
      let offset = 0;
      while (offset < count) {
        const chunk = this.pending[0];
        const take = Math.min(chunk.length, count - offset);
        output.set(chunk.subarray(0, take), offset);
        offset += take;
        if (take === chunk.length) this.pending.shift();
        else this.pending[0] = chunk.slice(take);
      }
      this.pendingSamples -= count;
      const buffer = this.context.createBuffer(1, count, rate);
      buffer.copyToChannel(output, 0);
      const source = this.context.createBufferSource();
      source.buffer = buffer;
      source.connect(this.gate);
      source.onended = () => { this.sources.delete(source); source.disconnect(); };
      this.sources.add(source);
      source.start(this.nextTime);
      this.nextTime += count / rate;
    }
    return true;
  }

  reset() {
    this.resets++;
    this.gate.gain.cancelScheduledValues(this.context.currentTime);
    this.gate.gain.setValueAtTime(0, this.context.currentTime);
    for (const source of this.sources) {
      source.onended = null;
      source.stop();
      source.disconnect();
    }
    this.sources.clear();
    this.pending = [];
    this.pendingSamples = 0;
    this.nextTime = 0;
    this.started = false;
  }

  get stats() {
    return { targetMs: Math.round(this.target * 1000), buffering: !this.started,
      queuedMs: Math.round((Math.max(0, this.nextTime - this.context.currentTime) + this.pendingSamples / this.rate) * 1000),
      underruns: this.underruns, resets: this.resets };
  }

  close() {
    if (this.closed) return;
    this.reset();
    this.closed = true;
    this.session = null;
    this.gate.disconnect();
    this.gain.disconnect();
    void this.context.close();
  }
}

export function decodePcm16(encoded: string) {
  const binary = atob(encoded);
  if (binary.length % 2) throw new Error("Incomplete audio block");
  const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
  const view = new DataView(bytes.buffer);
  const samples = new Float32Array(bytes.length / 2);
  for (let index = 0; index < samples.length; index++) samples[index] = view.getInt16(index * 2, true) / 32768;
  return samples;
}
