// Local mono PCM monitor. Resampling uses the negotiated input rate.
class PcmPlayer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.queue = [];
    this.offset = 0;
    this.phase = 0;
    this.rate = 48000;
    this.port.onmessage = ({ data }) => {
      if (data.reset) {
        this.queue = [];
        this.offset = 0;
        this.phase = 0;
        return;
      }
      this.rate = data.rate;
      if (this.queue.length >= 10) {
        this.queue = [];
        this.offset = 0;
        this.phase = 0;
      }
      this.queue.push(new Float32Array(data.samples));
    };
  }
  process(inputs, outputs) {
    const output = outputs[0][0];
    if (!output) return true;
    for (let i = 0; i < output.length; i++) {
      if (!this.queue.length) {
        output[i] = 0;
        this.phase = 0;
        continue;
      }
      output[i] = this.queue[0][this.offset] || 0;
      this.phase += this.rate;
      const advance = Math.floor(this.phase / sampleRate);
      this.phase -= advance * sampleRate;
      this.offset += advance;
      while (this.queue.length && this.offset >= this.queue[0].length)
        this.offset -= this.queue.shift().length;
      if (!this.queue.length) this.offset = 0;
    }
    return true;
  }
}
registerProcessor("pcm-player", PcmPlayer);
