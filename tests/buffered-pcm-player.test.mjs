import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "typescript";

const exports = {};
const code = ts.transpileModule(readFileSync("lib/buffered-pcm-player.ts", "utf8"), { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText;
new Function("exports", code)(exports);
const { BufferedPcmPlayer, decodePcm16 } = exports;

function fixture(rate = 48000) {
  const parameter = () => ({ value: 1, setTargetAtTime(value) { this.value = value; }, setValueAtTime(value) { this.value = value; }, linearRampToValueAtTime(value) { this.value = value; }, cancelScheduledValues() {} });
  const buffers = [], sources = [], gains = [];
  const context = { currentTime: 0, state: "running", destination: {},
    resume: async () => {}, close: async () => { context.state = "closed"; },
    createGain: () => { const node = { gain: parameter(), connect() {}, disconnect() {} }; gains.push(node); return node; },
    createBuffer: (channels, count, inputRate) => {
      const data = new Float32Array(count);
      const buffer = { duration: count / inputRate, sampleRate: inputRate, data, copyToChannel: samples => data.set(samples) };
      buffers.push(buffer); return buffer;
    },
    createBufferSource: () => {
      const source = { buffer: null, stopped: false, connect() {}, disconnect() {}, start: at => { source.at = at; }, stop: () => { source.stopped = true; } };
      sources.push(source); return source;
    },
  };
  const player = new BufferedPcmPlayer(context);
  player.setSession("s");
  let seq = 0;
  const send = (value = .25) => player.push(new Float32Array(rate / 50).fill(value), rate, seq++, "s");
  return { player, context, buffers, sources, gains, send };
}

test("LAN player schedules audio without AudioWorklet and preserves negotiated rates", async () => {
  for (const rate of [16000, 44100, 48000]) {
    const f = fixture(rate);
    await f.player.resume();
    for (let i = 0; i < 9; i++) f.send();
    assert.equal(f.sources.length, 0);
    f.send();
    assert.equal(f.sources.length, 2);
    assert.equal(f.buffers[0].sampleRate, rate);
    assert.equal(f.buffers[0].duration, .1);
    assert.equal(f.sources[0].at, .02);
    assert.ok(Math.abs(f.sources[1].at - .12) < 1e-8);
    assert.ok(f.buffers[0].data.every(value => value === .25));
  }
});

test("underrun increases buffering and cancels stale scheduled sound", () => {
  const f = fixture();
  for (let i = 0; i < 10; i++) f.send();
  f.context.currentTime = 1;
  f.send();
  assert.equal(f.player.stats.underruns, 1);
  assert.equal(f.player.stats.targetMs, 250);
  assert.equal(f.player.stats.buffering, true);
  assert.ok(f.sources.every(source => source.stopped));
  for (let cycle = 0; cycle < 10; cycle++) {
    for (let i = 0; i < 25; i++) f.send();
    f.context.currentTime += 2;
    f.send();
  }
  assert.equal(f.player.stats.targetMs, 500);
});

test("session changes, discontinuities, mute/close never replay old audio", () => {
  const f = fixture();
  for (let i = 0; i < 10; i++) f.send();
  const prior = f.sources.slice();
  f.player.setSession("next");
  assert.ok(prior.every(source => source.stopped));
  assert.equal(f.player.push(new Float32Array(960), 48000, 11, "s"), false);
  for (let seq = 0; seq < 10; seq++) f.player.push(new Float32Array(960).fill(.5), 48000, seq, "next");
  assert.ok(f.buffers.at(-1).data.every(value => value === .5));
  f.player.reset();
  assert.ok(f.sources.every(source => source.stopped));
  f.player.close();
  assert.equal(f.context.state, "closed");
  assert.equal(f.player.push(new Float32Array(960), 48000, 100, "next"), false);
});

test("queue overflow is bounded and gaps/duplicate sequences are handled", () => {
  const f = fixture();
  for (let i = 0; i < 80; i++) f.send();
  assert.ok(f.player.stats.queuedMs <= 600);
  assert.ok(f.player.stats.resets > 1);
  assert.equal(f.player.push(new Float32Array(960), 48000, 0, "s"), false);
  f.player.push(new Float32Array(960), 48000, 100, "s");
  assert.equal(f.player.stats.buffering, true);
  assert.throws(() => f.player.push(new Float32Array(3), 48000, 101, "s"), /Invalid/);
  assert.throws(() => f.player.push(new Float32Array(960).fill(NaN), 48000, 101, "s"), /Invalid/);
  assert.throws(() => f.player.push(new Float32Array(160), 8000, 101, "s"), /Invalid/);
});

test("suspended browser audio is reported and a closed player cannot resume", async () => {
  const f = fixture();
  f.context.state = "suspended";
  await assert.rejects(f.player.resume(), /suspended/);
  f.player.close();
  await assert.rejects(f.player.resume(), /closed/);
});

test("volume belongs to each player; PCM decoding is little-endian", () => {
  const a = fixture(), b = fixture();
  a.player.setVolume(.1);
  assert.equal(a.gains[0].gain.value, .1);
  assert.equal(b.gains[0].gain.value, 1);
  a.player.setVolume(9);
  assert.equal(a.gains[0].gain.value, 1);
  const samples = decodePcm16(Buffer.from([0, 64, 0, 192]).toString("base64"));
  assert.deepEqual([...samples], [.5, -.5]);
  assert.throws(() => decodePcm16("AA=="), /Incomplete/);
});
