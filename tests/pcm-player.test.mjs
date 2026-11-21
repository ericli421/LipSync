import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import vm from "node:vm";

function player() {
  let Player;
  vm.runInNewContext(readFileSync("public/pcm-player.js", "utf8"), {
    AudioWorkletProcessor: class {
      constructor() {
        this.port = {};
      }
    },
    sampleRate: 48000,
    Float32Array,
    registerProcessor: (_, implementation) => {
      Player = implementation;
    },
  });
  const instance = new Player();
  return {
    instance,
    send: (data) => instance.port.onmessage({ data }),
    render: (count) => {
      const result = new Float32Array(count);
      instance.process([], [[result]]);
      return result;
    },
  };
}

test("reset prevents playback of the previous session", () => {
  const p = player();
  p.send({ rate: 48000, samples: new Float32Array(960).fill(0.5) });
  p.send({ reset: true });
  assert.ok(p.render(128).every((sample) => sample === 0));
});

test("overflow discards old blocks", () => {
  const p = player();
  for (let i = 0; i <= 10; i++)
    p.send({ rate: 48000, samples: new Float32Array(960).fill(i) });
  assert.equal(p.instance.queue.length, 1);
  assert.equal(p.render(1)[0], 10);
});

test("44100 Hz input occupies 20 ms at 48000 Hz output", () => {
  const p = player();
  p.send({ rate: 44100, samples: new Float32Array(882).fill(0.25) });
  const output = p.render(961);
  assert.equal(output[0], 0.25);
  assert.equal(output[959], 0.25);
  assert.equal(output[960], 0);
});
