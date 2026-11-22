import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import ts from "typescript";

const code = ts.transpileModule(readFileSync("lib/processor-url.ts", "utf8"), {
  compilerOptions: { module: ts.ModuleKind.CommonJS },
}).outputText;

function urls(origin, override) {
  const exports = {};
  new Function("exports", "process", "window", code)(exports,
    { env: { NEXT_PUBLIC_PROCESSOR_URL: override } }, { location: { origin } });
  return exports;
}

test("LAN viewers use the page hostname and port for both media endpoints", () => {
  const u = urls("http://192.168.30.48:3000");
  assert.equal(u.viewerUrl(), "ws://192.168.30.48:3000/live/viewer");
  assert.equal(u.listenUrl(), "ws://192.168.30.48:3000/live/listen");
});

test("HTTPS pages use secure same-origin WebSockets", () => {
  const u = urls("https://demo.example:3443");
  assert.equal(u.viewerUrl(), "wss://demo.example:3443/live/viewer");
  assert.equal(u.listenUrl(), "wss://demo.example:3443/live/listen");
});

test("explicit presenter override preserves compatibility without redirecting judges", () => {
  const u = urls("http://demo.local:3000", "ws://processor.local:8000/ws");
  assert.equal(u.viewerUrl(), "ws://processor.local:8000/ws/viewer");
  assert.equal(u.listenUrl(), "ws://demo.local:3000/live/listen");
});
