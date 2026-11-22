export const PROCESSOR_URL =
  process.env.NEXT_PUBLIC_PROCESSOR_URL ?? "ws://127.0.0.1:8000/ws";
export function viewerUrl() {
  if (process.env.NEXT_PUBLIC_PROCESSOR_URL) {
    const url = new URL(process.env.NEXT_PUBLIC_PROCESSOR_URL);
    url.pathname = "/ws/viewer";
    return url.toString();
  }
  return liveUrl("viewer");
}
export function listenUrl() {
  return liveUrl("listen");
}
function liveUrl(kind: "listen" | "viewer") {
  const url = new URL(`/live/${kind}`, typeof window === "undefined" ? "http://localhost:3000" : window.location.origin);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  return url.toString();
}
