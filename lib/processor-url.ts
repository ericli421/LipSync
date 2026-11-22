export const PROCESSOR_URL =
  process.env.NEXT_PUBLIC_PROCESSOR_URL ?? "ws://127.0.0.1:8000/ws";
export function viewerUrl() {
  const fallback = new URL("ws://127.0.0.1:8000/ws");
  if (typeof window !== "undefined") {
    fallback.hostname = window.location.hostname;
    fallback.protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  }
  const url = new URL(process.env.NEXT_PUBLIC_PROCESSOR_URL ?? fallback.toString());
  url.pathname = "/ws/viewer";
  return url.toString();
}
