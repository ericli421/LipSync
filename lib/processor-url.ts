export const PROCESSOR_URL =
  process.env.NEXT_PUBLIC_PROCESSOR_URL ?? "ws://127.0.0.1:8000/ws";
export function viewerUrl() {
  const url = new URL(PROCESSOR_URL);
  url.pathname = "/ws/viewer";
  return url.toString();
}
