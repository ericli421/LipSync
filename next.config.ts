import type { NextConfig } from "next";
import { networkInterfaces } from "node:os";
import { resolve } from "node:path";

const processor = new URL(process.env.PROCESSOR_HTTP_URL ?? "http://127.0.0.1:8000");
if (!["http:", "https:"].includes(processor.protocol)) {
  throw new Error("PROCESSOR_HTTP_URL must use http or https");
}
const nextConfig: NextConfig = {
  cacheComponents: false,
  partialPrefetching: false,
  turbopack: { root: resolve(__dirname) },
  // Next dev also guards its scripts/fonts when opened via a LAN address.
  // Allow only addresses actually assigned to this machine, not arbitrary origins.
  allowedDevOrigins: ["127.0.0.1", ...Object.values(networkInterfaces()).flatMap(addresses =>
    (addresses ?? []).filter(address => !address.internal && address.family === "IPv4").map(address => address.address))],
  async rewrites() {
    return [
      { source: "/live/listen", destination: new URL("/ws/listen", processor).toString() },
      { source: "/live/viewer", destination: new URL("/ws/viewer", processor).toString() },
      { source: "/live/health", destination: new URL("/", processor).toString() },
      { source: "/live/media/:path*", destination: new URL("/media/:path*", processor).toString() },
    ];
  },
};

export default nextConfig;
