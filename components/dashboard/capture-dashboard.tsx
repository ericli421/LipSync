"use client";
import { useState } from "react";
import { CameraDashboard } from "./camera-dashboard";
import { PhoneDashboard } from "./phone-dashboard";
import { Button } from "@/components/ui/button";
export function CaptureDashboard() {
  const [source, setSource] = useState("phone");
  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold">Local capture</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            Camera tracking and raw audio passthrough
          </p>
        </div>
        <div className="flex gap-2" role="group" aria-label="Capture source">
          <Button
            aria-pressed={source === "phone"}
            variant={source === "phone" ? "default" : "outline"}
            onClick={() => setSource("phone")}
          >
            Phone
          </Button>
          <Button
            aria-pressed={source === "browser"}
            variant={source === "browser" ? "default" : "outline"}
            onClick={() => setSource("browser")}
          >
            Browser camera · debug
          </Button>
        </div>
      </div>
      {source === "phone" ? <PhoneDashboard /> : <CameraDashboard />}
    </div>
  );
}
