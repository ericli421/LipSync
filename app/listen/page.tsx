import type { Metadata } from "next";
import { ListeningPage } from "@/components/listen/listening-page";

export const metadata: Metadata = { title: "LipSync · Live listening", description: "Listen to the presenter's live LipSync demonstration." };

export default function Listen() {
  return <ListeningPage />;
}
