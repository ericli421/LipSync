import Link from "next/link";
import {
  ArrowRight,
  Cctv,
  Plug,
  ScanFace,
  ShieldCheck,
  Users,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { SiteHeader } from "@/components/site-header";
import { SiteFooter } from "@/components/site-footer";
import { CameraPreview } from "@/components/landing/camera-preview";

const FEATURES = [
  {
    icon: Cctv,
    title: "Native phone capture",
    body: "Send camera frames and microphone audio from the Android APK over your local network.",
  },
  {
    icon: ScanFace,
    title: "Mouth tracking",
    body: "Inspect the selected face and mouth movement in the laptop dashboard.",
  },
  {
    icon: Plug,
    title: "Raw audio monitoring",
    body: "Listen through headphones on the phone or dashboard. Separation is a later milestone.",
  },
  {
    icon: ShieldCheck,
    title: "Local operation",
    body: "Run the processor on your laptop. No account, database, or cloud connection is required.",
  },
  {
    icon: Users,
    title: "Shared demonstration",
    body: "Connect one phone and open the dashboard for your demonstration audience.",
  },
  {
    icon: Zap,
    title: "Built for experiments",
    body: "Keep the capture bridge while testing voice separation and lip matching separately.",
  },
];
const STEPS = [
  {
    icon: Plug,
    title: "Start the local server",
    body: "Run the Python processor and web dashboard on your laptop.",
  },
  {
    icon: Cctv,
    title: "Connect the APK",
    body: "Enter your laptop�s LAN address and allow camera and microphone access.",
  },
  {
    icon: ScanFace,
    title: "Watch and listen",
    body: "Open the dashboard to inspect mouth tracking and enable raw audio monitoring.",
  },
];

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <SiteHeader
        links={[
          { href: "#features", label: "Features" },
          { href: "#how-it-works", label: "How it works" },
        ]}
      />

      <main className="flex-1">
        {/* Hero */}
        <section className="relative overflow-hidden">
          <div className="bg-grid pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_top,black_30%,transparent_70%)]" />
          <div className="pointer-events-none absolute left-1/2 top-[-12rem] h-[32rem] w-[56rem] -translate-x-1/2 rounded-full bg-primary/20 blur-3xl" />

          <div className="relative mx-auto grid w-full max-w-6xl items-center gap-14 px-5 pb-24 pt-20 lg:grid-cols-2 lg:pt-28">
            <div className="flex flex-col items-start gap-6">
              <span className="inline-flex items-center gap-2 rounded-full border border-primary/30 bg-primary/10 px-3 py-1 text-xs font-medium text-primary-deep">
                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                Local selective listening prototype
              </span>
              <h1 className="text-4xl font-bold leading-[1.1] tracking-tight sm:text-5xl lg:text-6xl">
                Hear what your{" "}
                <span className="text-gradient">cameras see</span>.
              </h1>
              <p className="max-w-lg text-lg text-muted-foreground">
                LipSync connects to your cameras, reads lips in real time, and
                turns every spoken word into an accurate, searchable transcript
                — no microphone needed.
              </p>
              <div className="flex flex-wrap gap-3">
                <Button
                  asChild
                  size="lg"
                  className="h-11 px-6 shadow-lg shadow-primary/30"
                >
                  <Link href="/dashboard">
                    Open dashboard
                    <ArrowRight />
                  </Link>
                </Button>
                <Button
                  asChild
                  size="lg"
                  variant="outline"
                  className="h-11 px-6"
                >
                  <Link href="#how-it-works">See how it works</Link>
                </Button>
              </div>
            </div>

            <CameraPreview />
          </div>
        </section>

        {/* Features */}
        <section
          id="features"
          className="scroll-mt-20 border-t border-border/60 bg-muted/40"
        >
          <div className="mx-auto w-full max-w-6xl px-5 py-24">
            <div className="mx-auto mb-14 max-w-2xl text-center">
              <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-primary-deep">
                Features
              </p>
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                A working bridge for the next experiment
              </h2>
            </div>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
              {FEATURES.map(({ icon: Icon, title, body }) => (
                <div
                  key={title}
                  className="group rounded-xl border bg-card p-6 transition-all hover:-translate-y-0.5 hover:border-primary/50 hover:shadow-lg hover:shadow-primary/10"
                >
                  <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-primary/15 text-primary-deep transition-colors group-hover:bg-primary group-hover:text-primary-foreground">
                    <Icon className="h-5 w-5" />
                  </div>
                  <h3 className="mb-2 font-semibold">{title}</h3>
                  <p className="text-sm leading-relaxed text-muted-foreground">
                    {body}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* How it works */}
        <section id="how-it-works" className="scroll-mt-20">
          <div className="mx-auto w-full max-w-6xl px-5 py-24">
            <div className="mx-auto mb-14 max-w-2xl text-center">
              <p className="mb-3 text-sm font-semibold uppercase tracking-widest text-primary-deep">
                How it works
              </p>
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                From phone to headphones in three steps
              </h2>
            </div>
            <div className="relative">
              <div className="absolute left-[16.66%] right-[16.66%] top-7 hidden h-px bg-gradient-to-r from-primary/0 via-primary/60 to-primary/0 md:block" />
              <ol className="relative grid gap-10 md:grid-cols-3">
                {STEPS.map(({ icon: Icon, title, body }, i) => (
                  <li
                    key={title}
                    className="relative flex flex-col items-center text-center"
                  >
                    <div className="relative mb-5 flex h-14 w-14 items-center justify-center rounded-2xl border border-primary/40 bg-background text-primary-deep shadow-[0_0_30px_-6px_hsl(var(--primary)/0.6)]">
                      <Icon className="h-6 w-6" />
                      <span className="absolute -right-2 -top-2 flex h-6 w-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-foreground">
                        {i + 1}
                      </span>
                    </div>
                    <h3 className="mb-2 text-lg font-semibold">{title}</h3>
                    <p className="max-w-xs text-sm leading-relaxed text-muted-foreground">
                      {body}
                    </p>
                  </li>
                ))}
              </ol>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="px-5 pb-24">
          <div className="relative mx-auto w-full max-w-6xl overflow-hidden rounded-3xl bg-gradient-to-br from-[hsl(193_82%_34%)] via-[hsl(190_74%_50%)] to-[hsl(205_85%_55%)] px-8 py-16 text-center text-white shadow-xl shadow-primary/25">
            <div className="bg-grid pointer-events-none absolute inset-0 opacity-30 [--foreground:0_0%_100%]" />
            <div className="relative mx-auto flex max-w-2xl flex-col items-center gap-6">
              <h2 className="text-3xl font-bold tracking-tight sm:text-4xl">
                Start your local demonstration
              </h2>
              <p className="text-lg text-white/85">
                Start the server, connect your phone, and see the live feed.
                Voice isolation and captions will follow.
              </p>
              <div className="flex flex-wrap justify-center gap-3">
                <Button
                  asChild
                  size="lg"
                  className="h-11 bg-white px-6 text-[hsl(200_65%_14%)] hover:bg-white/90"
                >
                  <Link href="/dashboard">
                    Open dashboard
                    <ArrowRight />
                  </Link>
                </Button>
                <Button
                  asChild
                  size="lg"
                  variant="outline"
                  className="h-11 border-white/50 bg-transparent px-6 text-white hover:bg-white/10 hover:text-white"
                >
                  <Link href="/dashboard">View local capture</Link>
                </Button>
              </div>
            </div>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
}
