import Link from "next/link";
import { cn } from "@/lib/utils";

export function LogoMark({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "relative inline-flex h-8 w-8 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-[0_0_20px_-4px_hsl(var(--primary)/0.7)]",
        className,
      )}
    >
      <svg
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
        className="h-5 w-5"
        aria-hidden
      >
        {/* Stylised lips */}
        <path d="M3 12c2.5-3 4.5-4 6-4 1.2 0 2.2.8 3 1.6.8-.8 1.8-1.6 3-1.6 1.5 0 3.5 1 6 4" />
        <path d="M3 12c3 4 6 5 9 5s6-1 9-5" />
        <path d="M3 12h18" strokeOpacity={0.55} />
      </svg>
    </span>
  );
}

export function Logo({
  className,
  href = "/",
}: {
  className?: string;
  href?: string;
}) {
  return (
    <Link
      href={href}
      className={cn(
        "flex items-center gap-2 text-lg font-semibold tracking-tight",
        className,
      )}
    >
      <LogoMark />
      <span>
        Lip<span className="text-primary-deep">Sync</span>
      </span>
    </Link>
  );
}
