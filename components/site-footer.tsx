import { Logo } from "@/components/logo";

export function SiteFooter() {
  return (
    <footer className="w-full border-t border-border/60">
      <div className="mx-auto flex w-full max-w-6xl flex-col items-center justify-between gap-4 px-5 py-10 text-sm text-muted-foreground sm:flex-row">
        <Logo className="text-base text-foreground" />
        <p>&copy; LipSync. Local selective listening prototype.</p>
      </div>
    </footer>
  );
}
