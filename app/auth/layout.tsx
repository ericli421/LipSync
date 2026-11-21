import { Logo } from "@/components/logo";
import { ThemeSwitcher } from "@/components/theme-switcher";

export default function AuthLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="relative min-h-svh overflow-hidden">
      <div className="bg-grid pointer-events-none absolute inset-0 [mask-image:radial-gradient(ellipse_at_center,black_20%,transparent_70%)]" />
      <div className="pointer-events-none absolute left-1/2 top-1/2 h-[28rem] w-[28rem] -translate-x-1/2 -translate-y-1/2 rounded-full bg-primary/20 blur-3xl" />
      <header className="absolute inset-x-0 top-0 z-10 mx-auto flex h-16 w-full max-w-6xl items-center justify-between px-5">
        <Logo />
        <ThemeSwitcher />
      </header>
      <div className="relative">{children}</div>
    </div>
  );
}
