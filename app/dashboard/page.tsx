import { redirect } from "next/navigation";
import { Suspense } from "react";
import { createClient } from "@/lib/supabase/server";
import { CameraDashboard } from "@/components/dashboard/camera-dashboard";

async function RequireUser({ children }: { children: React.ReactNode }) {
  const supabase = await createClient();
  const { data, error } = await supabase.auth.getClaims();

  if (error || !data?.claims) {
    redirect("/");
  }

  return <>{children}</>;
}

export default function DashboardPage() {
  return (
    <Suspense
      fallback={
        <div className="py-24 text-center text-sm text-muted-foreground">
          Loading your cameras…
        </div>
      }
    >
      <RequireUser>
        <CameraDashboard />
      </RequireUser>
    </Suspense>
  );
}
