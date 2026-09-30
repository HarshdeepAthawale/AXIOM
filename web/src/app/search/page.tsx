import type { Metadata } from "next";
import { Suspense } from "react";

import { Console } from "@/components/console";

export const metadata: Metadata = {
  title: "Console · Axiom",
};

export default function SearchPage() {
  return (
    <Suspense>
      <Console />
    </Suspense>
  );
}
