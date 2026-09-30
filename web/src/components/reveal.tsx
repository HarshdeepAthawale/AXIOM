"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";

import { cx } from "./ui";

/**
 * Fades and lifts its children in the first time they scroll into view.
 * Content is visible without JavaScript and under prefers-reduced-motion.
 */
export function Reveal({
  children,
  className,
  delay = 0,
  as: Tag = "div",
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
  as?: "div" | "li";
}) {
  const ref = useRef<HTMLElement>(null);
  const [state, setState] = useState<"idle" | "hidden" | "shown">("idle");

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    // Only hide what starts below the fold, so nothing above it ever flickers.
    if (node.getBoundingClientRect().top < window.innerHeight * 0.92) return;
    const hide = requestAnimationFrame(() => setState("hidden"));
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setState("shown");
          observer.disconnect();
        }
      },
      { rootMargin: "0px 0px -8% 0px" },
    );
    observer.observe(node);
    return () => {
      cancelAnimationFrame(hide);
      observer.disconnect();
    };
  }, []);

  return (
    <Tag
      ref={ref as never}
      style={state === "shown" ? { transitionDelay: `${delay}ms` } : undefined}
      className={cx(
        "transition-[opacity,transform] duration-[900ms] ease-[cubic-bezier(0.2,0.7,0.2,1)]",
        state === "hidden" && "translate-y-5 opacity-0",
        className,
      )}
    >
      {children}
    </Tag>
  );
}
