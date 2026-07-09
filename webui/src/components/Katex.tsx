import { useEffect, useRef } from "react";

declare global { interface Window { katex?: any; } }

export function Katex({ tex, block }: { tex: string; block?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    const k = window.katex;
    if (ref.current && k) {
      try { k.render(tex, ref.current, { throwOnError: false, displayMode: !!block }); }
      catch { if (ref.current) ref.current.textContent = tex; }
    } else if (ref.current) {
      ref.current.textContent = tex; // fallback пока katex грузится
    }
  }, [tex, block]);
  return <span ref={ref} className="formula" style={{ color: "var(--text)" }} />;
}
