// webui/src/components/CompareParallel.tsx
import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";

export function CompareParallel({ rows }: { rows: { hypothesis: string; r2: number | null }[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const r2 = rows.map((r) => r.r2 ?? 0);
    Plotly.newPlot(ref.current, [{
      type: "parcoords",
      dimensions: [{ label: "R²", values: r2 }],
    }] as any, { margin: { t: 30 } });
  }, [rows]);
  return <div ref={ref} />;
}
