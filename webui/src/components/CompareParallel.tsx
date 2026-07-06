import { useEffect, useRef } from "react";
import Plotly from "plotly.js-dist-min";
import type { CompareRow } from "../types";

export function CompareParallel({ rows }: { rows: CompareRow[] }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current || !rows.length) return;
    const r2 = rows.map((r) => r.r2 ?? 0);
    const idx = rows.map((_, i) => i);
    Plotly.newPlot(
      ref.current,
      [{
        type: "parcoords",
        line: { color: r2, colorscale: [[0, "#6b6250"], [1, "#e0913a"]] },
        dimensions: [
          { label: "Гипотеза", tickvals: idx, ticktext: rows.map((r) => r.hypothesis), values: idx, range: [0, rows.length - 1] },
          { label: "R²", values: r2, range: [0, 1] },
        ],
      }] as any,
      {
        margin: { t: 40, l: 60, r: 40, b: 20 },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        font: { color: "#9c9077", family: "IBM Plex Mono, monospace", size: 11 },
        height: 300,
      } as any,
      { displayModeBar: false } as any
    );
  }, [rows]);
  if (!rows.length) return <div className="empty">Нет данных для сравнения.</div>;
  return <div ref={ref} />;
}
