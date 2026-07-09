import type { RealData } from "../types";

type Pt = { n: number; count: number; law: number | string };

function Chart({ title, law, lawTex, points, note }: { title: string; law: string; lawTex: string; points: Pt[]; note: string }) {
  const W = 300, H = 180, pad = 34;
  const xs = points.map((p) => p.n);
  const ys = points.flatMap((p) => [p.count, typeof p.law === "number" ? p.law : p.count]);
  const xmax = Math.max(...xs), ymax = Math.max(...ys) * 1.05;
  const X = (n: number) => pad + (n / xmax) * (W - pad - 8);
  const Y = (v: number) => H - pad - (v / ymax) * (H - pad - 8);
  // теоретическая кривая (по закону) — плотная линия
  const curve: string = points.map((p, i) => {
    const lv = typeof p.law === "number" ? p.law : p.count;
    return `${i ? "L" : "M"} ${X(p.n).toFixed(1)} ${Y(lv).toFixed(1)}`;
  }).join(" ");
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 4 }}>
        <b style={{ fontSize: 13 }}>{title}</b><span className="formula" style={{ fontSize: 12 }}>{law}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        <line x1={pad} y1={H - pad} x2={W - 4} y2={H - pad} stroke="var(--line-2)" />
        <line x1={pad} y1={8} x2={pad} y2={H - pad} stroke="var(--line-2)" />
        <path d={curve} fill="none" stroke="var(--line-2)" strokeWidth={2} strokeDasharray="4 3" />
        {points.map((p) => (
          <circle key={p.n} cx={X(p.n)} cy={Y(p.count)} r={4} fill="var(--accent)" />
        ))}
        <text x={W - 4} y={H - pad + 14} textAnchor="end" className="gsub" style={{ fontSize: 9 }}>n = |H|</text>
        <text x={pad - 4} y={12} textAnchor="end" className="gsub" style={{ fontSize: 9 }}>оп.</text>
      </svg>
      <div className="muted" style={{ fontSize: 11 }}>
        <span style={{ color: "var(--accent)" }}>● счётчик</span> ложится на <span className="muted">┄ закон {lawTex}</span>. {note}
      </div>
    </div>
  );
}

export function ComplexityBars({ real }: { real: RealData }) {
  const cx = real.demos?.complexity;
  if (!cx) return null;
  const meta: Record<string, { t: string; lawTex: string }> = {
    alg1: { t: "Алгоритм 1 / Лемма 1", lawTex: "n(n−1)/2" },
    alg2: { t: "Алгоритм 2 / Лемма 2", lawTex: "n" },
    alg4: { t: "Алгоритм 4", lawTex: "V+E" },
  };
  return (
    <div className="row c3">
      {Object.entries(cx).map(([k, v]) => (
        <div className="panel" key={k} style={{ background: "var(--panel-2)" }}>
          <Chart title={meta[k]?.t ?? k} law={v.law} lawTex={meta[k]?.lawTex ?? ""} points={v.points} note={v.note} />
        </div>
      ))}
    </div>
  );
}
