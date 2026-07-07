import type { RealData } from "../types";

type Pt = { n: number; count: number; law: number | string };

function Group({ title, law, points, note }: { title: string; law: string; points: Pt[]; note: string }) {
  const nums = points.flatMap((p) => [p.count, typeof p.law === "number" ? p.law : 0]);
  const max = Math.max(1, ...nums);
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 8 }}>
        <b>{title}</b><span className="formula" style={{ fontSize: 12 }}>{law}</span>
      </div>
      <div style={{ display: "flex", gap: 18, alignItems: "flex-end", height: 130 }}>
        {points.map((p) => {
          const lawN = typeof p.law === "number" ? p.law : 0;
          return (
            <div key={p.n} style={{ textAlign: "center", flex: 1 }}>
              <div style={{ display: "flex", gap: 5, alignItems: "flex-end", height: 100, justifyContent: "center" }}>
                <div title={`счётчик ${p.count}`} style={{ width: 18, height: `${p.count / max * 100}%`, background: "var(--accent)", borderRadius: "3px 3px 0 0" }} />
                {lawN > 0 && <div title={`закон ${lawN}`} style={{ width: 18, height: `${lawN / max * 100}%`, background: "var(--line-2)", borderRadius: "3px 3px 0 0" }} />}
              </div>
              <div className="num" style={{ fontSize: 11, marginTop: 4 }}>n={p.n}</div>
              <div className="muted" style={{ fontSize: 10 }}>{p.count}{lawN ? ` / ${lawN}` : ""}</div>
            </div>
          );
        })}
      </div>
      <div className="muted" style={{ fontSize: 11, marginTop: 6 }}>{note}</div>
    </div>
  );
}

export function ComplexityBars({ real }: { real: RealData }) {
  const cx = real.demos?.complexity;
  if (!cx) return null;
  const titles: Record<string, string> = { alg1: "Алгоритм 1 — построение", alg2: "Алгоритм 2 — добавление", alg4: "Алгоритм 4 — планирование" };
  return (
    <div className="row c3">
      {Object.entries(cx).map(([k, v]) => (
        <div className="panel" key={k} style={{ background: "var(--panel-2)" }}>
          <Group title={titles[k] ?? k} law={v.law} points={v.points} note={v.note} />
        </div>
      ))}
    </div>
  );
}
