import { useMemo } from "react";
import type { RealData } from "../types";
import { usePanZoom } from "../components/usePanZoom";

const BW = 168, BH = 64, COL = 220, ROW = 92;

export function Workflow({ real }: { real: RealData }) {
  const c = real.graph_conceptual;
  const tasks = c.tasks;
  const edges = c.task_edges;
  const preds = c.task_preds ?? {};
  const labelOf = Object.fromEntries(c.nodes.map((n) => [n.id, n.label]));

  const { pos, viewW, viewH } = useMemo(() => {
    const inc: Record<string, string[]> = {};
    tasks.forEach((t) => (inc[t.id] = []));
    edges.forEach(([a, b]) => inc[b]?.push(a));
    const dc: Record<string, number> = {};
    const depth = (id: string, seen = new Set<string>()): number => {
      if (dc[id] != null) return dc[id];
      if (seen.has(id)) return 0;
      seen.add(id);
      const ps = inc[id] ?? [];
      return (dc[id] = ps.length ? Math.max(...ps.map((p) => depth(p, seen) + 1)) : 0);
    };
    const byD: Record<number, string[]> = {};
    tasks.forEach((t) => (byD[depth(t.id)] ??= []).push(t.id));
    const pos: Record<string, { x: number; y: number }> = {};
    let maxD = 0, maxRows = 0;
    Object.entries(byD).forEach(([d, ids]) => {
      maxD = Math.max(maxD, +d); maxRows = Math.max(maxRows, ids.length);
      ids.forEach((id, i) => (pos[id] = { x: 20 + +d * COL, y: 20 + i * ROW }));
    });
    return { pos, viewW: 40 + maxD * COL + BW, viewH: 40 + maxRows * ROW };
  }, [tasks, edges]);

  const pz = usePanZoom(viewW, viewH);

  function path(a: string, b: string) {
    const p = pos[a], q = pos[b];
    const x1 = p.x + BW, y1 = p.y + BH / 2, x2 = q.x, y2 = q.y + BH / 2, mx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  }

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент W · поток работ (ОАГ задач, параллельные ветви)</div>
        <h1>Поток работ и принадлежность гипотез задачам</h1>
        <p className="lead">
          Поток работ — ациклический граф задач глубины {c.depth}. Ветви <b>LPR</b> (жидкость) и{" "}
          <b>WCT</b> (обводнённость) выполняются <b>параллельно</b> и сходятся в задаче прогноза нефти
          (AND-join: обе ветви обязательны). Каждая задача содержит гипотезы (<span className="formula">hasHypothesis</span>).
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <span className="muted" style={{ fontSize: 12, marginLeft: "auto" }}>колесо — масштаб · тянуть фон — сдвиг</span>
          {pz.zoomed && <button className="btn ghost" onClick={pz.reset}>⟲ масштаб</button>}
        </div>
        <div className="graph-frame" style={{ height: viewH + 40 }}>
          <svg ref={pz.svgRef} viewBox={pz.viewBox} width="100%" height="100%" preserveAspectRatio="xMidYMid meet"
               style={{ touchAction: "none", cursor: pz.zoomed ? "grab" : "default" }}
               onPointerDown={pz.onPointerDown} onPointerMove={pz.onPointerMove}
               onPointerUp={pz.onPointerUp} onPointerLeave={pz.onPointerUp}>
            <defs>
              <marker id="tarw" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
                <path d="M0,0 L7,3 L0,6 Z" fill="var(--accent-dim)" />
              </marker>
            </defs>
            {edges.map(([a, b], i) => (
              <path key={i} d={path(a, b)} fill="none" stroke="var(--accent-dim)" strokeWidth={1.6} markerEnd="url(#tarw)" />
            ))}
            {tasks.map((t) => {
              const p = pos[t.id]; const andJoin = (preds[t.id]?.length ?? 0) > 1;
              return (
                <g key={t.id}>
                  <rect x={p.x} y={p.y} width={BW} height={BH} rx={9}
                        fill="var(--panel-2)" stroke={andJoin ? "var(--accent)" : "var(--line-2)"} strokeWidth={andJoin ? 2 : 1.3} />
                  <text x={p.x + 10} y={p.y + 17} className="glabel" style={{ fontSize: 11 }}>{t.id}{andJoin ? "  ⋀ AND" : ""}</text>
                  <text x={p.x + 10} y={p.y + 31} className="gsub" style={{ fontSize: 8.5 }}>{t.label.slice(0, 24)}</text>
                  <text x={p.x + 10} y={p.y + 48} className="gsub" style={{ fontSize: 9, fill: "var(--accent)" }}>
                    {t.hypotheses.join(", ")}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      <div className="row c2">
        <div className="panel">
          <div className="label">Принадлежность гипотез задачам (hasHypothesis)</div>
          <table className="data">
            <thead><tr><th>Задача</th><th>Гипотезы</th></tr></thead>
            <tbody>
              {tasks.map((t) => (
                <tr key={t.id}>
                  <td className="num">{t.id} <span className="muted">{t.label}</span></td>
                  <td>{t.hypotheses.map((h) => <span key={h} className="tag">{h} · {labelOf[h]}</span>)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <div className="label">Формальное описание (CWL-подобный DAG)</div>
          <pre style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)", background: "var(--panel-2)",
                        padding: 14, borderRadius: 8, overflowX: "auto", margin: 0, lineHeight: 1.5 }}>
            {c.formal_text}
          </pre>
          <p className="muted" style={{ fontSize: 12 }}>
            Потоки работ формально описываются в CWL (Common Workflow Language, YAML), BPMN (шлюзы AND/OR)
            или YAWL (OR-join). Здесь — CWL-подобный вид: две параллельные ветви с AND-join в прогнозе нефти.
          </p>
        </div>
      </div>
    </div>
  );
}
