import { useMemo, useState } from "react";
import type { RealData } from "../types";
import { GraphBuild, GEdge, GNode } from "./GraphBuild";

export function PlanDemo({ real }: { real: RealData }) {
  const nodes = real.graph_conceptual.nodes;
  const edges = real.graph_conceptual.edges;
  const [changed, setChanged] = useState<Set<string>>(new Set());

  const succ = useMemo(() => {
    const s: Record<string, string[]> = {};
    nodes.forEach((n) => (s[n.id] = []));
    edges.forEach(([a, b]) => s[a]?.push(b));
    return s;
  }, [nodes, edges]);

  // P_ne = изменённые + все их потомки по derived_by (каскад, Алгоритм 4)
  const pne = useMemo(() => {
    const p = new Set<string>(changed);
    const st = [...changed];
    while (st.length) { const u = st.pop()!; for (const v of succ[u] ?? []) if (!p.has(v)) { p.add(v); st.push(v); } }
    return p;
  }, [changed, succ]);

  function toggle(id: string) {
    const s = new Set(changed); s.has(id) ? s.delete(id) : s.add(id); setChanged(s);
  }

  // цвет: изменённые — красный, их потомки (пересчёт) — янтарь, остальное — кэш (зелёный)
  const statusOverride: Record<string, string> = {};
  nodes.forEach((n) => {
    statusOverride[n.id] = changed.has(n.id) ? "REFUTED" : pne.has(n.id) ? "CONFIRMED" : "SUPPORTED";
  });

  const gnodes: GNode[] = nodes.map((n) => ({ id: n.id, label: n.label }));
  const gedges: GEdge[] = real.graph_conceptual.derivation.map((d) => ({ src: d.src, dst: d.dst, via: d.via, reason: d.reason }));
  const cascadeOnly = [...pne].filter((x) => !changed.has(x));

  return (
    <div>
      <div className="gb-toolbar">
        <span className="muted" style={{ fontSize: 13 }}>
          Кликните гипотезу на графе — вы её <b>изменили</b>. Подсветится всё, что придётся пересчитать.
        </span>
        <button className="btn ghost" onClick={() => setChanged(new Set())}>Сброс</button>
      </div>

      <GraphBuild nodes={gnodes} edges={gedges} statusOverride={statusOverride} onNodeClick={toggle} />

      <div className="panel" style={{ background: "var(--panel-2)", marginTop: 12 }}>
        {changed.size === 0 ? (
          <div className="muted">Пока ничего не изменено — весь эксперимент берётся из кэша (P_ne = ∅).</div>
        ) : (
          <div style={{ fontSize: 13 }}>
            <div style={{ marginBottom: 6 }}>
              <span className="chip r">изменено: {[...changed].join(", ")}</span>{" "}
              <span className="chip a">пересчёт-каскад: {cascadeOnly.length ? cascadeOnly.join(", ") : "—"}</span>{" "}
              <span className="chip g">из кэша: {nodes.length - pne.size}</span>
            </div>
            <div className="muted">
              План пересчёта <span className="formula">P_ne</span> = <b>{pne.size} из {nodes.length}</b>{" "}
              ({Math.round(pne.size / nodes.length * 100)}%): изменённые гипотезы и все их потомки по{" "}
              <span className="formula">derived_by</span>. Остальное берётся из кэша. По <b>Теореме 1</b> это
              множество минимально — оно содержится в любом корректном плане (и по числу вершин, и по стоимости).
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
