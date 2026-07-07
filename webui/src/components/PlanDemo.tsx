import { useMemo, useState } from "react";
import type { RealData } from "../types";

export function PlanDemo({ real }: { real: RealData }) {
  const nodes = real.graph_conceptual.nodes;
  const edges = real.graph_conceptual.edges;
  const [cached, setCached] = useState<Set<string>>(new Set());
  const [proposed, setProposed] = useState<Set<string>>(new Set());
  const [mode, setMode] = useState<"cache" | "propose">("cache");

  const succ = useMemo(() => {
    const s: Record<string, string[]> = {};
    nodes.forEach((n) => (s[n.id] = []));
    edges.forEach(([a, b]) => s[a]?.push(b));
    return s;
  }, [nodes, edges]);

  function descendants(start: string): Set<string> {
    const out = new Set<string>(); const st = [start];
    while (st.length) { const u = st.pop()!; for (const v of succ[u] ?? []) if (!out.has(v)) { out.add(v); st.push(v); } }
    return out;
  }

  // минимальный план: некэшированные + все их потомки (Алгоритм 4 / оракул)
  const pne = useMemo(() => {
    const p = new Set<string>();
    for (const n of nodes) if (!cached.has(n.id)) { p.add(n.id); descendants(n.id).forEach((d) => p.add(d)); }
    return p;
  }, [cached, nodes]);

  // корректность предложенного плана P: содержит все некэшированные и замкнут по потомкам
  const nonCached = nodes.filter((n) => !cached.has(n.id)).map((n) => n.id);
  const containsAllNonCached = nonCached.every((id) => proposed.has(id));
  const closed = [...proposed].every((id) => (succ[id] ?? []).every((v) => proposed.has(v)));
  const validProposed = containsAllNonCached && closed;
  const extra = [...proposed].filter((id) => !pne.has(id));

  function toggle(id: string) {
    if (mode === "cache") {
      const s = new Set(cached); s.has(id) ? s.delete(id) : s.add(id); setCached(s);
    } else {
      const s = new Set(proposed); s.has(id) ? s.delete(id) : s.add(id); setProposed(s);
    }
  }

  function cls(id: string) {
    if (mode === "cache") {
      if (cached.has(id)) return "g";          // кэш
      return pne.has(id) ? "a" : "";           // пересчёт (минимальный план)
    }
    return proposed.has(id) ? (pne.has(id) ? "a" : "b") : "";
  }

  return (
    <div>
      <div className="gb-toolbar">
        <div className="seg">
          <button className={mode === "cache" ? "on" : ""} onClick={() => setMode("cache")}>Каскад: отметить кэш</button>
          <button className={mode === "propose" ? "on" : ""} onClick={() => setMode("propose")}>Предложи свой план P</button>
        </div>
        <button className="btn ghost" onClick={() => { setCached(new Set()); setProposed(new Set()); }}>Сброс</button>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(8, 1fr)", gap: 8, marginBottom: 14 }}>
        {nodes.map((n) => (
          <button key={n.id} className={"chip " + cls(n.id)} style={{ cursor: "pointer", justifyContent: "center", padding: "8px 4px" }}
                  title={n.label} onClick={() => toggle(n.id)}>{n.id}</button>
        ))}
      </div>

      {mode === "cache" ? (
        <div className="panel" style={{ background: "var(--panel-2)" }}>
          <div className="muted" style={{ fontSize: 13 }}>
            Зелёные — <b>кэш</b>, янтарные — <b>P_ne</b> (пересчёт). Минимальный план:{" "}
            <b>{pne.size} из {nodes.length}</b> ({Math.round(pne.size / nodes.length * 100)}%).
            По теореме 1 это множество содержится в любом корректном плане — минимально и по числу вершин,
            и по любой неотрицательной стоимости.
          </div>
        </div>
      ) : (
        <div className="panel" style={{ background: "var(--panel-2)" }}>
          <div style={{ fontSize: 13 }}>
            Ваш план P = {"{" + [...proposed].join(", ") + "}"} ({proposed.size} вершин).{" "}
            {validProposed
              ? <span className="chip g">корректен</span>
              : <span className="chip r">некорректен</span>}
          </div>
          <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
            {!containsAllNonCached && <div>✕ не содержит все некэшированные гипотезы (нарушено условие полноты).</div>}
            {containsAllNonCached && !closed && <div>✕ не замкнут по потомкам derived_by (нарушена каскадность A2).</div>}
            {validProposed && <div>✓ содержит минимальный план P_ne ⊆ P. {extra.length
              ? <>Лишние вершины P∖P_ne: {extra.map((h) => <span key={h} className="tag">{h}</span>)}</>
              : <>P совпадает с минимальным.</>}</div>}
          </div>
        </div>
      )}
    </div>
  );
}
