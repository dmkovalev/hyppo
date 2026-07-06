import { useEffect, useState } from "react";
import { get } from "../api";
import type { VE } from "../types";
import { HypothesisGraph, GNode } from "../components/HypothesisGraph";

const LEGEND = [
  { s: "SUPPORTED", c: "g", t: "подтверждена" },
  { s: "REFUTED", c: "r", t: "отвергнута" },
  { s: "SUPERSEDED", c: "b", t: "замещена" },
  { s: "PROPOSED", c: "", t: "предложена" },
];

export function Graph({ pid }: { pid: string }) {
  const [ve, setVe] = useState<VE | null>(null);
  useEffect(() => { get<VE>(`/api/projects/${pid}/ve`).then(setVe).catch(() => setVe(null)); }, [pid]);

  const nodes: GNode[] = (ve?.hypotheses ?? []).map((h) => ({
    id: h.id, label: h.label ?? h.id, status: h.epistemic_status,
  }));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Граф гипотез · алгоритм 1</div>
        <h1>Граф гипотез L</h1>
        <p className="lead">
          Ориентированный ациклический граф с отношением <span className="formula">derived_by</span>.
          Цвет узла — эпистемический статус гипотезы. Каскадная инвалидация распространяется по рёбрам.
        </p>
      </div>
      <div className="panel">
        {nodes.length
          ? <HypothesisGraph nodes={nodes} edges={ve?.workflow_edges ?? []} />
          : <div className="empty">Граф не построен.</div>}
        <div style={{ display: "flex", gap: 16, marginTop: 14 }}>
          {LEGEND.map((l) => <span key={l.s} className={"chip " + l.c}>{l.t}</span>)}
        </div>
      </div>
    </div>
  );
}
