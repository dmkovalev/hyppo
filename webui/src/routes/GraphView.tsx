import { useState } from "react";
import type { RealData } from "../types";
import { GraphBuild, GEdge, GNode } from "../components/GraphBuild";

function Seg({ field, setField }: { field: string; setField: (s: string) => void }) {
  return (
    <div className="seg">
      {["Brugge", "Norne"].map((f) => (
        <button key={f} className={field === f ? "on" : ""} onClick={() => setField(f)}>{f}</button>
      ))}
    </div>
  );
}

export function GraphView({ real, field, setField }: { real: RealData; field: string; setField: (s: string) => void }) {
  const [mode, setMode] = useState<"concept" | "wells">("concept");

  const c = real.graph_conceptual;
  const fr = real.fields[field];
  const g = fr.graph;

  let nodes: GNode[]; let edges: GEdge[];
  if (mode === "concept") {
    nodes = c.nodes.map((n) => ({ id: n.id, label: n.label, status: n.status }));
    edges = c.derivation.map((d) => ({ src: d.src, dst: d.dst, via: d.via, reason: d.reason }));
  } else {
    const st = fr.epistemic_status ?? {};
    nodes = g.nodes.map((n) => ({ id: n.id, label: n.label, kind: n.kind, status: st[n.id] }));
    edges = g.derivation.map((d) => ({ src: d.src, dst: d.dst, via: d.via, reason: d.reason }));
  }

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Алгоритм 1 · причинное упорядочение из уравнений</div>
        <h1>Граф гипотез</h1>
        <p className="lead">
          Рёбра <span className="formula">derived_by</span> выводятся алгоритмом 1: ребро h→h′
          добавляется, когда выходная переменная уравнения h входит в уравнение h′. Нажмите
          «Построить граф» — вывод по шагам; клик по вершине — каскад (алгоритм 4).
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 10 }}>
          <div className="seg">
            <button className={mode === "concept" ? "on" : ""} onClick={() => setMode("concept")}>
              концептуальный (16, алгоритм 1)
            </button>
            <button className={mode === "wells" ? "on" : ""} onClick={() => setMode("wells")}>
              по скважинам (данные {field})
            </button>
          </div>
          {mode === "wells" && <Seg field={field} setField={setField} />}
        </div>

        <div className="muted" style={{ fontSize: 13, marginBottom: 10 }}>
          {mode === "concept"
            ? <>Построено настоящим алгоритмом 1 (<span className="formula">HypothesisLattice</span>): {c.nodes.length} гипотез, {c.edges.length} рёбер, DAG глубины {c.depth}. Сплошная нумерация статьи: жидкость H1–H8, обводнённость H9–H14, ГРП H15, нефть H16.</>
            : <>Инстанция на данных {field}: {g.nodes.length} гипотез-скважин, {g.edges.length} рёбер из матрицы связности CRM (gain &gt; перцентиль).</>}
        </div>

        <GraphBuild key={mode + field} nodes={nodes} edges={edges} />
      </div>
    </div>
  );
}
