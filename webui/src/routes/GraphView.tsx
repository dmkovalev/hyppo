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
  const status = real.fields[field]?.epistemic_status ?? {};
  const nodes: GNode[] = real.ve.hypotheses.map((h) => ({ id: h.id, label: h.label, status: status[h.id] }));
  const edges: GEdge[] = real.graph.derivation.map((d) => ({ src: d.src, dst: d.dst, via: d.via, reason: d.reason }));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Алгоритм 1 · причинное упорядочение</div>
        <h1>Граф гипотез строится из уравнений</h1>
        <p className="lead">
          Рёбра <span className="formula">derived_by</span> выводятся алгоритмом 1: ребро
          h→h′ добавляется, когда выходная переменная уравнения h входит в уравнение h′.
          Нажмите «Построить граф», чтобы увидеть вывод по шагам.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div className="label" style={{ margin: 0 }}>Месторождение (влияет на статусы вершин)</div>
          <Seg field={field} setField={setField} />
        </div>
        <GraphBuild nodes={nodes} edges={edges} />
      </div>
    </div>
  );
}
