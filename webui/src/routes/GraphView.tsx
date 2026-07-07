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
  const c = real.graph_conceptual;
  const status = real.fields[field].concept_status ?? {};
  const nodes: GNode[] = c.nodes.map((n) => ({ id: n.id, label: n.label, status: status[n.id] ?? n.status }));
  const edges: GEdge[] = c.derivation.map((d) => ({ src: d.src, dst: d.dst, via: d.via, reason: d.reason }));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Алгоритм 1 · причинное упорядочение из уравнений</div>
        <h1>Граф гипотез HybridCRM</h1>
        <p className="lead">
          Единый граф модели (16 гипотез H1–H16), построенный настоящим алгоритмом 1
          (<span className="formula">HypothesisLattice</span>) из уравнений: ребро h→h′ — когда выход
          уравнения h входит в уравнение h′. Структура одна для обоих месторождений; цвет вершины —
          эпистемический статус на выбранных данных. Нажмите «Построить граф» — вывод по шагам; клик — каскад.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10, flexWrap: "wrap", gap: 10 }}>
          <div className="muted" style={{ fontSize: 13 }}>
            {c.nodes.length} гипотез · {c.edges.length} рёбер · DAG глубины {c.depth} · статусы на данных <b>{field}</b>
          </div>
          <Seg field={field} setField={setField} />
        </div>
        <GraphBuild key={field} nodes={nodes} edges={edges} />
      </div>
    </div>
  );
}
