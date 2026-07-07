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
  const fr = real.fields[field];
  const g = fr.graph;
  const status = fr.epistemic_status ?? {};
  const nodes: GNode[] = g.nodes.map((n) => ({ id: n.id, label: n.label, kind: n.kind, status: status[n.id] }));
  const edges: GEdge[] = g.derivation.map((d) => ({ src: d.src, dst: d.dst, via: d.via, reason: d.reason }));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Алгоритм 1 · граф из реальной связности CRM</div>
        <h1>Граф гипотез · {field}</h1>
        <p className="lead">
          Каждая скважина — гипотеза. Рёбра <span className="formula">derived_by</span> выведены
          из матрицы связности CRM (pywaterflood): инжектор→продюсер, если коэффициент{" "}
          <span className="formula">gain</span> выше 75-го перцентиля; продюсеры сходятся в слиянии OPR.
          Нажмите «Построить граф» — рёбра проявятся по одному с пояснением.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
          <div className="muted" style={{ fontSize: 13 }}>
            {fr.producers} добывающих + {fr.injectors} нагнетательных · {g.nodes.length} гипотез · {g.edges.length} рёбер · R:M→H {g.r_map}
          </div>
          <Seg field={field} setField={setField} />
        </div>
        <GraphBuild nodes={nodes} edges={edges} />
      </div>
    </div>
  );
}
