import type { RealData } from "../types";

export function Workflow({ real }: { real: RealData }) {
  const c = real.graph_conceptual;
  const labelOf = Object.fromEntries(c.nodes.map((n) => [n.id, n.label]));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент W · поток работ (ОАГ задач)</div>
        <h1>Поток работ и принадлежность гипотез задачам</h1>
        <p className="lead">
          Поток работ — ациклический граф задач (<span className="formula">WorkflowTask</span>),
          глубины {c.depth}. Каждая задача содержит гипотезы (<span className="formula">hasHypothesis</span>),
          исполняется в топологическом порядке. Эта принадлежность используется алгоритмом 1 при
          построении графа и проверяется условием 2 корректной определённости.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", gap: 10, alignItems: "stretch", overflowX: "auto", paddingBottom: 8 }}>
          {c.tasks.map((t, i) => (
            <div key={t.id} style={{ display: "flex", alignItems: "center", gap: 10, flex: "none" }}>
              <div className="wf-task">
                <div className="label" style={{ marginBottom: 6 }}>{t.id}</div>
                <div style={{ fontWeight: 600, marginBottom: 10, fontSize: 13 }}>{t.label}</div>
                <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>гипотез: {t.hypotheses.length}</div>
                {t.hypotheses.map((h) => (
                  <div key={h} className="tag" style={{ display: "block", marginBottom: 3 }}>
                    {h} · {labelOf[h]}
                  </div>
                ))}
              </div>
              {i < c.tasks.length - 1 && <span className="wf-arrow">→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
