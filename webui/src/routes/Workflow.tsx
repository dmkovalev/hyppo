import type { RealData } from "../types";

function Seg({ field, setField }: { field: string; setField: (s: string) => void }) {
  return (
    <div className="seg">
      {["Brugge", "Norne"].map((f) => (
        <button key={f} className={field === f ? "on" : ""} onClick={() => setField(f)}>{f}</button>
      ))}
    </div>
  );
}

export function Workflow({ real, field, setField }: { real: RealData; field: string; setField: (s: string) => void }) {
  const g = real.fields[field].graph;
  const labelOf = Object.fromEntries(g.nodes.map((n) => [n.id, n.label]));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент W · поток работ (ОАГ задач)</div>
        <h1>Поток работ и принадлежность гипотез задачам</h1>
        <p className="lead">
          Поток работ — ациклический граф задач. Каждая задача (<span className="formula">WorkflowTask</span>)
          содержит гипотезы (<span className="formula">hasHypothesis</span>); задачи исполняются в
          топологическом порядке. Именно эта принадлежность определяет корректную определённость (условие 2).
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: 12 }}>
          <Seg field={field} setField={setField} />
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "stretch", overflowX: "auto" }}>
          {g.tasks.map((t, i) => (
            <div key={t.id} style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div className="panel" style={{ minWidth: 220, background: "var(--panel-2)" }}>
                <div className="label" style={{ marginBottom: 6 }}>{t.id}</div>
                <div style={{ fontWeight: 600, marginBottom: 10 }}>{t.label}</div>
                <div className="muted" style={{ fontSize: 11, marginBottom: 6 }}>гипотез: {t.hypotheses.length}</div>
                <div style={{ maxHeight: 220, overflowY: "auto" }}>
                  {t.hypotheses.map((h) => (
                    <div key={h} className="tag" style={{ display: "block", marginBottom: 3 }}>
                      {h} · {labelOf[h]}
                    </div>
                  ))}
                </div>
              </div>
              {i < g.tasks.length - 1 && <span className="formula" style={{ fontSize: 22 }}>→</span>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
