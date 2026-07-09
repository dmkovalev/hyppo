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

export function Fields({ real, field, setField }: { real: RealData; field: string; setField: (s: string) => void }) {
  const fr = real.fields[field];
  const refuted = fr.physics_verdict === "REFUTED";
  const mlabel = Object.fromEntries(real.ve.models.map((m) => [m.id, m.label]));
  const tiles = [
    { k: "CRM (физика)", v: fr.r2.CRM, good: fr.r2.CRM > 0.5 },
    { k: "Hybrid", v: fr.r2.Hybrid, good: fr.r2.Hybrid > 0.5 },
    { k: "WCT", v: fr.r2.WCT, good: fr.r2.WCT > 0.5 },
    { k: "OPR (нефть)", v: fr.r2.OPR, good: fr.r2.OPR > 0.5 },
  ];

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Апробация · pywaterflood на реальных данных</div>
        <h1>Результаты на данных {field}</h1>
        <p className="lead">
          Реальный прогон CRM на промысловых данных. R² и байесовский фактор определяют
          эпистемический статус гипотез.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div>
            <div className="stage-name">{field}</div>
            <div className="muted">{fr.producers} добывающих + {fr.injectors} нагнетательных · {fr.months} мес. · CRM: {fr.fit}</div>
          </div>
          <Seg field={field} setField={setField} />
        </div>
        <div className="tiles">
          {tiles.map((t) => (
            <div key={t.k} className={"tile " + (t.good ? "good" : "bad")}>
              <div className="k">{t.k}</div>
              <div className="v">{t.v.toFixed(2)}</div>
              <div className="muted" style={{ fontSize: 11 }}>R²</div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="label">Вердикт по физической CRM (байесовский фактор)</div>
        <div className={"verdict-big " + (refuted ? "ref" : "inc")}>
          {refuted ? "CRM ОПРОВЕРГНУТА" : "CRM не опровергнута"}
        </div>
        <div className="muted">
          BF(CRM-UTO / POS) = <span className="num">{fr.bayes_factor.toExponential(2)}</span> · порог REFUTED: BF &lt; 0.1
        </div>
        <p className="muted" style={{ maxWidth: "62ch" }}>
          {refuted
            ? `На ${field} чистая физика (CRM R²=${fr.r2.CRM.toFixed(2)}) систематически хуже гибрида (R²=${fr.r2.Hybrid.toFixed(2)}) — физическая гипотеза опровергается, гибрид подтверждается.`
            : `На ${field} физическая CRM (R²=${fr.r2.CRM.toFixed(2)}) неокончательна; гибрид (R²=${fr.r2.Hybrid.toFixed(2)}) и прогноз нефти (OPR R²=${fr.r2.OPR.toFixed(2)}) устойчивы.`}
        </p>
      </div>

      <div className="panel">
        <div className="label">
          Гипотезы модели (H1–H16) · модели (R : M→H, ≥1) · статус на {field}
        </div>
        <table className="data">
          <thead><tr><th>Гипотеза</th><th>Ветвь</th><th>Название</th><th>Модели (R)</th><th>Статус на {field}</th></tr></thead>
          <tbody>
            {real.graph_conceptual.nodes.map((n) => {
              const s = fr.concept_status[n.id] ?? n.status;
              const cls = s === "REFUTED" ? "r" : s === "CONFIRMED" ? "a" : s === "SUPERSEDED" ? "b" : "g";
              return (
                <tr key={n.id}>
                  <td className="num">{n.id}</td>
                  <td className="muted">{n.branch}</td>
                  <td>{n.label}</td>
                  <td>
                    {n.models.map((m) => <span key={m} className="tag">{mlabel[m] ?? m}</span>)}
                    <span className="muted" style={{ fontSize: 11 }}> · {n.models.length}</span>
                  </td>
                  <td><span className={"chip " + cls}>{s}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: 12 }}>
          Граф модели один для обоих месторождений; на {field} физика жидкости{" "}
          {fr.r2.CRM >= 0.5 ? "подтверждается" : "опровергается"} (CRM R²={fr.r2.CRM.toFixed(2)}).
          Свойство <span className="formula">is_implemented_by_model</span> типа «some» (≥1): у физических
          гипотез несколько конкурирующих моделей (CRMP/CRMT). Связность скважин ({fr.producers}+{fr.injectors})
          — в разделе связности, не в графе гипотез.
        </p>
      </div>
    </div>
  );
}
