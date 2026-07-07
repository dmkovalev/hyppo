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
        <div className="label">Эпистемический статус гипотез-скважин на {field}</div>
        <table className="data">
          <thead><tr><th>Гипотеза</th><th>Тип</th><th>Скважина</th><th>Статус</th></tr></thead>
          <tbody>
            {fr.graph.nodes.map((n) => {
              const s = fr.epistemic_status[n.id];
              const cls = s === "REFUTED" ? "r" : s === "CONFIRMED" ? "a" : s === "SUPERSEDED" ? "b" : "g";
              const kind = n.kind === "injector" ? "нагнетательная" : n.kind === "producer" ? "добывающая" : "слияние";
              return (
                <tr key={n.id}>
                  <td className="num">{n.id}</td>
                  <td className="muted">{kind}</td>
                  <td>{n.label}</td>
                  <td><span className={"chip " + cls}>{s}</span></td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
