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
  const tiles: { k: string; v: number; good: boolean }[] = [
    { k: "CRM (физика)", v: fr.r2.CRM, good: fr.r2.CRM > 0.5 },
    { k: "Hybrid", v: fr.r2.Hybrid, good: fr.r2.Hybrid > 0.5 },
    { k: "WCT", v: fr.r2.WCT, good: fr.r2.WCT > 0.5 },
    { k: "OPR (нефть)", v: fr.r2.OPR, good: fr.r2.OPR > 0.5 },
  ];

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Апробация · реальные данные заводнения</div>
        <h1>Сравнение гипотез на данных {field}</h1>
        <p className="lead">
          Реальный прогон CRM (pywaterflood) на промысловых данных. Метрика R² и байесовский
          фактор определяют эпистемический статус: физика-CRM подтверждается или опровергается.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div>
            <div className="stage-name">{field}</div>
            <div className="muted">{fr.producers} добывающих + {fr.injectors} нагнетательных · {fr.months} мес. · обучение [{fr.train[0]}:{fr.train[1]}]</div>
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
        <div className="label">Вердикт по физической гипотезе (байесовский фактор)</div>
        <div className="ring-wrap">
          <div>
            <div className={"verdict-big " + (refuted ? "ref" : "inc")}>
              {refuted ? "CRM ОПРОВЕРГНУТА" : "CRM не опровергнута"}
            </div>
            <div className="muted">
              BF(CRM-UTO / POS) = <span className="num">{fr.bayes_factor.toExponential(2)}</span>
              {" · "}порог REFUTED: BF &lt; 0.1
            </div>
            <p className="muted" style={{ maxWidth: "60ch", marginBottom: 0 }}>
              {refuted
                ? `На ${field} чистая физическая CRM (R²=${fr.r2.CRM.toFixed(2)}) систематически хуже, чем гибрид (R²=${fr.r2.Hybrid.toFixed(2)}) — гипотеза h_CRM опровергается, h_LPR (слияние) подтверждается.`
                : `На ${field} физическая CRM (R²=${fr.r2.CRM.toFixed(2)}) сопоставима с альтернативой — решение неокончательно, гипотеза не опровергнута.`}
            </p>
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="label">Эпистемический статус гипотез на {field}</div>
        <table className="data">
          <thead><tr><th>Гипотеза</th><th>Ветвь</th><th>Статус</th></tr></thead>
          <tbody>
            {real.ve.hypotheses.map((h) => {
              const s = fr.epistemic_status[h.id];
              const cls = s === "REFUTED" ? "r" : s === "CONFIRMED" ? "a" : s === "SUPERSEDED" ? "b" : "g";
              return (
                <tr key={h.id}>
                  <td className="num">{h.id}</td>
                  <td className="muted">{h.branch}</td>
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
