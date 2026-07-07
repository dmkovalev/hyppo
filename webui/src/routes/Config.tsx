import type { RealData } from "../types";

export function Config({ real }: { real: RealData }) {
  const cfg = real.ve.configuration;
  const size = real.ve.config_space_size;
  const sections = Array.from(new Set(cfg.map((a) => a.section)));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент 𝒞 · пространство конфигураций</div>
        <h1>Конфигурация 𝒞 = ∏ Qᵢ</h1>
        <p className="lead">
          Пространство гиперпараметров HybridCRM: <span className="formula">{size.toLocaleString("ru")}</span>{" "}
          конфигураций по <span className="formula">{cfg.length}</span> осям. Идентично для Brugge и Norne
          (модельно-определено). По утверждению об изоморфизме 𝒞 ≅ графу Хэмминга H(q₁,…,qₙ).
        </p>
      </div>

      <div className="panel">
        <div className="ring-wrap" style={{ marginBottom: 8 }}>
          <div>
            <div className="stage-name formula">|𝒞| = {size.toLocaleString("ru")}</div>
            <div className="muted">{cfg.map((a) => a.levels.length).join(" × ")}</div>
          </div>
        </div>
      </div>

      {sections.map((sec) => (
        <div className="panel" key={sec}>
          <div className="label">Секция: {sec}</div>
          <table className="data">
            <thead><tr><th>Ось (гиперпараметр)</th><th>Уровни</th><th>|Qᵢ|</th></tr></thead>
            <tbody>
              {cfg.filter((a) => a.section === sec).map((a) => (
                <tr key={a.name}>
                  <td className="num">{a.name}</td>
                  <td>{a.levels.map((l) => <span key={String(l)} className="tag">{String(l)}</span>)}</td>
                  <td className="num">{a.levels.length}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
