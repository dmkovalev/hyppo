import type { RealData } from "../types";

export function Experiment({ real, field }: { real: RealData; field: string }) {
  const ve = real.ve;
  const fr = real.fields[field];
  const g = fr.graph;

  const els = [
    { sym: "O", name: "Онтология", v: `${ve.ontology.total_classes} классов`, note: "owlready2, HermiT" },
    { sym: "H", name: "Гипотезы", v: `${g.nodes.length}`, note: `скважины + слияние (${field})` },
    { sym: "M", name: "Модели", v: `${ve.models.length}`, note: "Physics · DataDriven · Hybrid" },
    { sym: "R", name: "R : M → H", v: g.r_map, note: "is_implemented_by_model" },
    { sym: "W", name: "Поток работ", v: `${g.tasks.length} задач`, note: "hasHypothesis" },
    { sym: "𝒞", name: "Конфигурации", v: ve.config_space_size.toLocaleString("ru"), note: `${ve.configuration.length} осей` },
  ];

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Определение 1 · реальный ВЭ HybridCRM</div>
        <h1>{real.domain}</h1>
        <p className="lead">
          Кортеж <span className="formula">⟨O, H, M, R, W, 𝒞⟩</span> из настоящей библиотеки:
          онтология owlready2, гипотезы-скважины из реальной связности CRM, модели физика/ML/гибрид,
          поток работ с задачами, {ve.config_space_size.toLocaleString("ru")} конфигураций.
          Детали — в разделах «Граф», «Онтология», «Поток работ».
        </p>
      </div>

      <div className="panel">
        <div className="tuple">
          <span className="br">⟨</span>
          {els.map((e, i) => (
            <span key={e.sym}>
              <span className="el active" style={{ cursor: "default" }}>{e.sym}</span>
              {i < els.length - 1 && <span className="comma">,</span>}
            </span>
          ))}
          <span className="br">⟩</span>
        </div>
        <div className="el-cards">
          {els.map((e) => (
            <div key={e.sym} className="el-card" style={{ cursor: "default" }}>
              <div className="sym">{e.sym}</div>
              <div className="nm">{e.name}</div>
              <div className="ct" style={{ fontSize: 16 }}>{e.v}</div>
              <div className="muted" style={{ fontSize: 10, marginTop: 4 }}>{e.note}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        <div className="label">Конфигурация 𝒞 — {ve.configuration.length} осей гиперпараметров</div>
        <table className="data">
          <thead><tr><th>Ось</th><th>Секция</th><th>Уровни</th><th>|Qᵢ|</th></tr></thead>
          <tbody>
            {ve.configuration.map((a, i) => (
              <tr key={i}>
                <td className="num">{a.name}</td>
                <td className="muted">{a.section}</td>
                <td>{a.levels.map((l) => <span key={String(l)} className="tag">{String(l)}</span>)}</td>
                <td className="num">{a.levels.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
