import type { RealData } from "../types";
import { LifecycleCycle } from "../components/LifecycleCycle";

export function Experiment({ real, field, onNavigate }:
  { real: RealData; field: string; onNavigate: (tab: string) => void }) {
  const ve = real.ve;
  const c = real.graph_conceptual;

  const els = [
    { sym: "O", name: "Онтология", v: `${ve.ontology.total_classes} классов`, note: "owlready2, VOWL", tab: "onto" },
    { sym: "H", name: "Гипотезы", v: `${c.nodes.length}`, note: "граф H1–H16 (алг. 1)", tab: "graph" },
    { sym: "M", name: "Модели", v: `${ve.models.length}`, note: "Physics · ML · Hybrid", tab: "models" },
    { sym: "R", name: "R : M → H", v: "≥1 : 1", note: "is_implemented_by_model", tab: "models" },
    { sym: "W", name: "Поток работ", v: `${c.tasks.length} задач`, note: "hasHypothesis, глубина " + c.depth, tab: "wf" },
    { sym: "𝒞", name: "Конфигурации", v: ve.config_space_size.toLocaleString("ru"), note: `${ve.configuration.length} осей`, tab: "config" },
  ];

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Определение 1 · реальный ВЭ HybridCRM</div>
        <h1>{real.domain}</h1>
        <p className="lead">
          Кортеж <span className="formula">⟨O, H, M, R, W, 𝒞⟩</span> из настоящей библиотеки.
          Кликните любой элемент — откроется его вкладка со всем содержимым.
        </p>
      </div>

      <div className="panel">
        <LifecycleCycle real={real} />
      </div>

      <div className="panel">
        <div className="tuple">
          <span className="br">⟨</span>
          {els.map((e, i) => (
            <span key={e.sym}>
              <span className="el" style={{ cursor: "pointer" }} onClick={() => onNavigate(e.tab)}>{e.sym}</span>
              {i < els.length - 1 && <span className="comma">,</span>}
            </span>
          ))}
          <span className="br">⟩</span>
        </div>
        <div className="el-cards">
          {els.map((e) => (
            <div key={e.sym} className="el-card" style={{ cursor: "pointer" }} onClick={() => onNavigate(e.tab)}>
              <div className="sym">{e.sym}</div>
              <div className="nm">{e.name}</div>
              <div className="ct" style={{ fontSize: 16 }}>{e.v}</div>
              <div className="muted" style={{ fontSize: 10, marginTop: 4 }}>{e.note} →</div>
            </div>
          ))}
        </div>
        <p className="muted" style={{ fontSize: 12, marginTop: 14 }}>
          Данные месторождения ({field}) влияют на статусы гипотез в графе и метрики в разделе «Данные».
        </p>
      </div>
    </div>
  );
}
