import type { RealData } from "../types";
import { OntologyCyto } from "../components/OntologyCyto";

export function Ontology({ real }: { real: RealData }) {
  const o = real.ve.ontology;
  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент O · онтология owlready2 (нотация VOWL)</div>
        <h1>Онтология виртуального эксперимента</h1>
        <p className="lead">
          Интерактивная схема онтологии: <b>синие круги</b> — классы платформы (гипотезы, модели, правила),
          <b> бордовые прямоугольники</b> — классы <b>предметной области</b> (Петро-домен: Well, Injector,
          Producer, ReservoirParameter…). Стрелки с подписью — объектные свойства (domain→range),
          пунктир&nbsp;⊑ — подклассы. Всего <span className="formula">{o.total_classes}</span> классов, из них
          {" "}{o.classes.filter((c) => c.group === "domain").length} доменных. Кликните класс — подсветятся связи.
        </p>
      </div>

      <div className="panel">
        <OntologyCyto classes={o.classes} relations={o.relations} />
      </div>
    </div>
  );
}
