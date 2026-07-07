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
          Интерактивная схема ядра онтологии: <b>круги</b> — классы, <b>бордовые стрелки с подписью</b> —
          объектные свойства (domain→range), <b>пунктир&nbsp;⊑</b> — иерархия подклассов. Всего в
          онтологии комплекса <span className="formula">{o.total_classes}</span> классов. Кликните класс —
          подсветятся его связи и откроется панель свойств.
        </p>
      </div>

      <div className="panel">
        <OntologyCyto classes={o.classes} relations={o.relations} />
      </div>
    </div>
  );
}
