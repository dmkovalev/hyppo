import type { RealData } from "../types";
import { OntologyGraph } from "../components/OntologyGraph";

export function Ontology({ real }: { real: RealData }) {
  const o = real.ve.ontology;
  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент O · онтология owlready2</div>
        <h1>Онтология виртуального эксперимента</h1>
        <p className="lead">
          Ядро онтологии: сплошные стрелки — иерархия классов (⊑), пунктирные — объектные
          свойства с областью и диапазоном. Всего в онтологии комплекса{" "}
          <span className="formula">{o.total_classes}</span> классов; ниже — ядровой фрагмент.
        </p>
      </div>

      <div className="panel">
        <div className="label">Граф классов и свойств</div>
        <OntologyGraph classes={o.classes} relations={o.relations} />
      </div>

      <div className="panel">
        <div className="label">Объектные свойства (область → диапазон)</div>
        <table className="data">
          <thead><tr><th>Свойство</th><th>Domain</th><th></th><th>Range</th></tr></thead>
          <tbody>
            {o.relations.map((r, i) => (
              <tr key={i}>
                <td className="num">{r.property}</td>
                <td>{r.domain}</td>
                <td className="formula">→</td>
                <td>{r.range}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
