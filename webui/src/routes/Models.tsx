import type { RealData } from "../types";

export function Models({ real }: { real: RealData }) {
  const models = real.ve.models;
  const nodes = real.graph_conceptual.nodes;
  // R : which hypotheses each model implements
  const usedBy = (mid: string) => nodes.filter((n) => n.models.includes(mid)).map((n) => n.id);
  const mlabel = Object.fromEntries(models.map((m) => [m.id, m.label]));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элементы M и R · модели и отображение</div>
        <h1>Модели M и отображение R : M → H</h1>
        <p className="lead">
          Каждая гипотеза реализуется ≥1 моделью (<span className="formula">is_implemented_by_model</span>,
          тип «some»). Модель может реализовать несколько гипотез; у гипотезы может быть несколько
          конкурирующих моделей.
        </p>
      </div>

      <div className="panel">
        <div className="label">M — каталог моделей</div>
        <table className="data">
          <thead><tr><th>Модель</th><th>Название</th><th>Класс (онтология)</th><th>Реализует гипотезы (R⁻¹)</th></tr></thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.id}>
                <td className="num">{m.id}</td>
                <td>{m.label}</td>
                <td><span className="chip">{m.class}</span></td>
                <td>{usedBy(m.id).map((h) => <span key={h} className="tag">{h}</span>)}
                  {usedBy(m.id).length === 0 && <span className="muted">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <div className="label">R : H → M(H) · модели каждой гипотезы (один-ко-многим)</div>
        <table className="data">
          <thead><tr><th>Гипотеза</th><th>Ветвь</th><th>Модели R(h)</th><th>Кол-во</th></tr></thead>
          <tbody>
            {nodes.map((n) => (
              <tr key={n.id}>
                <td className="num">{n.id}</td>
                <td className="muted">{n.branch}</td>
                <td>{n.models.map((m) => <span key={m} className="tag">{mlabel[m] ?? m}</span>)}</td>
                <td className="num">{n.models.length}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
