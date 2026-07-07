import type { RealData } from "../types";

export function Algorithms({ real }: { real: RealData; field?: string }) {
  const c = real.graph_conceptual;
  const a2 = real.algorithm2_example;
  const a4 = real.algorithm4;
  const th = real.theorems;
  const labelOf = Object.fromEntries(c.nodes.map((n) => [n.id, n.label]));
  const g = { nodes: c.nodes, edges: c.edges };

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Методы главы 2 · на графе HybridCRM (16 гипотез)</div>
        <h1>Алгоритмы и теоремы</h1>
        <p className="lead">Четыре алгоритма управления виртуальным экспериментом с доказанными оценками.</p>
      </div>

      <div className="panel">
        <div className="label">Алгоритм 1 — построение графа гипотез</div>
        <p style={{ marginTop: 0 }}>
          Настоящим алгоритмом 1 (<span className="formula">HypothesisLattice</span>) выведено{" "}
          <b>{g.edges.length} рёбер</b> derived_by для <b>{g.nodes.length} гипотез</b> (DAG глубины {c.depth}).
          Пошаговое построение — в разделе «Граф гипотез».
        </p>
        <div className="kv"><dt>Лемма 1</dt><dd>{th.lemma1}</dd></div>
      </div>

      <div className="panel">
        <div className="label">Алгоритм 2 — инкрементальное добавление гипотезы</div>
        <p style={{ marginTop: 0 }}>
          Добавление <span className="num">{a2.add}</span> ({a2.label}) не требует полной перестройки графа.
        </p>
        <div className="kv"><dt>Лемма 2</dt><dd>{th.lemma2}</dd><dt>Смысл</dt><dd>{a2.note}</dd></div>
      </div>

      <div className="panel">
        <div className="label">Алгоритм 3 — проверка корректной определённости</div>
        <table className="data">
          <thead><tr><th>№</th><th>Условие</th><th>Статус</th></tr></thead>
          <tbody>
            {real.algorithm3_conditions.map((c) => (
              <tr key={c.n}>
                <td className="num">{c.n}</td><td>{c.text}</td>
                <td><span className={"chip " + (c.ok ? "g" : "r")}>{c.ok ? "✓ выполнено" : "✕ нарушено"}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="panel">
        <div className="label">Алгоритм 4 — планирование с каскадным переиспользованием</div>
        <p style={{ marginTop: 0 }}>
          При изменении гипотезы пересчёту (<span className="formula">P_ne</span>) подлежат она и потомки;
          остальное — из кэша (<span className="formula">P_e</span>).
        </p>
        <table className="data">
          <thead><tr><th>Изменена</th><th>P_ne</th><th>Доля пересчёта</th></tr></thead>
          <tbody>
            {Object.entries(a4).map(([k, v]) => (
              <tr key={k}>
                <td className="num">{v.changed.map((h) => `${h} · ${labelOf[h] ?? ""}`).join(", ")}</td>
                <td>{v.p_ne.map((h) => <span key={h} className="tag">{h}</span>)}</td>
                <td className="num">{Math.round(v.recompute_frac * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="kv" style={{ marginTop: 14 }}>
          <dt>Теорема 1</dt><dd>{th.theorem1}</dd>
          <dt>Утверждение</dt><dd>{th.prop_hamming}</dd>
        </div>
        <p className="muted" style={{ fontSize: 12 }}>Интерактивный каскад по клику на вершину — в разделе «Граф гипотез».</p>
      </div>

      <div className="panel">
        <div className="label">Масштабируемость — алгоритм 1 и онтологический вывод до 10 000 гипотез</div>
        <p style={{ marginTop: 0 }}>{real.scale.note}</p>
        <table className="data">
          <thead><tr><th>Гипотез</th><th>Масштаб</th><th>ELK (OWL 2 EL), с</th><th>HermiT (OWL 2 DL), с</th></tr></thead>
          <tbody>
            {real.scale.points.map((p) => (
              <tr key={p.hypotheses}>
                <td className="num">{p.hypotheses.toLocaleString("ru")}</td>
                <td className="muted">{p.wells}</td>
                <td className="num">{p.ELK_s.toFixed(2)}</td>
                <td className="num">{p.HermiT_s.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="muted" style={{ fontSize: 12 }}>
          На <b>10 166</b> гипотез профиль OWL 2 EL (ELK) быстрее HermiT в <b>{real.scale.speedup_10k}</b> —
          полиномиальная сложность вместо 2-EXPTIME. Граф на любом масштабе строится алгоритмом 1 из уравнений.
        </p>
      </div>
    </div>
  );
}
