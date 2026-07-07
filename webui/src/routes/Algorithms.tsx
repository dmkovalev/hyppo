import type { RealData } from "../types";
import { PlanDemo } from "../components/PlanDemo";
import { ComplexityBars } from "../components/ComplexityBars";
import { Battery } from "../components/Battery";

export function Algorithms({ real }: { real: RealData; field?: string }) {
  const c = real.graph_conceptual;
  const a2 = real.algorithm2_example;
  const a4 = real.algorithm4;
  const th = real.theorems;
  const d = real.demos;
  const labelOf = Object.fromEntries(c.nodes.map((n) => [n.id, n.label]));
  const g = { nodes: c.nodes, edges: c.edges };

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Методы главы 2 · на графе гибридной модели заводнения (16 гипотез)</div>
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
        <div className="label">Алгоритм 2 — инкрементальное добавление гипотезы (реальный вызов)</div>
        {d ? (
          <p style={{ marginTop: 0 }}>
            <span className="formula">add_hypothesis({d.alg2.added})</span> даёт тот же граф, что полная
            перестройка (golden-тест), но за O(|H|) объединений. Появившиеся рёбра:{" "}
            {d.alg2.new_edges.map((e, i) => <span key={i} className="tag">{e[0]} → {e[1]}</span>)}.
          </p>
        ) : <p style={{ marginTop: 0 }}>Добавление {a2.add} ({a2.label}).</p>}
        <div className="kv"><dt>Лемма 2</dt><dd>{th.lemma2}</dd></div>
      </div>

      <div className="panel">
        <div className="label">Алгоритм 3 — двухэтапная проверка корректной определённости (реальный check_consistency)</div>
        {d ? (
          <>
            <table className="data">
              <thead><tr><th>Сценарий</th><th>Статус</th><th>Детали</th></tr></thead>
              <tbody>
                {d.alg3.scenarios.map((s, i) => (
                  <tr key={i}>
                    <td>{s.case}</td>
                    <td><span className={"chip " + (s.ok ? "g" : "r")}>{s.status}</span></td>
                    <td className="muted" style={{ fontSize: 12 }}>{s.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
              <b>OWA:</b> {d.alg3.owa_note}
            </p>
          </>
        ) : (
          <table className="data"><tbody>
            {real.algorithm3_conditions.map((cc) => (
              <tr key={cc.n}><td className="num">{cc.n}</td><td>{cc.text}</td>
                <td><span className={"chip " + (cc.ok ? "g" : "r")}>{cc.ok ? "✓" : "✕"}</span></td></tr>
            ))}
          </tbody></table>
        )}
      </div>

      {d?.c7 && (
        <div className="panel">
          <div className="label">Условие C7 — доменное грунтирование переменных (новое правило · SHACL-семантика)</div>
          <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
            Каждая свободная переменная уравнения гипотезы обязана быть объявленным термином
            предметной онтологии (словарь из {d.c7.vocabulary_size} терминов, извлечён из pywaterflood).
            Реальный вызов <span className="formula">check_consistency(…, domain_terms, hypothesis_vars)</span>:
          </p>
          <table className="data">
            <thead><tr><th>Сценарий</th><th>Статус</th><th>Детали</th></tr></thead>
            <tbody>
              <tr>
                <td>Все переменные грунтированы</td>
                <td><span className="chip g">{d.c7.grounded.status}</span></td>
                <td className="muted" style={{ fontSize: 12 }}>{d.c7.grounded.detail}</td>
              </tr>
              <tr>
                <td>Есть негрунтированная переменная</td>
                <td><span className="chip r">{d.c7.ungrounded.status}</span></td>
                <td className="muted" style={{ fontSize: 12 }}>
                  гипотеза h{d.c7.ungrounded.hypothesis}: переменные вне домена{" "}
                  {d.c7.ungrounded.vars.map((v) => <span key={v} className="tag">{v}</span>)}
                </td>
              </tr>
            </tbody>
          </table>
          <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>{d.c7.note}</p>
          <pre style={{ fontFamily: "var(--mono)", fontSize: 11, color: "var(--text)",
                        background: "var(--panel-2)", padding: "12px 14px", borderRadius: 6,
                        overflowX: "auto", margin: "8px 0 0" }}>{d.c7.shacl}</pre>
        </div>
      )}

      {real.graph_conceptual.competes_derivation && real.graph_conceptual.competes_derivation.length > 0 && (
        <div className="panel">
          <div className="label">
            Вывод отношения competes из предметной области{" "}
            {real.graph_conceptual.competes_matches_ref &&
              <span className="chip g" style={{ fontSize: 11 }}>совпало с эталоном</span>}
          </div>
          <p className="muted" style={{ fontSize: 12, marginTop: 0 }}>
            Правило: две гипотезы с одинаковым выходным доменным концептом и без пути derived_by
            между ними — конкурирующие объяснения. Отношение не задаётся вручную, а выводится.
          </p>
          <table className="data">
            <thead><tr><th>Гипотезы</th><th>Общий концепт</th><th>Обоснование</th></tr></thead>
            <tbody>
              {real.graph_conceptual.competes_derivation.map((c, i) => (
                <tr key={i}>
                  <td className="num">{c.a} ↔ {c.b}</td>
                  <td>{c.concept.replace(/_/g, " ")}</td>
                  <td className="muted" style={{ fontSize: 12 }}>{c.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

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

      {d?.cache && (
        <div className="panel">
          <div className="label">
            Кэш результатов — реальный прогон на гибридной модели{" "}
            {d.cache.planner_sees_runner &&
              <span className="chip g" style={{ fontSize: 11 }}>planner ↔ runner: общий SQLite</span>}
          </div>
          <p style={{ marginTop: 0 }}>
            {d.cache.backend}. Ключ: <span className="formula">{d.cache.key}</span>.
            Планировщик и раннер пишут/читают один <span className="formula">SharedCache</span>:
            результат, вычисленный раннером, планировщик видит как кэш.
          </p>
          <div className="tiles" style={{ marginBottom: 12 }}>
            <div className="tile bad">
              <div className="k">Холодный прогон</div>
              <div className="v">{d.cache.cold.computed}/{d.cache.total}</div>
              <div className="muted" style={{ fontSize: 11 }}>вычислено · кэш пуст</div>
            </div>
            <div className="tile good">
              <div className="k">Тёплый прогон</div>
              <div className="v">{d.cache.warm.cached}/{d.cache.total}</div>
              <div className="muted" style={{ fontSize: 11 }}>из кэша · пересчёт {d.cache.warm.computed}</div>
            </div>
            <div className="tile">
              <div className="k">Инвалидация {d.cache.invalidate.changed}</div>
              <div className="v">{d.cache.invalidate.recompute.length}/{d.cache.total}</div>
              <div className="muted" style={{ fontSize: 11 }}>пересчёт · из кэша {d.cache.invalidate.reused}</div>
            </div>
          </div>
          <div className="kv">
            <dt>Смена конфигурации {d.cache.invalidate.changed}</dt>
            <dd>
              пересчитываются только{" "}
              {d.cache.invalidate.recompute.map((h) => <span key={h} className="tag">{h}</span>)}{" "}
              (гипотеза и её замыкание вниз); остальные {d.cache.invalidate.reused} — из кэша.{" "}
              {d.cache.invalidate.matches_descendants &&
                <span className="chip g" style={{ fontSize: 11 }}>= потомки {d.cache.invalidate.changed} ✓</span>}
            </dd>
          </div>
          <p className="muted" style={{ fontSize: 12 }}>{d.cache.note}</p>
        </div>
      )}

      <div className="panel">
        <div className="label">Демо 5 — минимальность плана (Теорема 1, интерактивно)</div>
        <p style={{ marginTop: 0 }}>
          Отметьте кэшированные гипотезы — GUI покажет минимальный план пересчёта <span className="formula">P_ne</span>.
          Или предложите свой план P — проверим корректность (полнота + каскадность) и что P_ne ⊆ P.
        </p>
        <PlanDemo real={real} />
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

      {d && (
        <div className="panel">
          <div className="label">Сложности — операционными счётчиками (янтарь = счётчик, серый = теоретический закон)</div>
          <ComplexityBars real={real} />
          <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
            Детерминированно, не по времени: алг. 1 — n(n−1)/2 проверок полноты (×4 при ×2), алг. 2 — n
            объединений (×2), алг. 4 — ~V+E обходов (×2). Счётчик совпадает с законом.
          </p>
        </div>
      )}

      {d && (
        <div className="panel">
          <div className="label">Правило 5 — процедурная ацикличность (Kahn)</div>
          <p style={{ marginTop: 0 }}>
            Ацикличный граф: <span className="chip g">цикла нет</span>. Циклический — обнаружен со свидетелем:{" "}
            <span className="chip r">цикл {d.rule5.cyclic_witness.join(" → ")}</span>.
            Транзитивное свойство в OWL нельзя объявить асимметричным (ограничение simplicity) — ацикличность
            проверяется процедурно при каждом добавлении связи.
          </p>
        </div>
      )}

      <Battery real={real} />
    </div>
  );
}
