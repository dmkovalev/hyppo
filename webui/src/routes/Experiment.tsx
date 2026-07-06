import { useState } from "react";
import type { RealData } from "../types";

const ELS = [
  { key: "O", sym: "O", name: "Онтология" },
  { key: "H", sym: "H", name: "Гипотезы" },
  { key: "M", sym: "M", name: "Модели" },
  { key: "R", sym: "R", name: "Отображение" },
  { key: "W", sym: "W", name: "Поток работ" },
  { key: "C", sym: "𝒞", name: "Конфигурации" },
];

function chip(s?: string) {
  const m: Record<string, string> = { SUPPORTED: "g", CONFIRMED: "a", REFUTED: "r", SUPERSEDED: "b" };
  return <span className={"chip " + (m[s ?? ""] ?? "")}>{s ?? "—"}</span>;
}

export function Experiment({ real, field }: { real: RealData; field: string }) {
  const ve = real.ve;
  const status = real.fields[field]?.epistemic_status ?? {};
  const [el, setEl] = useState("H");
  const [sub, setSub] = useState("");

  const models = Array.from(new Set(ve.hypotheses.map((h) => h.model).filter(Boolean))) as string[];
  const counts: Record<string, number | string> = {
    O: ve.ontology.classes.length, H: ve.hypotheses.length, M: models.length,
    R: ve.mapping.length, W: real.graph.edges.length, C: ve.config_space_size,
  };

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Определение 1 · реальный ВЭ (owlready2)</div>
        <h1>{real.domain}</h1>
        <p className="lead">
          Кортеж <span className="formula">⟨O, H, M, R, W, 𝒞⟩</span> из настоящей онтологии
          комплекса. Кликните элемент, затем отдельные сущности.
        </p>
      </div>

      <div className="panel">
        <div className="tuple">
          <span className="br">⟨</span>
          {ELS.map((e, i) => (
            <span key={e.key}>
              <span className={"el" + (el === e.key ? " active" : "")}
                    onClick={() => { setEl(e.key); setSub(""); }}>{e.sym}</span>
              {i < ELS.length - 1 && <span className="comma">,</span>}
            </span>
          ))}
          <span className="br">⟩</span>
        </div>
        <div className="el-cards">
          {ELS.map((e) => (
            <div key={e.key} className={"el-card" + (el === e.key ? " active" : "")}
                 onClick={() => { setEl(e.key); setSub(""); }}>
              <div className="sym">{e.sym}</div>
              <div className="nm">{e.name}</div>
              <div className="ct">{counts[e.key]}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="panel">
        {el === "O" && (
          <div>
            <div className="label">O — онтология {ve.ontology.name} · {ve.ontology.classes.length} классов</div>
            <div style={{ marginBottom: 14 }}>
              {ve.ontology.classes.map((c) => <span key={c} className="tag">{c}</span>)}
            </div>
            <div className="row c2">
              <div>
                <div className="label">Объектные свойства ({ve.ontology.object_properties.length})</div>
                {ve.ontology.object_properties.map((p) => <span key={p.name} className="tag">{p.name}</span>)}
              </div>
              <div>
                <div className="label">Свойства-данные ({ve.ontology.data_properties.length})</div>
                {ve.ontology.data_properties.map((p) => <span key={p.name} className="tag">{p.name}</span>)}
              </div>
            </div>
          </div>
        )}

        {el === "H" && (() => {
          const h = ve.hypotheses.find((x) => x.id === sub) ?? ve.hypotheses[0];
          return (
            <div>
              <div className="label">H — гипотезы (ветви LPR / WCT)</div>
              <div className="row split">
                <div className="list">
                  {ve.hypotheses.map((x) => (
                    <div key={x.id} className={"item" + (h.id === x.id ? " active" : "")}
                         onClick={() => setSub(x.id)}>
                      <span className="t">{x.id}</span>
                      {chip(status[x.id])}
                    </div>
                  ))}
                </div>
                <div className="detail">
                  <h3>{h.label}</h3>
                  <div className="sub">{h.description}</div>
                  <dl className="kv">
                    <dt>Ветвь</dt><dd>{h.branch}</dd>
                    <dt>Статус ({field})</dt><dd>{chip(status[h.id])}</dd>
                    <dt>Уравнение</dt><dd className="formula">{h.equation?.formula}</dd>
                    <dt>Выход</dt><dd className="num">{h.equation?.output}</dd>
                    <dt>Переменные</dt><dd>{(h.variables ?? []).map((v) => <span key={v} className="tag">{v}</span>)}</dd>
                    <dt>Модель</dt><dd>{h.model_classes.slice(0, 2).map((m) => <span key={m} className="tag">{m}</span>)}</dd>
                    <dt>Гиперпараметры</dt><dd>{h.hyperparam_axes.map((a) => <span key={a} className="tag">{a}</span>)}</dd>
                  </dl>
                </div>
              </div>
            </div>
          );
        })()}

        {el === "M" && (
          <div>
            <div className="label">M — модели, реализующие гипотезы</div>
            <table className="data">
              <thead><tr><th>Модель</th><th>Класс (MRO)</th><th>Реализует</th></tr></thead>
              <tbody>
                {models.map((m) => {
                  const hs = ve.hypotheses.filter((h) => h.model === m);
                  return (
                    <tr key={m}>
                      <td className="num">{m}</td>
                      <td>{hs[0]?.model_classes.slice(0, 2).map((c) => <span key={c} className="tag">{c}</span>)}</td>
                      <td>{hs.map((h) => <span key={h.id} className="tag">{h.id}</span>)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {el === "R" && (
          <div>
            <div className="label">R : M → H — отображение (is_implemented_by_model, FunctionalProperty)</div>
            <table className="data">
              <thead><tr><th>Гипотеза</th><th></th><th>Модель R(h)</th></tr></thead>
              <tbody>
                {ve.mapping.map((r, i) => (
                  <tr key={i}>
                    <td className="num">{r.hypothesis}</td>
                    <td className="formula">←</td>
                    <td className="num">{r.model ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {el === "W" && (
          <div>
            <div className="label">W — поток работ (ОАГ, рёбра derived_by)</div>
            <div className="list">
              {real.graph.edges.map((e, i) => (
                <div key={i} className="item" style={{ cursor: "default" }}>
                  <span className="t">{e[0]}</span>
                  <span className="formula">→ derived_by →</span>
                  <span className="t">{e[1]}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {el === "C" && (
          <div>
            <div className="label">𝒞 = ∏ Qᵢ — {ve.config_space_size.toLocaleString("ru")} конфигураций · {ve.configuration.length} осей</div>
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
        )}
      </div>
    </div>
  );
}
