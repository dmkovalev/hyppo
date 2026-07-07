import { useState } from "react";
import type { RealData } from "../types";
import { Katex } from "../components/Katex";

export function Hypotheses({ real, field }: { real: RealData; field: string }) {
  const nodes = real.graph_conceptual.nodes;
  const edges = real.graph_conceptual.edges;
  const mById = Object.fromEntries(real.ve.models.map((m) => [m.id, m]));
  const status = real.fields[field]?.concept_status ?? {};
  const [sel, setSel] = useState<string>(nodes[0]?.id ?? "");
  const h = nodes.find((n) => n.id === sel) ?? nodes[0];

  const derivedFrom = edges.filter(([, b]) => b === h.id).map(([a]) => a);
  const impacts = edges.filter(([a]) => a === h.id).map(([, b]) => b);
  const labelOf = Object.fromEntries(nodes.map((n) => [n.id, n.label]));
  const chip = (s?: string) => {
    const cl = s === "REFUTED" ? "r" : s === "CONFIRMED" ? "a" : s === "SUPERSEDED" ? "b" : "g";
    return <span className={"chip " + cl}>{s}</span>;
  };

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент H · гипотезы</div>
        <h1>Гипотезы виртуального эксперимента</h1>
        <p className="lead">
          {nodes.length} гипотез. Кликните гипотезу — описание, уравнение, порождающие и зависимые
          гипотезы, конкурирующие и реализующие модели. Статус — на данных {field}.
        </p>
      </div>

      <div className="row split">
        <div className="list">
          {nodes.map((n) => (
            <div key={n.id} className={"item" + (h.id === n.id ? " active" : "")} onClick={() => setSel(n.id)}>
              <span className="t">{n.id} <span className="muted" style={{ fontWeight: 400 }}>{n.label}</span></span>
              {chip(status[n.id])}
            </div>
          ))}
        </div>
        <div className="detail">
          <h3>{h.id} · {h.label}</h3>
          {h.desc && <p style={{ marginTop: 0 }}>{h.desc}</p>}
          <div style={{ margin: "10px 0 14px" }}><Katex tex={h.equation.latex ?? h.equation.formula} block /></div>
          <dl className="kv">
            <dt>Ветвь</dt><dd>{h.branch}</dd>
            <dt>Статус ({field})</dt><dd>{chip(status[h.id])}</dd>
            <dt>Вход → Выход</dt>
            <dd>{(h.equation.inputs ?? []).map((v) => <span key={v} className="tag">{v}</span>)}
              <span className="formula"> → </span>
              <span className="tag" style={{ borderColor: "var(--accent)", color: "var(--accent)" }}>{h.equation.output}</span></dd>
            <dt>Порождается из (derived_by)</dt>
            <dd>{derivedFrom.length ? derivedFrom.map((id) => (
              <span key={id} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(id)}>{id} · {labelOf[id]}</span>
            )) : <span className="muted">— корень (сырые данные)</span>}</dd>
            <dt>Влияет на (зависимые)</dt>
            <dd>{impacts.length ? impacts.map((id) => (
              <span key={id} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(id)}>{id} · {labelOf[id]}</span>
            )) : <span className="muted">— терминальная</span>}</dd>
            <dt>Конкурирующие (competes)</dt>
            <dd>{(h.competes ?? []).length ? h.competes!.map((id) => (
              <span key={id} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(id)}>{id} · {labelOf[id]}</span>
            )) : <span className="muted">—</span>}</dd>
            <dt>Реализующие модели (R)</dt>
            <dd>{h.models.map((mid) => <span key={mid} className="tag">{mById[mid]?.label ?? mid}</span>)}</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}
