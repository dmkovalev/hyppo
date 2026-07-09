import { useState } from "react";
import type { RealData } from "../types";
import { Katex } from "../components/Katex";

export function Models({ real }: { real: RealData }) {
  const models = real.ve.models;
  const nodes = real.graph_conceptual.nodes;
  const usedBy = (mid: string) => nodes.filter((n) => n.models.includes(mid)).map((n) => n.id);
  const mById = Object.fromEntries(models.map((m) => [m.id, m]));
  const [sel, setSel] = useState<string>(models[0]?.id ?? "");
  const m = mById[sel];
  const implH = usedBy(sel).map((id) => nodes.find((n) => n.id === id)!);

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элементы M и R · модели, их Python-реализация и отображение</div>
        <h1>Модели M и отображение R : M → H</h1>
        <p className="lead">
          {models.length} гранулярных моделей — своя реализация на каждую гипотезу (≥1, конкурирующие
          где осмысленно). У каждой указана реализация в Python-библиотеке, конфигурация и меняемые параметры.
          Кликните модель — детали и формулы реализуемых гипотез.
        </p>
      </div>

      <div className="row split">
        <div className="list">
          {models.map((mo) => (
            <div key={mo.id} className={"item" + (sel === mo.id ? " active" : "")} onClick={() => setSel(mo.id)}>
              <span className="t" style={{ fontSize: 13 }}>{mo.label}</span>
              <span className="chip">{mo.class}</span>
            </div>
          ))}
        </div>
        <div className="detail">
          {m && (() => {
            const h0 = implH[0];
            return (
            <>
              <h3>{m.label}</h3>
              <div className="sub"><span className="chip">{m.class}</span></div>
              {m.desc && <p style={{ marginTop: 0, fontSize: 14 }}>{m.desc}</p>}
              {h0 && (
                <div className="tile" style={{ marginBottom: 14, background: "var(--panel-2)" }}>
                  <div className="k">Вход → Выход</div>
                  <div style={{ marginTop: 6 }}>
                    {(h0.equation.inputs ?? []).map((v) => <span key={v} className="tag">{v}</span>)}
                    <span className="formula"> → </span>
                    <span className="tag" style={{ borderColor: "var(--accent)", color: "var(--accent)" }}>{h0.equation.output}</span>
                  </div>
                </div>
              )}
              <dl className="kv">
                <dt>Реализация (Python)</dt><dd className="formula" style={{ fontSize: 12 }}>{m.python_ref}</dd>
                <dt>Конфигурация</dt><dd className="muted">{m.config}</dd>
                <dt>Меняемые параметры</dt>
                <dd>{(m.params ?? []).length ? m.params!.map((p) => <span key={p} className="tag">{p}</span>) : <span className="muted">—</span>}</dd>
                <dt>Реализует гипотезы (R⁻¹)</dt>
                <dd>
                  {implH.map((h) => (
                    <div key={h.id} style={{ marginBottom: 8 }}>
                      <span className="num" style={{ color: "var(--accent)" }}>{h.id}</span> {h.label}
                      <div style={{ marginTop: 4 }}><Katex tex={h.equation.latex ?? h.equation.formula} /></div>
                    </div>
                  ))}
                </dd>
              </dl>
            </>
            );
          })()}
        </div>
      </div>

      <div className="panel">
        <div className="label">R : H → M(h) · модели каждой гипотезы (один-ко-многим) + формула</div>
        <table className="data">
          <thead><tr><th>Гипотеза</th><th>Уравнение</th><th>Модели R(h)</th></tr></thead>
          <tbody>
            {nodes.map((n) => (
              <tr key={n.id}>
                <td className="num">{n.id}<div className="muted" style={{ fontSize: 11 }}>{n.label}</div></td>
                <td><Katex tex={n.equation.latex ?? n.equation.formula} /></td>
                <td>{n.models.map((mid) => (
                  <span key={mid} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(mid)}>
                    {mById[mid]?.label ?? mid}
                  </span>
                ))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
