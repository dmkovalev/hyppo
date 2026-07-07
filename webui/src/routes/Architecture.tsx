import { useState } from "react";
import type { RealData, ArchComponent } from "../types";

const LAYER_COLOR: Record<string, string> = {
  "Интерфейс": "#3b6fb0", "Оркестрация": "#7c1d2b", "Методы (гл. 2)": "#2f7d43",
  "Исполнение": "#b0324f", "Онтология": "#8a5a1f", "Хранение": "#5f5c55", "Инфраструктура": "#928e85",
};

export function Architecture({ real }: { real: RealData }) {
  const a = real.architecture;
  const byId = Object.fromEntries(a.components.map((c) => [c.id, c]));
  const [sel, setSel] = useState<string>("");
  const s = sel ? byId[sel] : null;

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Программный комплекс · глава 3</div>
        <h1>Архитектура комплекса Hyppo</h1>
        <p className="lead">{a.note} Кликните компонент — его назначение, модуль и зависимости.</p>
      </div>

      <div className="panel">
        {a.layers.map((layer) => {
          const comps = a.components.filter((c) => c.layer === layer);
          return (
            <div key={layer} style={{ display: "flex", gap: 14, alignItems: "center", marginBottom: 12 }}>
              <div style={{ width: 150, flex: "none", fontFamily: "var(--mono)", fontSize: 11,
                            textTransform: "uppercase", letterSpacing: 1, color: LAYER_COLOR[layer] ?? "var(--muted)" }}>
                {layer}
              </div>
              <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                {comps.map((c) => (
                  <div key={c.id} onClick={() => setSel(c.id)}
                       style={{ cursor: "pointer", padding: "10px 14px", borderRadius: 9,
                                border: `1.6px solid ${sel === c.id ? "var(--accent)" : (LAYER_COLOR[layer] ?? "var(--line-2)")}`,
                                background: sel === c.id ? "var(--accent-soft)" : "var(--panel-2)", minWidth: 150 }}>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{c.name}</div>
                    <div className="muted" style={{ fontFamily: "var(--mono)", fontSize: 10 }}>{c.module}</div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {s && (
        <div className="panel">
          <h3 style={{ fontFamily: "var(--serif)", fontWeight: 500, marginTop: 0 }}>{s.name}</h3>
          <dl className="kv">
            <dt>Слой</dt><dd>{s.layer}</dd>
            <dt>Модуль</dt><dd className="num">{s.module}</dd>
            <dt>Назначение</dt><dd>{s.desc}</dd>
            <dt>Использует</dt>
            <dd>{s.deps.length ? s.deps.map((d) => (
              <span key={d} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(d)}>{byId[d]?.name ?? d}</span>
            )) : <span className="muted">—</span>}</dd>
            <dt>Используется в</dt>
            <dd>{a.components.filter((c: ArchComponent) => c.deps.includes(s.id)).map((c) => (
              <span key={c.id} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(c.id)}>{c.name}</span>
            ))}</dd>
          </dl>
        </div>
      )}
    </div>
  );
}
