import { useEffect, useRef, useState } from "react";
import type { RealData, ArchComponent } from "../types";

declare global { interface Window { cytoscape?: any; cytoscapeDagre?: any; } }

const LAYER_COLOR: Record<string, string> = {
  "Интерфейс": "#3b6fb0", "Оркестрация": "#7c1d2b", "Методы (гл. 2)": "#2f7d43",
  "Исполнение": "#b0324f", "Онтология": "#8a5a1f", "Хранение": "#5f5c55", "Инфраструктура": "#928e85",
};

export function Architecture({ real }: { real: RealData }) {
  const a = real.architecture;
  const byId = Object.fromEntries(a.components.map((c) => [c.id, c]));
  const ref = useRef<HTMLDivElement>(null);
  const cyRef = useRef<any>(null);
  const [sel, setSel] = useState<string>("");
  const s = sel ? byId[sel] : null;

  useEffect(() => {
    const cytoscape = window.cytoscape;
    if (!ref.current || !cytoscape) return;
    if (window.cytoscapeDagre) { try { cytoscape.use(window.cytoscapeDagre); } catch { /* ok */ } }

    const elements: any[] = [];
    a.components.forEach((c) => elements.push({ data: { id: c.id, label: c.name, layer: c.layer } }));
    a.components.forEach((c) => c.deps.forEach((dep) => {
      if (byId[dep]) elements.push({ data: { id: `${c.id}_${dep}`, source: c.id, target: dep } });
    }));
    (a.realizes ?? []).forEach((r) => {
      if (byId[r.src] && byId[r.dst])
        elements.push({ data: { id: `${r.src}_rz_${r.dst}`, source: r.src, target: r.dst, rz: 1 } });
    });

    const styles: any[] = [
      { selector: "node", style: {
        shape: "round-rectangle", width: 128, height: 44,
        "background-color": "#fbfaf7", "border-width": 2,
        label: "data(label)", "text-valign": "center", "text-halign": "center",
        "text-wrap": "wrap", "text-max-width": 118, "font-size": 10, "font-family": "IBM Plex Sans", color: "#1c1b19",
      }},
      { selector: "edge", style: {
        "curve-style": "bezier", width: 1.7, "line-color": "#b9742a",
        "target-arrow-shape": "triangle", "target-arrow-color": "#b9742a", "arrow-scale": 0.9,
      }},
      { selector: "edge[rz]", style: {
        "line-style": "dashed", "line-color": "#5f5c55", width: 1.4,
        "target-arrow-shape": "triangle-tee", "target-arrow-color": "#5f5c55", "arrow-scale": 0.9,
      }},
      { selector: ".faded", style: { opacity: 0.14 } },
      { selector: "node.hl", style: { "border-width": 4, "border-color": "#7c1d2b" } },
      { selector: "edge.hl", style: { "line-color": "#7c1d2b", "target-arrow-color": "#7c1d2b", width: 3 } },
    ];
    // цвет рамки по слою
    Object.entries(LAYER_COLOR).forEach(([ly, col]) =>
      styles.push({ selector: `node[layer="${ly}"]`, style: { "border-color": col } }));

    const cy = cytoscape({ container: ref.current, elements, style: styles });
    cyRef.current = cy;
    const opts = window.cytoscapeDagre
      ? { name: "dagre", rankDir: "LR", nodeDimensionsIncludeLabels: true, rankSep: 90, nodeSep: 26, fit: true, padding: 24 }
      : { name: "breadthfirst", directed: true, fit: true, padding: 24 };
    try { cy.layout(opts).run(); } catch { cy.layout({ name: "grid", fit: true }).run(); }

    cy.on("tap", "node", (e: any) => {
      const n = e.target; const neigh = n.closedNeighborhood();
      cy.elements().addClass("faded"); neigh.removeClass("faded").addClass("hl");
      setSel(n.id());
    });
    cy.on("tap", (e: any) => { if (e.target === cy) { cy.elements().removeClass("faded hl"); setSel(""); } });
    return () => cy.destroy();
  }, [a]);

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Программный комплекс · глава 3</div>
        <h1>Архитектура комплекса Hyppo</h1>
        <p className="lead">
          Схема компонентов и зависимостей: <b>сплошная A → B</b> — «A обращается к интерфейсу B»,
          <b>пунктир A ⇢ B</b> — «A реализует абстрактный интерфейс B» (напр. MetadataRepository ⇢ Storage).
          Цвет рамки — слой. Кликните компонент — подсветятся связи и откроются детали.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 10 }}>
          {a.layers.map((ly) => (
            <span key={ly} style={{ fontFamily: "var(--mono)", fontSize: 11, color: LAYER_COLOR[ly] }}>■ {ly}</span>
          ))}
        </div>
        {!window.cytoscape
          ? <div className="empty">Загрузка библиотеки визуализации…</div>
          : <div ref={ref} className="cy-frame" style={{ height: 560 }} />}
      </div>

      {s && (
        <div className="panel">
          <h3 style={{ fontFamily: "var(--serif)", fontWeight: 500, marginTop: 0 }}>{s.name}</h3>
          <dl className="kv">
            <dt>Слой</dt><dd><span className="chip" style={{ color: LAYER_COLOR[s.layer], borderColor: LAYER_COLOR[s.layer] }}>{s.layer}</span></dd>
            <dt>Модуль</dt><dd className="num">{s.module}</dd>
            <dt>Назначение</dt><dd>{s.desc}</dd>
            <dt>Использует →</dt>
            <dd>{s.deps.length ? s.deps.map((d) => (
              <span key={d} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(d)}>{byId[d]?.name ?? d}</span>
            )) : <span className="muted">—</span>}</dd>
            <dt>← Используется в</dt>
            <dd>{a.components.filter((c: ArchComponent) => c.deps.includes(s.id)).map((c) => (
              <span key={c.id} className="tag" style={{ cursor: "pointer" }} onClick={() => setSel(c.id)}>{c.name}</span>
            ))}</dd>
          </dl>
        </div>
      )}
    </div>
  );
}
