import { useEffect, useRef, useState } from "react";
import type { OntoClass, OntoRel } from "../types";

/* cytoscape + dagre loaded from CDN (see index.html) */
declare global { interface Window { cytoscape?: any; cytoscapeDagre?: any; } }

type Sel = { cls: string; supers: string[]; out: OntoRel[]; inc: OntoRel[] } | null;

export function OntologyCyto({ classes, relations }: { classes: OntoClass[]; relations: OntoRel[] }) {
  const ref = useRef<HTMLDivElement>(null);
  const cyRef = useRef<any>(null);
  const [sel, setSel] = useState<Sel>(null);
  const [layout, setLayout] = useState<"dagre" | "cose">("dagre");

  useEffect(() => {
    const cytoscape = window.cytoscape;
    if (!ref.current || !cytoscape) return;
    if (window.cytoscapeDagre && !cyRef.current) {
      try { cytoscape.use(window.cytoscapeDagre); } catch { /* already registered */ }
    }

    const names = new Set(classes.map((c) => c.name));
    const elements: any[] = [];
    classes.forEach((c) => elements.push({ data: { id: c.name, label: c.name, type: "class" } }));
    classes.forEach((c) => {
      if (c.parent && names.has(c.parent))
        elements.push({ data: { id: `sub_${c.name}`, source: c.name, target: c.parent, kind: "subClassOf" } });
    });
    relations.forEach((r, i) => {
      if (names.has(r.domain) && names.has(r.range))
        elements.push({ data: { id: `op_${i}`, source: r.domain, target: r.range, kind: "objectProperty", label: r.property } });
    });

    const cy = cytoscape({
      container: ref.current,
      elements,
      style: [
        { selector: 'node[type="class"]', style: {
          shape: "ellipse", width: 78, height: 78,
          "background-color": "#e8f0f9", "border-width": 2, "border-color": "#3b6fb0",
          label: "data(label)", "text-valign": "center", "text-halign": "center",
          "text-wrap": "wrap", "text-max-width": 70, "font-size": 10, "font-family": "IBM Plex Sans",
          color: "#1c1b19",
        }},
        { selector: 'edge[kind="objectProperty"]', style: {
          "curve-style": "bezier", width: 1.8, "line-color": "#7c1d2b",
          "target-arrow-shape": "triangle", "target-arrow-color": "#7c1d2b",
          label: "data(label)", "text-rotation": "autorotate",
          "text-background-color": "#ffffff", "text-background-opacity": 1, "text-background-padding": 2,
          "font-size": 9, "font-family": "IBM Plex Mono", color: "#5f5c55",
        }},
        { selector: 'edge[kind="subClassOf"]', style: {
          "curve-style": "bezier", width: 1.4, "line-color": "#b8b4aa", "line-style": "dashed",
          "target-arrow-shape": "triangle-tee", "target-arrow-fill": "hollow", "target-arrow-color": "#b8b4aa",
          label: "⊑", "font-size": 12, color: "#928e85",
        }},
        { selector: ".faded", style: { opacity: 0.12 } },
        { selector: "node.hl", style: { "border-width": 4, "border-color": "#7c1d2b", "background-color": "#f6e7ea" } },
        { selector: "edge.hl", style: { "line-color": "#7c1d2b", "target-arrow-color": "#7c1d2b", width: 3, opacity: 1 } },
      ],
    });
    cyRef.current = cy;
    runLayout(cy, layout);

    cy.on("tap", "node", (evt: any) => {
      const n = evt.target; const neigh = n.closedNeighborhood();
      cy.elements().addClass("faded"); neigh.removeClass("faded").addClass("hl");
      const rel = (e: any): OntoRel => ({ property: e.data("label"), domain: e.source().data("label"), range: e.target().data("label") });
      setSel({
        cls: n.data("label"),
        supers: n.outgoers('edge[kind="subClassOf"]').targets().map((t: any) => t.data("label")),
        out: n.outgoers('edge[kind="objectProperty"]').map(rel),
        inc: n.incomers('edge[kind="objectProperty"]').map(rel),
      });
    });
    cy.on("tap", (evt: any) => {
      if (evt.target === cy) { cy.elements().removeClass("faded hl"); setSel(null); }
    });

    return () => cy.destroy();
  }, [classes, relations]);

  function runLayout(cy: any, name: string) {
    const dagreOk = !!window.cytoscapeDagre;
    let opts: any;
    if (name === "dagre" && dagreOk) {
      opts = { name: "dagre", rankDir: "BT", nodeDimensionsIncludeLabels: true, rankSep: 80, nodeSep: 40, fit: true, padding: 30 };
    } else if (name === "cose") {
      opts = { name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 120, fit: true, padding: 30 };
    } else {
      opts = { name: "breadthfirst", directed: true, spacingFactor: 1.3, fit: true, padding: 30 };
    }
    try { cy.layout(opts).run(); }
    catch { cy.layout({ name: "grid", fit: true, padding: 30 }).run(); }
  }

  useEffect(() => { if (cyRef.current) runLayout(cyRef.current, layout); }, [layout]);

  if (!window.cytoscape) {
    return <div className="empty">Загрузка библиотеки визуализации (cytoscape)… обновите страницу, если не появилось.</div>;
  }

  return (
    <div>
      <div className="gb-toolbar">
        <span className="label" style={{ margin: 0 }}>Раскладка:</span>
        <div className="seg">
          <button className={layout === "dagre" ? "on" : ""} onClick={() => setLayout("dagre")}>иерархия</button>
          <button className={layout === "cose" ? "on" : ""} onClick={() => setLayout("cose")}>карта связей</button>
        </div>
        <span className="muted" style={{ fontSize: 12 }}>клик по классу — подсветить связи</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 16, alignItems: "stretch" }}>
        <div ref={ref} className="cy-frame" />
        <div className="panel" style={{ minHeight: 0, background: "var(--panel-2)" }}>
          {!sel && <div className="muted">Круги — классы, сплошные бордовые стрелки — объектные свойства (domain→range),
            пунктир ⊑ — подкласс. Кликните класс, чтобы увидеть его связи.</div>}
          {sel && (
            <div className="detail">
              <h3>{sel.cls}</h3>
              {sel.supers.length > 0 && (
                <div style={{ marginBottom: 10 }}>
                  <span className="label">⊑ подкласс</span>
                  {sel.supers.map((s) => <span key={s} className="tag">{s}</span>)}
                </div>
              )}
              <div className="label">Свойства из класса (domain → range)</div>
              {sel.out.length ? sel.out.map((r, i) => (
                <div key={i} style={{ marginBottom: 4 }}>
                  <span className="num" style={{ color: "var(--accent)" }}>{r.property}</span>
                  <span className="muted"> → {r.range}</span>
                </div>
              )) : <div className="muted" style={{ marginBottom: 8 }}>—</div>}
              <div className="label" style={{ marginTop: 12 }}>Свойства в класс</div>
              {sel.inc.length ? sel.inc.map((r, i) => (
                <div key={i} style={{ marginBottom: 4 }}>
                  <span className="muted">{r.domain} → </span>
                  <span className="num" style={{ color: "var(--accent)" }}>{r.property}</span>
                </div>
              )) : <div className="muted">—</div>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default OntologyCyto;
