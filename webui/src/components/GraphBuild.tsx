import { useMemo, useRef, useState } from "react";

export type GNode = { id: string; label: string; status?: string; kind?: string };
export type GEdge = { src: string; dst: string; via?: string; reason?: string };

const KIND_ICON: Record<string, string> = { injector: "▼ нагн.", producer: "▲ доб.", fusion: "◆ OPR" };

const STATUS_STROKE: Record<string, string> = {
  SUPPORTED: "var(--st-supported)", CONFIRMED: "var(--st-confirmed)",
  REFUTED: "var(--st-refuted)", SUPERSEDED: "var(--st-superseded)",
  PROPOSED: "var(--st-proposed)",
};
const STATUS_FILL: Record<string, string> = {
  SUPPORTED: "var(--st-supported-bg)", CONFIRMED: "var(--st-confirmed-bg)",
  REFUTED: "var(--st-refuted-bg)", SUPERSEDED: "var(--st-superseded-bg)",
  PROPOSED: "var(--st-proposed-bg)",
};

const NW = 156, NH = 48, COLW = 230, ROWH = 74;

export function GraphBuild({ nodes, edges, onNodeClick, statusOverride }:
  { nodes: GNode[]; edges: GEdge[]; onNodeClick?: (id: string) => void; statusOverride?: Record<string, string> }) {
  const [revealed, setRevealed] = useState(edges.length); // built by default
  const [annot, setAnnot] = useState("");
  const [invalid, setInvalid] = useState<{ nodes: Set<string>; edges: Set<string> }>(
    { nodes: new Set(), edges: new Set() });
  const timers = useRef<number[]>([]);

  // ── layered layout by longest-path depth ──
  const { pos, ordered, viewW, viewH } = useMemo(() => {
    const inc: Record<string, string[]> = {};
    nodes.forEach((n) => (inc[n.id] = []));
    edges.forEach((e) => inc[e.dst]?.push(e.src));
    const dc: Record<string, number> = {};
    const depth = (id: string, seen = new Set<string>()): number => {
      if (dc[id] != null) return dc[id];
      if (seen.has(id)) return 0;
      seen.add(id);
      const ps = inc[id] ?? [];
      const d = ps.length ? Math.max(...ps.map((p) => depth(p, seen) + 1)) : 0;
      return (dc[id] = d);
    };
    const byDepth: Record<number, string[]> = {};
    nodes.forEach((n) => (byDepth[depth(n.id)] ??= []).push(n.id));
    const pos: Record<string, { x: number; y: number }> = {};
    let maxD = 0, maxRows = 0;
    Object.entries(byDepth).forEach(([d, ids]) => {
      const di = +d; maxD = Math.max(maxD, di); maxRows = Math.max(maxRows, ids.length);
      ids.forEach((id, i) => {
        pos[id] = { x: 40 + di * COLW, y: 40 + i * ROWH - ((ids.length - 1) * ROWH) / 2 };
      });
    });
    // center each column vertically around a common midline
    const mid = 40 + ((maxRows - 1) * ROWH) / 2 + NH / 2;
    Object.values(pos).forEach((p) => (p.y += mid - NH / 2));
    const ordered = [...edges].sort((a, b) => (dc[a.src] - dc[b.src]) || (dc[a.dst] - dc[b.dst]));
    return { pos, ordered, viewW: 40 + maxD * COLW + NW + 40, viewH: mid * 2 + 20 };
  }, [nodes, edges]);

  function clearTimers() { timers.current.forEach(clearTimeout); timers.current = []; }

  function build() {
    clearTimers();
    setInvalid({ nodes: new Set(), edges: new Set() });
    setRevealed(0); setAnnot("Алгоритм 1: причинное упорядочение уравнений…");
    ordered.forEach((e, i) => {
      const t = window.setTimeout(() => {
        setRevealed(i + 1);
        setAnnot(e.reason ?? `${e.src} → ${e.dst}`);
        if (i === ordered.length - 1)
          timers.current.push(window.setTimeout(() => setAnnot(
            `Граф построен: ${nodes.length} гипотез, ${ordered.length} рёбер derived_by.`), 700));
      }, 200 + i * 620);
      timers.current.push(t);
    });
  }

  function cascade(start: string) {
    clearTimers();
    setRevealed(edges.length);
    // BFS depths from start along edges
    const adj: Record<string, string[]> = {};
    edges.forEach((e) => (adj[e.src] ??= []).push(e.dst));
    const depth: Record<string, number> = { [start]: 0 };
    const q = [start];
    while (q.length) {
      const u = q.shift()!;
      for (const v of adj[u] ?? []) if (depth[v] == null) { depth[v] = depth[u] + 1; q.push(v); }
    }
    const invN = new Set<string>(), invE = new Set<string>();
    setAnnot(`Каскадная инвалидация от ${start}: пересчёту подлежат потомки по derived_by.`);
    Object.entries(depth).forEach(([id, d]) => {
      const t = window.setTimeout(() => {
        setInvalid((prev) => {
          const nn = new Set(prev.nodes); nn.add(id);
          const ne = new Set(prev.edges);
          edges.forEach((e, i) => { if (depth[e.src] != null && depth[e.dst] != null && depth[e.dst] === depth[e.src] + 1) ne.add(String(i)); });
          return { nodes: nn, edges: ne };
        });
      }, d * 500);
      timers.current.push(t);
      invN.add(id);
    });
    const cnt = Object.keys(depth).length;
    timers.current.push(window.setTimeout(() => setAnnot(
      `P_ne = ${cnt} из ${nodes.length} (${Math.round(cnt / nodes.length * 100)}%): изменение ${start} инвалидирует ${cnt - 1} потомков.`),
      (Math.max(...Object.values(depth)) + 1) * 500));
  }

  function reset() { clearTimers(); setInvalid({ nodes: new Set(), edges: new Set() }); setRevealed(edges.length); setAnnot(""); }

  function edgePath(e: GEdge) {
    const a = pos[e.src], b = pos[e.dst];
    const x1 = a.x + NW, y1 = a.y + NH / 2, x2 = b.x, y2 = b.y + NH / 2;
    const mx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  }

  return (
    <div>
      <div className="gb-toolbar">
        <button className="btn" onClick={build}>▶ Построить граф (алг. 1)</button>
        <button className="btn ghost" onClick={reset}>Сброс</button>
        <span className="muted" style={{ fontSize: 12 }}>клик по вершине — каскад (алг. 4)</span>
      </div>
      <div className="graph-frame">
        <svg className="gb-svg" viewBox={`0 0 ${viewW} ${viewH}`} preserveAspectRatio="xMidYMid meet">
          <defs>
            <marker id="arw" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
              <path d="M0,0 L7,3 L0,6 Z" fill="var(--accent-dim)" />
            </marker>
            <marker id="arwr" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
              <path d="M0,0 L7,3 L0,6 Z" fill="var(--st-refuted)" />
            </marker>
          </defs>
          {ordered.slice(0, revealed).map((e, i) => {
            const idx = String(edges.indexOf(e));
            const inv = invalid.edges.has(idx);
            const a = pos[e.src], b = pos[e.dst];
            const mx = (a.x + NW + b.x) / 2, my = (a.y + b.y) / 2 + NH / 2;
            return (
              <g key={i}>
                <path className={"gedge" + (inv ? " invalid" : "")} d={edgePath(e)}
                      pathLength={1} markerEnd={`url(#${inv ? "arwr" : "arw"})`}
                      style={{ strokeDasharray: 1, strokeDashoffset: 1, animationDelay: "0ms" }} />
                {e.via && (
                  <text x={mx} y={my - 3} textAnchor="middle" className="gsub"
                        style={{ fontSize: 9, fill: inv ? "var(--st-refuted)" : "var(--accent)" }}>
                    <tspan style={{ paintOrder: "stroke", stroke: "var(--paper)", strokeWidth: 3 }}>{e.via}</tspan>
                  </text>
                )}
              </g>
            );
          })}
          {nodes.map((n) => {
            const p = pos[n.id];
            const inv = invalid.nodes.has(n.id);
            const st = statusOverride?.[n.id] ?? (inv ? "REFUTED" : (n.status ?? "PROPOSED"));
            return (
              <g key={n.id} className={"gnode" + (inv ? " wave" : "")}
                 style={{ cursor: "pointer" }} onClick={() => onNodeClick ? onNodeClick(n.id) : cascade(n.id)}>
                <rect x={p.x} y={p.y} width={NW} height={NH} rx={9}
                      fill={STATUS_FILL[st]} stroke={STATUS_STROKE[st]} strokeWidth={1.6} />
                <text className="gsub" x={p.x + 10} y={p.y + 14} textAnchor="start">
                  {KIND_ICON[n.kind ?? ""] ?? ""}
                </text>
                <text className="glabel" x={p.x + NW / 2} y={p.y + 28} textAnchor="middle">{n.id}</text>
                <text className="gsub" x={p.x + NW / 2} y={p.y + 41} textAnchor="middle">
                  {n.label.length > 24 ? n.label.slice(0, 23) + "…" : n.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="gb-annot">{annot || " "}</div>
    </div>
  );
}
