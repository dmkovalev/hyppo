import { useMemo } from "react";
import type { OntoClass, OntoRel } from "../types";

const NW = 150, NH = 40, COLW = 210, ROWH = 62;

export function OntologyGraph({ classes, relations }: { classes: OntoClass[]; relations: OntoRel[] }) {
  const { pos, viewW, viewH } = useMemo(() => {
    const depthOf: Record<string, number> = {};
    const byName = Object.fromEntries(classes.map((c) => [c.name, c]));
    const depth = (n: string, seen = new Set<string>()): number => {
      if (depthOf[n] != null) return depthOf[n];
      if (seen.has(n)) return 0;
      seen.add(n);
      const p = byName[n]?.parent;
      return (depthOf[n] = p && byName[p] ? depth(p, seen) + 1 : 0);
    };
    const byDepth: Record<number, string[]> = {};
    classes.forEach((c) => (byDepth[depth(c.name)] ??= []).push(c.name));
    const pos: Record<string, { x: number; y: number }> = {};
    let maxD = 0, maxRows = 0;
    Object.entries(byDepth).forEach(([d, ns]) => {
      maxD = Math.max(maxD, +d); maxRows = Math.max(maxRows, ns.length);
      ns.forEach((n, i) => (pos[n] = { x: 30 + +d * COLW, y: 30 + i * ROWH }));
    });
    return { pos, viewW: 30 + maxD * COLW + NW + 40, viewH: 30 + maxRows * ROWH + 20 };
  }, [classes]);

  const has = (n: string) => pos[n] != null;

  function sub(a: string, b: string) {
    const p = pos[a], q = pos[b];
    const x1 = p.x + NW, y1 = p.y + NH / 2, x2 = q.x, y2 = q.y + NH / 2, mx = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mx} ${y1}, ${mx} ${y2}, ${x2} ${y2}`;
  }
  function rel(a: string, b: string) {
    const p = pos[a], q = pos[b];
    if (a === b) { // self loop
      const x = p.x + NW / 2, y = p.y;
      return `M ${x - 16} ${y} C ${x - 26} ${y - 34}, ${x + 26} ${y - 34}, ${x + 16} ${y}`;
    }
    const x1 = p.x + NW / 2, y1 = p.y, x2 = q.x + NW / 2, y2 = q.y;
    return `M ${x1} ${y1} C ${x1} ${y1 - 40}, ${x2} ${y2 - 40}, ${x2} ${y2}`;
  }

  return (
    <div className="graph-frame" style={{ height: viewH > 460 ? 520 : viewH + 20 }}>
      <svg className="gb-svg" viewBox={`0 0 ${viewW} ${viewH}`} preserveAspectRatio="xMidYMid meet"
           style={{ height: "100%" }}>
        <defs>
          <marker id="oarw" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--line-2)" />
          </marker>
          <marker id="oarwp" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent-dim)" />
          </marker>
        </defs>
        {/* subclass edges (solid) */}
        {classes.filter((c) => c.parent && has(c.parent)).map((c) => (
          <path key={"s" + c.name} d={sub(c.parent!, c.name)} fill="none"
                stroke="var(--line-2)" strokeWidth={1.4} markerEnd="url(#oarw)" />
        ))}
        {/* object-property edges (dashed, labeled) */}
        {relations.filter((r) => has(r.domain) && has(r.range)).map((r, i) => (
          <g key={"r" + i}>
            <path d={rel(r.domain, r.range)} fill="none" stroke="var(--accent-dim)"
                  strokeWidth={1.2} strokeDasharray="4 3" markerEnd="url(#oarwp)" opacity={0.85} />
          </g>
        ))}
        {classes.map((c) => (
          <g key={c.name}>
            <rect x={pos[c.name].x} y={pos[c.name].y} width={NW} height={NH} rx={7}
                  fill="var(--panel-2)" stroke="var(--line-2)" strokeWidth={1.3} />
            <text className="glabel" x={pos[c.name].x + NW / 2} y={pos[c.name].y + NH / 2 + 4}
                  textAnchor="middle" style={{ fontSize: 11 }}>{c.name}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}
