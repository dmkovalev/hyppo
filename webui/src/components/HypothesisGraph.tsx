import ReactFlow, { Background, Controls } from "reactflow";
import "reactflow/dist/style.css";

export type GNode = { id: string; label: string; status?: string };

const STATUS_COLOR: Record<string, string> = {
  SUPPORTED: "#7bab77",
  REFUTED: "#c25a33",
  SUPERSEDED: "#6f97c0",
  PROPOSED: "#9c9077",
};

export function HypothesisGraph({ nodes, edges }: { nodes: GNode[]; edges: string[][] }) {
  // layered layout: depth = longest path from a source
  const inc: Record<string, string[]> = {};
  nodes.forEach((n) => (inc[n.id] = []));
  edges.forEach(([a, b]) => { if (inc[b]) inc[b].push(a); });
  const depthCache: Record<string, number> = {};
  const depth = (id: string, seen: Set<string> = new Set()): number => {
    if (depthCache[id] != null) return depthCache[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    const ps = inc[id] ?? [];
    const d = ps.length ? Math.max(...ps.map((p) => depth(p, seen) + 1)) : 0;
    depthCache[id] = d;
    return d;
  };
  const byDepth: Record<number, string[]> = {};
  nodes.forEach((n) => {
    const d = depth(n.id);
    (byDepth[d] ??= []).push(n.id);
  });

  const rfNodes = nodes.map((n) => {
    const d = depth(n.id);
    const col = byDepth[d];
    const idx = col.indexOf(n.id);
    const color = STATUS_COLOR[n.status ?? ""] ?? "#9c9077";
    return {
      id: n.id,
      data: { label: n.label },
      position: { x: 70 + d * 260, y: 60 + idx * 130 - (col.length - 1) * 65 + 140 },
      style: {
        background: "#211c14",
        color: "#ece4d2",
        border: `1px solid ${color}`,
        borderLeft: `4px solid ${color}`,
        borderRadius: 8,
        padding: "10px 14px",
        fontFamily: "IBM Plex Sans, sans-serif",
        fontSize: 12,
        fontWeight: 500,
        width: 180,
      },
    };
  });
  const rfEdges = edges.map(([a, b], i) => ({
    id: `e${i}`, source: a, target: b, animated: true,
    style: { stroke: "#b9742a", strokeWidth: 1.5 },
  }));

  return (
    <div className="graph-frame">
      <ReactFlow nodes={rfNodes} edges={rfEdges} fitView
                 proOptions={{ hideAttribution: true }}>
        <Background color="#322b20" gap={22} />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
