// webui/src/components/HypothesisGraph.tsx
import ReactFlow, { Background } from "reactflow";
import "reactflow/dist/style.css";

export function HypothesisGraph({ nodes, edges }: { nodes: string[]; edges: string[][] }) {
  const rfNodes = nodes.map((id, i) => ({
    id, data: { label: id }, position: { x: 60, y: i * 80 },
  }));
  const rfEdges = edges.map(([a, b], i) => ({ id: `e${i}`, source: a, target: b }));
  return (
    <div style={{ height: 360, border: "1px solid #ddd" }}>
      <ReactFlow nodes={rfNodes} edges={rfEdges} fitView><Background /></ReactFlow>
    </div>
  );
}
