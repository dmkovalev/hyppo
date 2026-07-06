// webui/src/routes/Graph.tsx
import { useEffect, useState } from "react";
import { get } from "../api";
import { HypothesisGraph } from "../components/HypothesisGraph";

export function Graph({ pid }: { pid: string }) {
  const [g, setG] = useState<{ nodes: string[]; edges: string[][] }>({ nodes: [], edges: [] });
  useEffect(() => { get(`/api/projects/${pid}/graph`).then(setG as any); }, [pid]);
  return <HypothesisGraph nodes={g.nodes} edges={g.edges} />;
}
