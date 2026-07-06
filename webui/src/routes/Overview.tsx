// webui/src/routes/Overview.tsx
import { useEffect, useState } from "react";
import { get } from "../api";
import { LifecycleRing } from "../components/LifecycleRing";
import { IterationHistory, Iteration } from "../components/IterationHistory";

export function Overview({ pid }: { pid: string }) {
  const [items, setItems] = useState<Iteration[]>([]);
  useEffect(() => {
    get<Iteration[]>(`/api/projects/${pid}/runs`).then(setItems).catch(() => setItems([]));
  }, [pid]);
  const stage = items.length ? 4 : 1;
  return (
    <div>
      <LifecycleRing stage={stage} iteration={items.length} />
      <h3>История итераций</h3>
      <IterationHistory items={items} />
    </div>
  );
}
