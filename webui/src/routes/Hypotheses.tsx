// webui/src/routes/Hypotheses.tsx
import { useEffect, useState } from "react";
import { get } from "../api";

type VEView = { hypotheses: { id: string }[]; config_space_size: number };

export function Hypotheses({ pid }: { pid: string }) {
  const [ve, setVe] = useState<VEView | null>(null);
  useEffect(() => { get<VEView>(`/api/projects/${pid}/hypotheses`).then(setVe).catch(() => setVe(null)); }, [pid]);
  if (!ve) return <div>ВЭ не определён</div>;
  return (
    <div>
      <p>Пространство конфигураций: <b>{ve.config_space_size}</b></p>
      <ul>{ve.hypotheses.map((h) => <li key={h.id}>{h.id}</li>)}</ul>
    </div>
  );
}
