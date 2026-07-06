// webui/src/routes/Comparison.tsx
import { useEffect, useState } from "react";
import { get } from "../api";
import { CompareParallel } from "../components/CompareParallel";
import { RunsTable, RunRow } from "../components/RunsTable";

export function Comparison({ pid }: { pid: string }) {
  const [rows, setRows] = useState<RunRow[]>([]);
  useEffect(() => {
    get<{ rows: RunRow[] }>(`/api/projects/${pid}/comparison`)
      .then((d) => setRows(d.rows)).catch(() => setRows([]));
  }, [pid]);
  return (
    <div>
      <RunsTable rows={rows} />
      <CompareParallel rows={rows} />
    </div>
  );
}
