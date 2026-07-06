// webui/src/routes/Runs.tsx
import { useState } from "react";
import { post, get } from "../api";
import { RunsTable, RunRow } from "../components/RunsTable";

export function Runs({ pid }: { pid: string }) {
  const [rows, setRows] = useState<RunRow[]>([]);
  async function runNow() {
    await post(`/api/projects/${pid}/runs`);
    const cmp = await get<{ rows: RunRow[] }>(`/api/projects/${pid}/comparison`);
    setRows(cmp.rows);
  }
  return (
    <div>
      <button onClick={runNow}>Запустить итерацию</button>
      <RunsTable rows={rows} />
    </div>
  );
}
