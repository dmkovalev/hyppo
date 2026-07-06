import { useEffect, useState } from "react";
import { post, get } from "../api";
import type { CompareRow, Iteration } from "../types";
import { RunsTable } from "../components/RunsTable";

export function Runs({ pid }: { pid: string }) {
  const [rows, setRows] = useState<CompareRow[]>([]);
  const [hist, setHist] = useState<Iteration[]>([]);
  const [busy, setBusy] = useState(false);

  async function load() {
    const [cmp, h] = await Promise.all([
      get<{ rows: CompareRow[] }>(`/api/projects/${pid}/comparison`).catch(() => ({ rows: [] })),
      get<Iteration[]>(`/api/projects/${pid}/runs`).catch(() => []),
    ]);
    setRows(cmp.rows); setHist(h);
  }
  useEffect(() => { load(); }, [pid]);

  async function runNow() {
    setBusy(true);
    try { await post(`/api/projects/${pid}/runs`); await load(); }
    finally { setBusy(false); }
  }

  const last = hist[0];
  return (
    <div>
      <div className="page-head">
        <div className="kicker">Исполнение · этап 4</div>
        <h1>Запуски</h1>
        <p className="lead">
          Модели вызываются в топологическом порядке; кэшированные гипотезы
          переиспользуются, бесперспективные маршруты отсекаются.
        </p>
      </div>

      <div className="panel">
        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <button className="btn" onClick={runNow} disabled={busy}>
            {busy ? "Выполняется…" : "Запустить итерацию"}
          </button>
          {last && (
            <span className="muted">
              последняя · итерация <span className="num">{last.iteration}</span>
              {" · "}переиспользовано <span className="num">{last.reused}</span>
            </span>
          )}
        </div>
      </div>

      <div className="panel">
        <div className="label">Результаты последней итерации</div>
        <RunsTable rows={rows} />
      </div>
    </div>
  );
}
