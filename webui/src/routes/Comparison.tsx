import { useEffect, useState } from "react";
import { get } from "../api";
import type { CompareRow } from "../types";
import { CompareParallel } from "../components/CompareParallel";
import { RunsTable } from "../components/RunsTable";

export function Comparison({ pid }: { pid: string }) {
  const [rows, setRows] = useState<CompareRow[]>([]);
  useEffect(() => {
    get<{ rows: CompareRow[] }>(`/api/projects/${pid}/comparison`)
      .then((d) => setRows(d.rows)).catch(() => setRows([]));
  }, [pid]);

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Ранжирование · раздел 2.6</div>
        <h1>Сравнение гипотез</h1>
        <p className="lead">
          Конкурирующие гипотезы ранжируются по метрикам качества и
          информационным критериям. Параллельные координаты — связь параметров с R².
        </p>
      </div>

      <div className="panel">
        <div className="label">Параллельные координаты</div>
        <CompareParallel rows={rows} />
      </div>

      <div className="panel">
        <div className="label">Метрики по гипотезам</div>
        <RunsTable rows={rows} />
      </div>
    </div>
  );
}
