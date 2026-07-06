import { useEffect, useState } from "react";
import { get } from "../api";
import type { Iteration } from "../types";
import { LifecycleRing } from "../components/LifecycleRing";
import { IterationHistory } from "../components/IterationHistory";

export function Overview({ pid }: { pid: string }) {
  const [items, setItems] = useState<Iteration[]>([]);
  useEffect(() => {
    get<Iteration[]>(`/api/projects/${pid}/runs`).then(setItems).catch(() => setItems([]));
  }, [pid]);

  const stage = items.length ? 5 : 1;
  return (
    <div>
      <div className="page-head">
        <div className="kicker">Жизненный цикл · раздел 2.2</div>
        <h1>Обзор эксперимента</h1>
        <p className="lead">
          Итеративный цикл: гипотезы → граф → план → выполнение → результаты →
          новая итерация. Каскадное переиспользование сокращает пересчёт на каждом витке.
        </p>
      </div>

      <div className="panel">
        <LifecycleRing stage={stage} iteration={items.length} />
      </div>

      <div className="panel">
        <div className="label">История итераций</div>
        <IterationHistory items={items} />
      </div>
    </div>
  );
}
