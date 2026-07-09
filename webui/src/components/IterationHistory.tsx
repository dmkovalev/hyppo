import type { Iteration } from "../types";

export function IterationHistory({ items }: { items: Iteration[] }) {
  if (!items.length) return <div className="empty">Итераций пока нет — запустите эксперимент.</div>;
  return (
    <div>
      {items.map((it, i) => (
        <div key={it.iteration} className={"iter" + (i === 0 ? " cur" : "")}>
          <div className="h">
            <span className="n">Итерация {it.iteration}</span>
            {i === 0 && <span className="chip a">текущая</span>}
          </div>
          <div className="meta">
            лучшая гипотеза <span className="num">{it.best.hypothesis ?? "—"}</span>
            {it.best.r2 != null && <> · R² <span className="num">{it.best.r2.toFixed(2)}</span></>}
            {" · "}переиспользовано из кэша <span className="num">{it.reused}</span>
            {it.note && <> · {it.note}</>}
          </div>
        </div>
      ))}
    </div>
  );
}
