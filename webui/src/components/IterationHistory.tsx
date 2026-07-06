// webui/src/components/IterationHistory.tsx
export type Iteration = {
  iteration: number;
  reused: number;
  best: { hypothesis: string | null; r2: number | null };
};

export function IterationHistory({ items }: { items: Iteration[] }) {
  return (
    <div>
      {items.map((it) => (
        <div key={it.iteration}
             style={{ padding: 6, borderLeft: "3px solid #22c55e", marginBottom: 6 }}>
          <b>Итерация {it.iteration}</b> — лучшая: {it.best.hypothesis ?? "—"}
          {it.best.r2 != null ? `, R² ${it.best.r2.toFixed(2)}` : ""} · кэш: {it.reused}
        </div>
      ))}
    </div>
  );
}
