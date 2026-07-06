// webui/src/components/LifecycleRing.tsx
const STAGES = ["Гипотезы", "Граф", "План", "Выполнение", "Результаты", "Итерация"];

export function LifecycleRing({ stage, iteration }: { stage: number; iteration: number }) {
  const pct = Math.round((stage / STAGES.length) * 100);
  return (
    <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
      <div style={{
        width: 64, height: 64, borderRadius: "50%",
        background: `conic-gradient(#3b82f6 ${pct}%, #e5e7eb 0)`,
        display: "flex", alignItems: "center", justifyContent: "center",
      }}>
        <span style={{ fontSize: 11 }}>ит. {iteration}<br />{stage}/{STAGES.length}</span>
      </div>
      <div><b>{STAGES[stage - 1] ?? STAGES[0]}</b></div>
    </div>
  );
}
