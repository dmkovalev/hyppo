const STAGES = ["Гипотезы", "Граф", "План", "Выполнение", "Результаты", "Итерация"];

export function LifecycleRing({ stage, iteration }: { stage: number; iteration: number }) {
  const total = STAGES.length;
  const R = 58, C = 2 * Math.PI * R;
  const frac = Math.min(stage, total) / total;
  return (
    <div className="ring-wrap">
      <div className="ring">
        <svg width="132" height="132" viewBox="0 0 132 132">
          <circle cx="66" cy="66" r={R} fill="none" stroke="var(--line)" strokeWidth="6" />
          <circle cx="66" cy="66" r={R} fill="none" stroke="var(--accent)" strokeWidth="6"
                  strokeLinecap="round" strokeDasharray={C}
                  strokeDashoffset={C * (1 - frac)}
                  transform="rotate(-90 66 66)" />
        </svg>
        <div className="cap">
          <div className="it">итерация</div>
          <div className="st">{iteration}</div>
          <div className="of">этап {Math.min(stage, total)}/{total}</div>
        </div>
      </div>
      <div>
        <div className="stage-name">{STAGES[Math.min(stage, total) - 1] ?? STAGES[0]}</div>
        <div className="stage-next">
          {stage < total
            ? <>следующий этап · {STAGES[stage]}</>
            : "цикл замкнут · возможна новая итерация"}
        </div>
      </div>
    </div>
  );
}
