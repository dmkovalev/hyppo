import { useRef, useState } from "react";
import type { RealData } from "../types";

const STAGES = [
  { key: 0, name: "Инициализация", short: "⟨O,H,M,R,W,𝒞⟩" },
  { key: 1, name: "Построение графа", short: "алгоритм 1" },
  { key: 2, name: "Планирование", short: "P_ne / P_e" },
  { key: 3, name: "Исполнение и анализ", short: "R², сравнение" },
  { key: 4, name: "Итерация", short: "↺ кэш" },
];

export function LifecycleCycle({ real }: { real: RealData }) {
  const [active, setActive] = useState<number>(-1);
  const [running, setRunning] = useState(false);
  const timers = useRef<number[]>([]);
  const c = real.graph_conceptual;
  const fr = real.fields["Brugge"];
  const plan = real.algorithm4?.change_H1;

  const R = 92, cx = 130, cy = 130, NR = 30;
  const angle = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / STAGES.length;
  const px = (i: number) => cx + R * Math.cos(angle(i));
  const py = (i: number) => cy + R * Math.sin(angle(i));

  function run() {
    timers.current.forEach(clearTimeout); timers.current = [];
    setRunning(true); setActive(-1);
    STAGES.forEach((_, i) => {
      timers.current.push(window.setTimeout(() => {
        setActive(i);
        if (i === STAGES.length - 1) timers.current.push(window.setTimeout(() => setRunning(false), 1400));
      }, 200 + i * 1400));
    });
  }

  const detail = [
    <>Определён кортеж <span className="formula">⟨O, H, M, R, W, 𝒞⟩</span>: {c.nodes.length} гипотез,
       {" "}{real.ve.models.length} моделей, {real.ve.configuration.length} осей конфигурации.</>,
    <>Алгоритм 1 (<span className="formula">HypothesisLattice</span>) построил граф: <b>{c.nodes.length} гипотез,
       {" "}{c.edges.length} рёбер</b> derived_by, DAG глубины {c.depth}.</>,
    <>Планировщик разбил на <span className="formula">P_ne</span> (пересчёт) и <span className="formula">P_e</span> (кэш).
       При изменении H1 пересчёту подлежат <b>{plan?.p_ne.length ?? 0} из {c.nodes.length}</b>.</>,
    <>Модели исполнены, гипотезы сравнены: на Brugge OPR R² = <b>{fr.r2.OPR.toFixed(2)}</b>,
       гибрид R² = <b>{fr.r2.Hybrid.toFixed(2)}</b>, физика-CRM {fr.r2.CRM >= 0.5 ? "подтверждена" : "опровергнута"}.</>,
    <>Эксперт меняет гипотезу или параметр → возврат к этапу 2; ранее вычисленные фрагменты
       <b> переиспользуются из кэша</b>, цикл повторяется с меньшими затратами.</>,
  ];

  return (
    <div className="row" style={{ gridTemplateColumns: "290px 1fr", alignItems: "center" }}>
      <svg viewBox="0 0 260 260" width="290">
        {/* стрелки по кругу */}
        {STAGES.map((_, i) => {
          const j = (i + 1) % STAGES.length;
          const a0 = angle(i) + 0.42, a1 = angle(j) - 0.42;
          const x0 = cx + R * Math.cos(a0), y0 = cy + R * Math.sin(a0);
          const x1 = cx + R * Math.cos(a1), y1 = cy + R * Math.sin(a1);
          const done = running && active >= 0 && (i < active || (i === active));
          return (
            <path key={i} d={`M ${x0} ${y0} A ${R} ${R} 0 0 1 ${x1} ${y1}`} fill="none"
                  stroke={done ? "var(--accent)" : "var(--line-2)"} strokeWidth={2} markerEnd="url(#lcarw)" />
          );
        })}
        <defs><marker id="lcarw" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent-dim)" /></marker></defs>
        {/* центр — кнопка */}
        <circle cx={cx} cy={cy} r={40} fill="var(--panel-2)" stroke="var(--line-2)" />
        <text x={cx} y={cy - 2} textAnchor="middle" style={{ fontFamily: "var(--serif)", fontSize: 13, fill: "var(--text)", cursor: "pointer" }}
              onClick={run}>{running ? "…идёт" : "▶ Запустить"}</text>
        <text x={cx} y={cy + 14} textAnchor="middle" className="gsub" style={{ fontSize: 8, cursor: "pointer" }} onClick={run}>эксперимент</text>
        {/* этапы */}
        {STAGES.map((s, i) => {
          const on = active === i;
          return (
            <g key={i} style={{ cursor: "pointer" }} onClick={() => setActive(i)}>
              <circle cx={px(i)} cy={py(i)} r={NR} fill={on ? "var(--accent-soft)" : "var(--panel)"}
                      stroke={on ? "var(--accent)" : "var(--line-2)"} strokeWidth={on ? 2.4 : 1.4} />
              <text x={px(i)} y={py(i) - 2} textAnchor="middle" style={{ fontFamily: "var(--serif)", fontSize: 15, fill: on ? "var(--accent)" : "var(--muted)" }}>{i + 1}</text>
              <text x={px(i)} y={py(i) + 11} textAnchor="middle" className="gsub" style={{ fontSize: 7 }}>{s.short}</text>
            </g>
          );
        })}
      </svg>

      <div>
        <div className="label">Жизненный цикл виртуального эксперимента (раздел 2.2)</div>
        {active < 0 ? (
          <p className="muted" style={{ marginTop: 0 }}>
            Итеративный цикл из 5 этапов. Нажмите <b>«Запустить»</b> в центре — цикл пройдёт по этапам,
            показывая реальный результат каждого. Или кликните любой этап.
          </p>
        ) : (
          <div>
            <div className="stage-name" style={{ color: "var(--accent)" }}>{active + 1}. {STAGES[active].name}</div>
            <p style={{ marginTop: 6 }}>{detail[active]}</p>
          </div>
        )}
        <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginTop: 6 }}>
          {STAGES.map((s, i) => (
            <span key={i} className={"chip" + (active === i ? " a" : "")} style={{ cursor: "pointer" }} onClick={() => setActive(i)}>
              {i + 1}. {s.name}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
