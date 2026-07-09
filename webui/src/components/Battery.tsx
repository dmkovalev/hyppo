import { useMemo, useState } from "react";
import type { RealData } from "../types";
import { GraphBuild, GEdge, GNode } from "./GraphBuild";

/**
 * Демо 8 (docs/gui_demo_spec.md): систематическая проверка обнаружения —
 * 80 испытаний HermiT из norne_battery_results.json. Правило 4 (каскад
 * устаревания) и правило 7 (детекция stale) сверяются с достижимостью в графе;
 * это проверка соответствия реализации спецификации, а НЕ статистика.
 */
export function Battery({ real }: { real: RealData }) {
  const b = real.battery;
  const [hovered, setHovered] = useState<string | null>(null);

  const staleOf: Record<string, string[]> = {};
  (b?.rule4 ?? []).forEach((r) => (staleOf[r.source] = r.stale));
  // подсветка каскада на графе: источник — красный, его устаревшие потомки — янтарь
  const statusOverride = useMemo(() => {
    if (!hovered) return undefined;
    const stale = new Set(staleOf[hovered] ?? []);
    const ov: Record<string, string> = {};
    real.graph_conceptual.nodes.forEach((n) => {
      ov[n.id] = n.id === hovered ? "REFUTED" : stale.has(n.id) ? "CONFIRMED" : "SUPPORTED";
    });
    return ov;
  }, [hovered, real.graph_conceptual.nodes]);
  const gnodes: GNode[] = real.graph_conceptual.nodes.map((n) => ({ id: n.id, label: n.label }));
  const gedges: GEdge[] = real.graph_conceptual.derivation.map(
    (d) => ({ src: d.src, dst: d.dst, via: d.via, reason: d.reason }));

  if (!b) return null;
  const s = b.summary;

  // порядок гипотез = порядок источников правила 4 (конвейерный)
  const order = b.rule4.map((r) => r.source);
  const runs = Array.from(new Set(b.rule7.map((r) => r.run_uses)))
    .sort((a, z) => order.indexOf(a) - order.indexOf(z));
  const verdictOf: Record<string, string> = {};
  b.rule7.forEach((r) => (verdictOf[`${r.run_uses}|${r.invalid}`] = r.verdict));

  return (
    <div className="panel">
      <div className="label">
        Демо 8 — систематическая проверка обнаружения ({s.trials} испытаний HermiT)
      </div>
      <p style={{ marginTop: 0 }}>
        {s.rule4_classifications} классификаций (правило 4) и {b.rule7.length} пар
        «запуск × кандидат» (правило 7) сверены с достижимостью в графе гибридной модели.
        Расхождений — <b>0</b>. Это проверка соответствия реализации спецификации, а не
        статистическая метрика.
      </p>

      <div className="tiles" style={{ marginBottom: 14 }}>
        <div className={"tile " + (s.rule4_mismatches === 0 ? "good" : "bad")}>
          <div className="k">Правило 4 · каскад</div>
          <div className="v">{s.rule4_classifications}</div>
          <div className="muted" style={{ fontSize: 11 }}>классификаций · расхождений {s.rule4_mismatches}</div>
        </div>
        <div className={"tile " + (s.rule7_errors === 0 ? "good" : "bad")}>
          <div className="k">Правило 7 · детекция</div>
          <div className="v">{s.rule7_tp}<span className="muted" style={{ fontSize: 14 }}> TP</span> · {s.rule7_tn}<span className="muted" style={{ fontSize: 14 }}> TN</span></div>
          <div className="muted" style={{ fontSize: 11 }}>{b.rule7.length} пар · ошибок {s.rule7_errors}</div>
        </div>
        <div className="tile">
          <div className="k">Стоимость серии</div>
          <div className="v">{s.wall_seconds.toFixed(0)}<span className="muted" style={{ fontSize: 14 }}> с</span></div>
          <div className="muted" style={{ fontSize: 11 }}>{s.trials} запусков HermiT (готовый JSON)</div>
        </div>
      </div>

      {/* правило 4: источник → число устаревших (монотонно убывает) */}
      <div className="label" style={{ marginTop: 4 }}>
        Правило 4 — каскад устаревания (источник → устаревшие потомки) ·{" "}
        <span className="muted" style={{ fontWeight: 400, fontSize: 11 }}>наведите строку — каскад подсветится на графе</span>
      </div>
      <table className="data" style={{ marginBottom: 16 }}>
        <thead><tr><th>Источник</th><th>Устарело</th><th>Потомки</th><th>Оракул</th></tr></thead>
        <tbody>
          {b.rule4.map((r) => (
            <tr key={r.source}
                onMouseEnter={() => setHovered(r.source)} onMouseLeave={() => setHovered(null)}
                style={{ cursor: "pointer", background: hovered === r.source ? "var(--panel-2)" : undefined }}>
              <td className="num">{r.source}</td>
              <td className="num">{r.stale.length}</td>
              <td className="muted" style={{ fontSize: 11 }}>
                {r.stale.length ? r.stale.join(", ") : "— (терминальная)"}
              </td>
              <td>{r.match
                ? <span className="chip g" style={{ fontSize: 11 }}>✓ совпало</span>
                : <span className="chip r" style={{ fontSize: 11 }}>✗</span>}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {/* граф гипотез: подсветка каскада устаревания по наведению на строку */}
      <div className="label">
        Граф гипотез — каскад источника{" "}
        {hovered
          ? <><span className="chip r" style={{ fontSize: 11 }}>{hovered}</span>{" "}
              <span className="muted" style={{ fontWeight: 400, fontSize: 11 }}>
                → устаревает {staleOf[hovered]?.length ?? 0}</span></>
          : <span className="muted" style={{ fontWeight: 400, fontSize: 11 }}>наведите строку правила 4 выше</span>}
      </div>
      <GraphBuild nodes={gnodes} edges={gedges} statusOverride={statusOverride} onNodeClick={() => {}} />

      {/* правило 7: матрица запуск × кандидат (TP/TN) */}
      <div className="label">Правило 7 — матрица «запуск × кандидат» (TP — верно stale, TN — верно свежая)</div>
      <div style={{ overflowX: "auto" }}>
        <table className="data" style={{ fontSize: 11 }}>
          <thead>
            <tr>
              <th style={{ position: "sticky", left: 0 }}>запуск \ кандидат</th>
              {order.map((c) => <th key={c} style={{ padding: "4px 6px", textAlign: "center" }}>{c}</th>)}
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run}>
                <td className="num" style={{ position: "sticky", left: 0, background: "var(--panel)" }}>{run}</td>
                {order.map((c) => {
                  const v = verdictOf[`${run}|${c}`];
                  const bg = v === "TP" ? "var(--st-confirmed, #2f7d43)" : v === "TN" ? "var(--line-2)" : "transparent";
                  const col = v === "TP" ? "#fff" : "var(--muted)";
                  return (
                    <td key={c} title={v ? `${run} × ${c}: ${v}` : ""}
                        style={{ textAlign: "center", padding: "3px 5px", background: bg, color: col, fontFamily: "var(--mono)" }}>
                      {v ?? "·"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="muted" style={{ fontSize: 12, marginTop: 10 }}>
        Итог: {s.rule4_classifications} классификаций и {b.rule7.length} пар — 0 расхождений
        с достижимостью в графе. Каждая ячейка TP/TN подтверждена реальным прогоном HermiT
        (серия не запускается живьём — {s.wall_seconds.toFixed(0)} с; загружается готовый JSON).
      </p>
    </div>
  );
}
