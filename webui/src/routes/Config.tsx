import type { RealData } from "../types";

// H(2,2,2) — 3-куб: вершины = бинарные тройки, ребро = отличие в одном бите.
const CUBE = {
  pos: {
    "000": [60, 210], "100": [60, 90], "010": [180, 210], "110": [180, 90],
    "001": [140, 160], "101": [140, 40], "011": [260, 160], "111": [260, 40],
  } as Record<string, number[]>,
  edges: [
    ["000", "100"], ["000", "010"], ["000", "001"],
    ["100", "110"], ["100", "101"], ["010", "110"], ["010", "011"],
    ["001", "101"], ["001", "011"], ["110", "111"], ["101", "111"], ["011", "111"],
  ],
};

function HammingCube() {
  return (
    <svg viewBox="0 0 320 250" width="100%" style={{ maxWidth: 360 }}>
      {CUBE.edges.map(([a, b], i) => {
        const p = CUBE.pos[a], q = CUBE.pos[b];
        const hl = a === "000" && b === "100"; // подсветить один переход
        return <line key={i} x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]}
          stroke={hl ? "var(--accent)" : "var(--line-2)"} strokeWidth={hl ? 2.5 : 1.3} />;
      })}
      {Object.entries(CUBE.pos).map(([id, p]) => (
        <g key={id}>
          <circle cx={p[0]} cy={p[1]} r={14} fill="var(--panel-2)" stroke="var(--accent-dim)" strokeWidth={1.4} />
          <text x={p[0]} y={p[1] + 3} textAnchor="middle" fontFamily="IBM Plex Mono" fontSize={9} fill="var(--text)">{id}</text>
        </g>
      ))}
      <text x={60} y={158} fontFamily="IBM Plex Mono" fontSize={9} fill="var(--accent)">q₁: 0→1</text>
    </svg>
  );
}

export function Config({ real }: { real: RealData }) {
  const cfg = real.ve.configuration;
  const size = real.ve.config_space_size;
  const dbar = cfg.reduce((s, a) => s + (a.levels.length - 1), 0);
  const sections = Array.from(new Set(cfg.map((a) => a.section)));

  return (
    <div>
      <div className="page-head">
        <div className="kicker">Элемент 𝒞 · пространство конфигураций ≅ граф Хэмминга</div>
        <h1>Конфигурация 𝒞 = ∏ Qᵢ</h1>
        <p className="lead">
          Пространство гиперпараметров HybridCRM. По <span className="formula">утверждению об
          изоморфизме</span> оно совпадает с обобщённым графом Хэмминга{" "}
          <span className="formula">H(q₁,…,qₙ) = K_q₁ □ … □ K_qₙ</span>: вершины — конфигурации,
          ребро — отличие ровно в одном параметре. Идентично для Brugge и Norne.
        </p>
      </div>

      <div className="row c2">
        <div className="panel">
          <div className="label">Геометрия пространства = граф Хэмминга</div>
          <div className="tiles" style={{ gridTemplateColumns: "1fr 1fr", marginBottom: 12 }}>
            <div className="tile"><div className="k">N = |𝒞| = ∏ qᵢ</div><div className="v" style={{ fontSize: 24 }}>{size.toLocaleString("ru")}</div><div className="muted" style={{ fontSize: 11 }}>вершин</div></div>
            <div className="tile"><div className="k">d̄ = Σ(qᵢ−1)</div><div className="v" style={{ fontSize: 24 }}>{dbar}</div><div className="muted" style={{ fontSize: 11 }}>степень (регулярный)</div></div>
          </div>
          <p className="muted" style={{ fontSize: 13, margin: 0 }}>
            Смена одного гиперпараметра = переход по <b>ребру</b>; число различающихся параметров =
            <b> расстояние Хэмминга</b> d_H. На этом стоят «вычислительные маршруты» (путь в графе)
            и каскадное переиспользование: близкие конфигурации делят кэш.
          </p>
        </div>
        <div className="panel">
          <div className="label">Иллюстрация: H(2, 2, 2) — 3-куб</div>
          <HammingCube />
          <p className="muted" style={{ fontSize: 12, margin: 0 }}>
            8 вершин, степень 3. Янтарное ребро — смена параметра q₁ (000→100).
          </p>
        </div>
      </div>

      {sections.map((sec) => (
        <div className="panel" key={sec}>
          <div className="label">Секция: {sec}</div>
          <table className="data">
            <thead><tr><th>Ось (гиперпараметр)</th><th>Уровни Qᵢ</th><th>qᵢ</th></tr></thead>
            <tbody>
              {cfg.filter((a) => a.section === sec).map((a) => (
                <tr key={a.name}>
                  <td className="num">{a.name}</td>
                  <td>{a.levels.map((l) => <span key={String(l)} className="tag">{String(l)}</span>)}</td>
                  <td className="num">{a.levels.length}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}
