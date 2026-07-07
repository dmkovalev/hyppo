import { useEffect, useState } from "react";
import { get } from "./api";
import type { RealData } from "./types";
import { Experiment } from "./routes/Experiment";
import { GraphView } from "./routes/GraphView";
import { Ontology } from "./routes/Ontology";
import { Workflow } from "./routes/Workflow";
import { Algorithms } from "./routes/Algorithms";
import { Fields } from "./routes/Fields";

const NAV = [
  { key: "ve", glyph: "⟨⟩", label: "Обзор ВЭ" },
  { key: "graph", glyph: "⋔", label: "Граф гипотез" },
  { key: "onto", glyph: "◈", label: "Онтология" },
  { key: "wf", glyph: "▤", label: "Поток работ" },
  { key: "algo", glyph: "∑", label: "Алгоритмы и теоремы" },
  { key: "fields", glyph: "◉", label: "Данные: Brugge / Norne" },
];

export default function App() {
  const [real, setReal] = useState<RealData | null>(null);
  const [field, setField] = useState<string>("Brugge");
  const [tab, setTab] = useState<string>("graph");

  useEffect(() => {
    get<RealData>("/api/real").then(setReal).catch(() => setReal(null));
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="mark"><b>H</b>yppo</div>
          <div className="sub">virtual experiments</div>
        </div>

        <div className="side-label">Проект</div>
        <div className="proj active"><span className="dot" /><span>HybridCRM · заводнение</span></div>

        <div className="side-label">Месторождение</div>
        <div style={{ padding: "4px 22px 8px" }}>
          <div className="seg">
            {["Brugge", "Norne"].map((f) => (
              <button key={f} className={field === f ? "on" : ""} onClick={() => setField(f)}>{f}</button>
            ))}
          </div>
        </div>

        <div className="side-label">Разделы</div>
        <div className="nav">
          {NAV.map((n) => (
            <div key={n.key} className={"nav-item" + (n.key === tab ? " active" : "")}
                 onClick={() => setTab(n.key)}>
              <span className="g">{n.glyph}</span><span>{n.label}</span>
            </div>
          ))}
        </div>

        <div className="foot">
          {real ? <>реальные данные · pywaterflood</> : "загрузка…"}<br />ФИЦ ИУ РАН · 2.3.5
        </div>
      </aside>

      <main className="main">
        {!real && <div className="empty">Загрузка реальных данных…</div>}
        {real && tab === "ve" && <Experiment real={real} field={field} />}
        {real && tab === "graph" && <GraphView real={real} field={field} setField={setField} />}
        {real && tab === "onto" && <Ontology real={real} />}
        {real && tab === "wf" && <Workflow real={real} field={field} setField={setField} />}
        {real && tab === "algo" && <Algorithms real={real} field={field} />}
        {real && tab === "fields" && <Fields real={real} field={field} setField={setField} />}
      </main>
    </div>
  );
}
