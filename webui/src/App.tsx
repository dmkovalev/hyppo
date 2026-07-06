// webui/src/App.tsx
import { useEffect, useState } from "react";
import { get } from "./api";
import { Overview } from "./routes/Overview";
import { Hypotheses } from "./routes/Hypotheses";
import { Graph } from "./routes/Graph";
import { Runs } from "./routes/Runs";
import { Comparison } from "./routes/Comparison";

type Project = { id: string; name: string };

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [pid, setPid] = useState<string>("");
  const [tab, setTab] = useState("Обзор");
  const TABS = ["Обзор", "Гипотезы", "Граф", "Запуски", "Сравнение"];
  useEffect(() => {
    get<Project[]>("/api/projects").then((ps) => {
      setProjects(ps);
      if (ps[0]) setPid(ps[0].id);
    });
  }, []);
  return (
    <div style={{ display: "flex" }}>
      <nav style={{ width: 180, borderRight: "1px solid #ddd", padding: 8 }}>
        <b>Проекты</b>
        {projects.map((p) => (
          <div key={p.id} onClick={() => setPid(p.id)}
               style={{ cursor: "pointer", fontWeight: p.id === pid ? 700 : 400 }}>
            {p.name}
          </div>
        ))}
      </nav>
      <main style={{ flex: 1, padding: 16 }}>
        <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
          {TABS.map((t) => (
            <button key={t} onClick={() => setTab(t)}
                    style={{ fontWeight: t === tab ? 700 : 400 }}>{t}</button>
          ))}
        </div>
        {pid && tab === "Обзор" && <Overview pid={pid} />}
        {pid && tab === "Гипотезы" && <Hypotheses pid={pid} />}
        {pid && tab === "Граф" && <Graph pid={pid} />}
        {pid && tab === "Запуски" && <Runs pid={pid} />}
        {pid && tab === "Сравнение" && <Comparison pid={pid} />}
      </main>
    </div>
  );
}
