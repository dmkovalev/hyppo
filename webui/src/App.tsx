// webui/src/App.tsx
import { useEffect, useState } from "react";
import { get } from "./api";
import { Overview } from "./routes/Overview";

type Project = { id: string; name: string };

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [pid, setPid] = useState<string>("");
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
        {pid && <Overview pid={pid} />}
      </main>
    </div>
  );
}
