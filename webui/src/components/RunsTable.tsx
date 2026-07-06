import type { CompareRow } from "../types";

function chip(status: string, epi?: string) {
  const s = epi ?? status;
  const cls = s === "SUPPORTED" ? "g" : s === "REFUTED" ? "r"
    : s === "SUCCESS" ? "g" : s === "SUPERSEDED" ? "b" : "";
  return <span className={"chip " + cls}>{s}</span>;
}

export function RunsTable({ rows }: { rows: CompareRow[] }) {
  if (!rows.length) return <div className="empty">Нет результатов — запустите итерацию.</div>;
  return (
    <table className="data">
      <thead><tr><th>Гипотеза</th><th>Статус</th><th>R²</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.hypothesis}>
            <td className="num">{r.hypothesis}</td>
            <td>{chip(r.status)}</td>
            <td className="num">{r.r2 != null ? r.r2.toFixed(2) : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
