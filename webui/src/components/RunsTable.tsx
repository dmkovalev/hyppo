// webui/src/components/RunsTable.tsx
export type RunRow = { hypothesis: string; status: string; r2: number | null };

export function RunsTable({ rows }: { rows: RunRow[] }) {
  return (
    <table>
      <thead><tr><th>Гипотеза</th><th>Статус</th><th>R²</th></tr></thead>
      <tbody>
        {rows.map((r) => (
          <tr key={r.hypothesis}>
            <td>{r.hypothesis}</td><td>{r.status}</td>
            <td>{r.r2 != null ? r.r2.toFixed(2) : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
