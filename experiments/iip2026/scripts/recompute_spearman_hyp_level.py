"""Пересчёт коэффициента Спирмена rho(|H|, ρ@r=0.7) на hypothesis-level:
146 subworkflow + 20 hand-curated = 166 точек (было: 146+5=151)."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from scipy.stats import spearmanr

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent   # experiments/iip2026/
CACHE = ROOT / "cache"
OUT   = ROOT / "out"
DATA  = ROOT / "data"
CACHE.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(parents=True, exist_ok=True)

R_GRID = ["0.3", "0.5", "0.7", "0.9"]


def main():
    # === subworkflow-level: 146 точек ===
    sub = json.loads(
        (DATA / "cascade_hypothesis_results.json").read_text())
    sub_pts = []
    for wf in sub["per_workflow"]:
        n = wf["n_hypotheses"]
        rho_07 = wf["cascade"]["0.7"]["median"]
        sub_pts.append((n, rho_07, wf.get("name", "?")))
    print(f"Subworkflow: {len(sub_pts)} точек")

    # === hand-curated: 20 точек ===
    cur = json.loads(
        (DATA / "cascade_curated_results.json").read_text())
    cur_pts = []
    for key, info in cur["results"].items():
        n = info["n_hypotheses"]
        rho_07 = info["cascade"]["0.7"]["median_rho"]
        cur_pts.append((n, rho_07, key))
    print(f"Hand-curated: {len(cur_pts)} точек")

    # === Объединённый hypothesis-level: 166 точек ===
    all_pts = sub_pts + cur_pts
    print(f"Hypothesis-level (combined): {len(all_pts)} точек")

    ns = [p[0] for p in all_pts]
    rhos = [p[1] for p in all_pts]
    rs, p = spearmanr(ns, rhos)
    print(f"\nSpearman r_s={rs:.3f}, p={p:.4f}  (N={len(all_pts)})")

    # === Для сравнения: только subworkflow ===
    ns_sub = [p[0] for p in sub_pts]
    rhos_sub = [p[1] for p in sub_pts]
    rs_sub, p_sub = spearmanr(ns_sub, rhos_sub)
    print(f"  Subworkflow only:  r_s={rs_sub:.3f}, p={p_sub:.4f}  "
          f"(N={len(sub_pts)})")

    # === Для сравнения: только hand-curated 20 ===
    ns_cur = [p[0] for p in cur_pts]
    rhos_cur = [p[1] for p in cur_pts]
    rs_cur, p_cur = spearmanr(ns_cur, rhos_cur)
    print(f"  Hand-curated only: r_s={rs_cur:.3f}, p={p_cur:.4f}  "
          f"(N={len(cur_pts)})")

    # === Task-level для контраста ===
    task = json.loads(
        (DATA / "wfcommons_validation_results.json").read_text())
    if "per_workflow" in task:
        ns_task = [w["n_hypotheses"] for w in task["per_workflow"]]
        rhos_task = [w["rho_real"] for w in task["per_workflow"]]
        rs_t, p_t = spearmanr(ns_task, rhos_task)
        print(f"\nTask-level (WfCommons): r_s={rs_t:.3f}, p={p_t:.4f}  "
              f"(N={len(ns_task)})")

    # === Сохраняем ===
    out = DATA / "spearman_hyp_level_n20.json"
    out.write_text(json.dumps({
        "hypothesis_level_combined": {
            "n_points": len(all_pts),
            "r_s": float(rs), "p_value": float(p),
        },
        "subworkflow_only": {
            "n_points": len(sub_pts),
            "r_s": float(rs_sub), "p_value": float(p_sub),
        },
        "hand_curated_only": {
            "n_points": len(cur_pts),
            "r_s": float(rs_cur), "p_value": float(p_cur),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
