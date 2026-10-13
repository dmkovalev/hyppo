"""Рисунки статьи о минимальном безопасном перевычислении (E1, E2, E3, E5, E6).

Каждая функция fig_*() читает свой JSON из data/ и пишет векторный PDF в figures/.
Запуск: python -m examples.research.safe_recompute.figures
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from examples.research.safe_recompute.common import DATA  # noqa: E402

FIG = Path(__file__).resolve().parent / "figures"
CM = 1 / 2.54
W1 = 8 * CM   # одна колонка
W2 = 16 * CM  # две колонки


def _setup():
    names = {f.name for f in font_manager.fontManager.ttflist}
    serif = "Times New Roman" if "Times New Roman" in names else "DejaVu Serif"
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [serif, "DejaVu Serif"],
        "mathtext.fontset": "stix" if serif == "Times New Roman" else "dejavuserif",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "pdf.fonttype": 42, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "hatch.linewidth": 0.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def _num(x, nd=2):
    """Число с десятичной запятой."""
    return f"{x:.{nd}f}".replace(".", ",")


def _mnum(x):
    """Число для mathtext с десятичной запятой: 0.01 -> 0{,}01."""
    return f"{x:g}".replace(".", "{,}")


_comma = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))
_pct = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def _load(data_dir, name):
    p = Path(data_dir) / name
    if not p.exists():
        print(f"[figures] {p} missing, figure skipped")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


TEXTWIDTH_CM = 17.0  # статья: a4, поля 2 см
MIN_PT = 7.0


def min_effective_pt(fig, width_cm: float) -> tuple[float, float]:
    """(наименьший кегль текста после масштабирования рисунка к ширине width_cm, масштаб)."""
    from matplotlib.text import Text
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    scale = width_cm * CM / (bbox.width + 2 * plt.rcParams["savefig.pad_inches"])
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]
    return min(sizes) * scale, scale


def _save(fig, out_dir, name, width_cm=0.48 * TEXTWIDTH_CM):
    """width_cm -- ширина рисунка в статье (minipage 0.48 textwidth или textwidth)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    pt, scale = min_effective_pt(fig, width_cm)
    fig.savefig(path)
    plt.close(fig)
    print(f"[figures] {path}: min {pt:.2f} pt at {width_cm:.1f} cm (x{scale:.2f})")
    if pt < MIN_PT - 1e-6:
        print(f"[figures] WARNING: text below {MIN_PT} pt in {name}")
    return path


# --------------------------------------------------------------------- E1
def fig_e1(data_dir=DATA, out_dir=FIG):
    """Экономия безопасного режима относительно каскада от ёмкости репозитория (LRU / случайное).
    Между ёмкостями 0,75 и 1 режим меняется (при ёмкости >= |V| вытеснения нет), поэтому этот
    участок показан штрихом."""
    d = _load(data_dir, "e1_eviction.json")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(W1, 5.5 * CM))
    styles = {"lru": ("LRU", "C0", "o"), "random": ("случайное", "C3", "s")}
    for policy, (label, color, marker) in styles.items():
        pts = []
        for tag, e in d.items():
            pol, _, mult = tag.partition(":")
            if pol != policy or not isinstance(e, dict):
                continue
            saved = [1 - g["safe_runs_median"] / g["cascade_runs_median"]
                     for g in e.get("graphs", {}).values() if g.get("cascade_runs_median")]
            if not saved:
                continue
            q1, med, q3 = np.percentile(saved, [25, 50, 75])
            pts.append((float(mult), 100 * med, 100 * q1, 100 * q3))
        if not pts:
            continue
        pts.sort()
        x, m, lo, hi = map(np.array, zip(*pts))
        below, above = x < 1, x >= 1
        for sel in (below, above):
            if sel.any():
                ax.fill_between(x[sel], lo[sel], hi[sel], color=color, alpha=0.18, lw=0)
                ax.plot(x[sel], m[sel], color=color, lw=1.0)
        if below.any() and above.any():
            i, j = np.flatnonzero(below)[-1], np.flatnonzero(above)[0]
            ax.plot(x[[i, j]], m[[i, j]], color=color, lw=0.8, ls="--")
        ax.plot(x, m, color=color, marker=marker, ms=3.5, ls="none", label=f"вытеснение: {label}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.25, 0.5, 0.75, 1, 2])
    ax.xaxis.set_major_formatter(_comma)
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.set_xlabel(r"ёмкость репозитория, доля $|V|$")
    ax.set_ylabel("экономия запусков, %")
    ax.set_ylim(0, 66)  # место для легенды над точками
    ax.legend(title="медиана и МКР по графам", title_fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, lw=0.4)
    return _save(fig, out_dir, "e1_eviction.pdf")


# --------------------------------------------------------------------- E2
def fig_e2(data_dir=DATA, out_dir=FIG):
    """Выигрыш плана минимальной стоимости: распределение по случаям (ящики 10--90 %)."""
    d = _load(data_dir, "e2_mincost.json")
    if d is None:
        return None
    rows = d["rows"]
    rhos = sorted({r["rho"] for r in rows})
    bases = [("N_read", r"$N$", "C0"), ("cascade_read", "каскад", "C3"),
             ("best_simple", "лучший простой", "C2")]
    fig, axes = plt.subplots(2, 1, figsize=(W1, 9 * CM), sharex=True, sharey=True)
    width = 0.24
    for ax, (tgt, title) in zip(axes, [("all", "(а) цели: все гипотезы"), ("sinks", "(б) цели: стоки")]):
        for i, rho in enumerate(rhos):
            sel = [r for r in rows if r["rho"] == rho and r["targets"] == tgt]
            for j, (b, _, color) in enumerate(bases):
                g = [100 * (1 - r["cost_mincost"] / r[f"cost_{b}"]) for r in sel if r[f"cost_{b}"] > 0]
                if not g:
                    continue
                pos = i + (j - 1) * width
                ax.boxplot([g], positions=[pos], widths=width * 0.8, whis=(10, 90),
                           showfliers=False, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=0.35, lw=0.6, edgecolor=color),
                           medianprops=dict(color="k", lw=1.0),
                           whiskerprops=dict(color=color, lw=0.7), capprops=dict(color=color, lw=0.7))
                ax.plot(pos, np.mean(g), marker="D", ms=2.5, color="k", zorder=5)
                share = sum(x > 1e-6 for x in g) / len(g)
                ax.text(pos, -9, f"{round(100 * share)}", ha="center", va="center", fontsize=7.5,
                        color=color)
        ax.set_title(title, loc="left", pad=2)
        ax.set_ylabel("выигрыш, %")
        ax.set_ylim(-14, 100)
        ax.axhline(0, color="0.5", lw=0.4)
        ax.grid(axis="y", alpha=0.3, lw=0.4)
    axes[-1].set_xticks(range(len(rhos)))
    axes[-1].set_xticklabels([rf"$\rho={_mnum(r)}$" for r in rhos])
    handles = [Patch(facecolor=c, alpha=0.35, edgecolor=c, label=f"отн. {lbl}") for _, lbl, c in bases]
    handles.append(plt.Line2D([], [], marker="D", ms=2.5, color="k", lw=0, label="среднее"))
    axes[0].legend(handles=handles, loc="upper left", ncol=2, fontsize=7.5, handlelength=1.2,
                   columnspacing=0.8)
    fig.text(0.02, 0.005, "Ящик: квартили, усы: 10 и 90 %;\nцифры под ящиками: доля случаев\n"
             "с выигрышем > 0, %.", fontsize=7.5, ha="left")
    fig.tight_layout(rect=(0, 0.09, 1, 1), h_pad=0.4)
    return _save(fig, out_dir, "e2_mincost.pdf")


# --------------------------------------------------------------------- E3
def fig_e3(data_dir=DATA, out_dir=FIG):
    """Причинное уточнение: доля строгих случаев и среднее |N_caus|/|N| от доли многовыходных гипотез."""
    d = _load(data_dir, "e3_causal.json")
    if d is None:
        return None
    syn = sorted(d["synthetic"].values(), key=lambda e: e["multi_output_share"])
    x = np.array([100 * e["multi_output_share"] for e in syn])
    strict = np.array([100 * e["frac_strict"] for e in syn])
    inst = np.array([100 * e["frac_instances_strict"] for e in syn])
    ratio = np.array([e["mean_ratio"] for e in syn])
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(W1, 8 * CM), sharex=True,
                                  gridspec_kw=dict(height_ratios=[1.3, 1], hspace=0.3))
    ax.plot(x, strict, "o-", color="C0", ms=3.5, label=r"случаи с $N_{\mathrm{caus}}\subsetneq N$")
    ax.plot(x, inst, "s--", color="C0", ms=3, mfc="white",
            label=r"системы, где такой случай есть")
    ax.set_ylabel("доля, %")
    ax.set_ylim(0, max(30, inst.max() * 1.15))
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("(а)", loc="left", pad=2)
    ax2.plot(x, ratio, "^-", color="C3", ms=3.5)
    ax2.set_ylabel(r"$|N_{\mathrm{caus}}|/|N|$")
    ax2.set_ylim(0.915, 1.004)  # место для врезки CRM под кривой
    ax2.set_yticks([0.92, 0.96, 1.0])
    ax2.yaxis.set_major_formatter(_comma)
    ax2.set_xlabel("доля многовыходных гипотез, %")
    ax2.set_title("(б)", loc="left", pad=2)
    for a in (ax, ax2):
        a.grid(alpha=0.3, lw=0.4)
    fine, coarse = d.get("crm_fine"), d.get("crm_coarse")
    if fine and coarse:
        # На графе компонент N_caus совпадает с N (n_caus_components_mean == n_mean);
        # укрупнение увеличивает сам план N относительно мелкого графа.
        rc = coarse.get("rows", [])
        n = len(rc)
        n_gt = sum(r["N_coarse"] > r["N_fine"] for r in rc)
        n_eq = sum(r["N_caus_coarse_components"] == r["N_coarse"] for r in rc)
        all_eq = "во всех" if n_eq == n else f"в {n_eq} из"
        nc = r"N_{\mathrm{caus}}"
        txt = (f"CRM, мелкие гипотезы: ${nc}=N$ во всех {fine['cases']} случаях;\n"
               f"CRM, укрупнённые: $N$ на графе компонент больше,\n"
               f"чем на мелком, в {_num(100 * n_gt / n, 1)} % ({n_gt}/{n}); ${nc}=N$\n"
               f"{all_eq} {n} случаях")
        ax2.text(0.03, 0.06, txt, transform=ax2.transAxes, ha="left", va="bottom", fontsize=7.5,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.5))
    return _save(fig, out_dir, "e3_causal.pdf")


# --------------------------------------------------------------------- E5
def fig_e5(data_dir=DATA, out_dir=FIG):
    """Время планирования (log-log): цепочечные ключи против прямого сравнения контекстов."""
    d = _load(data_dir, "e5_planning_time.json")
    if d is None:
        return None
    fig, ax = plt.subplots(figsize=(W1, 5.5 * CM))
    for key, label, color, marker in [("chained", "цепочечные ключи", "C0", "o"),
                                      ("direct", "прямое сравнение контекстов", "C3", "s")]:
        e = d.get(key)
        if not e:
            continue
        n, t = np.array(e["sizes"], float), np.array(e["times"], float)
        k, b = np.polyfit(np.log10(n), np.log10(t), 1)
        slope = e.get("slope", k)
        ax.plot(n, t, marker=marker, ls="none", color=color, ms=3.5,
                label=f"{label}, наклон {_num(slope, 2)}")
        xx = np.array([n.min(), n.max()])
        ax.plot(xx, 10 ** (b + k * np.log10(xx)), color=color, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"число гипотез $|V|$")
    ax.set_ylabel("время планирования, с")
    ax.grid(alpha=0.3, lw=0.4, which="both")
    ax.legend(loc="upper left", fontsize=7.5)
    return _save(fig, out_dir, "e5_planning_time.pdf")


# --------------------------------------------------------------------- E6
E6_SYSTEMS = [  # (ключ, подпись, группа)
    ("snakemake", "Snakemake", 0), ("nextflow", "Nextflow", 0), ("dagster_eager", "Dagster eager", 0),
    ("dvc", "DVC", 1), ("dagster", "Dagster", 1), ("nextflow_deep", "Nextflow deep", 1),
    ("bazel", "Bazel", 1), ("hyppo_cascade", "комплекс, каскад", 2),
    ("hyppo_safe", "комплекс, безопасный", 3),
]
E6_GROUPS = [("файловый каскад", "#9e9e9e"), ("раннее отсечение", "C0"),
             ("комплекс: каскад", "C1"), ("комплекс: безопасный", "C2")]


def fig_e6(data_dir=DATA, out_dir=FIG):
    """(а) перевычислено и пропущено по системам на корпусе; (б) чувствительность к P(выход изменился)."""
    d = _load(data_dir, "e6_systems.json")
    if d is None:
        return None
    per = d["per_system"]
    sim_only = set(d.get("meta", {}).get("not_validated_by_real_run", []))
    for k, v in per.items():
        if v.get("validated_by_real_run") is False:
            sim_only.add(k)
    systems = [s for s in E6_SYSTEMS if s[0] in per]
    ps = sorted(per[systems[0][0]]["by_p"], key=float)
    P0 = d.get("corpus_meta", {}).get("p_output_changed", 0.7)

    fig = plt.figure(figsize=(W2, 8 * CM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], height_ratios=[2.2, 1],
                          hspace=0.12, wspace=0.28)
    ax_r = fig.add_subplot(gs[0, 0])
    ax_m = fig.add_subplot(gs[1, 0], sharex=ax_r)
    ax_s = fig.add_subplot(gs[:, 1])

    nb = len(systems)
    width = 0.8 / nb
    for i, p in enumerate(ps):
        for j, (key, _, grp) in enumerate(systems):
            e = per[key]["by_p"][p]
            pos = i + (j - (nb - 1) / 2) * width
            color = E6_GROUPS[grp][1]
            hatch = "////" if key in sim_only else None
            kw = dict(width=width * 0.92, color=color, edgecolor="k", lw=0.3, hatch=hatch)
            ax_r.bar(pos, e["recomputed_total"] / 1e3, **kw)
            ax_m.bar(pos, e["missed_total"] / 1e3, **kw)
    ax_r.set_ylabel("перевычислено, тыс. запусков")
    ax_m.set_ylabel("пропущено,\nтыс.")
    ax_m.set_xticks(range(len(ps)))
    ax_m.set_xticklabels([f"$p={_mnum(float(p))}$" for p in ps])
    ax_m.set_xlabel(r"доля объявленных пар $p$")
    plt.setp(ax_r.get_xticklabels(), visible=False)
    for a in (ax_r, ax_m):
        a.grid(axis="y", alpha=0.3, lw=0.4)
        a.set_axisbelow(True)
    ax_r.set_title(f"(а) корпус, P(выход изменился) = {_num(P0, 1)}", loc="left", pad=2)
    ax_r.set_ylim(0, max(per[k]["by_p"][p]["recomputed_total"] for k, _, _ in systems for p in ps)
                  / 1e3 * 1.1)
    handles = [Patch(facecolor=c, edgecolor="k", lw=0.3, label=lbl) for lbl, c in E6_GROUPS]
    handles.append(Patch(facecolor="white", edgecolor="k", lw=0.3, hatch="////",
                         label="правило не сверено с реальным прогоном"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5, handlelength=1.4,
               columnspacing=1.2, bbox_to_anchor=(0.5, -0.165))
    order = "; ".join(lbl for _, lbl, _ in systems)
    fig.text(0.5, -0.175, f"Порядок столбцов в группе: {order}.", fontsize=7.5, ha="center",
             va="top")

    # (б) чувствительность
    sens = d.get("sensitivity", {}).get("by_P", {})
    Ps = sorted(sens, key=float)
    cutoff = [k for k, _, g in systems if g == 1]
    if Ps:
        for p, style, mk in [("0.5", "--", "o"), ("1.0", "-", "s")]:
            curves = {k: [sens[P]["per_system"][k][p]["recomputed"] / 1e3 for P in Ps] for k in cutoff}
            same = all(np.allclose(curves[k], curves[cutoff[0]]) for k in cutoff)
            groups = [(cutoff, curves[cutoff[0]])] if same else [([k], curves[k]) for k in cutoff]
            for ks, y in groups:
                names = "раннее отсечение" if same else dict((k, l) for k, l, _ in systems)[ks[0]]
                ax_s.plot([float(P) for P in Ps], y, ls=style, marker=mk, ms=3.5, color="C0",
                          mfc="white" if p == "0.5" else "C0",
                          label=f"{names}, $p={_mnum(float(p))}$")
            if p == "0.5":
                for P, yy in zip(Ps, curves[cutoff[0]]):
                    miss = sens[P]["per_system"][cutoff[0]][p]["missed"]
                    ax_s.annotate(f"−{miss / 1e3:.1f}".replace(".", ","), (float(P), yy),
                                  xytext=(0, -9), textcoords="offset points", ha="center",
                                  fontsize=7.5, color="C3")
        for key, color, ls in [("hyppo_safe", "C2", "-"), ("hyppo_cascade", "C1", ":")]:
            if key in sens[Ps[0]]["per_system"]:
                y = sens[Ps[0]]["per_system"][key]["1.0"]["recomputed"] / 1e3
                lbl = dict((k, l) for k, l, _ in systems)[key]
                ax_s.axhline(y, color=color, ls=ls, lw=1.0, label=lbl)
        ax_s.set_xticks([float(P) for P in Ps])
        ax_s.xaxis.set_major_formatter(_comma)
        ax_s.set_xlabel("P(выход изменился)")
        ax_s.set_ylabel("перевычислено, тыс. запусков")
        ax_s.set_title("(б) чувствительность раннего отсечения", loc="left", pad=2)
        lo = min(min(sens[P]["per_system"][cutoff[0]]["0.5"]["recomputed"] for P in Ps) / 1e3, 1e9)
        ax_s.set_ylim(lo * 0.85, sens[Ps[0]]["per_system"]["hyppo_cascade"]["1.0"]["recomputed"]
                      / 1e3 * 1.08 if "hyppo_cascade" in sens[Ps[0]]["per_system"] else None)
        ax_s.grid(alpha=0.3, lw=0.4)
        ax_s.legend(loc="center right", fontsize=7.5, bbox_to_anchor=(1.0, 0.62))
        ax_s.text(0.03, 0.83, "красным: пропущено при $p=0{,}5$, тыс.", transform=ax_s.transAxes,
                  fontsize=7.5, color="C3", ha="left", va="top")
    return _save(fig, out_dir, "e6_systems.pdf", width_cm=TEXTWIDTH_CM)


def main(data_dir=DATA, out_dir=FIG):
    _setup()
    return [p for p in (f(data_dir, out_dir) for f in (fig_e1, fig_e2, fig_e3, fig_e5, fig_e6))
            if p is not None]


if __name__ == "__main__":
    main()
