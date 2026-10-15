# -*- coding: utf-8 -*-
"""Рисунок fig:crm_rivals (гл. 2, сквозной пример): переоценка соперников по q_c
на открытых данных Brugge. (а) суммарный дебит жидкости: замер и прогнозы
многоканальной цепочки H2--H4 и одноканального соперника H4'; (б) веса Акаике
и сводка критериев. Числа --- из hyppo-ref
examples/research/chapter2/crm_rivals_brugge.py (копия JSON рядом со скриптом).

Вариант _r2tuned: данные из crm_rivals_r2tuned.json (crm_rivals_r2tuned.py).
Запуск: python make_crm_rivals_r2tuned.py -> thesis/images/crm_rivals_r2tuned.pdf (+ PNG рядом)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
THESIS = Path("F:/git-repos/diss/thesis")
d = json.loads((HERE / "crm_rivals_r2tuned.json").read_text(encoding="utf-8"))
A, B = "H2-H4 (per-pair)", "H4' (per-producer)"
ra, rb = d["rivals"][A], d["rivals"][B]
t = np.array(d["series"]["t"]) / 365.25
obs = np.array(d["series"]["observed_total"]) / 1e3
pa = np.array(d["series"][A]) / 1e3
pb = np.array(d["series"][B]) / 1e3
s0, hi = d["fit_window"]

COL = {"A": "#0072B2", "B": "#E69F00", "obs": "#222222"}
plt.rcParams.update({
    "font.size": 14, "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 3.9),
                               gridspec_kw={"width_ratios": [1.5, 1.0]})


def fmt(x, nd=3):
    return f"{x:.{nd}f}".replace(".", ",")


LIQ = np.array(d["perwell"]["observed"])
PA = np.array(d["perwell"][A]); PB = np.array(d["perwell"][B])
m = LIQ > 1; m[:s0] = False
fitm = m.copy(); fitm[hi:] = False
holdm = m.copy(); holdm[:hi] = False
# скважина с наибольшим различием соперников по сумме квадратов остатков
diff = (((LIQ - PA) ** 2 - (LIQ - PB) ** 2) * m).sum(0)
j = int(np.argmax(np.abs(diff)))
axA.axvspan(t[s0], t[hi - 1], color="#999999", alpha=0.13, lw=0)
axA.text(t[hi - 1] - 0.1, 0.92, "окно подгонки", transform=axA.get_xaxis_transform(),
         fontsize=11, color="#555555", ha="right")
mj = m[:, j]
axA.scatter(t[mj], LIQ[mj, j], s=12, color=COL["obs"], zorder=5, alpha=0.75,
            label=f"замер, скважина № {j + 1} из {d['Np']}")
axA.plot(t[s0:], PA[s0:, j], color=COL["A"], lw=2.6, ls="-", label=r"$H_2$–$H_4$ (многоканальная)")
axA.plot(t[s0:], PB[s0:, j], color=COL["B"], lw=2.4, dashes=(6, 2), label=r"$H'_4$ (одноканальная)")
axA.set_xlabel("время, годы")
axA.set_ylabel("дебит жидкости, усл. ед.")
axA.set_title(r"(а) Соперники по величине $q_c$: прогноз и замер", fontsize=14, loc="left")
axA.legend(fontsize=10.5, loc="lower center", framealpha=1.0)
print("well", j + 1)

# (б) веса Акаике: 100%-стековая полоса + сводка критериев
wa, wb = ra["akaike_weight"], rb["akaike_weight"]
axB.barh(0.0, wb, 0.5, color=COL["B"], hatch="//", edgecolor="white", lw=1.4)
axB.barh(0.0, wa, 0.5, left=wb, color=COL["A"], edgecolor="white", lw=1.4)
axB.text(wb / 2, 0.0, r"$w(H'_4)\approx 1$", ha="center", va="center", fontsize=13,
         color="#111111", fontweight="bold")
summary = (rf"$H_2$–$H_4$: $k={ra['k']}$, $\Delta\mathrm{{AIC}}={fmt(ra['dAIC'], 1)}$" "\n"
           rf"$\quad\Rightarrow$ SUPERSEDED ($w\approx 10^{{{int(np.floor(np.log10(wa)))}}}$)" "\n"
           rf"$H'_4$: $k={rb['k']}$, $\Delta\mathrm{{AIC}}=0$ (лучшая)" "\n"
           rf"$\Delta\mathrm{{BIC}}={fmt(ra['BIC'] - rb['BIC'], 0)}$;  $n={ra['n_fit']}$")
axB.text(0.0, -0.62, summary, transform=axB.get_yaxis_transform(), ha="left", va="top",
         fontsize=11.5)
axB.set_yticks([])
axB.set_xlim(0, 1.0); axB.set_ylim(-1.9, 0.5)
axB.set_xticks([0, 0.5, 1.0]); axB.set_xticklabels(["0", "0,5", "1"])
axB.set_xlabel(r"вес Акаике $w$ (сумма $=1$)", fontsize=12.5)
axB.set_title(r"(б) Переоценка пары $H_4 \parallel H'_4$", fontsize=14, loc="left")
axB.grid(axis="y", alpha=0)
for sp in ("top", "right", "left"):
    axB.spines[sp].set_visible(False)

fig.tight_layout()
out = THESIS / "images" / "crm_rivals_r2tuned.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(HERE / "crm_rivals_r2tuned.png", dpi=150, bbox_inches="tight")
print("saved", out)
