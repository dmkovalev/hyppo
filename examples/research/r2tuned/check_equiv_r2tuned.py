"""Проверка: q_CRM_perpair из models_r2tuned совпадает с pywaterflood.crm.q_CRM_perpair."""
import sys
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
import numpy as np
from pywaterflood.crm import q_CRM_perpair as ref
from examples.research.r2tuned.models_r2tuned import q_CRM_perpair as mine

R = r"F:\git-repos\diss\thesis\papers\brugge_run"
rng = np.random.default_rng(0)
for f, key in [("norne_perwell.npz", "inj"), ("brugge_perwell.npz", "injection")]:
    d = np.load(R + "/" + f, allow_pickle=True)
    I = np.ascontiguousarray(d[key], float); t = d["time"].astype(float)
    for _ in range(20):
        g = rng.uniform(0, 1, I.shape[1]); tau = np.exp(rng.uniform(0, 8, I.shape[1]))
        a, b = ref(I, t, g, tau), mine(I, t, g, tau)
        assert np.abs(a - b).max() <= 1e-9 * max(1, np.abs(a).max()), (f, np.abs(a - b).max())
    print(f, "OK: max |diff| =", np.abs(a - b).max())
