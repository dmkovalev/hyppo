"""Воспроизведение базовых R^2 tab:r2_fields без записи timing JSON."""
import os, sys, time, json
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
os.environ.setdefault("HYPPO_BRUGGE_RUN", r"F:\git-repos\diss\thesis\papers\brugge_run")
import numpy as np
from examples.brugge_reuse_timing import load_field, run_components
for f in sys.argv[1:] or ["Brugge", "Norne"]:
    data = load_field(f)
    print(f, data[0].shape, "s0", data[5], "hi", data[6])
    t, r2 = run_components({"CRM","ML","FUS","MB","COREY","BL","WCTF","OPR"}, data, {})
    print(f, round(t,2), {k: round(v,3) for k,v in r2.items()}, flush=True)
