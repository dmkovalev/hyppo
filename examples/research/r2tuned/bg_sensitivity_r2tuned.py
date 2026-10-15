"""Чувствительность R^2 (Norne, in-sample) к принятому B_g закачки газа."""
import os, sys, json
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
import numpy as np
import examples.research.r2tuned.models_r2tuned as M
from examples.research.r2tuned.brugge_reuse_timing_r2tuned import load_field_r2tuned, run_components
out = {}
for bg in [0.0, 0.002, 0.0043, 0.008]:
    M.BG_NORNE = bg
    import examples.research.r2tuned.brugge_reuse_timing_r2tuned as B; B.BG_NORNE = bg
    d = load_field_r2tuned("Norne", with_gas=bg > 0)
    _, r2 = run_components({"CRM", "ML", "FUS", "MB", "COREY", "BL", "WCTF", "OPR"}, d, {})
    out[str(bg)] = {k: round(v, 3) for k, v in r2.items()}; print(bg, out[str(bg)], flush=True)
json.dump(out, open(os.path.join(os.path.dirname(__file__), "bg_sensitivity_r2tuned.json"), "w"), indent=1)
