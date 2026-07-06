import json

DEMO_NAME = "norne-brugge"

_DEMO_VE = {
    "hypotheses": [
        {"id": "H_liquid", "params": {"crm": ["base", "skin"]}},
        {"id": "H_wct", "params": {"model": ["frac_flow", "nn"]}},
        {"id": "H_opr", "params": {"combine": ["sum", "weighted"]}},
    ],
    "workflow_edges": [["H_liquid", "H_opr"], ["H_wct", "H_opr"]],
}


def seed_demo(store) -> None:
    if any(p["name"] == DEMO_NAME for p in store.list()):
        return
    pid = store.create(name=DEMO_NAME,
                       description="Demo VE over Norne/Brugge subset")
    store.save_ve(pid, json.dumps(_DEMO_VE))
