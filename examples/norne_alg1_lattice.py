"""Run the ACTUAL Algorithm 1 (HypothesisLattice.build_lattice) on Norne data."""
import os, sys
sys.path.insert(0, r"F:\git-repos\diss\hyppo-ref")
import networkx as nx
from hyppo.coa._base import Equation, Structure
from hyppo.lattice_constructor._base import HypothesisLattice

# --- 1. Define hypotheses with their equation structures ---
# Each hypothesis: (name, equation_formula, output_var)
HYPS = [
    ("H1",  "I_agg = w_ij * I_j", "I_agg"),
    ("H2",  "q_f = a_f*q_f_prev + b_f*I_agg", "q_f"),
    ("H3",  "q_s = a_s*q_s_prev + b_s*I_agg", "q_s"),
    ("H4",  "q_c = w_f*q_f + (1-w_f)*q_s", "q_c"),
    ("H5",  "q_liq_phys = J*q_c + q_prim", "q_liq_phys"),
    ("H6",  "q_prim = q_prev*exp(-dt*taup)", "q_prim"),
    ("H7",  "l_ml = MLP(x_hist)", "l_ml"),
    ("H8",  "l = g*q_liq_phys + (1-g)*l_ml", "l"),
    ("H11", "Sw = Sw_prev + (Winj - l)*dt/Vp", "Sw"),
    ("H12",  "krw = ((Sw-Swc)/(1-Swc-Sor))**nw", "krw"),
    ("H12b", "kro = ((1-Sw-Sor)/(1-Swc-Sor))**no", "kro"),
    ("H13", "fw = 1/(1 + kro*muw/(krw*muo))", "fw"),
    ("H14", "o_p = 1 - fw", "o_p"),
    ("H15", "o = gw*o_p + (1-gw)*o_m", "o"),
    ("GRP", "J = J0 + dJ_grp", "J"),
    ("H19", "q_oil = l * o", "q_oil"),
]

# Build Structure objects from equations
class Hyp:
    """Plain hypothesis with .name and .structure."""
    def __init__(self, name, formula):
        self.name = name
        self.structure = Structure([Equation(formula=formula)])
    def __repr__(self): return self.name

hyp_objs = {name: Hyp(name, formula) for name, formula, _ in HYPS}
out_vars = {name: out for name, _, out in HYPS}

# --- 2. Define the WORKFLOW (tasks = groups of hypotheses) ---
TASKS = [
    ["H1"],                    # t1: parse injection
    ["H2", "H3"],              # t2: CRM channels
    ["H4"],                    # t3: mixing
    ["H5", "H6"],              # t4: productivity + decline
    ["H7"],                    # t5: ML correction
    ["H8"],                    # t6: LPR fusion
    ["H11"],                   # t7: material balance
    ["H12", "H12b"],           # t8: Corey rel perms
    ["H13"],                   # t9: fractional flow
    ["H14", "H15"],            # t10: WCT
    ["H19"],                   # t11: oil forecast
    ["GRP"],                   # t12: GTM modulation
]

class Workflow:
    def __init__(self, tasks, hyp_map):
        self._tasks = [[hyp_map[h] for h in task] for task in tasks]
    def get_tasks(self):
        return self._tasks

wf = Workflow(TASKS, hyp_objs)
all_hyps = [hyp_objs[name] for name, _, _ in HYPS]

# --- 3. Run Algorithm 1 ---
print("=== Algorithm 1: build_lattice (HypothesisLattice) ===")
print(f"Hypotheses: {len(all_hyps)}, Tasks: {len(TASKS)}")
print(f"Task pairs to check: {len(TASKs) if False else len(TASKS)*(len(TASKS)-1)//2}")

lattice_obj = HypothesisLattice(all_hyps, wf)
G = lattice_obj.lattice

print(f"\nResult: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"DAG: {nx.is_directed_acyclic_graph(G)}")

# Show edges
print("\nEdges (derived_by: h_i -> h_j = h_j depends on h_i):")
for u, v in sorted(G.edges()):
    print(f"  {u} -> {v}")

# Compare with our COA variable-flow graph
print("\n=== Comparison: Algorithm 1 vs COA variable-flow ===")
from hyppo.coa._base import Equation as Eq2
eqs = {n: Eq2(formula=f) for n, f, _ in HYPS}
ov = {n: o for n, _, o in HYPS}
G_coa = nx.DiGraph()
for n, _, _ in HYPS: G_coa.add_node(n)
for n, _, _ in HYPS:
    for b, _, _ in HYPS:
        if n != b and ov[n] in {x.name for x in eqs[b].get_vars()}:
            G_coa.add_edge(n, b)
print(f"COA graph: {G_coa.number_of_nodes()} nodes, {G_coa.number_of_edges()} edges")
print(f"Algorithm 1 graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Edges match: {set(G.edges()) == set(G_coa.edges())}")
if set(G.edges()) != set(G_coa.edges()):
    only_alg1 = set(G.edges()) - set(G_coa.edges())
    only_coa = set(G_coa.edges()) - set(G.edges())
    if only_alg1: print(f"  Only in Alg1: {sorted(only_alg1)}")
    if only_coa: print(f"  Only in COA: {sorted(only_coa)}")

# Test derived_by and impacts
print("\n=== Lattice queries ===")
for h_name in ["H1", "H8", "H19"]:
    h = hyp_objs[h_name]
    deps = lattice_obj.derived_by(h)
    impacts = lattice_obj.impacts(h)
    print(f"  {h_name}: derived_by (predecessors) = {sorted(d.name for d in deps)}")
    print(f"       impacts (descendants) = {sorted(d.name for d in impacts)}")

# Add hypothesis (Algorithm 2)
print("\n=== Algorithm 2: add_hypothesis ===")
new_hyp = Hyp("H20", "q_water = q_oil * fw / (1 - fw)")
lattice_obj.add_hypothesis(new_hyp)
print(f"After adding H20: {lattice_obj.lattice.number_of_nodes()} nodes, {lattice_obj.lattice.number_of_edges()} edges")
print(f"H20 derived_by: {sorted(d.name for d in lattice_obj.derived_by(new_hyp))}")
