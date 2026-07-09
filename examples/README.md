# Hyppo Examples

## Causal Ordering Analysis (COA) Example

The `coa_example.py` demonstrates how to use Hyppo's COA module to analyze causal relationships in a system of equations. The example uses a simple chain system to show:

- How to define equations using LaTeX format
- Creating and working with Equation objects
- Building a Structure from equations
- Analyzing exogenous and endogenous variables
- Generating causal mappings and directed causal graphs
- Computing transitive closures of dependencies
- Measuring algorithm complexity with different system sizes

### Running the Example

To run the COA example:

```bash
python coa_example.py
```

The example will output:
- The system's exogenous and endogenous variables
- Full causal mapping showing which equations determine which variables
- A directed causal graph in DOT format
- Transitive closure of variable dependencies
- A complexity analysis plot saved as 'complexity_analysis.png'

## Other Examples

- `all_rules.py`: All-17-rules demo (owlready2/HermiT) on the real Brugge-fitted hypothesis graph.
- `all_rules_16.py`: All 17 rules on the 16-node formula-derived (COA) HybridCRM graph; demonstrates a staleness cascade restricted to the liquid branch.
- `benchmark_341.py`: Scalability benchmark — all 17 rules via HermiT on a synthetic 341-node per-pair/per-well graph.
- `brugge_demo.py`: Brugge integrity-control demo — real HermiT run of rule 4 (cascade) and rule 7 (hidden staleness).
- `norne_alg1_lattice.py`: Runs the actual Algorithm 1 (`HypothesisLattice.build_lattice`) on Norne data; reference for the 16-node/18-edge golden lattice.
- `norne_markers_real.py`: Layer 2 (markers) procedural rule run on a real Norne ABox.
- `three_layers_v2.py`: Fits `pywaterflood` CRM per-well on real Brugge/Norne production data.
- `volve_run.py`: Real Volve field-level CRM fit + integrity-control demo.
- `functional_connectivity/`: HCP brain functional-connectivity example (legacy, see its own README).
- `random_structures/`: Generates random complete COA structures (see its own README).