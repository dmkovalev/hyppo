# Hyppo Examples

## Causal Ordering Analysis (COA) Example

The `coa_example.py` demonstrates how to use Hyppo's COA module to analyze causal relationships in a system of equations. The example uses a simple physical system based on Newton's laws of motion to show:

- How to define equations using LaTeX format
- Creating and working with Equation objects
- Building a Structure from equations
- Analyzing exogenous and endogenous variables
- Generating causal mappings and directed causal graphs
- Computing transitive closures of dependencies

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

## Other Examples

- `functional_connectivity/`: Examples demonstrating functional connectivity analysis