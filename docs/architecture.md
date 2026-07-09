# Architecture

`hyppo` implements the virtual-experiment management platform of
dissertation Chapter 3: a layered pipeline from user-facing surfaces down
to the OWL domain ontology.

```
gui / mcp
    -> manager
        -> runner / planner
            -> lattice_constructor / coa
                -> core (OWL ontology, Definition 1)
```

Cross-cutting packages (`storage`, `versioning`, `metadata_repository`,
`comparison`, `ontology`, `actions`, `adapters`, `generator`) are consumed
by one or more layers above without being layers themselves.

## Layers

### `hyppo.core`

The domain core: the virtual-experiment OWL ontology (`virtual_experiment_onto`,
Definition 1) built on owlready2, plus the epistemic-status transition
function (Section 2) that classifies a hypothesis as confirmed / rejected /
undetermined given its evidence.

### `hyppo.coa`

Causal-ordering algorithms (COA constructor). A pure Dulmage-Mendelsohn
core (`hyppo.coa.causal`: matching, strongly-connected components,
transitive closure) importable without sympy/owlready, plus the
`Structure`/`Equation` data model consumed by the lattice constructor.

### `hyppo.lattice_constructor`

Algorithm 1 (and its incremental variant, Algorithm 2): builds the
hypothesis dependency graph (`HypothesisLattice`) from a set of hypotheses'
equations. The output variable of a hypothesis is the left-hand side of
its formula; edges `h_i -> h_j` mean `h_j` depends on `h_i`. This is the
only place graph edges are derived — no component re-implements Algorithm 1.

### `hyppo.planner`

Algorithm 4: builds an `ExecutionPlan` as a cascaded closure over the
lattice, proven correct and ⊆-minimal (Theorem 1).

### `hyppo.runner`

Executes an `ExecutionPlan`: runs each hypothesis's model, assigns
epistemic status, cascades skips to dependents when a prerequisite fails.

### `hyppo.manager`

Top-level orchestrator (`Manager`) coordinating generator, lattice
constructor, planner and runner into one virtual-experiment lifecycle.

### `hyppo.generator`

`HypothesisGenerator`: proposes candidate hypotheses (including a
genetic-programming backend, `deap`, optional extra `gp`).

### `hyppo.storage`

Content-addressed pickle object store (`Database`, cloudpickle-backed) for
arbitrary experiment artifacts.

### `hyppo.versioning`

Hypothesis-version persistence: content-addressed snapshots and run
provenance (SQLAlchemy models + async query functions). Deliberately
independent of `hyppo.mcp` and `hyppo.actions` — `hyppo.actions` depends on
`hyppo.versioning`, not the other way around, breaking what used to be an
`actions <-> mcp` import cycle.

### `hyppo.metadata_repository`

`MetadataRepository` + `SharedCache`: persisted run/hypothesis metadata and
an in-process cache shared across a virtual-experiment session.

### `hyppo.comparison`

Statistical comparison of competing hypotheses/models: AIC/BIC, Bayesian
posterior, Benjamini-Yekutieli correction, combined ranking.

### `hyppo.ontology`

16 OWL 2 DL reasoning rules (no SWRL, no arithmetic) extending the base
ontology from `hyppo.core._base`: consistency checks, workflow validation,
quality gates, multi-experiment rules, model compatibility, lifecycle
rules. Procedural acyclicity (rule 5) is one of the golden-test claims.

### `hyppo.actions`

Typed operations callable by agents over MCP (register/diff/resolve
hypothesis versions, build/run a virtual experiment). **Known limitation**:
the domain identifier is a `Literal["oil_waterflood"]` rather than a
registry — only one domain exists so far and a registry abstraction was
judged premature (rule of three not met); revisit if/when a second domain
is added.

### `hyppo.adapters`

Bridges to external systems, e.g. `hyppo.adapters.wfopt_adapter` builds an
oil-waterflow virtual experiment from `pywaterflood` data (optional `data`
extra).

### `hyppo.gui`

FastAPI application (`create_app`) serving the packaged single-page web UI
(`hyppo/gui/static`, committed build output) plus a JSON API for the
virtual-experiment lifecycle. Entry point: `hyppo-gui`.

### `hyppo.mcp`

MCP server package (stdio and streamable-HTTP transports) exposing
`hyppo.actions` as typed MCP tools and a `Lattice Steward` persona
resource. Entry point: `hyppo-mcp`.

## Golden-test contract

Every algorithmic claim above (Algorithm 1 graph shape, Algorithm 2
incremental-equals-rebuild, Algorithm 4 correctness/minimality, Theorem 1,
operation-count complexity, rule-5 acyclicity) is pinned to a real platform
call in `tests/test_golden_claims.py`. See [Home](index.md#golden-test-contract).
