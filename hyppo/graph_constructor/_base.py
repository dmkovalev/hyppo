from __future__ import annotations

import re
from typing import TYPE_CHECKING

import networkx as nx
from sympy import Symbol, sympify

# Identifier token that is NOT part of a longer word or a numeric literal
# (the negative look-behind rejects the ``e`` in ``1e5``, etc.). Used to read
# the *left-hand side* (output) of a formula literally — no function call can
# appear there — so physical names that collide with SymPy singletons
# (``E, I, N, S, O, Q``) are never swallowed as constants.
_IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*")

# Right-hand sides are parsed through SymPy so that *function applications*
# (``sin(...)``, ``exp(...)``, ``log(...)``, ...) are recognised as functions,
# not variables. To keep that benefit without losing physical quantities whose
# names coincide with reserved SymPy singletons, those six names are shadowed
# by plain Symbols. Genuine constants (``pi``, ``oo``, ``zoo``, ``nan``) are
# deliberately NOT shadowed and stay constants.
_SHADOW = {n: Symbol(n) for n in ("E", "I", "N", "S", "O", "Q")}


def _rhs_variable_names(rhs):
    """Return the set of variable names on the RHS of a formula string.

    Uses ``sympify(rhs, locals=_SHADOW)`` then ``free_symbols`` so that applied
    functions (``sin``, ``cos``, ``exp``, ``log``, ``sqrt``, ``tan`` …) and
    undefined-function heads (``f`` in ``f(E, I, N)``) are excluded, while the
    six shadowed collision names remain variables. Falls back to the literal
    identifier scan if the expression cannot be parsed by SymPy.
    """
    try:
        expr = sympify(rhs, locals=_SHADOW)
    except Exception:
        return set(_IDENT_RE.findall(rhs))
    return {s.name for s in expr.free_symbols}

from hyppo.core._base import virtual_experiment_onto

if TYPE_CHECKING:
    pass


with virtual_experiment_onto:

    class GraphConstructor:
        def __init__(self, hypotheses, workflow):
            self.hypotheses = hypotheses
            self.workflow = workflow
            self.hyp_graph = self.build_graph()

        def build_graph(self):
            """Algorithm 1: Build hypothesis graph from hypothesis structures.

            An edge (h_i, h_j) is derived when the output variable of h_i
            appears among the input variables of h_j's equations (i.e. h_j is
            derived_by h_i); see :meth:`_build_hypothesis_var_mapping`.
            """
            hyp_graph = nx.DiGraph()
            # check if all hypotheses are in workflow
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")

            for dep in self._build_hypothesis_var_mapping():
                hyp_graph.add_edge(dep[0], dep[1])

            return hyp_graph

        def add_hypothesis(self, hypothesis):
            """Algorithm 2: add a hypothesis to an existing graph incrementally.

            Registers the new hypothesis and decides only the pairs it forms with
            the hypotheses of tasks related to its task by reachability, by the
            same rule as :meth:`build_graph` -- O(|H| * v_max), no rebuild.
            """
            if hypothesis not in self.hypotheses:
                self.hypotheses.append(hypothesis)
            if not self._is_correct():
                raise Exception("Hypotheses not found in workflow")
            self.hyp_graph.add_node(hypothesis)
            # Only pairs involving the new hypothesis are decided (Theorem
            # thm:build (III)): reachability between existing tasks is assumed
            # preserved, so all other pairs and their edges are unchanged.
            precedes = self._tau_precedes()
            new = hypothesis
            out_n = self._output_variables(new) if hasattr(new, "structure") else set()
            in_n = self._input_variables(new) if hasattr(new, "structure") else set()
            for h in self.hypotheses:
                if h is new or not hasattr(h, "structure"):
                    continue
                if precedes(h, new):
                    a, b, out_a, in_a, in_b = h, new, self._output_variables(h), self._input_variables(h), in_n
                    out_b = out_n
                elif precedes(new, h):
                    a, b, out_a, in_a, in_b = new, h, out_n, in_n, self._input_variables(h)
                    out_b = self._output_variables(h)
                else:
                    continue
                if not out_a or out_a & out_b or in_a & out_b:
                    continue  # condition У3 has priority: no edge
                if out_a & in_b:
                    self.hyp_graph.add_edge(a, b)

        def derived_by(self, hypothesis):
            """Return hypotheses that are derived by the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            return set(self.hyp_graph.predecessors(hypothesis))

        def _build_hypothesis_var_mapping(self):
            """Derive hypothesis-level edges from equation variables.

            Edge (h_i, h_j) added iff an output variable of h_i appears among
            the input variables of h_j, i.e. ``Out(h_i) ∩ In(h_j) ≠ ∅``
            (h_j is derived_by h_i). Full output *sets* are used, so a
            hypothesis with several output variables (``|Out| > 1``) feeds
            every consumer of any of them.
            """
            # Only pairs with τ(h_i) ≺ τ(h_j) (the workflow routes h_i before
            # h_j) are considered, and condition У3 has priority: an output
            # collision (competes) or a counter-supply (order conflict) gives
            # NO edge -- definition def:derived / Algorithm 1.
            precedes = self._tau_precedes()
            dependencies = []
            for h_i in self.hypotheses:
                if not hasattr(h_i, "structure") or not h_i.structure.equations:
                    continue
                out_i = self._output_variables(h_i)
                if not out_i:
                    continue
                in_i = self._input_variables(h_i)
                for h_j in self.hypotheses:
                    if h_i is h_j or not hasattr(h_j, "structure"):
                        continue
                    if not precedes(h_i, h_j):
                        continue
                    out_j = self._output_variables(h_j)
                    if out_i & out_j or in_i & out_j:
                        continue
                    if out_i & self._input_variables(h_j):
                        dependencies.append((h_i, h_j))
            return dependencies

        def _tau_precedes(self):
            """Return a predicate ``precedes(h_a, h_b)`` for τ(h_a) ≺ τ(h_b).

            The task of h_b must be reachable from the (different) task of h_a.
            If the workflow exposes task edges (``get_task_edges()`` returning
            ``(i, j)`` index pairs over ``get_tasks()``), reachability over that
            DAG is used; otherwise the ordered task list is a chain, so a task
            precedes every later one. Hypotheses outside the workflow precede
            nothing."""
            tasks = []
            try:
                tasks = list(self.workflow.get_tasks())
            except Exception:
                pass
            task_of = {}
            for idx, task in enumerate(tasks):
                for h in task:
                    task_of.setdefault(id(h), idx)
            get_edges = getattr(self.workflow, "get_task_edges", None)
            if callable(get_edges):
                g = nx.DiGraph()
                g.add_nodes_from(range(len(tasks)))
                g.add_edges_from(get_edges())
                later = {t: nx.descendants(g, t) for t in g.nodes}
            else:
                later = {t: set(range(t + 1, len(tasks))) for t in range(len(tasks))}

            def precedes(h_a, h_b):
                ta, tb = task_of.get(id(h_a)), task_of.get(id(h_b))
                return ta is not None and tb is not None and tb in later[ta]

            return precedes

        @staticmethod
        def _var_name(v):
            """Canonical variable name (works for sympy Symbols and plain str)."""
            name = getattr(v, "name", None)
            return name if name is not None else str(v)

        @classmethod
        def _output_variables(cls, hypothesis):
            """Return the set of output variable *names* of a hypothesis.

            An output is the left-hand side of an equation written as
            ``out = f(...)``. The LHS is read as a **literal token**, not via
            ``sympify`` — otherwise SymPy built-ins (``E``, ``I``, ``pi``,
            ``S``, ``N``) would be swallowed as constants and the quantity would
            silently vanish. Each ``out = ...`` equation contributes one output,
            so ``|Out|`` may exceed 1. When no equation carries an explicit LHS,
            fall back to the exogenous-variable heuristic.
            """
            outs = set()
            has_formula = False
            for eq in hypothesis.structure.equations:
                formula = getattr(eq, "formula", None)
                if formula is not None and "=" in str(formula):
                    has_formula = True
                    lhs = str(formula).split("=", 1)[0]
                    outs |= set(_IDENT_RE.findall(lhs))
            if has_formula:
                return outs
            # Fallback (no explicit LHS): endogenous = first non-exogenous var.
            exo_vars = {
                cls._var_name(e.vars[0])
                for e in hypothesis.structure.equations
                if len(e.vars) == 1
            }
            for eq in hypothesis.structure.equations:
                if len(eq.vars) > 1:
                    candidates = [
                        cls._var_name(v)
                        for v in eq.vars
                        if cls._var_name(v) not in exo_vars
                    ]
                    return {candidates[0]} if candidates else {cls._var_name(eq.vars[0])}
            return set()

        @classmethod
        def _input_variables(cls, hypothesis):
            """Return the set of input variable *names* of a hypothesis.

            For formula equations ``out = f(...)`` the inputs are the RHS
            variables parsed via SymPy (:func:`_rhs_variable_names`), so
            function calls such as ``sin(...)`` or ``exp(...)`` are excluded
            while collision names like ``E`` survive, minus the outputs. For
            bare vars-only equations (no explicit LHS) fall back to: every
            variable of a multi-variable equation that is not an output.
            """
            outs = cls._output_variables(hypothesis)
            ins = set()
            has_formula = False
            for eq in hypothesis.structure.equations:
                formula = getattr(eq, "formula", None)
                if formula is not None and "=" in str(formula):
                    has_formula = True
                    rhs = str(formula).split("=", 1)[1]
                    ins |= _rhs_variable_names(rhs)
            if has_formula:
                return ins - outs
            for eq in hypothesis.structure.equations:
                if len(eq.vars) <= 1:
                    continue
                for v in eq.vars:
                    name = cls._var_name(v)
                    if name not in outs:
                        ins.add(name)
            return ins

        # Backward-compatible single-output helper (returns one name or None).
        @classmethod
        def _output_variable(cls, hypothesis):
            outs = sorted(cls._output_variables(hypothesis))
            return outs[0] if outs else None

        def competes(self, hypothesis):
            """Return hypotheses that compete with ``hypothesis``.

            Theory (symmetric ∥): ``h_a ∥ h_b ⟺ Out(h_a) ∩ Out(h_b) ≠ ∅`` —
            two hypotheses compute a common quantity. Computed over full output
            sets across all hypotheses; independent of graph edges, so isolated
            nodes are handled without touching ``predecessors``.
            """
            if hypothesis not in self.hypotheses:
                return set()
            out_h = self._output_variables(hypothesis)
            if not out_h:
                return set()
            competitors = set()
            for h in self.hypotheses:
                if h is hypothesis:
                    continue
                if out_h & self._output_variables(h):
                    competitors.add(h)
            return competitors

        def _tau_positions(self):
            """Map each hypothesis to its position in the workflow order τ
            (index of the earliest task that contains it)."""
            positions = {}
            try:
                tasks = self.workflow.get_tasks()
            except Exception:
                return positions
            for idx, task in enumerate(tasks):
                for h in task:
                    positions.setdefault(id(h), idx)
            return positions

        def order_conflicts(self):
            """Return directed order-conflict pairs (counter-supply / local
            2-cycles), a specification error distinct from ∥ competition.

            ``orderConflict(h_a, h_b) ⟺ In(h_a) ∩ Out(h_b) ≠ ∅`` **and** h_a
            precedes h_b in the workflow order τ (h_a needs a quantity that h_b
            only produces later). Returns a set of ``(h_a, h_b)`` tuples.
            """
            precedes = self._tau_precedes()
            conflicts = set()
            for h_a in self.hypotheses:
                if not hasattr(h_a, "structure"):
                    continue
                in_a = self._input_variables(h_a)
                if not in_a:
                    continue
                for h_b in self.hypotheses:
                    if h_a is h_b or not hasattr(h_b, "structure"):
                        continue
                    if not (in_a & self._output_variables(h_b)):
                        continue
                    # reachability, not list position: parallel (incomparable)
                    # tasks never form an order conflict
                    if precedes(h_a, h_b):
                        conflicts.add((h_a, h_b))
            return conflicts

        def impacts(self, hypothesis):
            """Return hypotheses that are impacted by the given hypothesis."""
            if hypothesis not in self.hypotheses:
                return set()
            # Impacted hypotheses are those that are reachable from this hypothesis
            return set(nx.descendants(self.hyp_graph, hypothesis))

        def remove_hypothesis(self, hypothesis):
            """Remove a hypothesis from the graph."""
            if hypothesis in self.hypotheses:
                self.hypotheses.remove(hypothesis)
                self.hyp_graph.remove_node(hypothesis)

        def _is_correct(self):
            """Check if all hypotheses are present in the workflow."""
            workflow_hypotheses = set()
            tasks = self.workflow.get_tasks()
            for task in tasks:
                workflow_hypotheses.update(task)

            return all(h in workflow_hypotheses for h in self.hypotheses)
