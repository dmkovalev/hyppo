"""Domain-layer hypothesis specification shared by the chapter-4 adapters.

A hypothesis of the formal model (dissertation §2.1.2) is a structure
``S_h = (E_h, V_h)`` with a declared input signature ``In(h)`` (condition У2);
its outputs are ``Out(h) = V_h \\ In(h)`` and condition У0 requires
``|E_h| + |In(h)| = |V_h|`` with a complete causal mapping over the
exogenised structure. :class:`HypothesisSpec` carries exactly this data plus
the domain-layer attributes the experiment needs (workflow task, model kind,
number of configuration variants).

Equations are written ``out = f(...)``. Function heads that are not SymPy
built-ins (``Eval``, ``Conv``, ``Jeans`` ...) denote operators of the domain
model and are *not* variables, so ``Vars(f)`` is the set of free symbols:
``Eval(f, x)`` reads "the tabulated law ``f`` evaluated at ``x``".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_IDENT_RE = re.compile(r"(?<![\w.])[A-Za-z_]\w*")


def _rhs_vars(rhs: str) -> set[str]:
    from hyppo.graph_constructor._base import _rhs_variable_names

    return set(_rhs_variable_names(rhs))


@dataclass(frozen=True)
class HypothesisSpec:
    """One hypothesis of a domain experiment (structure + signature + role)."""

    name: str
    label: str
    equations: tuple[str, ...]
    inputs: frozenset[str]
    task: int
    model_kind: str = "physics"  # physics | data | hybrid (rule 1)
    n_variants: int = 1  # |Q_h|, size of the hypothesis' configuration axis
    variant_param: str | None = None
    note: str = ""
    data_in: str | None = None  # data format consumed (rule 14), if typed
    data_out: str | None = None  # data format produced (rule 14), if typed
    extra: dict = field(default_factory=dict, compare=False, hash=False)

    # ---- structure ------------------------------------------------------
    def equation_vars(self) -> list[frozenset[str]]:
        """``Vars(f)`` for every equation ``f`` of ``E_h`` (LHS + RHS)."""
        out = []
        for eq in self.equations:
            lhs, rhs = eq.split("=", 1)
            out.append(frozenset(set(_IDENT_RE.findall(lhs)) | _rhs_vars(rhs)))
        return out

    @property
    def variables(self) -> frozenset[str]:
        """``V_h``."""
        return frozenset().union(*self.equation_vars()) if self.equations else frozenset()

    @property
    def outputs(self) -> frozenset[str]:
        """``Out(h) = V_h \\ In(h)``."""
        return self.variables - self.inputs

    def check_u0(self) -> None:
        """Raise ``ValueError`` unless У2 (In ⊆ V_h) and the У0 balance hold and
        each equation's left-hand side is one of the outputs (the causal mapping
        assigns every equation its LHS)."""
        if not self.inputs <= self.variables:
            raise ValueError(f"{self.name}: inputs not in V_h: {sorted(self.inputs - self.variables)}")
        if len(self.equations) + len(self.inputs) != len(self.variables):
            raise ValueError(
                f"{self.name}: У0 violated: |E|={len(self.equations)} + |In|={len(self.inputs)}"
                f" != |V|={len(self.variables)}"
            )
        lhs = [_IDENT_RE.findall(eq.split("=", 1)[0]) for eq in self.equations]
        if any(len(x) != 1 for x in lhs) or {x[0] for x in lhs} != set(self.outputs):
            raise ValueError(f"{self.name}: LHS of E_h must be exactly Out(h)")

    def formula_structure(self):
        """The structure as a :class:`hyppo.coa._base.Structure` of formula
        equations (the input format of ``GraphConstructor``)."""
        from hyppo.coa._base import Equation, Structure

        return Structure([Equation(formula=eq) for eq in self.equations])


class FormulaHypothesis:
    """Plain carrier with ``.name`` and ``.structure`` for ``GraphConstructor``."""

    def __init__(self, spec: HypothesisSpec):
        self.spec = spec
        self.name = spec.name
        self.structure = spec.formula_structure()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return self.name


class TaskWorkflow:
    """Workflow ``W`` as an ordered list of tasks with explicit task edges."""

    def __init__(self, tasks: list[list[FormulaHypothesis]], edges: list[tuple[int, int]]):
        self._tasks = tasks
        self._edges = edges

    def get_tasks(self):
        return self._tasks

    def get_task_edges(self):
        return list(self._edges)


def chain_edges(n_tasks: int) -> list[tuple[int, int]]:
    """Task edges of a sequential workflow ``t_0 -> t_1 -> ... -> t_{n-1}``."""
    return [(i, i + 1) for i in range(n_tasks - 1)]
