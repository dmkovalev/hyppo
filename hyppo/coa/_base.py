"""COA data types: Equation and Structure (plain Python, no owlready).

Formula parsing uses sympy.sympify (works on Python 3.13); all causal-ordering
algorithms delegate to the pure core in :mod:`hyppo.coa.causal`. Variables are
sympy Symbols on the public surface; the core works on their names internally.
"""

from __future__ import annotations

from sympy import sympify

from hyppo.coa import causal


class Equation:
    """A single equation, identified by its formula string, with the set of free
    variables it contains (sympy Symbols)."""

    def __init__(self, formula=None, vars=None):
        if formula is not None and vars is not None:
            raise ValueError("Equation accepts 'formula' or 'vars', not both")
        self.formula = formula
        if vars is not None:
            self.vars = sorted(vars, key=lambda s: s.name)
        elif formula is not None:
            self.vars = self._parse(formula)
        else:
            raise ValueError("Equation requires either 'formula' or 'vars'")
        self.equation = self._expr(formula) if formula is not None else None

    @staticmethod
    def _expr(formula):
        s = str(formula)
        if "=" in s:
            lhs, rhs = s.split("=", 1)
            s = f"({lhs})-({rhs})"
        return sympify(s, locals={})

    @classmethod
    def _parse(cls, formula):
        return sorted(cls._expr(formula).free_symbols, key=lambda x: x.name)

    def get_vars(self):
        return self.vars


class Structure:
    """A set of equations over a set of variables (sympy Symbols)."""

    def __init__(self, equations, vars=None):
        self.equations = list(equations)
        allv = set()
        for eq in self.equations:
            allv |= set(eq.vars)
        self.vars = set(vars) if vars is not None else allv
        self._name2sym = {s.name: s for s in (allv | self.vars)}
        self._eqsets = [frozenset(s.name for s in eq.vars) for eq in self.equations]

    # ---- predicates -----------------------------------------------------
    def is_complete(self):
        return causal.is_complete(self._eqsets)

    def is_structure(self):
        """True iff a matching saturating every equation exists (Hall's
        condition: every subset of equations references at least as many
        variables)."""
        return causal.perfect_matching(self._eqsets) is not None

    def is_minimal(self):
        return self.is_complete() and not self.find_minimal_structures()

    # ---- decomposition --------------------------------------------------
    def find_minimal_structures(self):
        """Inclusion-minimal complete *proper* substructures, as Structure objects.
        Returns [] when the structure is itself irreducible (no proper minimal
        substructure) -- matching the historical contract."""
        if not self.is_complete():
            return []
        blocks = causal.minimal_blocks(self._eqsets) or []
        result = []
        for b in blocks:
            if len(b) == len(self.equations):
                continue  # the whole structure is not a *proper* substructure
            result.append(Structure([self.equations[i] for i in b]))
        return result

    def build_full_causal_mapping(self):
        """{equation.formula: variable Symbol} -- each equation's computed var."""
        if not self.is_complete():
            raise Exception("Structure is not complete")
        m = causal.causal_mapping(self._eqsets)
        if m is None:
            raise Exception("Structure admits no perfect matching")
        return {self.equations[i].formula: self._name2sym[v] for i, v in m.items()}

    def build_transitive_closure(self):
        """{variable Symbol: set of transitively dependent variable Symbols}.

        Returns {} when the structure is incomplete OR structurally singular
        (no perfect matching). This is deliberate: lattice construction calls
        this over many candidate unions, some legitimately unsolvable, and must
        not raise. (build_full_causal_mapping, a direct query, raises instead.)
        """
        tc = causal.transitive_closure(self._eqsets)
        if tc is None:
            return {}
        return {
            self._name2sym[k]: {self._name2sym[n] for n in deps}
            for k, deps in tc.items()
        }

    # ---- set operations -------------------------------------------------
    def union(self, other):
        return Structure(self.equations + list(other.equations))

    def difference(self, others):
        drop = set()
        for s in others:
            drop |= {id(e) for e in s.equations}
        return Structure([e for e in self.equations if id(e) not in drop])

    # ---- variable roles -------------------------------------------------
    def exogenous(self):
        exo = set()
        for eq in self.equations:
            if len(eq.vars) == 1:
                exo |= set(eq.vars)
        return exo

    def endogenous(self):
        return self.vars - self.exogenous()
