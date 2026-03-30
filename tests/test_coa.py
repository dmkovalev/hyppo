"""Tests for COA (Causal Ordering Analysis) component.

These tests require latex2sympy2 and owlready2 which may not be available
on all platforms (antlr4 runtime is incompatible with Python 3.13+).
Tests are skipped automatically when dependencies are missing.
"""
import pytest

# Try to import the COA module; skip all tests if unavailable
coa = pytest.importorskip("hyppo.coa._base", reason="COA deps (latex2sympy) unavailable")


def test_powerset():
    """powerset should generate all subsets of the input."""
    from hyppo.coa._base import powerset
    result = list(powerset([1, 2, 3]))
    assert () in result
    assert (1,) in result
    assert (1, 2) in result
    assert (1, 2, 3) in result
    assert len(result) == 8  # 2^3


def test_equation_parses_vars():
    """Equation should extract free symbols from a LaTeX formula."""
    from hyppo.coa._base import Equation
    eq = Equation(formula=r"x_1 + x_2 + x_3 = 0")
    var_names = sorted(v.name for v in eq.vars)
    assert var_names == ["x_1", "x_2", "x_3"]


def test_structure_is_complete():
    """A structure with n equations and n variables is complete."""
    from hyppo.coa._base import Structure, Equation
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    assert s.is_complete()


def test_structure_not_complete():
    """A structure with fewer equations than variables is not complete."""
    from hyppo.coa._base import Structure, Equation
    eq1 = Equation(formula=r"x_1 + x_2 + x_3 = 0")
    s = Structure(equations=[eq1])
    assert not s.is_complete()


def test_structure_is_structure():
    """is_structure: for every subset of equations, #equations <= #variables."""
    from hyppo.coa._base import Structure, Equation
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    assert s.is_structure()


def test_exogenous_endogenous():
    """Exogenous vars appear in single-variable equations; rest are endogenous."""
    from hyppo.coa._base import Structure, Equation
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    exo_names = {v.name for v in s.exogenous()}
    endo_names = {v.name for v in s.endogenous()}
    assert "x_1" in exo_names
    assert "x_2" in endo_names


def test_transitive_closure_returns_dict():
    """build_transitive_closure should return a dict-like mapping."""
    from hyppo.coa._base import Structure, Equation
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    if s.is_complete():
        tc = s.build_transitive_closure()
        assert isinstance(tc, dict)


def test_find_minimal_structures():
    """find_minimal_structures returns Structure objects."""
    from hyppo.coa._base import Structure, Equation
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    mins = s.find_minimal_structures()
    for m in mins:
        assert hasattr(m, "equations")
        assert hasattr(m, "vars")
