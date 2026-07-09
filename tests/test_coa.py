"""Tests for COA (causal ordering analysis). Pure Python + sympy; runs on 3.13."""

from hyppo.coa._base import Equation, Structure


def test_equation_parses_vars():
    """Equation should extract free symbols from a LaTeX formula."""
    eq = Equation(formula=r"x_1 + x_2 + x_3 = 0")
    var_names = sorted(v.name for v in eq.vars)
    assert var_names == ["x_1", "x_2", "x_3"]


def test_structure_is_complete():
    """A structure with n equations and n variables is complete."""
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    assert s.is_complete()


def test_structure_not_complete():
    """A structure with fewer equations than variables is not complete."""
    eq1 = Equation(formula=r"x_1 + x_2 + x_3 = 0")
    s = Structure(equations=[eq1])
    assert not s.is_complete()


def test_structure_is_structure():
    """is_structure: for every subset of equations, #equations <= #variables."""
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    assert s.is_structure()


def test_exogenous_endogenous():
    """Exogenous vars appear in single-variable equations; rest are endogenous."""
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    exo_names = {v.name for v in s.exogenous()}
    endo_names = {v.name for v in s.endogenous()}
    assert "x_1" in exo_names
    assert "x_2" in endo_names


def test_transitive_closure_returns_dict():
    """build_transitive_closure should return a dict-like mapping."""
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    if s.is_complete():
        tc = s.build_transitive_closure()
        assert isinstance(tc, dict)


def test_find_minimal_structures():
    """find_minimal_structures returns Structure objects."""
    eq1 = Equation(formula=r"f_1(x_1)=0")
    eq2 = Equation(formula=r"x_1 + x_2 = 0")
    s = Structure(equations=[eq1, eq2])
    mins = s.find_minimal_structures()
    for m in mins:
        assert hasattr(m, "equations")
        assert hasattr(m, "vars")
