"""Domain layer of the astroinformatics experiment: Besancon Galaxy Model (BGM).

Dissertation §4.2 (part4.tex, sect4_bgm). The module supplies what the domain
expert provides in the life cycle of a virtual experiment -- and nothing of
the invariant core:

* ``BGM_HYPOTHESES`` -- the 10 hypotheses of the BGM mass model as structures
  ``(E_h, V_h)`` with declared inputs ``In(h)`` (conditions У1, У2), written at
  the variable level of the formal model (§2.1.2). The derivability graph is
  NOT given here: it is derived by Algorithm 1 from ``Out(h_a) ∩ In(h_b)``;
* ``BGM_TASKS`` -- the workflow ``W`` (5 sequential tasks);
* ``GalaxyOntology`` -- the OWL domain ontology (galaxy, thin/thick disc, halo,
  star, stellar population, age bin, and the physical quantities that ground
  every variable of the equations, condition C7);
* numeric domain invariants (layer 3, not expressible in OWL 2): IMF mass
  normalisation, positivity, contiguous age bins, admissible axis ratios and
  remnant fractions;
* the configuration space ``N = 11 x 2 x 2 x 3 x 3 x 3 = 1188``.

Model of the thin disc (Robin et al. 2003; Czekaj et al. 2014), seven age bins
``i`` with centres ``x_c(i)`` and widths ``d(i)``. Notation of the equations:
``Eval(f, x)`` is the tabulated law ``f`` evaluated at ``x``; ``Conv``,
``Poisson``, ``Jeans``, ``Tot``, ``Chi2`` are operators of the model (integral
over the mass function, vertical Poisson equation, vertical Jeans equation,
sum over age bins, chi-square misfit); they are not variables.

Two modelling decisions differ from the graph drawn in the dissertation
(images/BGM_graph_standalone.tex) and are reported by
``examples/research/chapter4/bgm_experiment.py``:

1. the evolutionary tracks are evaluated at the age of every bin, so the age
   bins supply ``x_c`` to ``evolutionary_tracks`` (edge AgeBins -> Tracks);
2. the local luminosity is computed from the isochrones of the evolutionary
   tracks (``L_iso``), so ``local_luminosity`` reads an output of
   ``evolutionary_tracks``. By condition 5 a supply inside one task is
   forbidden, hence the tracks form their own task and the luminosity joins the
   velocity dispersion in the next one (tasks 3 and 4 regrouped).

The remnant mass fraction ``W_trk`` of a coeval population is tabulated with
the track set (for its reference IMF); an IMF-consistent remnant fraction would
add the dependency Birthrate -> Tracks and enlarge the IMF cascade.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from owlready2 import AllDisjoint, DataProperty, FunctionalProperty, ObjectProperty, Thing
from pydantic import BaseModel, field_validator, model_validator

from hyppo.adapters._spec import HypothesisSpec, chain_edges
from hyppo.core._base import virtual_experiment_onto
from hyppo.ontology.core_rules import (
    DomainOntology,
    NeuroDomainOntology,
    OilDomainOntology,
)

__all__ = [
    "BGM_HYPOTHESES",
    "BGM_TASKS",
    "BGM_TASK_EDGES",
    "DISSERTATION_TASKS",
    "DISSERTATION_EDGES",
    "BGM_VARIABLES",
    "CONFIG_AXES",
    "CONFIG_SPACE_SIZE",
    "IMF_CHI2",
    "IMF_RIVALS",
    "RIVAL_CHI2",
    "rival_hypotheses",
    "AstroDomainOntology",
    "GalaxyOntology",
    "AgeBinsParams",
    "IMFParams",
    "SFRParams",
    "DensityParams",
    "RemnantParams",
    "nominal_invariant_checks",
]

# ---------------------------------------------------------------------------
# Hypotheses: structures (E_h, V_h) with declared inputs In(h)
# ---------------------------------------------------------------------------
_H = HypothesisSpec

BGM_HYPOTHESES: list[HypothesisSpec] = [
    # -- task 1: input laws ------------------------------------------------
    _H(
        "density_law", "Закон плотности",
        ("g_rho = exp(-sqrt(1/4 + (R**2 + z**2/eps**2)/h_p**2))"
         " - exp(-sqrt(1/4 + (R**2 + z**2/eps**2)/h_m**2))",),
        frozenset({"R", "z", "eps", "h_p", "h_m"}), task=0,
        note="Einasto law of the thin disc; g_rho(R, z; eps) is the profile shape "
             "for axis ratio eps, h_p/h_m the outer/inner scale lengths",
    ),
    _H(
        "age_cells", "Возрастные ячейки",
        ("x_c = (t_lo + t_hi)/2", "d = t_hi - t_lo"),
        frozenset({"t_lo", "t_hi"}), task=0, n_variants=3, variant_param="t_hi",
        note="7 thin-disc age bins; 3 variants of the thin-disc formation age "
             "(upper edge of the oldest bin)",
    ),
    _H(
        "SFR", "Закон SFR",
        ("psi = exp(gam*tau)",),
        frozenset({"gam", "tau"}), task=0, n_variants=2, variant_param="gam",
        note="star formation law psi(tau) = exp(gamma*tau) on the age grid tau; "
             "gamma in {0.12, 0.25} (eq:sfr)",
    ),
    _H(
        "IMF", "IMF",
        ("xi = A_xi*m**(-alpha)",
         "A_xi = (2 - alpha)/(m_up**(2 - alpha) - m_lo**(2 - alpha))"),
        frozenset({"m", "alpha", "m_lo", "m_up"}), task=0, n_variants=11,
        variant_param="alpha",
        note="power-law IMF xi(m) = A m^-alpha, mass-normalised on [m_lo, m_up] "
             "(integral of m*xi dm = 1); 11 variants of alpha",
    ),
    # -- task 2: local density and birthrate --------------------------------
    _H(
        "local_volume_density", "Локальная объёмная плотность",
        ("eps_i = Eval(eps_t, x_c)", "rho = rho_sun*Eval(g_rho, eps_i)"),
        frozenset({"eps_t", "x_c", "rho_sun", "g_rho"}), task=1, n_variants=3,
        variant_param="rho_sun",
        note="axis ratio of every bin from the age-axis-ratio relation eps_t at the "
             "bin centre; local density rho_i = rho_sun_i * g_rho(eps_i); 3 sets of "
             "local mass densities",
    ),
    _H(
        "star_birthrate", "Функция звёздной рождаемости",
        ("SFR_i = Eval(psi, x_c)*d", "b = SFR_i*xi"),
        frozenset({"psi", "x_c", "d", "xi"}), task=1,
        note="SFR(i) = exp(gamma x_c(i)) d (eq:sfr) and the birthrate "
             "b(m, i) = SFR(i) IMF(m) (eq:birthrate)",
    ),
    # -- task 3: evolutionary tracks ----------------------------------------
    _H(
        "evolutionary_tracks", "Эволюционные треки",
        ("w_i = Eval(W_trk, x_c)", "L_iso = Eval(L_trk, m, x_c)",
         "rho_dyn = rho/(1 - w_i)"),
        frozenset({"W_trk", "L_trk", "x_c", "m", "rho"}), task=2, n_variants=2,
        variant_param="L_trk",
        note="track set evaluated at the bin ages: isochrones L_iso(m, i) and the "
             "remnant mass fraction w_i; dynamical (stars + remnants) density "
             "rho_dyn = rho/(1 - w_i); 2 track sets",
    ),
    # -- task 4: luminosity and velocity dispersion -------------------------
    _H(
        "local_luminosity", "Локальная светимость",
        ("ell = Conv(b, L_iso, BC_atm)", "j_L = rho*ell"),
        frozenset({"b", "L_iso", "BC_atm", "rho"}), task=3, n_variants=3,
        variant_param="BC_atm",
        note="luminosity per unit mass formed, birthrate convolved with the "
             "isochrones and the bolometric corrections of the atmosphere models; "
             "local luminosity density j_L = rho * ell; 3 atmosphere sets",
    ),
    _H(
        "velocity_dispersion", "Дисперсия скоростей",
        ("Phi = Poisson(rho_dyn)", "sigma_W = Jeans(rho_dyn, Phi)"),
        frozenset({"rho_dyn"}), task=3,
        note="vertical potential from the dynamical density (Poisson) and the "
             "vertical velocity dispersion of every isothermal population from the "
             "vertical Jeans equation",
    ),
    # -- task 5: dynamical self-consistency ---------------------------------
    _H(
        "dynamical_self_consistency", "Динамическая самосогласованность",
        ("Upsilon = Tot(rho_dyn)/Tot(j_L)",
         "chi_dyn = Chi2(sigma_W, sigma_obs) + Chi2(Upsilon, Upsilon_obs)"),
        frozenset({"rho_dyn", "j_L", "sigma_W", "sigma_obs", "Upsilon_obs"}), task=4,
        note="local dynamical mass-to-light ratio and the misfit of the predicted "
             "dispersions and M/L against observations (Bienayme et al. 1987)",
    ),
]

#: Rival IMF hypotheses ranked in sect4_bgm by their published chi^2 on Tycho-2
#: (they compete with each other: Out = {xi, A_xi} collide). Used by the rival
#: scenario in place of the single parametrised IMF hypothesis.
IMF_RIVALS: list[HypothesisSpec] = [
    _H("IMF_Kroupa", "IMF Kroupa (2001)",
       ("xi = A_xi*BrokenPL(m, alpha_1, alpha_2, m_br)",
        "A_xi = 1/MassNorm(alpha_1, alpha_2, m_br, m_lo, m_up)"),
       frozenset({"m", "alpha_1", "alpha_2", "m_br", "m_lo", "m_up"}), task=0,
       note="two-segment power law, mass-normalised"),
    _H("IMF_Haywood", "IMF Haywood (1997)",
       ("xi = A_xi*BrokenPL(m, alpha_1, alpha_2, m_br)",
        "A_xi = 1/MassNorm(alpha_1, alpha_2, m_br, m_lo, m_up)"),
       frozenset({"m", "alpha_1", "alpha_2", "m_br", "m_lo", "m_up"}), task=0,
       note="two-segment power law (other slopes and break), mass-normalised"),
    _H("IMF_Salpeter", "IMF Salpeter (1955)",
       ("xi = A_xi*m**(-alpha_S)",
        "A_xi = (2 - alpha_S)/(m_up**(2 - alpha_S) - m_lo**(2 - alpha_S))"),
       frozenset({"m", "alpha_S", "m_lo", "m_up"}), task=0,
       note="single power law alpha = 2.35, mass-normalised"),
]
RIVAL_CHI2 = {"IMF_Kroupa": 187.0, "IMF_Haywood": 342.0, "IMF_Salpeter": 518.0}


def rival_hypotheses() -> tuple[list[HypothesisSpec], list[list[str]]]:
    """Hypotheses and workflow of the rival scenario: the IMF of task 1 is
    replaced by the three published rival IMFs."""
    specs = [h for h in BGM_HYPOTHESES if h.name != "IMF"] + IMF_RIVALS
    tasks = [list(t) for t in BGM_TASKS]
    tasks[0] = [n for n in tasks[0] if n != "IMF"] + [h.name for h in IMF_RIVALS]
    return specs, tasks


#: Workflow W: five sequential tasks (indices into BGM_HYPOTHESES by name).
BGM_TASKS: list[list[str]] = [
    ["density_law", "age_cells", "SFR", "IMF"],
    ["local_volume_density", "star_birthrate"],
    ["evolutionary_tracks"],
    ["local_luminosity", "velocity_dispersion"],
    ["dynamical_self_consistency"],
]
BGM_TASK_EDGES = chain_edges(len(BGM_TASKS))

#: Workflow as written in part4.tex (fig:mass_workflow): tracks and luminosity
#: share task 3. With the equations above it violates condition 5.
DISSERTATION_TASKS: list[list[str]] = [
    ["density_law", "age_cells", "SFR", "IMF"],
    ["local_volume_density", "star_birthrate"],
    ["evolutionary_tracks", "local_luminosity"],
    ["velocity_dispersion"],
    ["dynamical_self_consistency"],
]

#: The 12 derived_by edges drawn in images/BGM_graph_standalone.tex
#: (``u -> v``: v is derived_by u).
DISSERTATION_EDGES: list[tuple[str, str]] = [
    ("age_cells", "local_volume_density"),
    ("density_law", "local_volume_density"),
    ("SFR", "star_birthrate"),
    ("IMF", "star_birthrate"),
    ("age_cells", "star_birthrate"),
    ("local_volume_density", "evolutionary_tracks"),
    ("star_birthrate", "local_luminosity"),
    ("local_volume_density", "local_luminosity"),
    ("evolutionary_tracks", "velocity_dispersion"),
    ("local_luminosity", "dynamical_self_consistency"),
    ("evolutionary_tracks", "dynamical_self_consistency"),
    ("velocity_dispersion", "dynamical_self_consistency"),
]

# ---------------------------------------------------------------------------
# Domain vocabulary: every variable of the equations is a domain term (C7)
# ---------------------------------------------------------------------------
#: variable -> (quantity class of GalaxyOntology, meaning)
BGM_VARIABLES: dict[str, tuple[str, str]] = {
    "t_lo": ("Age", "lower edge of an age bin, Gyr"),
    "t_hi": ("Age", "upper edge of an age bin, Gyr"),
    "x_c": ("Age", "age at the centre of bin i"),
    "d": ("Age", "width of bin i"),
    "tau": ("Age", "age grid of the star formation law"),
    "R": ("Position", "galactocentric radius"),
    "z": ("Position", "height above the plane"),
    "h_p": ("ScaleLength", "outer scale length of the Einasto law"),
    "h_m": ("ScaleLength", "inner (hole) scale length of the Einasto law"),
    "eps": ("AxisRatio", "axis ratio argument of the density law"),
    "eps_t": ("AxisRatio", "age-axis-ratio relation of the thin disc"),
    "eps_i": ("AxisRatio", "axis ratio of bin i"),
    "g_rho": ("DensityProfile", "Einasto profile shape"),
    "rho_sun": ("MassDensity", "local mass density of bin i at the Sun"),
    "rho": ("MassDensity", "local volume density of living stars, bin i"),
    "rho_dyn": ("MassDensity", "dynamical density, stars + remnants, bin i"),
    "gam": ("ModelParameter", "exponent gamma of the SFR law"),
    "psi": ("StarFormationRate", "star formation law psi(tau)"),
    "SFR_i": ("StarFormationRate", "stars formed in bin i"),
    "m": ("StellarMass", "initial stellar mass grid"),
    "m_lo": ("StellarMass", "lower mass limit of the IMF"),
    "m_up": ("StellarMass", "upper mass limit of the IMF"),
    "alpha": ("ModelParameter", "IMF slope"),
    "A_xi": ("ModelParameter", "IMF mass normalisation"),
    "xi": ("MassFunction", "initial mass function xi(m)"),
    "b": ("BirthRate", "stellar birthrate b(m, i)"),
    "W_trk": ("RemnantFraction", "remnant mass fraction table of the track set"),
    "w_i": ("RemnantFraction", "remnant mass fraction of bin i"),
    "L_trk": ("Luminosity", "luminosity table L(m, t) of the track set"),
    "L_iso": ("Luminosity", "isochrone luminosity L(m) at the age of bin i"),
    "BC_atm": ("Luminosity", "bolometric corrections of the atmosphere models"),
    "ell": ("Luminosity", "luminosity per unit mass formed, bin i"),
    "j_L": ("Luminosity", "local luminosity density, bin i"),
    "Phi": ("GravitationalPotential", "vertical gravitational potential"),
    "sigma_W": ("VelocityDispersion", "predicted vertical dispersion, bin i"),
    "sigma_obs": ("VelocityDispersion", "observed vertical dispersion, bin i"),
    "Upsilon": ("MassToLight", "local dynamical mass-to-light ratio"),
    "Upsilon_obs": ("MassToLight", "observed local mass-to-light ratio"),
    "chi_dyn": ("GoodnessOfFit", "misfit of dynamical self-consistency"),
    # rival IMFs (scenario of competing hypotheses)
    "alpha_1": ("ModelParameter", "low-mass IMF slope"),
    "alpha_2": ("ModelParameter", "high-mass IMF slope"),
    "alpha_S": ("ModelParameter", "Salpeter IMF slope"),
    "m_br": ("StellarMass", "break mass of a two-segment IMF"),
}

# ---------------------------------------------------------------------------
# Configuration space (part4.tex: N = 11 x 2 x 2 x 3 x 3 x 3 = 1188)
# ---------------------------------------------------------------------------
#: axis -> (hypothesis, levels). Only the counts are fixed by the dissertation;
#: the levels below are the published BGM choices where they exist and an
#: illustrative grid otherwise (the IMF slopes, the formation ages).
CONFIG_AXES: dict[str, tuple[str, list]] = {
    "IMF slope alpha": ("IMF", [round(1.3 + 0.2 * k, 1) for k in range(11)]),
    "SFR gamma": ("SFR", [0.12, 0.25]),
    "evolutionary track set": ("evolutionary_tracks", ["tracks_A", "tracks_B"]),
    "atmosphere models": ("local_luminosity", ["atm_A", "atm_B", "atm_C"]),
    "thin-disc formation age, Gyr": ("age_cells", [9.0, 10.0, 11.0]),
    "local mass density set": ("local_volume_density", ["rho_A", "rho_B", "rho_C"]),
}
CONFIG_SPACE_SIZE = int(np.prod([len(v[1]) for v in CONFIG_AXES.values()]))

#: Published chi^2 of IMF variants on Tycho-2 thin-disc (B-V)_T (Czekaj+2014).
IMF_CHI2 = {"Kroupa (2001)": 187.0, "Haywood (1997)": 342.0, "Salpeter (1955)": 518.0}

# ---------------------------------------------------------------------------
# OWL domain ontology
# ---------------------------------------------------------------------------
_QUANTITIES = sorted({cls for cls, _ in BGM_VARIABLES.values()})


def define_galaxy_schema(onto, ns):
    """Declare the astro domain ontology in ``onto`` (Rule 6 hierarchy and the
    domain classes); register the created classes on ``ns``."""
    with onto:

        class AstroDomainOntology(ns.DomainOntology):
            """Astronomy / galactic-structure domain ontology."""

        AllDisjoint([ns.OilDomainOntology, ns.NeuroDomainOntology, AstroDomainOntology])

        class GalaxyOntology(AstroDomainOntology):
            """Besancon Galaxy Model domain ontology (part4.tex, sect4_bgm)."""

        class Galaxy(Thing):
            """A galaxy (the Milky Way)."""

        class GalacticComponent(Thing):
            """A structural population of the Galaxy."""

        class ThinDisc(GalacticComponent):
            pass

        class ThickDisc(GalacticComponent):
            pass

        class Halo(GalacticComponent):
            pass

        AllDisjoint([ThinDisc, ThickDisc, Halo])

        class Star(Thing):
            pass

        class StellarPopulation(Thing):
            """Coeval population of a component, one per age bin."""

        class AgeBin(Thing):
            pass

        class has_component(Galaxy >> GalacticComponent, ObjectProperty):
            pass

        class has_population(GalacticComponent >> StellarPopulation, ObjectProperty):
            pass

        class in_age_bin(StellarPopulation >> AgeBin, ObjectProperty, FunctionalProperty):
            pass

        class has_star(StellarPopulation >> Star, ObjectProperty):
            pass

        class age_center(AgeBin >> float, DataProperty, FunctionalProperty):
            pass

        class bin_width(AgeBin >> float, DataProperty, FunctionalProperty):
            pass

        class AstroQuantity(Thing):
            """Physical quantity named by a variable of a hypothesis structure."""

        quantity = {}
        for qname in _QUANTITIES:
            quantity[qname] = type(qname, (AstroQuantity,), {"namespace": onto})
        AllDisjoint(list(quantity.values()))

    ns.AstroDomainOntology = AstroDomainOntology
    ns.GalaxyOntology = GalaxyOntology
    ns.Galaxy = Galaxy
    ns.GalacticComponent = GalacticComponent
    ns.ThinDisc = ThinDisc
    ns.ThickDisc = ThickDisc
    ns.Halo = Halo
    ns.Star = Star
    ns.StellarPopulation = StellarPopulation
    ns.AgeBin = AgeBin
    ns.has_component = has_component
    ns.has_population = has_population
    ns.in_age_bin = in_age_bin
    ns.has_star = has_star
    ns.age_center = age_center
    ns.bin_width = bin_width
    ns.AstroQuantity = AstroQuantity
    ns.astro_quantity = quantity
    return ns


_ns = SimpleNamespace(
    DomainOntology=DomainOntology,
    OilDomainOntology=OilDomainOntology,
    NeuroDomainOntology=NeuroDomainOntology,
)
define_galaxy_schema(virtual_experiment_onto, _ns)

AstroDomainOntology = _ns.AstroDomainOntology
GalaxyOntology = _ns.GalaxyOntology
Galaxy = _ns.Galaxy
ThinDisc = _ns.ThinDisc
ThickDisc = _ns.ThickDisc
Halo = _ns.Halo
StellarPopulation = _ns.StellarPopulation
AgeBin = _ns.AgeBin
has_component = _ns.has_component
has_population = _ns.has_population
in_age_bin = _ns.in_age_bin
age_center = _ns.age_center
bin_width = _ns.bin_width
AstroQuantity = _ns.AstroQuantity
ASTRO_QUANTITY = _ns.astro_quantity

# ---------------------------------------------------------------------------
# Numeric domain invariants (layer 3; not expressible in OWL 2)
# ---------------------------------------------------------------------------
#: Thin-disc age-bin edges, Gyr (Robin et al. 2003); the last edge is the
#: formation age, a configuration axis.
THIN_DISC_EDGES = [0.0, 0.15, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
AGE_OF_UNIVERSE_GYR = 13.8


class AgeBinsParams(BaseModel):
    """Age bins: strictly increasing edges from 0, positive widths, the oldest
    edge not older than the Universe."""

    edges: list[float]

    @field_validator("edges")
    @classmethod
    def _edges(cls, v: list[float]) -> list[float]:
        if len(v) < 2 or v[0] != 0.0:
            raise ValueError("age-bin edges must start at 0 and define >= 1 bin")
        if any(b <= a for a, b in zip(v, v[1:])):
            raise ValueError("age-bin widths must be positive")
        if v[-1] > AGE_OF_UNIVERSE_GYR:
            raise ValueError(f"formation age {v[-1]} Gyr exceeds the age of the Universe")
        return v


class IMFParams(BaseModel):
    """IMF xi(m) = A m^-alpha: 0 < m_lo < m_up, xi > 0 and the mass
    normalisation integral of m*xi over [m_lo, m_up] equal to 1."""

    alpha: float
    m_lo: float = 0.08
    m_up: float = 120.0

    @model_validator(mode="after")
    def _normalised(self) -> "IMFParams":
        if not 0.0 < self.m_lo < self.m_up:
            raise ValueError("IMF mass range must satisfy 0 < m_lo < m_up")
        if abs(self.alpha - 2.0) < 1e-12:
            raise ValueError("alpha = 2 needs the logarithmic normalisation")
        a = (2 - self.alpha) / (self.m_up ** (2 - self.alpha) - self.m_lo ** (2 - self.alpha))
        m = np.geomspace(self.m_lo, self.m_up, 20001)
        xi = a * m ** (-self.alpha)
        if not np.all(xi > 0):
            raise ValueError("IMF must be positive")
        norm = float(np.trapezoid(m * xi, m))
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"IMF mass normalisation {norm:.6f} != 1")
        return self


class SFRParams(BaseModel):
    """Star formation law psi = exp(gamma tau): finite and positive on the grid."""

    gam: float
    t_max: float = 10.0

    @model_validator(mode="after")
    def _positive(self) -> "SFRParams":
        tau = np.linspace(0.0, self.t_max, 201)
        psi = np.exp(self.gam * tau)
        if not (np.all(np.isfinite(psi)) and np.all(psi > 0)):
            raise ValueError("SFR must be finite and positive")
        return self


class DensityParams(BaseModel):
    """Local densities positive; axis ratios in (0, 1]; Einasto inner scale
    length below the outer one."""

    rho_sun: list[float]
    eps: list[float]
    h_p: float = 2.53
    h_m: float = 1.32

    @model_validator(mode="after")
    def _admissible(self) -> "DensityParams":
        if any(r <= 0 for r in self.rho_sun):
            raise ValueError("local mass densities must be positive")
        if any(not 0 < e <= 1 for e in self.eps):
            raise ValueError("axis ratios must lie in (0, 1]")
        if not 0 < self.h_m < self.h_p:
            raise ValueError("Einasto scale lengths must satisfy 0 < h_m < h_p")
        return self


class RemnantParams(BaseModel):
    """Remnant mass fractions 0 <= w < 1 (rho_dyn = rho / (1 - w) finite)."""

    w: list[float]

    @field_validator("w")
    @classmethod
    def _unit(cls, v: list[float]) -> list[float]:
        if any(not 0.0 <= x < 1.0 for x in v):
            raise ValueError("remnant mass fraction must lie in [0, 1)")
        return v


#: Thin-disc axis ratios per bin (Robin et al. 2003, Table 3).
THIN_DISC_EPS = [0.0140, 0.0268, 0.0375, 0.0551, 0.0696, 0.0785, 0.0791]


def nominal_invariant_checks() -> dict[str, list[str]]:
    """Run the numeric invariants over the nominal configuration space; return
    ``{hypothesis: [violation messages]}`` (empty lists when all hold)."""
    from pydantic import ValidationError

    out: dict[str, list[str]] = {h.name: [] for h in BGM_HYPOTHESES}

    def guard(name, fn):
        try:
            fn()
        except ValidationError as exc:
            out[name].append(exc.errors()[0]["msg"])

    for alpha in CONFIG_AXES["IMF slope alpha"][1]:
        guard("IMF", lambda a=alpha: IMFParams(alpha=a))
    for gam in CONFIG_AXES["SFR gamma"][1]:
        guard("SFR", lambda g=gam: SFRParams(gam=g))
    for t_form in CONFIG_AXES["thin-disc formation age, Gyr"][1]:
        guard("age_cells", lambda t=t_form: AgeBinsParams(edges=THIN_DISC_EDGES[:-1] + [t]))
    # illustrative values of the order of Robin et al. (2003), Table 3
    guard("local_volume_density",
          lambda: DensityParams(rho_sun=[0.004, 0.008, 0.006, 0.004, 0.006, 0.005, 0.007],
                                eps=THIN_DISC_EPS))
    # illustrative remnant fractions growing with age
    guard("evolutionary_tracks", lambda: RemnantParams(w=[0.0, 0.01, 0.04, 0.07, 0.10, 0.13, 0.16]))
    return out
