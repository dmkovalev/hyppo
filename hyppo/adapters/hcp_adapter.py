"""Domain layer of the neuroinformatics experiment: HCP functional connectivity.

Dissertation §4.3 (part4.tex, sect4_neuro). Supplies the expert's part of the
virtual experiment:

* ``HCP_HYPOTHESES`` -- three hypotheses as structures with declared inputs:
  the atlas defines the ROI time series ``X = P Y`` (parcellation ``P`` of the
  voxel BOLD series ``Y``), the connectivity hypothesis computes the matrix
  ``C`` from ``X`` by the chosen estimator ``kappa``, the group hypothesis
  computes the group-level connectivity ``C_grp`` of the selected subject group.
  The derivability graph is derived by Algorithm 1 (atlas -> conn -> group);
* ``HCP_TASKS`` -- the workflow (3 sequential tasks);
* ``BrainOntology`` -- OWL domain ontology (Human, Brain, ROI, NeuroImage,
  Atlas, Connectivity, SubjectGroup; ``has_age``, ``has_gender``) and the
  quantity classes that ground every variable of the equations (C7);
* executable models of the three hypotheses (NumPy): 4 atlases, 4 connectivity
  estimators (Pearson, partial correlation, covariance, tangent embedding),
  3 subject groups -- ``N = 4 x 4 x 3 = 48`` configurations. HCP S1200 fMRI is
  access-restricted (DUA), so :func:`synthetic_cohort` provides a seeded
  surrogate cohort to execute and check the models; the published statistics
  of §4.3 are not recomputed;
* numeric invariants: parcellation rows are averaging weights, connectivity
  matrices are finite and symmetric (correlations in [-1, 1] with unit
  diagonal), every subject group is non-trivial.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from owlready2 import AllDisjoint, DataProperty, FunctionalProperty, ObjectProperty, Thing
from pydantic import BaseModel, ConfigDict, model_validator

from hyppo.adapters._spec import HypothesisSpec, chain_edges
from hyppo.core._base import virtual_experiment_onto
from hyppo.ontology.core_rules import NeuroDomainOntology

__all__ = [
    "HCP_HYPOTHESES",
    "HCP_TASKS",
    "HCP_RIVALS",
    "rival_hypotheses",
    "rival_aic",
    "HCP_TASK_EDGES",
    "HCP_VARIABLES",
    "ATLASES",
    "CONN_METHODS",
    "GROUPS",
    "CONFIG_SPACE_SIZE",
    "AIC_BY_REGION",
    "BrainOntology",
    "synthetic_cohort",
    "atlas_model",
    "conn_model",
    "group_model",
    "ParcellationParams",
    "ConnectivityParams",
    "GroupParams",
]

_H = HypothesisSpec

HCP_HYPOTHESES: list[HypothesisSpec] = [
    _H("h_atlas", "Выбор атласа анатомических регионов", ("X = P*Y",),
       frozenset({"P", "Y"}), task=0, model_kind="physics", n_variants=4,
       variant_param="P", data_out="timeseries",
       note="ROI time series = atlas parcellation (averaging weights) x voxel BOLD"),
    _H("h_conn", "Функциональная связность", ("C = Conn(X, kappa)",),
       frozenset({"X", "kappa"}), task=1, model_kind="data", n_variants=4,
       variant_param="kappa", data_in="timeseries", data_out="graph",
       note="connectivity matrix of the ROI series by estimator kappa"),
    _H("h_group", "Значимые различия по полу", ("C_grp = GroupMean(C, grp)",),
       frozenset({"C", "grp"}), task=2, model_kind="data", n_variants=3,
       variant_param="grp", data_in="graph",
       note="group-level connectivity of the subject group grp "
            "(men / women / mixed); compared across groups by the ranking module"),
]

#: Rival connectivity model added by the hypothesis generator (sect4_neuro_compare):
#: a nonlinear (GP symbolic-regression) model of the same connectivity C, so it
#: competes with h_conn (Out collision); ranked by AIC (tbl:aic_bic_gender).
HCP_RIVALS: list[HypothesisSpec] = [
    _H("h_conn_gp", "Нелинейная (ГП) модель связности", ("C = GPConn(X, theta_gp)",),
       frozenset({"X", "theta_gp"}), task=1, model_kind="data", data_in="timeseries",
       data_out="graph", note="GP symbolic regression seeded by the linear model"),
]

HCP_TASKS: list[list[str]] = [["h_atlas"], ["h_conn"], ["h_group"]]
HCP_TASK_EDGES = chain_edges(len(HCP_TASKS))
DISSERTATION_EDGES = [("h_atlas", "h_conn"), ("h_conn", "h_group")]

HCP_VARIABLES: dict[str, tuple[str, str]] = {
    "Y": ("VoxelTimeSeries", "rs-fMRI BOLD series of the voxels (NeuroImage)"),
    "P": ("Parcellation", "atlas ROI-by-voxel averaging weights"),
    "X": ("ROITimeSeries", "BOLD series averaged over every ROI"),
    "kappa": ("ConnectivityMethod", "connectivity estimator"),
    "C": ("ConnectivityMatrix", "subject connectivity matrix"),
    "grp": ("SubjectGroupSelector", "subject group by gender"),
    "C_grp": ("ConnectivityMatrix", "group-level connectivity"),
    "theta_gp": ("SymbolicExpression", "GP symbolic-regression expression (rival model)"),
}


def rival_hypotheses() -> tuple[list[HypothesisSpec], list[list[str]]]:
    """Hypotheses and workflow of the rival scenario (h_conn vs h_conn_gp)."""
    return HCP_HYPOTHESES + HCP_RIVALS, [["h_atlas"], ["h_conn", "h_conn_gp"], ["h_group"]]


def rival_aic() -> dict[str, float]:
    """Total AIC over the three reported regions per rival (lower is better)."""
    return {"h_conn": sum(v["linear"] for v in AIC_BY_REGION.values()),
            "h_conn_gp": sum(v["gp"] for v in AIC_BY_REGION.values())}

ATLASES = {"harvard_oxford": 48, "aal": 116, "schaefer_100": 100, "schaefer_200": 200}
CONN_METHODS = ["pearson", "partial", "covariance", "tangent"]
GROUPS = ["men", "women", "mixed"]
CONFIG_SPACE_SIZE = len(ATLASES) * len(CONN_METHODS) * len(GROUPS)

#: Published AIC of the competing connectivity models (tbl:aic_bic_gender).
AIC_BY_REGION = {
    "Precentral Gyrus": {"linear": -53058.0, "gp": -58638.0},
    "Postcentral Gyrus": {"linear": -46048.0, "gp": -52315.0},
    "Planum Polare": {"linear": -7109.0, "gp": -10658.0},
}

# ---------------------------------------------------------------------------
# OWL domain ontology
# ---------------------------------------------------------------------------
_QUANTITIES = sorted({cls for cls, _ in HCP_VARIABLES.values()})


def define_brain_schema(onto, ns):
    """Declare BrainOntology (fig:4_3) in ``onto``; register classes on ``ns``."""
    with onto:

        class BrainOntology(ns.NeuroDomainOntology):
            """Neuroimaging domain ontology of the HCP experiment."""

        class Human(Thing):
            pass

        class Brain(Thing):
            pass

        class ROI(Thing):
            """Region of interest of an atlas."""

        class NeuroImage(Thing):
            pass

        class RestingStateFMRI(NeuroImage):
            pass

        class Atlas(Thing):
            pass

        class Connectivity(Thing):
            pass

        class FunctionalConnectivity(Connectivity):
            pass

        class SubjectGroup(Thing):
            pass

        class has_age(Human >> int, DataProperty, FunctionalProperty):
            pass

        class has_gender(Human >> str, DataProperty, FunctionalProperty):
            pass

        class has_brain(Human >> Brain, ObjectProperty):
            pass

        class has_image(Brain >> NeuroImage, ObjectProperty):
            pass

        class has_roi(Atlas >> ROI, ObjectProperty):
            pass

        class has_connectivity(Brain >> Connectivity, ObjectProperty):
            pass

        class has_member(SubjectGroup >> Human, ObjectProperty):
            pass

        AllDisjoint([Human, Brain, ROI, NeuroImage, Atlas, Connectivity, SubjectGroup])

        class NeuroQuantity(Thing):
            """Quantity named by a variable of a hypothesis structure."""

        quantity = {}
        for qname in _QUANTITIES:
            quantity[qname] = type(qname, (NeuroQuantity,), {"namespace": onto})
        AllDisjoint(list(quantity.values()))

    for k, v in dict(
        BrainOntology=BrainOntology, Human=Human, Brain=Brain, ROI=ROI,
        NeuroImage=NeuroImage, RestingStateFMRI=RestingStateFMRI, Atlas=Atlas,
        Connectivity=Connectivity, FunctionalConnectivity=FunctionalConnectivity,
        SubjectGroup=SubjectGroup, has_age=has_age, has_gender=has_gender,
        has_brain=has_brain, has_image=has_image, has_roi=has_roi,
        has_connectivity=has_connectivity, has_member=has_member,
        NeuroQuantity=NeuroQuantity,
    ).items():
        setattr(ns, k, v)
    ns.neuro_quantity = quantity
    return ns


_ns = SimpleNamespace(NeuroDomainOntology=NeuroDomainOntology)
define_brain_schema(virtual_experiment_onto, _ns)

BrainOntology = _ns.BrainOntology
Atlas = _ns.Atlas
ROI = _ns.ROI
SubjectGroup = _ns.SubjectGroup
NeuroQuantity = _ns.NeuroQuantity
NEURO_QUANTITY = _ns.neuro_quantity

# ---------------------------------------------------------------------------
# Executable models (NumPy) on a seeded surrogate cohort
# ---------------------------------------------------------------------------


def synthetic_cohort(n_subjects: int = 12, n_voxels: int = 400, n_time: int = 240, seed: int = 0):
    """Surrogate rs-fMRI cohort: ``(Y, genders)`` with ``Y[s]`` of shape
    ``(n_voxels, n_time)`` -- a few latent networks plus noise, seeded."""
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((6, n_time))
    mixing = rng.standard_normal((n_voxels, 6))
    Y = np.stack([mixing @ latent + 0.8 * rng.standard_normal((n_voxels, n_time))
                  for _ in range(n_subjects)])
    genders = ["male" if s % 2 == 0 else "female" for s in range(n_subjects)]
    return Y, genders


def parcellation(atlas: str, n_voxels: int, seed: int = 0) -> np.ndarray:
    """Averaging weights ``P`` (n_roi x n_voxels) of a hard label map."""
    n_roi = min(ATLASES[atlas], n_voxels)
    rng = np.random.default_rng(seed + n_roi)
    labels = np.concatenate([np.arange(n_roi), rng.integers(0, n_roi, n_voxels - n_roi)])
    P = np.zeros((n_roi, n_voxels))
    P[labels, np.arange(n_voxels)] = 1.0
    return P / P.sum(axis=1, keepdims=True)


def atlas_model(atlas: str, Y: np.ndarray) -> np.ndarray:
    """h_atlas: ``X = P Y`` for every subject."""
    P = parcellation(atlas, Y.shape[1])
    return np.einsum("rv,svt->srt", P, Y)


def _spd_power(S: np.ndarray, p: float) -> np.ndarray:
    w, V = np.linalg.eigh(S)
    return (V * np.clip(w, 1e-12, None) ** p) @ V.T


def conn_model(method: str, X: np.ndarray) -> np.ndarray:
    """h_conn: ``C = Conn(X, kappa)`` for every subject (n_subj x n_roi x n_roi)."""
    covs = np.stack([np.cov(x) + 1e-6 * np.eye(x.shape[0]) for x in X])
    if method == "covariance":
        return covs
    if method == "pearson":
        sd = np.sqrt(np.einsum("sii->si", covs))
        return covs / sd[:, :, None] / sd[:, None, :]
    if method == "partial":
        prec = np.linalg.inv(covs)
        dg = np.sqrt(np.einsum("sii->si", prec))
        out = -prec / dg[:, :, None] / dg[:, None, :]
        idx = np.arange(out.shape[1])
        out[:, idx, idx] = 1.0
        return out
    if method == "tangent":
        ref = covs.mean(axis=0)
        w_isqrt = _spd_power(ref, -0.5)
        out = []
        for c in covs:
            m = w_isqrt @ c @ w_isqrt
            w, V = np.linalg.eigh((m + m.T) / 2)
            out.append((V * np.log(np.clip(w, 1e-12, None))) @ V.T)
        return np.stack(out)
    raise ValueError(f"unknown connectivity method {method!r}")


def group_model(group: str, C: np.ndarray, genders: list[str]) -> np.ndarray:
    """h_group: ``C_grp = GroupMean(C, grp)``."""
    sel = {"men": "male", "women": "female"}.get(group)
    mask = np.array([sel is None or g == sel for g in genders])
    return C[mask].mean(axis=0)


# ---------------------------------------------------------------------------
# Numeric domain invariants
# ---------------------------------------------------------------------------


class ParcellationParams(BaseModel):
    """Atlas weights: non-negative, every ROI row sums to 1."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    P: np.ndarray

    @model_validator(mode="after")
    def _weights(self) -> "ParcellationParams":
        if np.any(self.P < 0) or not np.allclose(self.P.sum(axis=1), 1.0):
            raise ValueError("parcellation rows must be averaging weights")
        return self


class ConnectivityParams(BaseModel):
    """Connectivity matrix: finite and symmetric; correlations in [-1, 1] with
    unit diagonal."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    C: np.ndarray
    method: str

    @model_validator(mode="after")
    def _matrix(self) -> "ConnectivityParams":
        C = self.C
        if not np.all(np.isfinite(C)) or not np.allclose(C, C.T, atol=1e-8):
            raise ValueError("connectivity matrix must be finite and symmetric")
        if self.method in ("pearson", "partial"):
            if np.any(np.abs(C) > 1 + 1e-9) or not np.allclose(np.diag(C), 1.0):
                raise ValueError("correlation matrix must lie in [-1, 1] with unit diagonal")
        return self


class GroupParams(BaseModel):
    """Every subject group used for comparison has at least two subjects."""

    sizes: dict[str, int]

    @model_validator(mode="after")
    def _sizes(self) -> "GroupParams":
        small = {g: n for g, n in self.sizes.items() if n < 2}
        if small:
            raise ValueError(f"subject groups too small: {small}")
        return self
