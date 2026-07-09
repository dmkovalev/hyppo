"""Python-layer physical validation constraints for the Oil domain.

These constraints involve arithmetic comparisons that *cannot* be expressed
in pure OWL 2 DL.  They are implemented as Pydantic validators and are
intended to guard hypothesis / model parameters at the application boundary.

Each constraint is a working hypothesis; see dissertation Section 4.10.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator, model_validator

__all__ = [
    "FractionalFlowParams",
    "TimeScaleParams",
    "SaturationParams",
    "CoreyExponentParams",
]


class FractionalFlowParams(BaseModel):
    """Fractional-flow allocation between injector-producer pairs.

    Working hypothesis (Section 4.10): each fractional-flow coefficient
    f_ij must lie in [0, 1] and the total allocation from any single
    injector must not exceed 1.
    """

    f_ij: list[float]
    """Fractional-flow coefficients for one injector."""

    @field_validator("f_ij", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> list[float]:
        if not isinstance(v, list):
            raise TypeError("f_ij must be a list of floats")
        return [float(x) for x in v]

    @field_validator("f_ij")
    @classmethod
    def _each_in_unit_interval(cls, v: list[float]) -> list[float]:
        for i, x in enumerate(v):
            if not 0.0 <= x <= 1.0:
                raise ValueError(f"f_ij[{i}] = {x} is outside [0, 1]")
        return v

    @model_validator(mode="after")
    def _sum_leq_one(self) -> "FractionalFlowParams":
        total = sum(self.f_ij)
        if total > 1.0 + 1e-12:
            raise ValueError(f"sum(f_ij) = {total:.6f} exceeds 1.0")
        return self


class TimeScaleParams(BaseModel):
    """CRM time-scale parameters.

    Working hypothesis (Section 4.10): the fast time constant must be
    strictly less than the slow time constant (tau_fast < tau_slow).
    """

    tau_fast: float
    """Fast (short-range) time constant."""

    tau_slow: float
    """Slow (long-range) time constant."""

    @model_validator(mode="after")
    def _fast_lt_slow(self) -> "TimeScaleParams":
        if self.tau_fast >= self.tau_slow:
            raise ValueError(
                f"tau_fast ({self.tau_fast}) must be < tau_slow ({self.tau_slow})"
            )
        return self


class SaturationParams(BaseModel):
    """Oil-water saturation balance.

    Working hypothesis (Section 4.10): S_o + S_w = 1 (two-phase system).
    """

    s_o: float
    """Oil saturation."""

    s_w: float
    """Water saturation."""

    @model_validator(mode="after")
    def _saturations_sum_to_one(self) -> "SaturationParams":
        total = self.s_o + self.s_w
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"S_o + S_w = {total:.9f}, expected 1.0")
        return self


class CoreyExponentParams(BaseModel):
    """Corey relative-permeability exponents.

    Working hypothesis (Section 4.10): all Corey exponents must be
    strictly positive (n > 0).
    """

    n_oil: float
    """Oil-phase Corey exponent."""

    n_water: float
    """Water-phase Corey exponent."""

    @field_validator("n_oil", "n_water")
    @classmethod
    def _positive(cls, v: float, info: object) -> float:
        if v <= 0:
            raise ValueError(
                f"{info.field_name} = {v} must be > 0"  # type: ignore[union-attr]
            )
        return v
