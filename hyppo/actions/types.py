"""Enumerations used by the hyppo action registry."""

from enum import Enum


class TrustLevel(str, Enum):
    SAFE = "safe"  # read-only, no side effects
    COMPUTE = "compute"  # heavy read, no writes
    STAGING = "staging"  # writes new artifacts
    PRODUCTION = "production"  # touches prod state — human approval required


class AgentRole(str, Enum):
    Coordinator = "Coordinator"
    Geologist = "Geologist"
    ReservoirEngineer = "ReservoirEngineer"
    ProductionEngineer = "ProductionEngineer"
    Economist = "Economist"
    Auditor = "Auditor"
