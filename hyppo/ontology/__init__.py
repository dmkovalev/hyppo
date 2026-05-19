"""OWL 2 DL reasoning rules for the Hyppo virtual experiment platform.

This package implements 16 OWL DL rules (no SWRL, no arithmetic) that extend
the base ontology defined in ``hyppo.core._base``.  Rules cover:

- Core classification and structural axioms (rules 1-6)
- Provenance and versioning (rules 7-8)
- Workflow validation (rules 9-10)
- Quality gates for hypothesis lattice pruning (rules 11-12)
- Multi-experiment sharing detection (rule 13)
- Model compatibility and data-format checks (rules 14-15)
- Lifecycle state management (rule 16)

Additionally, ``oil_constraints`` provides Python-layer physical validators
that cannot be expressed in pure OWL DL.
"""

from .core_rules import *  # noqa: F401,F403
from .provenance import *  # noqa: F401,F403
from .workflow_validation import *  # noqa: F401,F403
from .quality_gates import *  # noqa: F401,F403
from .multi_experiment import *  # noqa: F401,F403
from .model_compatibility import *  # noqa: F401,F403
from .lifecycle import *  # noqa: F401,F403
