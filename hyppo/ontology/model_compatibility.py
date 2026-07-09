"""OWL 2 DL model-compatibility rules (14-15).

Rule 14 -- FormatMismatch: detects when a time-series producer feeds into a
    graph consumer (incompatible data formats).
Rule 15 -- DatasetMissing: detects models that need a dataset but whose
    configuration does not provide one.
"""

from __future__ import annotations

from types import SimpleNamespace

from owlready2 import (
    AllDisjoint,
    ObjectProperty,
    PropertyChain,
    Thing,
)

from hyppo.core._base import Model, virtual_experiment_onto

__all__ = [
    "DataFormat",
    "TimeSeriesFormat",
    "GraphFormat",
    "TimeSeriesProducer",
    "GraphConsumer",
    "feedsInto",
    "FormatMismatch",
    "Dataset",
    "ModelConfig",
    "usedInConfig",
    "hasAvailableDataset",
    "hasAccessibleDataset",
    "ModelWithDatasetNeed",
    "DatasetNotInConfig",
]


def define_rules(onto, ns):
    """Declare this module's OWL rules in ``onto`` using base entities
    from ``ns``; register the created classes back onto ``ns``."""
    with onto:
        # ── Rule 14: FormatMismatch ───────────────────────────────────────────

        class DataFormat(Thing):
            """Abstract data-exchange format marker."""

        class TimeSeriesFormat(DataFormat):
            """Time-series tabular format."""

        class GraphFormat(DataFormat):
            """Graph / network format."""

        AllDisjoint([TimeSeriesFormat, GraphFormat])

        class TimeSeriesProducer(ns.Model):
            """A model that outputs time-series data."""

        class GraphConsumer(ns.Model):
            """A model that consumes graph-structured data."""

        class feedsInto(ns.Model >> ns.Model, ObjectProperty):
            """Data-flow edge: one model's output feeds another's input."""

        class FormatMismatch(ns.Model):
            """A time-series producer that feeds into a graph consumer --
            an incompatible format pairing.

            Formally: TimeSeriesProducer AND feedsInto SOME GraphConsumer.
            """

            equivalent_to = [TimeSeriesProducer & feedsInto.some(GraphConsumer)]

        # ── Rule 15: DatasetMissing ───────────────────────────────────────────

        class Dataset(Thing):
            """A named dataset resource."""

        class ModelConfig(Thing):
            """A model configuration bundle."""

        class usedInConfig(ns.Model >> ModelConfig, ObjectProperty):
            """Links a model to the configuration it is used in."""

        class hasAvailableDataset(ModelConfig >> Dataset, ObjectProperty):
            """Links a configuration to datasets it provides."""

        class hasAccessibleDataset(ns.Model >> Dataset, ObjectProperty):
            """Derived: datasets accessible to a model via its config.

            Inferred via PropertyChain(usedInConfig, hasAvailableDataset).
            """

        hasAccessibleDataset.property_chain.append(
            PropertyChain([usedInConfig, hasAvailableDataset])
        )

        class ModelWithDatasetNeed(ns.Model):
            """A model that requires at least one dataset."""

        class DatasetNotInConfig(ns.Model):
            """A model needing a dataset but whose configuration provides none.

            Note: ``Not(hasAccessibleDataset.some(Dataset))`` requires CWA —
            under OWA the reasoner cannot infer absence of a dataset link.
            The class is retained as a positive marker; dataset-missing
            detection is delegated to Python validation.
            """

    ns.DataFormat = DataFormat
    ns.TimeSeriesFormat = TimeSeriesFormat
    ns.GraphFormat = GraphFormat
    ns.TimeSeriesProducer = TimeSeriesProducer
    ns.GraphConsumer = GraphConsumer
    ns.feedsInto = feedsInto
    ns.FormatMismatch = FormatMismatch
    ns.Dataset = Dataset
    ns.ModelConfig = ModelConfig
    ns.usedInConfig = usedInConfig
    ns.hasAvailableDataset = hasAvailableDataset
    ns.hasAccessibleDataset = hasAccessibleDataset
    ns.ModelWithDatasetNeed = ModelWithDatasetNeed
    ns.DatasetNotInConfig = DatasetNotInConfig
    return ns


_ns = SimpleNamespace(
    Model=Model,
)
define_rules(virtual_experiment_onto, _ns)

DataFormat = _ns.DataFormat
TimeSeriesFormat = _ns.TimeSeriesFormat
GraphFormat = _ns.GraphFormat
TimeSeriesProducer = _ns.TimeSeriesProducer
GraphConsumer = _ns.GraphConsumer
feedsInto = _ns.feedsInto
FormatMismatch = _ns.FormatMismatch
Dataset = _ns.Dataset
ModelConfig = _ns.ModelConfig
usedInConfig = _ns.usedInConfig
hasAvailableDataset = _ns.hasAvailableDataset
hasAccessibleDataset = _ns.hasAccessibleDataset
ModelWithDatasetNeed = _ns.ModelWithDatasetNeed
DatasetNotInConfig = _ns.DatasetNotInConfig
