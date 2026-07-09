"""Workflow — DAG of tasks for virtual experiment execution."""

from __future__ import annotations

import networkx as nx


class Workflow:
    """Directed acyclic graph representing task dependencies in a virtual experiment.

    The Python-level counterpart of the OWL ``Workflow`` class in
    :mod:`hyppo.core._base`; used by the planner (Algorithm 4) to determine
    task execution order.
    """

    def __init__(self, tasks: list[str], edges: list[tuple[str, str]]) -> None:
        """Build the task graph and validate acyclicity.

        Args:
            tasks: Task node names.
            edges: Directed dependency edges ``(source, target)``.

        Raises:
            ValueError: If the resulting graph is not a DAG (contains a cycle).
        """
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(tasks)
        self.graph.add_edges_from(edges)
        if not self.is_dag():
            raise ValueError("Workflow must be a DAG (no cycles allowed)")

    def is_dag(self) -> bool:
        """Check whether the task graph is a directed acyclic graph.

        Returns:
            bool: True if the graph has no cycles.
        """
        return nx.is_directed_acyclic_graph(self.graph)

    def topological_order(self) -> list[str]:
        """Compute a valid execution order of the tasks.

        Returns:
            list[str]: Task names in an order that respects all dependency edges.
        """
        return list(nx.topological_sort(self.graph))

    def reachable_from(self, task: str) -> set[str]:
        """Find all tasks transitively dependent on a given task.

        Args:
            task: Name of the task to start from.

        Returns:
            set[str]: All tasks reachable from ``task`` following edges forward
            (i.e. its descendants in the DAG).
        """
        return nx.descendants(self.graph, task)

    @property
    def tasks(self) -> list[str]:
        """list[str]: All task names in the workflow."""
        return list(self.graph.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        """list[tuple[str, str]]: All dependency edges ``(source, target)``."""
        return list(self.graph.edges)
