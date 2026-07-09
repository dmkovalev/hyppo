"""Workflow — DAG of tasks for virtual experiment execution."""

from __future__ import annotations

import networkx as nx


class Workflow:
    """Directed acyclic graph representing task dependencies in a virtual experiment."""

    def __init__(self, tasks: list[str], edges: list[tuple[str, str]]) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(tasks)
        self.graph.add_edges_from(edges)
        if not self.is_dag():
            raise ValueError("Workflow must be a DAG (no cycles allowed)")

    def is_dag(self) -> bool:
        return nx.is_directed_acyclic_graph(self.graph)

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self.graph))

    def reachable_from(self, task: str) -> set[str]:
        return nx.descendants(self.graph, task)

    @property
    def tasks(self) -> list[str]:
        return list(self.graph.nodes)

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges)
