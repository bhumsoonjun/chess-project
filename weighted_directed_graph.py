from typing import *
class weighted_directed_graph:

    def get_edge_btw_nodes(self, u: str, v: str):
        return f"{u}|{v}"

    def __init__(self):
        self.weights: Dict[str, float] = dict()
        self.graph: Dict[str, Set[str]] = dict()
        self.nodes_name: Set = set()

    def add_edge(self, u: str, v: str, w: float):
        edge = self.get_edge_btw_nodes(u, v)
        self.nodes_name.add(u)
        self.nodes_name.add(v)

        if edge in self.weights:
            self.weights[edge] += w
        else:
            self.weights[edge] = w

        if u in self.graph:
            self.graph[u].add(v)
        else:
            self.graph[u] = set()
            self.graph[u].add(v)

    def get_node_degree(self, u: str):
        return len(self.graph[u])

    def get_edge_weight(self, u, v):
        edge = self.get_edge_btw_nodes(u, v)
        return self.weights[edge]

