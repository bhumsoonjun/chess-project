from __future__ import annotations

from typing import *


class tree_node:
    eval: float
    fen: str
    child: List[tree_node]

    def __init__(self, eval: float, fen: str, child: List):
        self.eval = eval;
        self.fen = fen
        self.child = child;

    def add_child(self, node: tree_node):
        self.child.append(node)

class tree:
    root: tree_node

    def __init__(self, root):
        self.root = root
