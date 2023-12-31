import pandas as pd

from stockfish import Stockfish
import platform
from typing import *
import numpy as np
from weighted_directed_graph import weighted_directed_graph
from abc import ABC, abstractmethod

class analyser(ABC):

    def __init__(
            self,
            states: List[Tuple[int, int]],
            num_variations: int = 3,
            analysis_depth: int = 5,
            stockfish_depth: int = 20,
            stockfish_config: Dict = None
    ):
        self.states: List[Tuple[int, int]] = [-1] + states + [1]
        self.states_as_string: List[str] = ["-1"] + list(map(lambda x: f"{x}", states)) + ["1"]
        self.analysis_depth: int = analysis_depth
        self.stockfish_depth = stockfish_depth
        self.num_variations: int = num_variations
        self.stockfish_config: Dict = stockfish_config
        self.stockfish: Stockfish = self.get_stockfish_for_platform(stockfish_config)

    def get_stockfish_for_platform(self, stockfish_config: Dict) -> Stockfish:
        platform_system = platform.system()
        if platform_system == "Windows" or platform_system == "Java":
            return Stockfish(
                path="stockfish/stockfish-windows-x86-64-avx2.exe",
                depth=self.stockfish_depth,
                parameters=stockfish_config
            )
        else:
            return Stockfish(
                depth=self.stockfish_depth,
                parameters=stockfish_config
            )

    def get_belonged_state(self, evaluation: float):
        for i in range(1, len(self.states) - 1):
            state = self.states[i]
            if state[0] <= evaluation <= state[1]:
                return self.states_as_string[i]
        return Exception(f"Evaluation: {evaluation} not found in any given states")

    def map_move_to_state(self, move: Dict):
        new_move = move["Move"]
        centipawn = move["Centipawn"]
        mate = move["Mate"]

        if centipawn is not None:
            mapped_state = self.get_belonged_state(centipawn / 100)
            return new_move, mapped_state
        else:
            if mate is not None:
                if mate < 0:
                    return new_move, "-1"
                else:
                    return new_move, "1"
            else:
                return Exception("Mate and Centipawn is none")

    def build_adjacency_matrix(self, graph: weighted_directed_graph) -> np.ndarray:
        num_states = len(self.states_as_string)
        all_states = self.states_as_string
        state_to_index_mapping = dict()
        matrix = np.zeros(shape=(num_states, num_states))

        print(graph.weights)

        for i, state_name in enumerate(all_states):
            state_to_index_mapping[state_name] = i

        for edge, weight in graph.weights.items():
            u, v = edge.split("|")
            u_ind = state_to_index_mapping[u]
            v_ind = state_to_index_mapping[v]
            matrix[u_ind, v_ind] = weight

        return matrix

    def build_stochastic_matrix(self, adjacency_matrix: np.ndarray) -> np.ndarray:
        n, d = adjacency_matrix.shape

        if n != d:
            return Exception(f"n: {n} is not equal to d: {d} in adjacency matrix.")

        stochastic_matrix = np.zeros(shape=(n, n))

        for i in range(n):
            total_in_row = 0
            for j in range(n):
                total_in_row += adjacency_matrix[i, j]
            if total_in_row != 0:
                for j in range(n):
                    stochastic_matrix[i, j] = adjacency_matrix[i, j] / total_in_row

        return stochastic_matrix

    def n_step_transition(self, stochastic_matrix: np.ndarray, n: int):
        return np.linalg.matrix_power(stochastic_matrix, n)

    def augment_stoch_matirx(self, matirx: np.ndarray, alpha: float = 0.97):
        return alpha * matirx + (1 - alpha) * 1/matirx.shape[0]

    @abstractmethod
    def analyse_one_path(self, stockfish: Stockfish, move: str = None):
        pass

    @abstractmethod
    def calculate_best_move(self, inputting_fen: str):
        pass

    @abstractmethod
    def eval_position(self, inputting_fen: str):
        pass


