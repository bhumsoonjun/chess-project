import pandas as pd

from stockfish import Stockfish
from stockfish import Stockfish
import platform
from typing import *
import numpy as np
from weighted_directed_graph import weighted_directed_graph
class analyser:

    def __init__(self, states: List[Tuple[int, int]], num_variations: int = 3, depth: int = 5, stockfish_depth:int = 20, stockfish_config: Dict = None):
        self.states: List[Tuple[int, int]] = states
        self.states_as_string: List[str] = list(map(lambda x: f"{x}", states))
        self.depth: int = depth
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
        for i in range(len(self.states)):
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

    def analyse_one_path(self, stockfish: Stockfish, move: str = None):
        if move != None:
            stockfish.make_moves_from_current_position([move])

        fen_frontiers = set()
        fen_frontiers.add(stockfish.get_fen_position())
        new_fen_frontiers = set()
        play_graph_one_path: weighted_directed_graph = weighted_directed_graph()
        depth_counter = 0

        # BFS
        while len(fen_frontiers) > 0 and depth_counter < self.depth:
            print(f"Current Depth: {depth_counter + 1}")
            print(f"Frontier Size: {len(fen_frontiers)}")
            frontier_counter = 1
            new_fen_frontiers.clear()
            for fen in fen_frontiers:
                print(f"Evaluating: {frontier_counter}/{len(fen_frontiers)}")
                print(f"This fen: {fen}")
                stockfish.set_fen_position(fen)
                current_eval = stockfish.get_evaluation()["value"] / 100
                top_moves = stockfish.get_top_moves(3 + self.num_variations)
                top_moves_from_fen = np.random.choice(top_moves, size=min(self.num_variations, len(top_moves)), replace=False)
                current_state = self.get_belonged_state(current_eval)
                print(top_moves_from_fen)
                for top_move in top_moves_from_fen:
                    stockfish.set_fen_position(fen)
                    new_move, new_state = self.map_move_to_state(top_move)
                    print(current_state, new_state)
                    play_graph_one_path.add_edge(current_state, new_state, 1)
                    if new_state == "-1" or new_state == "1":
                        pass
                    else:
                        stockfish.make_moves_from_current_position([new_move])
                        new_fen = stockfish.get_fen_position()
                        new_fen_frontiers.add(new_fen)
                frontier_counter += 1
            fen_frontiers = new_fen_frontiers.copy()
            depth_counter += 1

        return play_graph_one_path

    def build_adjacency_matrix(self, graph: weighted_directed_graph) -> np.ndarray:
        num_states = len(self.states_as_string) + 2
        all_states = ["-1"] + self.states_as_string + ["1"]
        state_to_index_mapping = dict()
        matrix = np.zeros(shape=(num_states, num_states))

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

    def calculate_best_move(self, inputting_fen: str):
        self.stockfish.set_fen_position(inputting_fen)
        top_moves: List[Dict] = self.stockfish.get_top_moves(self.num_variations)
        top_moves_states: List[str] = [top_move["Move"] for top_move in top_moves]
        graphs = []
        adjacency_matrices = []
        stochastic_matrices = []

        for i in range(len(top_moves)):
            print(f"======== move {i} ========")
            self.stockfish.set_fen_position(inputting_fen)
            move, new_state = self.map_move_to_state(top_moves[i])
            graph = self.analyse_one_path(self.stockfish, move)
            adjacency_matrix = self.build_adjacency_matrix(graph)
            stochastic_matrix = self.build_stochastic_matrix(adjacency_matrix)
            adjacency_matrices.append(adjacency_matrix)
            stochastic_matrices.append(stochastic_matrix)
            graphs.append(graph)

        num_states = len(self.states_as_string) + 2
        all_states = ["-1"] + self.states_as_string + ["1"]
        state_to_index_mapping = dict()
        res_matrix = np.zeros(shape=(self.num_variations, num_states))

        for i, state_name in enumerate(all_states):
            state_to_index_mapping[state_name] = i

        for i, top_move in enumerate(top_moves):
            move, eval = self.map_move_to_state(top_move)
            eval_state = self.get_belonged_state(eval)
            state_index = state_to_index_mapping[eval_state]
            res_matrix[i] = stochastic_matrices[i][state_index]

        res_df = pd.DataFrame(stochastic_matrices, columns=all_states, index=top_moves_states)

        return graphs, adjacency_matrices, stochastic_matrices, res_df

    def eval_position(self, inputting_fen: str):
        self.stockfish.set_fen_position(inputting_fen)
        initial_eval = self.stockfish.get_evaluation()["value"]/100
        initial_eval_state = self.get_belonged_state(initial_eval)

        graph = self.analyse_one_path(self.stockfish)
        adjacency_matrix = self.build_adjacency_matrix(graph)
        stochastic_matrix = self.build_stochastic_matrix(adjacency_matrix)

        all_states = ["-1"] + self.states_as_string + ["1"]
        state_to_index_mapping = dict()

        for i, state_name in enumerate(all_states):
            state_to_index_mapping[state_name] = i

        augmented_matrix = self.augment_stoch_matirx(stochastic_matrix)
        n_step_transition = self.n_step_transition(augmented_matrix, self.depth)
        init_dist = n_step_transition[state_to_index_mapping[initial_eval_state]]
        expectation = 0

        print(init_dist)
        for i in range(1, len(self.states) - 1):
            X = (self.states[i][0] + self.states[i][1])/2
            expectation += X * init_dist[1:-1][i]

        return adjacency_matrix, stochastic_matrix, init_dist, expectation


