import pandas as pd

from analyser import analyser
from stockfish import Stockfish
import platform
from typing import *
import numpy as np
from weighted_directed_graph import weighted_directed_graph

class analyser_base(analyser):

    def __init__(
            self,
            states: List[Tuple[int, int]],
            num_variations: int = 3,
            analysis_depth: int = 5,
            stockfish_depth: int = 20,
            stockfish_config: Dict = None
    ):
        super().__init__(
            states,
            num_variations=num_variations,
            analysis_depth=analysis_depth,
            stockfish_depth=stockfish_depth,
            stockfish_config=stockfish_config
        )

    def analyse_one_path(self, stockfish: Stockfish, move: str = None):
        if move != None:
            stockfish.make_moves_from_current_position([move])

        fen_frontiers = set()
        fen_frontiers.add(stockfish.get_fen_position())
        new_fen_frontiers = set()
        play_graph_one_path: weighted_directed_graph = weighted_directed_graph()
        depth_counter = 0

        # BFS
        while len(fen_frontiers) > 0 and depth_counter < self.analysis_depth:
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
                top_moves_from_fen = np.random.choice(top_moves, size=min(self.num_variations, len(top_moves)),
                                                      replace=False)
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
        initial_eval = self.stockfish.get_evaluation()["value"] / 100
        initial_eval_state = self.get_belonged_state(initial_eval)

        graph = self.analyse_one_path(self.stockfish)
        adjacency_matrix = self.build_adjacency_matrix(graph)
        stochastic_matrix = self.build_stochastic_matrix(adjacency_matrix)

        all_states = ["-1"] + self.states_as_string + ["1"]
        state_to_index_mapping = dict()

        for i, state_name in enumerate(all_states):
            state_to_index_mapping[state_name] = i

        augmented_matrix = self.augment_stoch_matirx(stochastic_matrix)
        n_step_transition = self.n_step_transition(augmented_matrix, self.analysis_depth)
        init_dist = n_step_transition[state_to_index_mapping[initial_eval_state]]
        expectation = 0

        print(init_dist)
        for i in range(1, len(self.states) - 1):
            X = (self.states[i][0] + self.states[i][1]) / 2
            expectation += X * init_dist[1:-1][i]

        return adjacency_matrix, stochastic_matrix, init_dist, expectation