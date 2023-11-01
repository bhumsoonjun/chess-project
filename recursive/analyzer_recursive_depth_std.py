from __future__ import annotations
from stockfish import Stockfish
import platform
from typing import *
import chess
import numpy as np
import scipy.stats as stats
from abc import ABC, abstractmethod

from stockfish.models import Capture

from recursive.tree import tree, tree_node


class analyzer_recursive_depth_std:

    def __init__(
            self,
            depth=6,
            stockfish_depth=15,
            num_variation=3,
            white_skill_level=20,
            black_skill_level=20,
            nnue="true",
            stockfish_config: Dict = None
    ):
        self.search_tree: tree = None
        self.stockfish_depth = stockfish_depth
        self.nnue = nnue
        self.depth = depth
        self.num_variation = num_variation
        self.white_skill_level = white_skill_level
        self.black_skill_level = black_skill_level
        self.stockfish = self.get_stockfish_for_platform(stockfish_config=stockfish_config, nnue=nnue)

    def get_stockfish_for_platform(self, stockfish_config: Dict, nnue) -> Stockfish:
        platform_system = platform.system()
        if platform_system == "Windows" or platform_system == "Java":
            return Stockfish(
                path="../stockfish/stockfish-windows-x86-64-avx2.exe",
                depth=self.stockfish_depth,
                parameters=stockfish_config,
                nnue=nnue
            )
        else:
            return Stockfish(
                depth=self.stockfish_depth,
                parameters=stockfish_config,
                nnue=nnue
            )

    def get_eval(self, move: Dict):
        centipawn = move["Centipawn"]
        mate = move["Mate"]
        if centipawn is not None:
            return centipawn
        else:
            if mate < 0:
                return -20 * 100
            else:
                return 20 * 100

    def look_ahead_eval(self, fen, move: str, is_white_move: bool):
        self.stockfish.set_fen_position(fen)
        self.stockfish.make_moves_from_current_position([move])
        top_moves = self.stockfish.get_top_moves(self.num_variation)
        evaluation = 0

        if len(top_moves) == 0:
            return -20 if is_white_move else 20

        for move in top_moves:
            evaluation += self.get_eval(move)

        return evaluation / len(top_moves) / 100

    def get_look_ahead_eval_prob(self, fen, move, is_white_move: bool):
        look_ahead_eval = self.look_ahead_eval(fen, move, is_white_move)
        print(f"look: {look_ahead_eval}")
        if is_white_move:
            return 0.3 - 0.3 * np.exp(-0.2 * (look_ahead_eval + 4))
        else:
            return 0.3 - 0.3 * np.exp(0.2 * (look_ahead_eval - 4))

    def logisitic(self, x):
        return -1 / (1 + np.exp(-0.5 * (x - 9))) + 1

    def get_first_depth_best_move_found_prob(self, moves_dict: Dict[int, List[Dict]], move: str):
        depth_found = 1
        evaluation_at_depth_i = []
        for i in range(1, self.stockfish_depth + 1):
            best_move = self.get_best_move_from_dict_at_depth(moves_dict, i)
            evaluation_at_depth_i.append(self.get_eval(best_move) / 100)
            depth_found = i
            print(f"best: {best_move} {move} {i}")
            if best_move["Move"] == move:
                break
        std = np.std(evaluation_at_depth_i)
        scaled_std = 0.5 * np.exp(-0.5 * std)
        first_depth_weight = self.logisitic(depth_found)
        print(evaluation_at_depth_i, std, scaled_std, first_depth_weight)
        self.stockfish.set_depth(self.stockfish_depth)
        return first_depth_weight

    # def get_first_depth_prob(self, moves_dict: Dict[int, List[Dict]], move: str):
    #     for i in range(1, self.stockfish_depth + 1):
    #         best_move = self.get_best_move_from_dict_at_depth(moves_dict, i)
    #         print(f"best: {best_move} {move} {i}")
    #         if best_move == move:
    #             return 0.5 * np.exp(-0.1 * (i - 1))
    #
    #     self.stockfish.set_depth(self.stockfish_depth)
    #     return 0.5 * np.exp(-0.1 * (self.stockfish_depth))

    def get_move_weight(self, moves_dict: Dict[int, List[Dict]], move: str, rank: int):
        for i in range(1, self.stockfish_depth + 1):
            best_move_multipv = moves_dict[i]
            best_moves_set = set(map(lambda x: x["Move"], best_move_multipv[:rank + 1]))
            print(f"get move: {best_move_multipv} {move} {i}")
            if move in best_moves_set:
                return 6 * np.exp(-0.1 * i)

        self.stockfish.set_depth(self.stockfish_depth)
        return 6 * np.exp(-0.1 * (self.stockfish_depth + 1))

    def is_move_capture(self, fen: str, move: str):
        self.stockfish.set_fen_position(fen)
        is_move_capture = self.stockfish.will_move_be_a_capture(move)
        return is_move_capture == Capture.DIRECT_CAPTURE or is_move_capture == Capture.EN_PASSANT

    def get_move_capture_weight(self, fen: str, move: str):
        if self.is_move_capture(fen, move):
            print(f"CAPTURE: {self.is_move_capture(fen, move)} {fen} {move}")
            return 0.5
        else:
            return 0

    def check_if_move_is_check(self, fen: str, move: str):
        self.stockfish.set_fen_position(fen)
        self.stockfish.make_moves_from_current_position([move])
        board = chess.Board(self.stockfish.get_fen_position())
        self.stockfish.set_fen_position(fen)
        return board.is_check()

    def get_move_check_weight(self, fen: str, move: str):
        if self.check_if_move_is_check(fen, move):
            print(f"CHECK: {self.check_if_move_is_check(fen, move)} {fen} {move}")
            return 0.3
        else:
            return 0

    def group_moves(self, moves: List[Dict]) -> Dict[int, List[dict]]:
        d = {}
        for move in moves:
            if move["Depth"] not in d:
                d[move["Depth"]] = []
                d[move["Depth"]].append(move)
            else:
                d[move["Depth"]].append(move)
        return d

    def get_best_move_from_dict_at_depth(self, moves: Dict[int, List[dict]], depth: int) -> Dict:
        return moves[depth][0]

    def get_next_move(self, fen: str, parent: tree_node) -> Set[Tuple[str, tree_node]]:
        is_white_move = self.is_white_move(fen)

        if is_white_move:
            self.stockfish.set_skill_level(self.white_skill_level)
        else:
            self.stockfish.set_skill_level(self.black_skill_level)

        temp_set = set()

        self.stockfish.set_fen_position(fen)
        top_moves_lines = self.stockfish.get_top_moves_lines(self.num_variation)
        top_moves_as_dict = self.group_moves(top_moves_lines)
        best_move_with_eval = self.get_best_move_from_dict_at_depth(top_moves_as_dict, self.stockfish_depth)

        if len(best_move_with_eval) == 0:
            evaluation = 0

            if is_white_move:
                evaluation = 20 * 100
            else:
                evaluation = -20 * 100

            node = tree_node(evaluation, 1, fen, [])
            parent.add_child(node)
            return set()

        p_play_best = self.get_first_depth_best_move_found_prob(top_moves_as_dict, best_move_with_eval["Move"]) \
                      + self.get_move_capture_weight(fen, best_move_with_eval["Move"]) \
                      + self.get_move_check_weight(fen, best_move_with_eval["Move"]) \

        print(f"best: {best_move_with_eval}, p_play_best: {p_play_best}")

        if np.random.uniform() < p_play_best:
            print("ACCEPTED BEST MOVE")
            self.stockfish.set_fen_position(fen)
            self.stockfish.make_moves_from_current_position([best_move_with_eval["Move"]])
            self.stockfish.set_depth(self.stockfish_depth)
            new_fen = self.stockfish.get_fen_position()
            evaluation = self.get_eval(best_move_with_eval)
            node = tree_node(evaluation, 1, new_fen, [])
            parent.add_child(node)
            temp_set.add((new_fen, node))
            return temp_set

        top_moves = top_moves_as_dict[self.stockfish_depth]
        top_moves_from_fen = np.random.choice(top_moves, size=min(self.num_variation, len(top_moves)), replace=False)
        weights = []

        for rank, move in enumerate(top_moves_from_fen):
            self.stockfish.set_fen_position(fen)
            weight = self.get_move_weight(top_moves_as_dict, move["Move"], rank)
            print(f"w: {weight}")
            weights.append(weight)

        scaler_weights = np.array(weights) / np.sum(weights)

        for move, scaler in zip(top_moves_from_fen, scaler_weights):
            self.stockfish.set_fen_position(fen)
            self.stockfish.make_moves_from_current_position([move["Move"]])
            new_fen = self.stockfish.get_fen_position()
            evaluation = self.get_eval(move)
            node = tree_node(evaluation, scaler, new_fen, [])
            temp_set.add((new_fen, node))
            parent.add_child(node)

        return temp_set

    def get_best_move_prob_with_skill_level(self, is_white_move: bool):
        if is_white_move:
            return 0.025 * self.white_skill_level
        else:
            return 0.025 * self.black_skill_level

    def evaluate(self, starting_fen: str):
        frontier: Set[Tuple[str, tree_node]] = set()

        self.stockfish.set_fen_position(starting_fen)
        print(f"STOCKFISH EVAL: {self.stockfish.get_evaluation()}")
        init_eval = -100
        self.search_tree = tree_node(init_eval, 1, starting_fen, [])
        frontier.add((starting_fen, self.search_tree))

        depth_counter = 0

        while (len(frontier)) > 0 and depth_counter <= self.depth:
            new_frontiers = set()
            processing_counter = 0

            for fen, parent in frontier:
                print(f"depth: {depth_counter}, evaluating fen: {fen}, size: {processing_counter}/{len(frontier)}")
                temp_set = self.get_next_move(fen, parent)
                new_frontiers.update(temp_set)
                processing_counter += 1

            depth_counter += 1
            frontier = new_frontiers

        return self.dfs_eval(self.search_tree) / 100

    def is_white_move(self, fen: str):
        return fen.split(" ")[1] == "w"

    def dfs_eval(self, root: tree_node):
        if len(root.child) == 0:
            print(f"Hit child: {root.eval}")
            return root.eval

        val = 0

        print("=====================================")
        weights = []
        evals = []
        for child in root.child:
            child_val = self.dfs_eval(child)
            # print(f"root: {root.eval}, weight: {child.weight} child eval: {child.eval}, child val: {child_val}")
            weights.append(child.weight)
            evals.append(child_val)
            val += child.weight * child_val

        print(f"total weight: {weights}, total eval: {evals}, {np.dot(weights, evals)}")
        return val
