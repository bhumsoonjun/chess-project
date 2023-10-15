from __future__ import annotations
from stockfish import Stockfish
import platform
from typing import *
import numpy as np
from abc import ABC, abstractmethod

from recursive.tree import tree, tree_node


class analyser_recursive_look_ahead:

    def __init__(
            self,
            depth=6,
            stockfish_depth=15,
            num_variation=3,
            white_skill_level=20,
            black_skill_level=20,
            nnue="false",
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
        print(move)
        if centipawn is not None:
            return centipawn
        else:
            if mate < 0:
                return -20 * 100
            else:
                return 20 * 100

    def look_ahead_eval(self, fen, move: str, is_white_move: bool):
        self.stockfish.set_fen_position(fen, send_ucinewgame_token=True)
        self.stockfish.make_moves_from_current_position([move])
        top_moves = self.stockfish.get_top_moves(self.num_variation + 2)
        evaluation = 0

        if len(top_moves) == 0:
            return 20 if is_white_move else -20

        for move in top_moves:
            evaluation += self.get_eval(move)

        return evaluation / len(top_moves) / 100

    def get_look_ahead_eval_prob(self, fen, move, is_white_move: bool):
        look_ahead_eval = self.look_ahead_eval(fen, move, is_white_move)
        print(f"look: {look_ahead_eval}")
        if is_white_move:
            return 0.2 - 0.2 * np.exp(-0.2 * (look_ahead_eval + 4))
        else:
            return 0.2 - 0.2 * np.exp(0.2 * (-look_ahead_eval - 4))

    def get_first_depth_prob(self, starting_fen: str, move: str):
        self.stockfish.set_fen_position(starting_fen, send_ucinewgame_token=True)
        for i in range(1, self.stockfish_depth + 1):
            self.stockfish.set_depth(i)
            best_move = self.stockfish.get_best_move()
            print(f"best: {best_move} {move} {i}")
            if best_move == move:
                return 0.5 * np.exp(-0.08 * i)

        self.stockfish.set_depth(self.stockfish_depth)
        return 0.5 * np.exp(-0.08 * (self.stockfish_depth + 1))
    def get_move_weight(self, starting_fen: str, move: str):
        self.stockfish.set_fen_position(starting_fen, send_ucinewgame_token=True)

        for i in range(1, self.stockfish_depth + 1):
            self.stockfish.set_depth(i)
            best_move = self.stockfish.get_best_move()
            if best_move == move:
                return 6 * np.exp(-0.1 * i)

        self.stockfish.set_depth(self.stockfish_depth)
        return 6 * np.exp(-0.1 * (self.stockfish_depth + 1))

    def get_next_move(self, fen: str, parent: tree_node) -> Set[Tuple[str, tree_node]]:
        temp_set = set()
        self.stockfish.set_fen_position(fen, send_ucinewgame_token=True)
        best_move_with_eval = self.stockfish.get_top_moves(1)[0]
        top_moves = self.stockfish.get_top_moves(self.num_variation + 2)
        is_white_move = self.is_white_move(fen)
        p_play_best = self.get_first_depth_prob(fen, best_move_with_eval["Move"])

        print(f"first: {p_play_best}")

        p_play_best += self.get_look_ahead_eval_prob(fen, best_move_with_eval["Move"], is_white_move)

        print(f"second: {p_play_best}")

        p_play_best += self.get_best_move_prob_with_skill_level(is_white_move)

        print(f"play: {p_play_best}")

        if np.random.uniform() < p_play_best:
            self.stockfish.set_fen_position(fen, send_ucinewgame_token=True)
            self.stockfish.make_moves_from_current_position([best_move_with_eval["Move"]])
            new_fen = self.stockfish.get_fen_position()
            evaluation = self.get_eval(best_move_with_eval)
            node = tree_node(evaluation, new_fen, [])
            parent.add_child(node)
            temp_set.add((new_fen, node))
            return temp_set

        top_moves_from_fen = np.random.choice(top_moves, size=min(self.num_variation, len(top_moves)), replace=False)

        for move in top_moves_from_fen:
            self.stockfish.set_fen_position(fen, send_ucinewgame_token=True)
            self.stockfish.make_moves_from_current_position([move["Move"]])

            new_fen = self.stockfish.get_fen_position()
            evaluation = self.get_eval(move)
            weight = self.get_move_weight(fen, move)
            weighted_eval = evaluation * weight

            node = tree_node(weighted_eval, new_fen, [])
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

        self.stockfish.set_fen_position(starting_fen, send_ucinewgame_token=True)
        init_eval = -100
        self.search_tree = tree_node(init_eval, starting_fen, [])
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

                print(len(new_frontiers))
            depth_counter += 1
            frontier = new_frontiers

        return self.dfs_eval(self.search_tree) / 100

    def is_white_move(self, fen: str):
        return fen.split(" ")[1] == "w"

    def dfs_eval(self, root: tree_node):
        if len(root.child) == 0:
            return root.eval

        val = 0

        for child in root.child:
            val += self.dfs_eval(child)

        return val / len(root.child)





