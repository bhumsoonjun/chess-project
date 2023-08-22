import pandas as pd

from analyser import analyser
from stockfish import Stockfish
import platform
from typing import *
import numpy as np
from weighted_directed_graph import weighted_directed_graph

class analyser_sampler(analyser):

    def __init__(
            self,
            states: List[Tuple[int, int]],
            white_elo: int,
            black_elo: int,
            num_variations: int = 3,
            white_depth: int = 20,
            black_depth: int = 20,
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
        self.white_elo = white_elo
        self.black_elo = black_elo
        self.white_depth = white_depth
        self.black_depth = black_depth

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