from stockfish import Stockfish

from recursive.analyser_recursive import analyser_recursive
from recursive.analyzer_recursive_depth_std import analyzer_recursive_depth_std
from recursive.ver_4 import analyzer_recursive_depth_pruned
import numpy as np

fen = "1r1q1r1k/p1p1ppb1/b4np1/2pP3p/2P4B/2NB3P/PP1Q1PP1/2KR3R w - - 1 16"
stockfish_conf = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 4, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 2048, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 5,
    "Skill Level": 20,
    "Move Overhead": 10,
    "Minimum Thinking Time": 0,
    "Slow Mover": 100,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 3500
}

# analyser = analyser_recursive(
#     depth=4,
#     stockfish_depth=15,
#     num_variation=3,
#     stockfish_config=stockfish_conf,
#     p_black_plat_best=0,
#     p_white_play_best=0,
#     nnue="true"
# )

np.random.seed(13)
#
# s = Stockfish(depth=20, parameters=stockfish_conf)
# s.set_fen_position(fen)
# lines = s.get_top_moves_lines(5)
# for move in lines:
#     print(move)
#
# print(len(lines))


num_times = 5
mean = 0

for i in range(num_times):
    analyser_1 = analyzer_recursive_depth_pruned(
        depth=5,
        stockfish_depth=18,
        num_variation=5,
        stockfish_config=stockfish_conf,
        white_skill_level=20,
        black_skill_level=20,
        nnue="true"
    )
    res = analyser_1.evaluate(fen)
    print(f"i: {res}")
    mean += res

print(mean / num_times)