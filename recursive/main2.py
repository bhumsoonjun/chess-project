from recursive.analyser_recursive import analyser_recursive
from recursive.analyser_recursive_look_ahead import analyser_recursive_look_ahead
import numpy as np

fen = "r2q1rk1/pbb2ppp/4p3/3pn3/NP1PnP2/3B4/PB4PP/R2Q1RK1 w - - 0 17"
stockfish_conf = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 8, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
    "Ponder": "false",
    "Hash": 128, # Default size is 16 MB. It's recommended that you increase this value, but keep it as some power of 2. E.g., if you're fine using 2 GB of RAM, set Hash to 2048 (11th power of 2).
    "MultiPV": 1,
    "Skill Level": 20,
    "Move Overhead": 10,
    "Minimum Thinking Time": 0,
    "Slow Mover": 100,
    "UCI_Chess960": "false",
    "UCI_LimitStrength": "false",
    "UCI_Elo": 3500
}

np.random.seed(1)

# analyser = analyser_recursive(
#     depth=4,
#     stockfish_depth=15,
#     num_variation=3,
#     stockfish_config=stockfish_conf,
#     p_black_plat_best=0,
#     p_white_play_best=0,
#     nnue="true"
# )

np.random.seed(1)

analyser_1 = analyser_recursive_look_ahead(
    depth=6,
    stockfish_depth=22,
    num_variation=3,
    stockfish_config=stockfish_conf,
    white_skill_level=20,
    black_skill_level=20,
    nnue="true"
)

res_1 = analyser_1.evaluate(fen)

print(f"res 1: {res_1}")

print(res_1)

# res = analyser.evaluate(fen)
#
# print(res)