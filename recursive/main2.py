from recursive.analyser_recursive import analyser_recursive
from recursive.analyser_recursive_look_ahead import analyser_recursive_look_ahead
import numpy as np

fen = "3r2rk/1p1nqpp1/2p1pn1p/p7/P2PPP2/2NQ1BR1/1P3P1P/6RK w - - 5 21"
stockfish_conf = {
    "Debug Log File": "",
    "Contempt": 0,
    "Min Split Depth": 0,
    "Threads": 1, # More threads will make the engine stronger, but should be kept at less than the number of logical processors on your computer.
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

analyser = analyser_recursive(
    depth=4,
    stockfish_depth=15,
    num_variation=3,
    stockfish_config=stockfish_conf,
    p_black_plat_best=0,
    p_white_play_best=0,
    nnue="true"
)

np.random.seed(1)

analyser_1 = analyser_recursive_look_ahead(
    depth=10,
    stockfish_depth=20,
    num_variation=3,
    stockfish_config=stockfish_conf,
    white_skill_level=15,
    black_skill_level=15,
    nnue="true"
)

res_1 = analyser_1.evaluate(fen)

print(f"res 1: {res_1}")

print(res_1)

# res = analyser.evaluate(fen)
#
# print(res)