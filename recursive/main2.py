from recursive.analyser_recursive import analyser_recursive

fen = "2kr3r/Qpp1n1p1/2b2pp1/4b3/8/6N1/PPP2P2/R1B2RK1 b - - 0 24"
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

analyser = analyser_recursive(
    depth=4,
    stockfish_depth=15,
    num_variation=3,
    stockfish_config=stockfish_conf,
    nnue="false"
)

print(analyser.evaluate(fen))
