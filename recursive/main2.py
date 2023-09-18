from recursive.analyser_recursive import analyser_recursive

fen = "5b1k/1p1n4/1P1P4/p3p2q/2B1Pp1n/R4PrP/5R1K/4BQ2 b - - 0 39"
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
    stockfish_config=stockfish_conf
)

print(analyser.evaluate(fen))
