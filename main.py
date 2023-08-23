
import json
from analyser_base import *
from analyser_sampler import analyser_sampler


def load_config(path: str) -> dict:
    with open(path) as file:
        raw_json_string = file.read()
        config = json.loads(raw_json_string)
        return config

def write_res_to_file(path: str, res: Dict):
    with open(path, "w") as file:
        file.write(res.__str__())

if __name__ == '__main__':
    fen = "1r2qb1k/1p4r1/pP1P1nn1/2BBp2p/2N1Pp2/5P2/6Pp/R2Q1R1K b - - 3 30"
    path = "output"

    states = [(-float("inf"), -20), (-20, -10), (-10, -5), (-5, -3), (-3, -1), (-1, -0.5), (-0.5, 0.5), (0.5, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, float("inf"))]
    num_variations = 3
    depth = 4
    stockfish_depth = 20
    config = load_config("stockfish_confg.json")

    model = analyser_sampler(
        states,
        white_elo=2400,
        black_elo=2700,
        sampling_amount=5,
        num_variations=num_variations,
        white_depth=20,
        black_depth=20,
        analysis_depth=depth,
        stockfish_depth=stockfish_depth,
        stockfish_config=config
    )
    adj, stoch, dist, expectation = model.eval_position(fen)

    print(expectation)

    write_res_to_file(f"{path}/test1/adj_", adj)
    write_res_to_file(f"{path}/test1/stoch_", pd.DataFrame(stoch, columns=model.states, index=model.states))
    write_res_to_file(f"{path}/test1/expectation_", expectation)
    write_res_to_file(f"{path}/test1/dist_", dist)




