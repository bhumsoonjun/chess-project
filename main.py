
import json
from analyser_base import *
from analyser_sampler import analyser_sampler
import pandas as pd


def load_config(path: str) -> dict:
    with open(path) as file:
        raw_json_string = file.read()
        config = json.loads(raw_json_string)
        return config

def write_res_to_file(path: str, res: Dict):
    with open(path, "w") as file:
        file.write(res.__str__())

if __name__ == '__main__':
    fen = "8/8/4kpp1/3p1b2/p6P/2B5/6P1/6K1 b - - 2 47"
    path = "output"

    states = [(-float("inf"), -20), (-20, -10), (-10, -5), (-5, -3), (-3, -1), (-1, -0.5), (-0.5, 0.5), (0.5, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, float("inf"))]
    num_variations = 3
    depth = 5
    stockfish_depth = 20
    config = load_config("stockfish_confg.json")

    model = analyser_base(
        states,
        num_variations=num_variations,
        analysis_depth=depth,
        stockfish_depth=stockfish_depth,
        stockfish_config=config
    )
    adj, stoch, dist, expectation = model.eval_position(fen)

    print(expectation)

    df = pd.DataFrame(stoch, columns=model.states, index=model.states)

    print(df.to_markdown)
    write_res_to_file(f"{path}/test1/adj_", adj)
    write_res_to_file(f"{path}/test1/stoch_", pd.DataFrame(stoch, columns=model.states, index=model.states))
    write_res_to_file(f"{path}/test1/expectation_", expectation)
    write_res_to_file(f"{path}/test1/dist_", dist)




