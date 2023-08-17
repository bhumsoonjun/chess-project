
import json
from analyser import *
def load_config(path: str) -> dict:
    with open(path) as file:
        raw_json_string = file.read()
        config = json.loads(raw_json_string)
        return config

def write_res_to_file(path: str, res: Dict):
    with open(path, "w") as file:
        file.write(res.__str__())

if __name__ == '__main__':
    fen = "2b5/pp3kp1/r3q1n1/4p2Q/3RB3/BP5P/P1P2PP1/7K b - - 0 27"
    path = "output"

    states = [(-float("inf"), -20), (-20, -10), (-10, -5), (-5, -3), (-3, -1), (-1, -0.5), (-0.5, 0.5), (0.5, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, float("inf"))]
    states_including_end_points = [-1] + states + [1]
    num_variations = 3
    depth = 5
    stockfish_depth = 20
    config = load_config("stockfish_confg.json")

    model = analyser(states, num_variations, depth, stockfish_depth, config)
    adj, stoch, dist, expectation = model.eval_position(fen)

    print(expectation)

    write_res_to_file(f"{path}/test/adj_", adj)
    write_res_to_file(f"{path}/test/stoch_", pd.DataFrame(stoch, columns=states_including_end_points, index=states_including_end_points))
    write_res_to_file(f"{path}/test/expectation_", expectation)
    write_res_to_file(f"{path}/test/dist_.", dist)




