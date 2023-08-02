
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
    fen = "8/8/2K5/6Q1/4k3/3r4/8/8 w - - 1 44"
    path = "output"

    states = [(-float("inf"), -20), (-20, -10), (-10, -5), (-5, -3), (-3, -1), (-1, -0.5), (-0.5, 0.5), (0.5, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, float("inf"))]
    num_variations = 5
    depth = 5
    stockfish_depth = 15
    config = load_config("stockfish_confg.json")

    model = analyser(states, num_variations, depth, stockfish_depth, config)
    graph, adj, stoch, res_df = model.calculate_best_move(fen)

    for i in range(len(graph)):
        write_res_to_file(f"{path}/test/graph_{i}.json", graph[i])
        write_res_to_file(f"{path}/test/adj_{i}.json", adj[i])
        write_res_to_file(f"{path}/test/stoch_{i}.json", stoch[i])

    res_df.to_csv(f"{path}/test/df_all.csv")



