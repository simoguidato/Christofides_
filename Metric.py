import itertools
from functools import lru_cache

import networkx as nx
import tsplib95

# Cache per evitare di ricalcolare le distanze tra le stesse città
@lru_cache(maxsize=None)
def get_distance(problem, city1, city2):
    return problem.get_weight(city1, city2)

def create_graph(tsp_problem):
    new_graph = nx.Graph()
    nodes = list(tsp_problem.get_nodes())
    for u, v in itertools.combinations(nodes, 2):
        new_graph.add_edge(u, v, weight=tsp_problem.get_weight(u, v))
    return new_graph


def is_euclidean_tsp(problem):
    nodes = list(problem.get_nodes())

    # Verifichiamo la disuguaglianza triangolare per ogni combinazione di tre nodi
    for a, b, c in itertools.combinations(nodes, 3):
        d_ab = get_distance(problem, a, b)
        d_ac = get_distance(problem, a, c)
        d_bc = get_distance(problem, b, c)

        # Verifica delle tre possibili versioni della disuguaglianza triangolare
        if d_ab > d_ac + d_bc or d_ac > d_ab + d_bc or d_bc > d_ab + d_ac:
            print(f"La disuguaglianza triangolare è violata per i nodi {a}, {b}, {c}")
            return False
    return True
def load_tsp_problem(filename):
    return tsplib95.load(filename)

def Main():
    if(is_euclidean_tsp(problem)):
        print("è metrico")

    else:
        print("non è metrico")

tsp_file = 'TSP/pla7397.tsp'
problem = load_tsp_problem(tsp_file)
graph = create_graph(problem)

if __name__ == "__main__":
    Main()
