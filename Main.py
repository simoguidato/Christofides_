import tsplib95
import networkx as nx
import time
from networkx.algorithms import matching
from itertools import combinations
import multiprocessing

start_time = time.process_time()


def load_tsp_problem(filename):
    return tsplib95.load(filename)


def christofides_tsp(input_graph):
    # Step 1: Calcolo del Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(input_graph)

    # Step 2: Trovare i vertici con grado dispari nel MST
    odd_degree_nodes = [v for v, degree in mst.degree() if degree % 2 == 1]

    # Step 3: Matching perfetto a costo minimo
    subgraph = input_graph.subgraph(odd_degree_nodes)
    matching_edges = matching.min_weight_matching(subgraph, maxcardinality=True)

    # Step 4: Creare un multigrafo aggiungendo gli spigoli del matching al MST
    multigraph = nx.MultiGraph(mst)
    multigraph.add_edges_from(matching_edges)

    # Step 5: Trovare un ciclo Euleriano nel multigrafo
    eulerian_circuit = list(nx.eulerian_circuit(multigraph))

    # Step 6: Convertire il ciclo Euleriano in un ciclo Hamiltoniano (rimuovere i duplicati)
    tspath = []
    visited = set()
    for u, v in eulerian_circuit:
        if u not in visited:
            tspath.append(u)
            visited.add(u)
        if v not in visited:
            tspath.append(v)
            visited.add(v)
    tspath.append(tspath[0])

    return tspath


def calcola_costo(h_graph, path):
    costo = 0
    for i in range(len(path) - 1):
        costo += h_graph[path[i]][path[i + 1]]['weight']
    return costo


def create_graph(tsp_problem):
    new_graph = nx.Graph()
    nodes = list(tsp_problem.get_nodes())
    for u, v in combinations(nodes, 2):
        new_graph.add_edge(u, v, weight=problem.get_weight(u, v))
    return new_graph


def solve_tsp_with_timeout(graph, timeout_sec):
    pool = multiprocessing.Pool(1)  # Crea un singolo processo nel pool
    async_result = pool.apply_async(christofides_tsp, (graph,))

    try:
        tsp_path = async_result.get(timeout=timeout_sec)  # Attendi il risultato con un timeout
    except multiprocessing.TimeoutError:
        print(f"Timeout di {timeout_sec} secondi raggiunto. Restituisco un percorso parziale.")
        tsp_path = []  # Restituisci un percorso vuoto o parziale se il timeout viene raggiunto
    finally:
        pool.terminate()  # Termina il processo nel pool

    return tsp_path


def parse_file(filename):
    risultati = {}
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    value_str = parts[1].strip()
                    try:
                        value = int(value_str.split()[0])
                        risultati[key] = value
                    except ValueError:
                        print(f"Valore non valido per la chiave '{key}': {value_str}")
    return risultati


def is_euclidean_tsp(prob):
    nodes = list(prob.get_nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                a, b, c = nodes[i], nodes[j], nodes[k]
                d_ab = problem.get_weight(a, b)
                d_ac = problem.get_weight(a, c)
                d_bc = problem.get_weight(b, c)

                if d_ab > d_ac + d_bc or d_ac > d_ab + d_bc or d_bc > d_ab + d_ac:
                    print(f"La disuguaglianza triangolare è violata per i nodi {a}, {b}, {c}")
                    return False
    return True


def cerca_valore(list, chiave):
    if chiave in list:
        print(f"Il valore corrispondente a '{chiave}' è: {list[chiave]}")
        if is_euclidean_tsp(problem):
            print("è euclideo")
        else:
            print("non è euclideo")
            if tsp_cost / list[chiave] > 3 / 2:
                print(f"non lo rispetta '{tsp_cost / list[chiave]}'.")
            else:
                print(f"Il problema TSP rispetta il rapporto di approssimazione, {tsp_cost / list[chiave]}.")
        error = (tsp_cost - list[chiave]) / tsp_cost
        report('/home/simo/PycharmProjects/Christofides/Report', tsp_cost, tsp_cost / list[chiave], error,
               is_euclidean_tsp(problem))
        print(f"Errore: {error}")
    else:
        return f"La chiave '{chiave}' non è stata trovata."


def report(file, tsp_value, approx, err, bool):
    with open(file, 'a') as file:
        file.write(
            f"{chiave_da_cercare}: Costo del percorso approssimato TSP: {tsp_value}, Rapporto di approssimazione: {approx}, Errore: {err}, Is Euclidean: {bool}\n")


def time_rep(file, time):
    with open(file, 'a') as f:
        f.write(f"{chiave_da_cercare}: {time}\n")


# Main
tsp_file = 'TSP/pla7397.tsp'
problem = load_tsp_problem(tsp_file)
graph = create_graph(problem)

# Risolvi con un timeout di 45 minuti (2700 secondi)
tsp_path = solve_tsp_with_timeout(graph, timeout_sec=2700)

tsp_cost = calcola_costo(graph, tsp_path)
chiave_da_cercare = 'pla7397'
print("Il percorso approssimato TSP è:")
print(tsp_path)
print(f"Il costo del percorso approssimato TSP è: {tsp_cost}")
filepath = 'SolutionsTSP'
dizionario = parse_file(filepath)

risultato = cerca_valore(dizionario, chiave_da_cercare)
print(risultato)
temp = time.process_time() - start_time
time_rep("Time", temp)
print(f"Tempo totale: {temp}")
