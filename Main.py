import time
import signal
import tsplib95
import networkx as nx
from networkx.algorithms import matching
from itertools import combinations
from multiprocessing import Pool

# Timeout exception
class TimeoutException(Exception): pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# Funzione per il matching per un sottoinsieme di nodi
def compute_matching(odd_degree_nodes, input_graph):
    subgraph = input_graph.subgraph(odd_degree_nodes)
    return matching.min_weight_matching(subgraph, maxcardinality=True)

# Funzione principale del TSP con parallelizzazione
def christofides_tsp_parallel(input_graph, num_workers=4):
    # Step 1: Calcolo del Minimum Spanning Tree (MST)
    mst = nx.minimum_spanning_tree(input_graph)

    # Step 2: Trovare i vertici con grado dispari nel MST
    odd_degree_nodes = [v for v, degree in mst.degree() if degree % 2 == 1]

    # Suddividi il lavoro in batch per processi paralleli
    chunk_size = max(1, len(odd_degree_nodes) // num_workers)
    node_batches = [odd_degree_nodes[i:i + chunk_size] for i in range(0, len(odd_degree_nodes), chunk_size)]
    
    # Step 3: Calcolo del matching con multiprocessing
    with Pool(num_workers) as pool:
        result = pool.starmap(compute_matching, [(batch, input_graph) for batch in node_batches])
    
    # Combinazione dei risultati del matching
    matching_edges = set()
    for match in result:
        matching_edges.update(match)

    # Step 4: Creare un multigrafo
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

# Funzione per calcolare il costo del percorso
def calcola_costo(graph, path):
    costo = 0
    for i in range(len(path) - 1):
        costo += graph[path[i]][path[i + 1]]['weight']
    return costo

# Funzione per creare il grafo dal problema TSP
def create_graph(tsp_problem):
    new_graph = nx.Graph()
    nodes = list(tsp_problem.get_nodes())
    for u, v in combinations(nodes, 2):
        new_graph.add_edge(u, v, weight=tsp_problem.get_weight(u, v))
    return new_graph

# Funzione per analizzare il file delle soluzioni e restituire i risultati
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

# Funzione per verificare se il TSP è euclideo
def is_euclidean_tsp(prob):
    nodes = list(prob.get_nodes())
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            for k in range(j + 1, len(nodes)):
                a, b, c = nodes[i], nodes[j], nodes[k]
                d_ab = prob.get_weight(a, b)
                d_ac = prob.get_weight(a, c)
                d_bc = prob.get_weight(b, c)

                if d_ab > d_ac + d_bc or d_ac > d_ab + d_bc or d_bc > d_ab + d_ac:
                    print(f"La disuguaglianza triangolare è violata per i nodi {a}, {b}, {c}")
                    return False
    return True

# Funzione per cercare il valore in base alla chiave
def cerca_valore(dizionario, chiave):
    if chiave in dizionario:
        print(f"Il valore corrispondente a '{chiave}' è: {dizionario[chiave]}")
        if is_euclidean_tsp(problem):
            print("è euclideo")
        else:
            print("non è euclideo")
            if tsp_cost / dizionario[chiave] > 3 / 2:
                print(f"Il rapporto approssimativo non è rispettato: {tsp_cost / dizionario[chiave]}.")
            else:
                print(f"Il problema TSP rispetta il rapporto di approssimazione: {tsp_cost / dizionario[chiave]}.")
        error = (tsp_cost - dizionario[chiave]) / tsp_cost
        report('/home/simo/PycharmProjects/Christofides/Report', tsp_cost, tsp_cost / dizionario[chiave], error,
               is_euclidean_tsp(problem))
        print(f"Errore: {error}")
    else:
        return f"La chiave '{chiave}' non è stata trovata."

# Funzione per scrivere il report
def report(file, tsp_value, approx, err, bool):
    with open(file, 'a') as file:
        file.write(
            f"{chiave_da_cercare}: Costo del percorso approssimato TSP: {tsp_value}, Rapporto di approssimazione: {approx}, Errore: {err}, Is Euclidean: {bool}\n")

# Funzione per registrare il tempo
def time_rep(file, time):
    with open(file, 'a') as f:
        f.write(f"{chiave_da_cercare}: {time}\n")

# Main
start_time = time.time()

tsp_file = 'TSP/rl5915.tsp'
problem = tsplib95.load(tsp_file)
graph = create_graph(problem)

# Imposta un timeout globale di 20 minuti (1200 secondi)
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(1200)

try:
    tsp_path = christofides_tsp_parallel(graph)
except TimeoutException:
    print("Timeout globale - restituisco il percorso parziale.")
    tsp_path = []
finally:
    signal.alarm(0)

# Calcolo del costo
tsp_cost = calcola_costo(graph, tsp_path)
chiave_da_cercare = 'rl5915'
print("Il percorso approssimato TSP è:")
print(tsp_path)
print(f"Il costo del percorso approssimato TSP è: {tsp_cost}")

# Lettura del file delle soluzioni e ricerca della chiave
filepath = 'SolutionsTSP'
dizionario = parse_file(filepath)

risultato = cerca_valore(dizionario, chiave_da_cercare)
print(risultato)

# Calcolo e registrazione del tempo
elapsed_time = time.time() - start_time
time_rep("Time", elapsed_time)
print(f"Tempo totale: {elapsed_time:.2f} secondi")

