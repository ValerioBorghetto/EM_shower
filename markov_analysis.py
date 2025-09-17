import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*
from build_shower.em_shower import*
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter

def markov_plot(markov_dic):
    df = pd.DataFrame(markov_dic).T  
    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.xlabel("Final state")
    plt.ylabel("Initial state")
    plt.title("Transition matrix averaged over all interactions")
    plt.savefig("plots/Average_markov.pdf")
    plt.show()
    plt.close()

def analyze_markov_vs_shower(depth=30, initial_energy=300, material_Z=20, n_avg=1000):
    """
    Analyze correspondence between Markov chain avg Markov simulation and shower graphs.
    The transition matrix is now averaged over n_avg showers.
    """
    # Initialize structures for averaged transition matrix
    transition_sum = defaultdict(lambda: defaultdict(float))
    all_states = set()
    
    # Generate n_avg showers to compute average transition matrix
    for _ in tqdm(range(n_avg), "m_vs_s_0"):
        _, _, transition_matrix = generate_shower(depth=depth, initial_energy=initial_energy,
                                                  Z=material_Z, initial_particle="electron")
        for state, transitions in transition_matrix.items():
            all_states.add(state)
            for next_state, prob in transitions.items():
                all_states.add(next_state)
                transition_sum[state][next_state] += prob
    
    # Normalize to get average probabilities
    transition_matrix_avg = {}
    for state in all_states:
        next_states = transition_sum[state]
        total = sum(next_states.values())
        if total > 0:
            transition_matrix_avg[state] = {s: p / total for s, p in next_states.items()}
        else:
            transition_matrix_avg[state] = {s: 1/len(all_states) for s in all_states}  # fallback uniform

    states = list(all_states)
    
    # Determine number of nodes from one shower (to set simulation length)
    shower_example, _, _ = generate_shower(depth=depth, initial_energy=initial_energy,
                                           Z=material_Z, initial_particle="electron")
    n_nodes = shower_example.number_of_nodes()

    # Function to pick next state in the Markov chain
    def next_state(current):
        probs = list(transition_matrix_avg[current].values())
        next_states = list(transition_matrix_avg[current].keys())
        return random.choices(next_states, weights=probs, k=1)[0]

    # Simulate a single Markov chain trajectory
    def simulate_markov_chain(start, steps):
        state = start
        path = [state]
        for _ in range(steps):
            state = next_state(state)
            path.append(state)
        return path

    # Count appearances of each state over all avg Markov simulation
    state_counts = defaultdict(list)
    for _ in range(n_avg):
        trajectory = simulate_markov_chain('brems', steps=n_nodes)
        counts = Counter(trajectory)
        for s in states:
            state_counts[s].append(counts.get(s, 0))

    state_means = {s: np.mean(vals) for s, vals in state_counts.items()}
    state_stds = {s: np.std(vals) for s, vals in state_counts.items()}

    # Plot average trajectory counts with error bars
    plt.figure(figsize=(8, 6))
    plt.bar(state_means.keys(), state_means.values(), yerr=[state_stds[s] for s in state_means.keys()],
            color="skyblue", edgecolor="black", capsize=5)
    plt.xlabel("States")
    plt.ylabel("Average appearances")
    plt.title(f"Average state frequency over {n_avg} Avg Markov simulation ({n_nodes} steps)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Count "kind" from shower graphs
    kind_counts = defaultdict(list)
    for _ in tqdm(range(n_avg), desc="m_vs_s_1"):
        shower, _, _ = generate_shower(depth=depth, initial_energy=initial_energy,
                                       Z=material_Z, initial_particle="electron")
        kinds = list(nx.get_node_attributes(shower, "kind").values())
        labels, counts = np.unique(kinds, return_counts=True)
        for label, count in zip(labels, counts):
            kind_counts[label].append(count)
    
    kind_means = {k: np.mean(v) for k, v in kind_counts.items()}
    kind_stds = {k: np.std(v) for k, v in kind_counts.items()}

    # Compare trajectory vs shower values
    types = list(state_means.keys())
    traj_values = [state_means[t] for t in types]
    traj_err = [state_stds[t] for t in types]
    shower_values = [kind_means.get(t, 0) for t in types]
    shower_err = [kind_stds.get(t, 0) for t in types]

    x = np.arange(len(types))
    width = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - width/2, traj_values, width, yerr=traj_err, label='Avg Markov simulation', 
            color='skyblue', edgecolor='black', capsize=5)
    plt.bar(x + width/2, shower_values, width, yerr=shower_err, label='Shower', 
            color='salmon', edgecolor='black', capsize=5)
    plt.xticks(x, types)
    plt.ylabel("Average appearances")
    plt.title("Comparison of average appearances per type with error bars")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

    # Relative frequencies with error bars
    traj_freq = np.array(traj_values) / np.sum(traj_values)
    shower_freq = np.array(shower_values) / np.sum(shower_values)
    
    traj_freq_err = np.array(traj_err) / np.sum(traj_values)
    shower_freq_err = np.array(shower_err) / np.sum(shower_values)

    plt.figure(figsize=(7,5))
    plt.bar(x - width/2, traj_freq, width, yerr=traj_freq_err, label='Avg Markov simulation',
            color='skyblue', edgecolor='black', capsize=5)
    plt.bar(x + width/2, shower_freq, width, yerr=shower_freq_err, label='Shower',
            color='salmon', edgecolor='black', capsize=5)
    plt.xticks(x, types)
    plt.ylabel("Relative frequency")
    plt.title("Comparison of relative frequencies per type (energy of 300 MeV)")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("plots/markov_vs_shower.pdf")
    plt.show()
    plt.close()

    return state_means, kind_means, traj_freq, shower_freq, transition_matrix_avg



def markov_power(markov_matrix, pow):
    states = list(markov_matrix.keys())
    n = len(states)

    # costruisci la matrice numpy
    mat = np.zeros((n, n))
    for i, s in enumerate(states):
        for j, t in enumerate(states):
            mat[i, j] = markov_matrix[s].get(t, 0.0)

    # eleva la matrice a potenza k
    mat_k = np.linalg.matrix_power(mat, pow)

    # ricostruisci un dict nello stesso formato
    powered_dict = {
        states[i]: {states[j]: mat_k[i, j] for j in range(n)} 
        for i in range(n)
    }
    

    df = pd.DataFrame(powered_dict).T  
    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.xlabel("Final state")
    plt.ylabel("Initial state")
    plt.title(f"Averaged transition matrix raised to the {pow}th")
    plt.savefig("plots/Markov_power.pdf")
    plt.show()
    plt.close()


def stationary_vector(norm):
    # Lista ordinata degli stati
    states = list(norm.keys())

    # Costruzione della matrice
    P = np.array([[norm[row][col] for col in states] for row in states], dtype=float)

    # Normalizzazione riga per riga
    P = P / P.sum(axis=1, keepdims=True)

    def stationary_distribution(P, tol=1e-12, max_iter=10000):
        """
        Calcola il vettore stazionario di una catena di Markov a partire dalla matrice P.
        """
        n = P.shape[0]
        pi = np.ones(n) / n  # distribuzione iniziale uniforme
        
        for _ in range(max_iter):
            pi_next = pi @ P
            if np.linalg.norm(pi_next - pi, 1) < tol:
                return pi_next
            pi = pi_next
        return pi

    # Calcolo del vettore stazionario
    pi = stationary_distribution(P)

    # Stampa risultato con etichette
    print("Vettore stazionario:")
    for s, val in zip(states, pi):
        print(f"{s}: {val:.4f}")
    print("Somma =", pi.sum())

#centrality measures
def centrality_meas(graph, kind="in_degree", show=True):
    if kind=="in_degree": #quanto capita quel nodo
        meas = nx.in_degree_centrality(graph)
    if kind=="out_degree": #quanto produce quel nodo
        meas = nx.out_degree_centrality(graph)
    if kind=="betweenness": #betweennes centrality
        meas = nx.betweenness_centrality(graph, normalized=False)
    if kind=="eigenvector": #eigenvector centrality
        meas = nx.pagerank(graph, weight='weight')
    if kind=="flow betweenness":
        undirect = graph.to_undirected()
        meas = nx.current_flow_betweenness_centrality(undirect, normalized=True)
    if kind=="random walk": #shows the degree of importance of the connections (e.g. brems-pp = 0.03 implies low importance, i.e. unlikely that brems produces pp in a few passages)
        graph=graph.to_undirected()
        meas=nx.edge_current_flow_betweenness_centrality(graph, weight='weight')
    if show:
        print("Measure of ", kind, "centrality :", meas)
        # Plot
        if kind == "random walk":
            labels = [f"{a}-{b}" for a, b in meas.keys()]
            values = list(meas.values())
            plt.figure(figsize=(10, 6))
            plt.bar(labels, values)
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("Random Walk Centrality")
            plt.title("Random Walk Centrality by Node Pair")
            plt.tight_layout()
            plt.savefig(f"plots/centrality_{kind.replace(' ', '_')}.pdf")
            plt.show()
        else:
            labels = list(meas.keys())
            values = list(meas.values())
            plt.figure(figsize=(8,5))
            plt.bar(labels, values, color='skyblue')
            plt.ylabel('Degree')
            plt.xlabel('Process')
            title=kind + " degree for each process"
            plt.title(title)
            #plt.savefig("plots/centrality.pdf")
            plt.savefig(f"plots/centrality_{kind.replace(' ', '_')}.pdf")
            plt.show()
            plt.close()
    return meas

def average_markov(markov_array):
    states = list(markov_array[0].keys())
    inter_number = len(markov_array)
    avg_matrix={s: {t: 0.0 for t in states} for s in states}

    for matrix in markov_array:
        for s in states:
            for t in states:
                avg_matrix[s][t] += matrix[s][t]

    for s in states:
        for t in states:
            avg_matrix[s][t] /= inter_number
    return avg_matrix

def interaction_show():
    G = nx.DiGraph()
    G.add_edges_from([
    ("L0", "L1"), ("L0", "L2"), ("L3", "L0"),
    ("R0", "R1"), ("R0", "R2"), ("R3", "R0"), ("R4", "R0"),
    ("C0", "C1"), ("C2", "C0"),
    ("P0", "P1"), ("P0", "P2"), ("P3", "P0"),
    ("D0", "D1"), ("D2", "D0")
    ])


    # Posizioni: prima L, poi R, poi C, poi P, poi D
    pos = {
    # L group
    "L0": (-4, 0), "L1": (-5, -1), "L2": (-3, -1), "L3": (-4, 1),
    # R group
    "R0": (-2, 0), "R1": (-3, -1), "R2": (-1, -1), "R3": (-3, 1), "R4": (-1, 1),
    # C group
    "C0": (0, 0), "C1": (0, -1), "C2": (0, 1),
    # P group
    "P0": (2, 0), "P1": (1, -1), "P2": (3, -1), "P3": (2, 1),
    # D group
    "D0": (4, 0), "D1": (4, -1), "D2": (4, 1)
    }


    # Colori personalizzati
    node_colors = []
    for n in G.nodes():
        if n == "L0":
            node_colors.append("#FF0000") # rosso
        elif n == "C0":
            node_colors.append("#FFD700") # oro
        elif n == "R0":
            node_colors.append("#FFA500") # arancione
        elif n == "P0":
            node_colors.append("#003FAB") # blu scuro
        elif n == "D0":
            node_colors.append("#87CEEB") # azzurro
        else:
            node_colors.append("white")

    labels = {
    "L0": "Brems",
    "C0": "Stay_e",
    "R0": "Ann",
    "P0": "pp",
    "D0": "Stay_p"
    }
    # Disegno del grafo con frecce orientate verso i nodi più bassi
    plt.figure(figsize=(10,5))
    nx.draw(
    G, pos,
    with_labels=False,
    node_size=1500,
    node_color=node_colors,
    font_size=10,
    font_color="black",
    edge_color="#555555",
    width=2,
    arrows=True,
    arrowsize=20,
    connectionstyle="arc3,rad=0.0"
    )
    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_color="black")
    plt.savefig("plots/Interactions.pdf")
    plt.axis("off")
    plt.show()
    plt.close()

