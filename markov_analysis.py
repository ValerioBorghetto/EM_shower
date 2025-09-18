import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*
from build_shower.em_shower import*
from tqdm import tqdm
import pandas as pd
import seaborn as sns
from collections import defaultdict, Counter
from scipy.stats import chi2

def markov_plot(markov_dic):
    df = pd.DataFrame(markov_dic).T  
    plt.figure(figsize=(8,6))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="viridis", cbar=True)
    plt.xlabel("Final state", fontsize=12)
    plt.ylabel("Initial state", fontsize=12)
    plt.title("Average transition matrix over all nodes", fontsize=14)
    plt.tight_layout()
    plt.savefig("plots/Average_markov.pdf")
    plt.show()
    plt.close()

def analyze_markov_vs_shower(depth=30, initial_energy=1000, material_Z=20, n_avg=5000):
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

    """# Plot average trajectory counts with error bars
    plt.figure(figsize=(8, 6))
    plt.bar(state_means.keys(), state_means.values(), yerr=[state_stds[s] for s in state_means.keys()],
            color="skyblue", edgecolor="black", capsize=5)
    plt.xlabel("States")
    plt.ylabel("Average appearances")
    plt.title(f"Average state frequency over {n_avg} Avg Markov simulation ({n_nodes} steps)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()
    """

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

    """plt.figure(figsize=(8,6)) #(figsize=(10,6))
    plt.bar(x - width/2, traj_values, width, yerr=traj_err, label='Avg Markov simulation', 
            color='skyblue', edgecolor='black', capsize=5)
    plt.bar(x + width/2, shower_values, width, yerr=shower_err, label='Shower', 
            color='salmon', edgecolor='black', capsize=5)
    plt.xticks(x, types)
    plt.ylabel("Average appearances")
    plt.title("Comparison of average appearances per type with error bars")
    plt.tight_layout()
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("plots/markov_vs_shower_appereances.pdf")
    plt.show()
    plt.close()
    """
    
    # Relative frequencies with error bars
    traj_freq = np.array(traj_values) / np.sum(traj_values)
    shower_freq = np.array(shower_values) / np.sum(shower_values)
    # Errori normalizzati
    traj_freq_err = np.array(traj_err) / np.sum(traj_values)
    shower_freq_err = np.array(shower_err) / np.sum(shower_values)
    
    # Chi-2
    chi2_bins = (traj_freq - shower_freq)**2 / (traj_freq_err**2 + shower_freq_err**2)
    chi2_sum = np.sum(chi2_bins)
    dof = len(shower_freq) - 1
    chi2_reduced = chi2_sum / dof

    print("Chi2 per bin:", chi2_bins)
    print("Chi2 totale:", chi2_sum)
    print("Chi2 ridotto:", chi2_reduced)
    p_value = 1 - chi2.cdf(chi2_sum, df=dof)
    print("p-value:", p_value)

    plt.figure(figsize=(8, 6)) #10,5
    plt.bar(x - width/2, traj_freq, width, yerr=traj_freq_err, label='Avg Markov simulation',
            color='skyblue', edgecolor='black', capsize=5, error_kw={'elinewidth':1.2, 'ecolor':'black'})
    plt.bar(x + width/2, shower_freq, width, yerr=shower_freq_err, label='Shower',
            color='salmon', edgecolor='black', capsize=5, error_kw={'elinewidth':1.2, 'ecolor':'black'})
    plt.xticks(x, types)
    plt.xlabel("Interaction type", fontsize = 12 )
    plt.ylabel("Relative frequency", fontsize = 12)
    plt.title("Relative frequencies per interaction type", fontsize = 14)
    plt.tight_layout()
    proxy_traj = mpatches.Patch(color='skyblue', label='Avg Markov simulation', edgecolor='none')
    proxy_shower = mpatches.Patch(color='salmon', label='Shower', edgecolor='none')
    plt.legend(handles=[proxy_traj, proxy_shower], framealpha=0.8, edgecolor='gray')
    #plt.legend(framealpha=0.8, edgecolor='gray')
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("plots/markov_vs_shower_frequencies.pdf")
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
    plt.tight_layout()
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
            plt.figure(figsize=(8, 6)) #(10, 6)
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
            plt.figure(figsize=(8,6)) #(8,5)
            plt.bar(labels, values, color='skyblue')
            plt.ylabel('Degree')
            plt.xlabel('Process')
            title=kind + " degree for each process"
            plt.title(title)
            #plt.savefig("plots/centrality.pdf")
            plt.tight_layout()
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
    states = ['brems', 'pp', 'ann', 'stay_e', 'stay_p']

    G = nx.DiGraph()
    #G.add_nodes_from(states)
    G.add_edges_from([
    ("brems", "brems1"), ("brems", "brems2"), ("brems3", "brems"),
    ("ann", "ann1"), ("ann", "ann2"), ("ann3", "ann"), ("ann4", "ann"),
    ("stay_e", "stay_e1"), ("stay_e2", "stay_e"),
    ("pp", "pp1"), ("pp", "pp2"), ("pp3", "pp"),
    ("stay_p", "stay_p1"), ("stay_p2", "stay_p")
    ])

    pos = {
    "brems": (-2, 0), "ann": (-1, 0), "stay_e": (0, 0), "pp": (1, 0), "stay_p": (2, 0),
    "brems1": (-2.5, -1), "brems2": (-1.5, -1), "brems3": (-2, 1),
    "ann1": (-1.5, -1), "ann2": (-0.5, -1), "ann3": (-1.5, 1), "ann4": (-0.5, 1),
    "stay_e1": (0, -1), "stay_e2": (0, 1),
    "pp1": (0.5, -1), "pp2": (1.5, -1), "pp3": (1, 1),
    "stay_p1": (2, -1), "stay_p2": (2, 1)
}

    # Colori personalizzati con questo DIZIONARIO
    color_map = {
        "brems": "#FF0000",    # rosso  
        "ann":   "#FFA500",    # arancione
        "stay_e":"#FFD700",    # giallo
        "pp":    "#003FAB",    # blu scuro
        "stay_p": "#87CEEB",    # azzurro
    }
    #node_colors = [color_map.get(n, "white") for n in states] #colori come vettore
    node_colors = [color_map[n] for n in states]

    # Disegno del grafo con frecce orientate verso i nodi più bassi
    plt.figure(figsize=(10,6)) #(10,5)
   
    # Disegna solo nodi principali
    nx.draw_networkx_nodes(G, pos, nodelist=states, node_size=1500, node_color=node_colors)

    # Etichette solo nodi principali
    nx.draw_networkx_labels(G, pos, labels={n: n for n in states}, font_size=12, font_color="black")

    # Disegna solo archi tra nodi principali e secondari (gli archi restano, i nodi secondari no)
    nx.draw_networkx_edges(G, pos, arrows=True, edge_color="#555555", width=2, arrowsize=20, connectionstyle="arc3,rad=0.0")
    plt.tight_layout() #adatta alla canvas

    readable_names = {
        "brems": "Bremsstrahlung",
        "pp": "Pair Production",
        "ann": "Annihilation",
        "stay_e": "No e interaction",
        "stay_p": "No p interaction"
    }
    patches = [mpatches.Patch(color=color_map[state], label=readable_names[state]) for state in color_map]

    plt.tight_layout()
    plt.legend(handles=patches, loc  = 'upper left', bbox_to_anchor=(0, 1), framealpha=0.8, edgecolor='gray') #, title="Interaction"
    plt.axis("off") #rimuove assi x, y e la cornice
    plt.tight_layout()
    plt.savefig("plots/Interactions.pdf")
    plt.show()
    plt.close()


"""nx.draw(
    G, pos, nodelist = states,
    with_labels=True,
    node_size=1500,
    node_color=node_colors,
    font_size=12,
    font_color="black",
    #font_weight="bold",
    edge_color="#555555",  
    width=2,
    arrows=True,
    arrowsize=20,
    connectionstyle="arc3,rad=0.0"
    )
    pos = { #necessario definire posizioni anche dei secondari altrimenti non posso disegnare gli edges
    "brems": (-4, 0), "ann": (-2, 0), "stay_e": (0, 0), "pp": (2, 0), "stay_p": (4, 0),
    "brems1": (-5, -0.5), "brems2": (-3, -0.5), "brems3": (-4, 0.5),
    "ann1": (-3, -0.5), "ann2": (-1, -0.5), "ann3": (-3, 0.5), "ann4": (-1, 0.5),
    "stay_e1": (0, -0.5), "stay_e2": (0, 0.5),
    "pp1": (1, -0.5), "pp2": (3, -0.5), "pp3": (2, 0.5),
    "stay_p1": (4, -0.5), "stay_p2": (4, 0.5)
}
    """
