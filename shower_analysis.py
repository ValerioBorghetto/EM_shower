import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*
from build_shower.em_shower import*
from tqdm import tqdm
import pandas as pd
import matplotlib.patches as mpatches

def plot_kinds(shower): #plot the interaction kind versus the time that process has occurred
    kinds = list(nx.get_node_attributes(shower, "kind").values())
    labels, counts = np.unique(kinds, return_counts=True)
    plt.bar(labels, counts, color="skyblue")
    plt.ylabel("Occurrence")
    plt.xlabel("Interaction kinds")
    plt.title("Occurrence per interaction kind")
    plt.savefig("plots/interactions_occurence.pdf")
    plt.show()
    plt.close()

#adjacency matrix plot and study 
def adj_matrix_study(graph):
    #get and plot adj matrix
    adjacency=nx.adjacency_matrix(graph)
    adjacency=adjacency.toarray()                 
    plot_adjacency_matrix(adjacency)
    #get and plot incidence matrix
    inc = nx.incidence_matrix(graph, oriented=True).toarray()
    plot_adjacency_matrix(inc, title="Incidence Matrix", labels=[])
    #plot how many nodes with that degree level
    in_degree = np.sum(adjacency, axis=0) #it sums for each coloumn over all the raws
    plt.hist(in_degree, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8,edgecolor='black')
    plt.xticks([0, 1, 2])
    plt.xlabel("Degree")
    plt.ylabel("N. nodes with that degree")
    plt.savefig("plots/degree.pdf")
    plt.show()
    plt.close()

#plot the width of the shower (the number of interactions per level)
def plot_width(shower):
    steps = list(nx.get_node_attributes(shower, "step").values())
    levels, counts = np.unique(steps, return_counts=True)
    plt.plot(levels, counts, marker="o")
    plt.title("Number of interactions per level")
    plt.xlabel("Depth")
    plt.ylabel("Number of interactions")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/depth.pdf")
    plt.show()
    
#It counts the different interaction kinds over the different shower levels
def level_count(shower):
    steps = list(nx.get_node_attributes(shower, "step").values())
    kinds = list(nx.get_node_attributes(shower, "kind").values())
    df = pd.DataFrame({'level': steps, 'interaction': kinds})
    tabella = pd.crosstab(df['level'], df['interaction'])

    color_map = {
        "brems": "#FF0000",    # rosso  
        "ann":   "#FFA500",    # arancione
        "stay_e":"#FFD700",    # giallo
        "pp":    "#003FAB",    # blu scuro
        "stay_p": "#87CEEB",    # azzurro
    }
    colors = [color_map.get(col, "#000000") for col in tabella.columns]
    tabella.plot(kind='line', figsize=(12, 6), marker='o', color=colors)
    plt.title("Number of different interactions over shower depth")
    plt.xlabel("Depth")
    plt.ylabel("Counts")

    readable_names = {
        "brems": "Bremsstrahlung",
        "pp": "Pair Production",
        "ann": "Annihilation",
        "stay_e": "No e interaction",
        "stay_p": "No p interaction"
    }
    patches = [mpatches.Patch(color=color_map[col], label=readable_names[col]) for col in tabella.columns]
    plt.legend(handles=patches, title="Interaction")
    plt.grid(True)
    plt.savefig("plots/interaction_vs_depth.pdf")
    plt.show()
    plt.close()

#it studies the width and the energy deposited in different showers
def shower_study(initial_energy, final_energy, times, energy=True, width=True):
    def mean_err(data):
        return np.mean(data), np.std(data, ddof=1) / np.sqrt(len(data))
    energies = np.linspace(initial_energy, final_energy, times)
    levels, levels_err, widths, widths_err = [], [], [], []
    n_iter = 100
    for e in tqdm(energies, desc="Simulation"):
        lvl, wth = [], []
        for _ in range(n_iter):
            shower, ener, _ = generate_shower(depth=40, initial_energy=e, Z=10, initial_particle="electron")
            if energy:
                lvl.append(np.argmax(ener))
            if width:
                steps = list(nx.get_node_attributes(shower, "step").values())
                wth.append(np.bincount(steps).argmax())
            
        if energy and lvl:
            m, err = mean_err(lvl)
            if m: levels.append(m); levels_err.append(err)
        if width and wth:
            m, err = mean_err(wth)
            if m: widths.append(m); widths_err.append(err)

    def plot(ax, x, y, yerr, title, xlabel, ylabel, fmt, color):
        ax.errorbar(x, y, yerr=yerr, fmt=fmt, capsize=5, color=color)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    plot(ax1, energies, levels, levels_err, "Max energy depth vs Initial energy",
         "Initial energy", "Depth with max energy deposit", 'o-', 'blue')
    plot(ax2, energies, widths, widths_err, "Max interaction depth vs Initial energy",
         "Initial energy", "Depth with max interactions", 's--', 'orange')
    plt.tight_layout()
    plt.savefig("plots/depth_vs_energy.pdf")
    plt.show()
    plt.close()

#studies the number of node, the number of edges, the depth over different energies
def study_properties(initial_energy, final_energy, times):
    def mean_err(data):
        return np.mean(data), np.std(data, ddof=1) / np.sqrt(len(data))

    energies = np.linspace(initial_energy, final_energy, times)
    depth, depth_err, node, node_err, edges, edges_err = [], [], [], [], [], []
    n_iter = 100
    for e in tqdm(energies, desc="Simulation 2"):
        dpt, nd, edg = [],[],[]
        for i in range(n_iter):
            shower, _, _ = generate_shower(depth=40, initial_energy=e, Z=10, initial_particle="electron")
            n = shower.number_of_nodes()
            l = shower.number_of_edges()
            s = max(list(nx.get_node_attributes(shower, "step").values()))
            dpt.append(s)
            nd.append(n)
            edg.append(l)
        m, err = mean_err(dpt)
        depth.append(m); depth_err.append(err)
        m, err = mean_err(nd)
        node.append(m); node_err.append(err)
        m, err = mean_err(edg)
        edges.append(m); edges_err.append(err)

    def plot(ax, x, y, yerr, title, xlabel, ylabel, style, color):
        ax.errorbar(x, y, yerr=yerr, fmt=style, capsize=5, color=color)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    plot(ax1, energies, node, node_err, "# Nodes vs Initial energy", "Initial energy", "# Nodes", 'o-', "orange")
    plot(ax2, energies, edges, edges_err, "# Edges vs Initial energy", "Initial energy", "# Edges", 's--', "orange")
    plt.tight_layout()
    plt.savefig("plots/nodes_edges_vs_energy.pdf")
    plt.show()
    plt.close()  
    fig, ax = plt.subplots(figsize=(8, 6))
    plot(ax, energies, depth, depth_err, "Shower depth vs Initial energy", "Initial energy", "Depth", 'o-', "blue")
    plt.tight_layout()
    plt.savefig("plots/depth_vs_energy.pdf")
    plt.show()
    plt.close()

def network_degree(initial_energy, n_iter=50):
    avg_clustering_list = []
    deg_assortativity_list = []
    avg_degree_list = []
    branching_factor_list = []
    assort_kind_list = []
    diameter_list = []
    for i in range(n_iter):
        shower, _, _ = generate_shower(depth=30, initial_energy=initial_energy, Z=20, initial_particle="electron")
        avg_clustering_list.append(nx.average_clustering(shower))
        deg_assortativity_list.append(nx.degree_assortativity_coefficient(shower))
        degrees = dict(shower.degree())
        avg_degree_list.append(sum(degrees.values()) / shower.number_of_nodes())
        internal_nodes = [n for n, d in shower.degree() if d > 1]
        if internal_nodes:
            branching_factor_list.append(sum(shower.degree(n) for n in internal_nodes) / (1.6 * len(internal_nodes)))
        else:
            branching_factor_list.append(0)
        assort_kind_list.append(nx.attribute_assortativity_coefficient(shower, "kind"))
        if nx.is_connected(shower.to_undirected()):
            diameter_list.append(nx.diameter(shower.to_undirected()))
        else:
            largest_cc = max(nx.connected_components(shower.to_undirected()), key=len)
            diameter_list.append(nx.diameter(shower.subgraph(largest_cc)))
    print("Average over", n_iter, "networks:")
    print("Average clustering:", sum(avg_clustering_list)/n_iter)
    print("Degree assortativity:", sum(deg_assortativity_list)/n_iter)
    print("Average degree:", sum(avg_degree_list)/n_iter)
    print("Branching factor:", sum(branching_factor_list)/n_iter)
    print("Attribute assortativity:", sum(assort_kind_list)/n_iter)
    print("Diameter:", sum(diameter_list)/n_iter)


"""
Interpreting the result: (Chatgpt)

(N.B. Questi valori sono più o meno stabili indipendentemente dall'energia iniziale. Tranne il diametro, che sarebbe il numero di steps già studiato)
Average clustering: 0.0
Degree assortativity: -0.053517835178208385
Average degree: 2.442962397071613
Branching factor: 1.7970650046098597
Attribute assortativity: -0.19741222213152493
Diameter: 17.34
ChatGPT ha detto:


🔹 Average clustering = 0.0
No triangles exist, confirming the networks are tree-like or acyclic.

🔹 Degree assortativity ≈ -0.054
Slightly negative → mild disassortative mixing.
High-degree nodes tend to connect slightly more often to low-degree nodes. Typical for hierarchical structures.

🔹 Average degree ≈ 2.44
Each node has about 2.44 neighbors on average.
Consistent with sparse tree-like graphs.

🔹 Branching factor ≈ 1.80
Internal nodes have roughly 1.8 children on average.
The trees are narrow and relatively deep rather than very bushy.

🔹 Attribute assortativity ≈ -0.197
Negative → nodes of different “kind” attributes tend to connect more than nodes of the same kind.
Suggests cross-type mixing is prevalent.

🔹 Diameter ≈ 17.34
Longest path between nodes is on average ~17 edges.
Confirms deep tree structure, consistent with narrow branching.

🔹 Overall Interpretation
Your shower networks are tree-like, narrow, and deep.
There’s weak disassortativity by degree and strong cross-kind mixing.
Most nodes are leaves or low-degree nodes, with a few internal nodes acting as hubs/bridges.

"""

    


