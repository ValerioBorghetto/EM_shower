import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*
from build_shower.em_shower import*
from tqdm import tqdm
import pandas as pd

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

