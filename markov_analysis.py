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
    plt.figure(figsize=(10,5))
   
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

    plt.legend(handles=patches, title="Interaction", loc  = 'upper left', bbox_to_anchor=(0, 1))
    plt.axis("off") #rimuove assi x, y e la cornice
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
