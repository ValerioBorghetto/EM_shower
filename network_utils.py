import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

#plot adj. matrix
def plot_adjacency_matrix(adj_matrix, title="Adjacency Matrix", labels=[]): 
    plt.figure(figsize=(6,6))
    plt.imshow(adj_matrix, cmap='gray_r', interpolation='none')
    plt.title(title)
    plt.colorbar(label="Edge Weight")
    # Imposta le etichette per righe e colonne
    plt.xticks(np.arange(len(labels)), labels, rotation=90)  # Etichette per le colonne
    plt.yticks(np.arange(len(labels)), labels)               # Etichette per le righe
    plt.show()


def plot_shower(shower, tree=False, color=False):
    # Setup posizione dei nodi
    pos = nx.nx_agraph.graphviz_layout(shower, prog='dot') if tree else nx.spring_layout(shower)

    # Mappa colori per tipi di kind
    color_map = {
        "brems": "#87CEEB",            # azzurro
        "pp":    "#00008B",            # blu scuro
        "ann":   "#FFD700",            # giallo
        "stay_e":"#FFA500",            # arancione
        "stay_p":"#FFA500"             # arancione
    }

    # Colori per i nodi se richiesto
    if color:
        node_kinds = nx.get_node_attributes(shower, "kind")
        node_colors = [color_map.get(node_kinds.get(n), "#CCCCCC") for n in shower.nodes]
    else:
        node_colors = "skyblue"

    # Disegno del grafo
    plt.figure(figsize=(14, 10))
    nx.draw(
        shower, pos,
        with_labels=False,
        node_color=node_colors,
        edge_color="gray",
        node_size=80,
        font_size=8,
        arrows=True
    )
    plt.title("EM shower", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    # Legenda se color è attivo
    if color:
        for kind, color_val in color_map.items():
            plt.plot([], [], marker="o", color=color_val, label=kind, linestyle="None")
        plt.legend(title="Kind")
        plt.title("Legenda tipi di interazione")
        plt.axis('off')
        plt.show()
    plt.show()


def plot_energy(energy_deposed):
    x = list(range(len(energy_deposed)))
    plt.figure(figsize=(10, 5))
    plt.plot(x, energy_deposed, marker='o', linestyle='-', color='b')
    plt.title('Energy deposed plot')
    plt.xlabel('X0')
    plt.ylabel('Energy (MeV)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

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
    if show:
        print("Measure of ", kind, "centrality :", meas)
        # Plot
        labels = list(meas.keys())
        values = list(meas.values())
        plt.figure(figsize=(8,5))
        plt.bar(labels, values, color='skyblue')
        plt.ylabel('Degree')
        plt.xlabel('Process')
        plt.title('Degree for each process')
        plt.show()
    return meas

#is symmetric?
def is_symmetric(A):
    return np.all(A - A.T == 0)

#is upper triangular
def is_upper_triangular(adj_matrix):
    return np.allclose(adj_matrix, np.triu(adj_matrix))


#adjacency matrix plot study 
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
    print(in_degree)
    plt.hist(in_degree, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8,edgecolor='black')
    plt.xticks([0, 1, 2])
    plt.xlabel("Degree")
    plt.ylabel("N. nodes with that degree")
    plt.show()