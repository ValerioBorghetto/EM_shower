import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*

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
        #plt.savefig(f"{kind}_centrality.pdf")  # salva in PDF
        plt.savefig("centrality.pdf")
    return meas

def plot_kinds(shower): #plot the interaction kind versus the time that process has occurred
    kinds = list(nx.get_node_attributes(shower, "kind").values())
    labels, counts = np.unique(kinds, return_counts=True)

    plt.bar(labels, counts, color="skyblue")
    plt.ylabel("Occurrence")
    plt.xlabel("Interaction kinds")
    plt.title("Occurrence per interaction kind")
    plt.savefig("kinds.pdf")


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
    plt.savefig("matrix_study.pdf")

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
    plt.savefig("width.pdf")