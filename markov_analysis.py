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