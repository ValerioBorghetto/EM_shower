import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*
from build_shower.em_shower import*
from tqdm import tqdm

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
        plt.show()
    return meas

def plot_kinds(shower): #plot the interaction kind versus the time that process has occurred
    kinds = list(nx.get_node_attributes(shower, "kind").values())
    labels, counts = np.unique(kinds, return_counts=True)

    plt.bar(labels, counts, color="skyblue")
    plt.ylabel("Occurrence")
    plt.xlabel("Interaction kinds")
    plt.title("Occurrence per interaction kind")
    plt.show()


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
    plt.show()

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
    plt.show()


def shower_study(initial_energy, final_energy, times, energy=True, width=True):
    step=(final_energy-initial_energy)/times
    energy=initial_energy
    levels_array=[]
    err_levels_array=[]
    energy_array=[]
    width_array=[]
    err_width_array=[]
    n_iter=30
    for _ in tqdm(range(times), desc="Simulation"):
        energy_array.append(energy)
        sub_level=[]
        sub_width=[]
        for i in range(n_iter):
            shower, energy_deposed=generate_shower(depth=40, initial_energy=energy, Z=10, initial_particle="electron")
            if energy:
                max_energy = max(energy_deposed)
                max_level = energy_deposed.index(max_energy)    
                sub_level.append(max_level)            
            if width:
                depth = list(nx.get_node_attributes(shower, "step").values())
                levels, counts = np.unique(depth, return_counts=True)
                max_level_index = np.argmax(counts)
                max_depth = levels[max_level_index]  
                sub_width.append(max_depth)  
        mean_level = np.mean(sub_level)
        err_level = np.std(sub_level, ddof=1) / np.sqrt(len(sub_level)) 
        mean_width = np.mean(sub_width)
        err_width = np.std(sub_width, ddof=1) / np.sqrt(len(sub_width)) 
        if mean_level != 0:           
            levels_array.append(mean_level)
            err_levels_array.append(err_level)
        if mean_width != 0:
            width_array.append(mean_width)
            err_width_array.append(err_width)
        energy += step
    
    #Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))  
    # Primo grafico
    ax1.errorbar(energy_array, levels_array, yerr=err_levels_array, fmt='o-', capsize=5, label='Level')
    ax1.set_title("Maximum energy depth vs Initial energy")
    ax1.set_xlabel("Initial energy")
    ax1.set_ylabel("Depth with max. energy deposit")
    ax1.grid(True)

    # Secondo grafico
    ax2.errorbar(energy_array, width_array, yerr=err_width_array, fmt='s--', capsize=5, color='orange', label='Level')
    ax2.set_title("Maximum interaction depth vs Initial energy")
    ax2.set_xlabel("Initial Energy")
    ax2.set_ylabel("Depth with max. #interaction")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    
    

        
    


