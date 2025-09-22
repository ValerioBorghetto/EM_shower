import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

#plot adj. matrix
def plot_adjacency_matrix(adj_matrix, title="Adjacency Matrix", labels=[]): 
    plt.figure(figsize=(8,6)) 
    plt.imshow(adj_matrix, cmap='gray_r', interpolation='none')
    plt.title(title)
    plt.colorbar(label="Edge Weight")
    plt.xticks(np.arange(len(labels)), labels, rotation=90) 
    plt.yticks(np.arange(len(labels)), labels)             
    plt.tight_layout()
    filename = f"plots/{title.replace(' ', '_')}.pdf" 
    plt.savefig(filename)
    plt.show()
    plt.close()

#plot the shower
def plot_shower(shower, tree=False, color=False, size=80):
    pos = nx.nx_agraph.graphviz_layout(shower, prog='dot') if tree else nx.spring_layout(shower)
    color_map = {
        "brems": "#FF0000",    # red  
        "ann":   "#FFA500",    # orange
        "stay_e":"#FFD700",    # yellow
        "pp":    "#003FAB",    # dark blue
        "stay_p": "#87CEEB",    # light blue
    }
    if color:
        node_kinds = nx.get_node_attributes(shower, "kind")
        node_colors = [color_map.get(node_kinds.get(n), "#CCCCCC") for n in shower.nodes]
    else:
        node_colors = "skyblue"

    plt.figure(figsize=(14, 10)) 
    nx.draw(
        shower, pos,
        with_labels=False,
        node_color=node_colors,
        edge_color="gray",
        node_size=size,
        font_size=8,
        arrows=True
    )
    plt.title("EM shower") 
    plt.axis('off')
    plt.tight_layout()
    readable_names = {
        "brems": "Bremsstrahlung",
        "pp": "Pair Production",
        "ann": "Annihilation",
        "stay_e": "No e interaction",
        "stay_p": "No p interaction"
    }
    if color:
        kinds_present = set(nx.get_node_attributes(shower, "kind").values())
        patches = [mpatches.Patch(color=color_map[k], label=readable_names[k]) for k in kinds_present]
        plt.legend(handles=patches, framealpha=0.8, edgecolor='gray', fontsize = 13) #, title="Interaction"
        plt.title("Legenda tipi di interazione")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("plots/shower.pdf")
        plt.show()
        plt.close()
    else:
        plt.savefig("plots/shower2.pdf")

#plot energy deposed vs the step of the tree
def plot_energy(energy_deposed):
    x = np.arange(len(energy_deposed))
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.plot(
        x, energy_deposed, "--",
        color="firebrick", linewidth=2, zorder=1
    )
    ax.plot(
        x, energy_deposed, "D",
        color="darkred", markersize=5,
        label="Deposited Energy", zorder=2
    )
    ax.set_title("Deposited energy per step", fontsize=14) 
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Deposited energy (MeV)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("plots/Energy_deposited.pdf")
    plt.show()
    plt.close()

#is symmetric?
def is_symmetric(A):
    return np.all(A - A.T == 0)

#is upper triangular
def is_upper_triangular(adj_matrix):
    return np.allclose(adj_matrix, np.triu(adj_matrix))



