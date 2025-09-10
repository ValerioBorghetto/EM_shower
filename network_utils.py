import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

#plot adj. matrix
def plot_adjacency_matrix(adj_matrix, title="Adjacency Matrix", labels=[]): 
    plt.figure(figsize=(6,6))
    plt.imshow(adj_matrix, cmap='gray_r', interpolation='none')
    plt.title(title)
    plt.colorbar(label="Edge Weight")
    # Imposta le etichette per righe e colonne
    plt.xticks(np.arange(len(labels)), labels, rotation=90)  # Etichette per le colonne
    plt.yticks(np.arange(len(labels)), labels)               # Etichette per le righe

    filename = f"plots/{title.replace(' ', '_')}.pdf" 
    plt.savefig(filename) #Adiacency matrix
    plt.show()
    plt.close()

def plot_shower(shower, tree=False, color=False, size=80):
    # Setup posizione dei nodi
    pos = nx.nx_agraph.graphviz_layout(shower, prog='dot') if tree else nx.spring_layout(shower)

    # Mappa colori per tipi di kind
    color_map = {
        "brems": "#FF0000",    # rosso  
        "ann":   "#FFA500",    # arancione
        "stay_e":"#FFD700",    # giallo
        "pp":    "#003FAB",    # blu scuro
        "stay_p": "#87CEEB",    # azzurro
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
        node_size=size,
        font_size=8,
        arrows=True
    )
    plt.title("EM shower", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    # Legenda se color è attivo
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
        plt.legend(handles=patches, title="Interaction")
        plt.title("Legenda tipi di interazione")
        plt.axis('off')
        plt.savefig("plots/shower.pdf")
        plt.show()
        plt.close()

    else:
        plt.savefig("plots/shower2.pdf")


def plot_energy(energy_deposed):
    x = list(range(len(energy_deposed)))
    plt.figure(figsize=(10, 5))
    plt.plot(x, energy_deposed, marker='o', linestyle='-', color='b')
    plt.title('Energy deposed plot')
    plt.xlabel('X0')
    plt.ylabel('Energy (MeV)')
    plt.grid(True)
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



