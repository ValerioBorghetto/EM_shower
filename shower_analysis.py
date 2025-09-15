import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*
from build_shower.em_shower import*
from tqdm import tqdm
import pandas as pd
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit

def plot_kinds(initial_energy, n_iter): #plot the interaction kind versus the time that process has occurred
    all_counts=[]
    for _ in tqdm(range(n_iter), desc="Simulation 4"):
        shower, _, _ = generate_shower(depth=40, initial_energy=initial_energy, Z=40, initial_particle="electron")
        kinds = list(nx.get_node_attributes(shower, "kind").values())
        labels, counts = np.unique(kinds, return_counts=True)
        all_counts.append(counts)
    all_counts=np.array(all_counts)
    type_means=np.mean(all_counts, axis=0)
    type_error=np.std(all_counts, axis=0, ddof=1) / np.sqrt(n_iter)
    colors = ["#FFA500","#FF0000","#003FAB","#FFD700","#87CEEB"]
    
    # Plot
    fig, ax = plt.subplots(figsize=(7,5))
    bars = ax.bar(range(5), type_means, width=0.6, color=colors, edgecolor='black')
    ax.set_xticks(range(len(type_means)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Average number of occurrence", fontsize=12)
    ax.set_xlabel("Interaction type", fontsize=12)
    ax.set_title(f"Interaction occurrence for showers of {initial_energy} MeV",
             fontsize=14, fontweight='bold')
    # Griglia leggera
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # Aggiungi textbox con valori medi ± errore in alto a sinistra
    textstr = '\n'.join([f'# {labels[i]}: {type_means[i]:.2f} ± {type_error[i]:.2f}' for i in range(5)])
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig("plots/interaction_occurrence.pdf")
    plt.show()
    plt.close()

#adjacency matrix plot and study 
def adj_matrix_study(initial_energy, n_iter):
    all_counts = []
    for _ in tqdm(range(n_iter), desc="Simulation 4"):
        graph, _, _ = generate_shower(depth=40, initial_energy=initial_energy, Z=40, initial_particle="electron")
        adjacency = nx.adjacency_matrix(graph).toarray()
        in_degree = np.sum(adjacency, axis=1)
        counts = [np.sum(in_degree == k) for k in range(3)]
        all_counts.append(counts)
    # Calcola la media
    all_counts=np.array(all_counts)
    degree_means = np.mean(all_counts, axis=0)
    degree_errors = np.std(all_counts, axis=0, ddof=1) / np.sqrt(n_iter)
    # Colori professionali per le barre
    colors = ['#4C72B0','#4C72B0','#4C72B0']

    # Plot
    fig, ax = plt.subplots(figsize=(7,5))
    bars = ax.bar(range(3), degree_means, width=0.6, color=colors, edgecolor='black')

    # Ticks e labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([0, 1, 2], fontsize=12)
    ax.set_ylabel("Average number of nodes", fontsize=12)
    ax.set_xlabel("Degree", fontsize=12)
    ax.set_title(f"Out-Degree Distribution for showers of {initial_energy} MeV",
             fontsize=14, fontweight='bold')
    # Griglia leggera
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    # Aggiungi textbox con valori medi ± errore in alto a sinistra
    textstr = '\n'.join([f'Degree {i}: {degree_means[i]:.2f} ± {degree_errors[i]:.2f}' for i in range(3)])
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig("plots/degree_professional.pdf")
    plt.show()
    plt.close()
#plot the width of the shower (the number of interactions per level)
def plot_width(shower):
    steps = list(nx.get_node_attributes(shower, "step").values())
    levels, counts = np.unique(steps, return_counts=True)

    # --- Preparazione figura ---
    fig, ax = plt.subplots(figsize=(7, 5))

    # linea più chiara sotto
    ax.plot(
        levels, counts, "--", 
        color="firebrick", linewidth=2, zorder=1
    )
    
    # punti con evidenza sopra
    ax.plot(
        levels, counts, "D", 
        color="darkred", markersize=5, 
        label="# interactions", zorder=2
    )

    # --- Stile e testi ---
    ax.set_title("# interactions per depth (X₀)", fontsize=18, weight="bold")
    ax.set_xlabel("X₀", fontsize=14)
    ax.set_ylabel("# interactions", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # --- Stile globale coerente ---
    plt.tight_layout()
    plt.savefig("plots/width.pdf")
    plt.show()
    plt.close()
    
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


    #Fit the max energy depth
    def log(x, a, b):
        return a * np.log(x / b)

    popt, pcov = curve_fit(log, energies, levels, sigma=levels_err, absolute_sigma=True)

    def plot(ax, x, y, yerr, title, xlabel, ylabel, style, color):
        # linea più chiara
        ax.plot(x, y, "--", color="royalblue", linewidth=2, zorder=1)
        # barre più scure in evidenza
        ax.errorbar(
            x, y, yerr=yerr, fmt="o", 
            capsize=6, elinewidth=2, 
            ecolor="darkblue", color="darkblue", 
            markersize=2, zorder=2
        )
        ax.set_title(title, fontsize=18, weight="bold")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)

    # --- errori sui parametri ---
    perr = np.sqrt(np.diag(pcov))

    # --- curva del fit ---
    x_fit = np.linspace(min(energies), max(energies), 200)
    y_fit = log(x_fit, *popt)

    # --- calcolo chi quadro ---
    residuals = levels - log(energies, *popt)
    chi2 = np.sum((residuals / levels_err) ** 2)
    ndof = len(energies) - len(popt)   # gradi di libertà
    chi2_red = chi2 / ndof

    # --- plotting ---
    fig, ax1 = plt.subplots(1, figsize=(7,5))
    plot(ax1, energies, levels, levels_err, 
        "Max energy depth vs Initial energy", 
        "Initial energy (MeV)", 
        "Depth with max energy deposit", 
        'o-', "orange")

    ax1.plot(x_fit, y_fit, "-", color="gray", linewidth=2, zorder=3)

    # --- Legenda con risultati del fit e chi2 ---
    textstr = '\n'.join([
        f'{name} = {val:.2f} ± {err:.2f}' 
        for name, val, err in zip(['a','b'], popt, perr)
    ])
    textstr += f"\nχ²/ndof = {chi2:.2f}/{ndof} = {chi2_red:.2f}"

    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.style.use("seaborn-v0_8-darkgrid")
    plt.tight_layout()
    plt.savefig("plots/depth_vs_energy_fit.pdf")
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

    def plot(ax, x, y, yerr, title, xlabel, ylabel, style="--", color="darkblue"):
        # linea più chiara sotto
        ax.plot(
            x, y, style, 
            color="royalblue", linewidth=2, zorder=1
        )

        # punti + barre errore sopra
        ax.errorbar(
            x, y, yerr=yerr, fmt="o",
            capsize=6, elinewidth=2,
            ecolor=color, color=color,
            markersize=4, zorder=2
        )

        # testi e griglia
        ax.set_title(title, fontsize=18, weight="bold")
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, linestyle="--", alpha=0.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
    plot(ax1, energies, node, node_err, "# Nodes vs Initial energy", "Initial energy", "# Nodes", 'o-', "orange")
    plot(ax2, energies, edges, edges_err, "# Edges vs Initial energy", "Initial energy", "# Edges", 's--', "orange")
    plt.tight_layout()
    plt.savefig("plots/nodes_edges_vs_energy.pdf")
    plt.show()
    plt.close()  
    fig, ax = plt.subplots(figsize=(7, 5))

    plot(
        ax, energies, depth, depth_err, 
        "Shower Depth vs Initial Energy", 
        "Initial Energy (MeV)", 
        "Depth (X₀)"
    )

    plt.tight_layout()
    ax.legend()
    plt.savefig("plots/depth_vs_energy.pdf")
    plt.show()
    plt.close()
    
def network_degree(initial_energy, n_iter=100):
    avg_clustering_list = []
    deg_assortativity_list = []
    avg_degree_list = []
    branching_factor_list = []
    assort_kind_list = []
    diameter_list = []
    for i in tqdm(range(n_iter), "Simulation3"):
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


