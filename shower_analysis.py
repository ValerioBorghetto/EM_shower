import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from network_utils import*
from build_shower.em_shower import*
from tqdm import tqdm
import pandas as pd
import matplotlib.patches as mpatches
from scipy.optimize import curve_fit

#plot the interaction kind versus the time that process has occurred
def plot_kinds(initial_energy, n_iter):  
    all_counts=[]
    for _ in tqdm(range(n_iter), desc="Interaction vs occurrence"):
        shower, _, _ = generate_shower(depth=40, initial_energy=initial_energy, Z=40, initial_particle="electron")
        kinds = list(nx.get_node_attributes(shower, "kind").values())
        labels, counts = np.unique(kinds, return_counts=True)
        all_counts.append(counts)
    all_counts=np.array(all_counts)
    type_means=np.mean(all_counts, axis=0)
    type_error=np.std(all_counts, axis=0, ddof=1) / np.sqrt(n_iter)
    colors = ["#FFA500","#FF0000","#003FAB","#FFD700","#87CEEB"] 
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(range(5), type_means, width=0.6, color=colors, edgecolor='black', yerr=type_error, capsize=5,
              error_kw={'elinewidth':1.2, 'ecolor':'black'})
    ax.set_xticks(range(len(type_means)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Occurrence", fontsize=12)
    ax.set_xlabel("Interaction type", fontsize=12)
    ax.set_title(f"Interaction occurrence", fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    readable_names = {
    "brems": "Bremsstrahlung",
    "pp": "Pair Production",
    "ann": "Annihilation",
    "stay_e": "No e interaction",
    "stay_p": "No p interaction"
    }
    readable_labels = [readable_names[l] for l in labels]
    textstr = '\n'.join([f'{readable_labels[i]}: {type_means[i]:.2f} ± {type_error[i]:.2f}' for i in range(5)])
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig("plots/interaction_occurrence.pdf")
    plt.show()
    plt.close()

#out-degree plot and study 
def out_degree_study(initial_energy, n_iter):
    all_counts = []
    for _ in tqdm(range(n_iter), desc="Out degree study 1"):
        graph, _, _ = generate_shower(depth=40, initial_energy=initial_energy, Z=40, initial_particle="electron")
        adjacency = nx.adjacency_matrix(graph).toarray()
        in_degree = np.sum(adjacency, axis=1)
        counts = [np.sum(in_degree == k) for k in range(3)]
        all_counts.append(counts)
    #Mean
    all_counts=np.array(all_counts)
    degree_means = np.mean(all_counts, axis=0)
    degree_errors = np.std(all_counts, axis=0, ddof=1) / np.sqrt(n_iter)
    colors = ['#4C72B0','#4C72B0','#4C72B0']
    # Plot
    fig, ax = plt.subplots(figsize=(8,6)) #(7,5)
    bars = ax.bar(range(3), degree_means, width=0.6, color=colors, edgecolor='black', alpha=0.8,
              yerr=degree_errors, capsize=5, error_kw={'elinewidth':1.2, 'ecolor':'black'})
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([0, 1, 2], fontsize=12)
    ax.set_ylabel("Average number of nodes", fontsize=12)
    ax.set_xlabel("Degree", fontsize=12)
    ax.set_title(f"Out-Degree Distribution for showers of {initial_energy} MeV", fontsize=14)
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    textstr = '\n'.join([f'Degree {i}: {degree_means[i]:.2f} ± {degree_errors[i]:.2f}' for i in range(3)])
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig("plots/degree_professional.pdf")
    plt.show()
    plt.close()

#studies the out-degree dividing the shower in two halves
def out_degree_study_max_width(initial_energy, n_iter):
    all_counts_first, all_counts_second = [], []

    for _ in tqdm(range(n_iter), desc="Out degree study 2"):
        graph, _, _ = generate_shower(depth=40, initial_energy=initial_energy, Z=40, initial_particle="electron")
        node_steps = nx.get_node_attributes(graph, "step")
        steps = np.array(list(node_steps.values()))
        levels, counts = np.unique(steps, return_counts=True)
        max_width_step = levels[np.argmax(counts)]
        first_half_nodes = [n for n, s in node_steps.items() if s <= max_width_step]
        second_half_nodes = [n for n, s in node_steps.items() if s > max_width_step]

        adjacency = nx.adjacency_matrix(graph).toarray()
        out_degrees = np.sum(adjacency, axis=1) 
        nodes_list = list(graph.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes_list)}
        def degree_counts(node_subset):
            indices = [node_to_idx[n] for n in node_subset]
            degs = out_degrees[indices]
            return [np.sum(degs == k) for k in range(3)]

        all_counts_first.append(degree_counts(first_half_nodes))
        all_counts_second.append(degree_counts(second_half_nodes))
    all_counts_first = np.array(all_counts_first)
    all_counts_second = np.array(all_counts_second)
    degree_means_first = np.mean(all_counts_first, axis=0)
    degree_errors_first = np.std(all_counts_first, axis=0, ddof=1) / np.sqrt(n_iter)
    degree_means_second = np.mean(all_counts_second, axis=0)
    degree_errors_second = np.std(all_counts_second, axis=0, ddof=1) / np.sqrt(n_iter)
    labels = [0, 1, 2]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,6)) 
    ax.bar(x - width/2, degree_means_first, width, yerr=degree_errors_first, label='_nolegend_', 
           color='#4C72B0', edgecolor='black', alpha=0.8, capsize=5, error_kw={'elinewidth':1.2, 'ecolor':'black'})
    ax.bar(x + width/2, degree_means_second, width, yerr=degree_errors_second, label='_nolegend_', 
           color='#55A868', edgecolor='black', alpha=0.8, capsize=5, error_kw={'elinewidth':1.2, 'ecolor':'black'})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Occurrence", fontsize=12)
    ax.set_xlabel("Out-Degree", fontsize=12)
    ax.set_title(f"Out-Degree Distribution split at max width", fontsize=14) 
    ax.yaxis.grid(True, linestyle='--', alpha=0.6)

    proxy1 = mpatches.Patch(color='#4C72B0', label='Before max width')
    proxy2 = mpatches.Patch(color='#55A868', label='After max width')
    ax.legend(handles=[proxy1, proxy2], framealpha=0.8, edgecolor='gray')
    textstr_first = '\n'.join([f'Degree {i}: {degree_means_first[i]:.2f} ± {degree_errors_first[i]:.2f}' for i in range(3)])
    ax.text(0.02, 0.95, textstr_first, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#4C72B0'))
    textstr_second = '\n'.join([f'Degree {i}: {degree_means_second[i]:.2f} ± {degree_errors_second[i]:.2f}' 
                                for i in range(3)])
    ax.text(0.02, 0.82, textstr_second, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#55A868'))

    plt.tight_layout()
    plt.savefig("plots/out_degree_comparison.pdf")
    plt.show()
    plt.close()

#plot the degree frequencies of the out degree versus different energy values
def plot_degree_vs_energy_with_error(initial_energies=None, n_iter=100, depth=40, Z=40):
    """
    initial_energies: array of energies
    n_iter: number of shower per energy value
    depth, Z: parameters for generate the showers
    """
    if initial_energies is None:
        initial_energies = np.linspace(100, 1000, 10)
    
    all_freqs = []
    all_errors = []
    for E in tqdm(initial_energies, desc="Degree versus energy"):
        degree_counts = []
        for _ in range(n_iter):
            graph, _, _ = generate_shower(depth=depth, initial_energy=E, Z=Z, initial_particle="electron")
            adjacency = nx.adjacency_matrix(graph).toarray()
            in_degree = np.sum(adjacency, axis=1)
            counts = [np.sum(in_degree == k) for k in range(3)]  # degree 0,1,2
            degree_counts.append(counts)
        
        degree_counts = np.array(degree_counts)
        degree_means = np.mean(degree_counts, axis=0)
        degree_std = np.std(degree_counts, axis=0, ddof=1)
        total_mean = np.sum(degree_means)
        freq = degree_means / total_mean
        freq_err = freq * np.sqrt((degree_std / degree_means)**3) 
        all_freqs.append(freq)
        all_errors.append(freq_err)

    all_freqs = np.array(all_freqs)
    all_errors = np.array(all_errors)

    plt.figure(figsize=(8,6)) 
    plt.errorbar(initial_energies, all_freqs[:,0], yerr=all_errors[:,0],
                 label='Degree 0', color='#4C72B0', marker='o', linestyle='-',
                 capsize=4, capthick=2, elinewidth=1.5)
    plt.errorbar(initial_energies, all_freqs[:,1], yerr=all_errors[:,1],
                 label='Degree 1', color='#55A868', marker='s', linestyle='-',
                 capsize=4, capthick=2, elinewidth=1.5)
    plt.errorbar(initial_energies, all_freqs[:,2], yerr=all_errors[:,2],
                 label='Degree 2', color='#C44E52', marker='^', linestyle='-',
                 capsize=4, capthick=2, elinewidth=1.5)
    plt.xlabel("Initial Energy (MeV)", fontsize=12)
    plt.ylabel("Degree frequency", fontsize=12)
    plt.title("Degree frequency vs. Initial energy", fontsize=14)
    plt.legend(framealpha=0.8, edgecolor='gray')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("plots/degree_vs_energy.pdf")
    plt.show()
    plt.close()
    
    return initial_energies, all_freqs, all_errors

#plot the width of the shower (the number of interactions per level)
def plot_width(shower):
    steps = list(nx.get_node_attributes(shower, "step").values())
    levels, counts = np.unique(steps, return_counts=True)
    fig, ax = plt.subplots(figsize=(8, 6)) 
    ax.plot(
        levels, counts, "--", 
        color="firebrick", linewidth=2, zorder=1
    )
    ax.plot(
        levels, counts, "D", 
        color="darkred", markersize=5, 
        label="# interactions", zorder=2
    )
    ax.set_title("Total interactions occurrences per step", fontsize=14) 
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Occurrences", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
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
        "brems": "#FF0000",    # red  
        "ann":   "#FFA500",    # orange
        "stay_e":"#FFD700",    # yellow
        "pp":    "#003FAB",    # dark blue
        "stay_p": "#87CEEB",   # light blue
    }
    colors = [color_map.get(col, "#000000") for col in tabella.columns]
    tabella.plot(kind='line', figsize=(10, 6), marker='o', color=colors) 
    plt.title("Interaction occurrence per step", fontsize = 14)
    plt.xlabel("Step", fontsize = 12)
    plt.ylabel("Occurence", fontsize = 12)
    readable_names = {
        "brems": "Bremsstrahlung",
        "pp": "Pair Production",
        "ann": "Annihilation",
        "stay_e": "No e interaction",
        "stay_p": "No p interaction"
    }
    patches = [mpatches.Patch(color=color_map[col], label=readable_names[col]) for col in tabella.columns]
    plt.legend(handles=patches, framealpha=0.8, edgecolor='gray')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/interaction_vs_depth.pdf")
    plt.show()
    plt.close()

#It studies the width and the energy deposited in different showers
def shower_study(initial_energy, final_energy, times, energy=True, width=True):
    def mean_err(data):
        return np.mean(data), np.std(data, ddof=1) / np.sqrt(len(data))
    energies = np.linspace(initial_energy, final_energy, times)
    levels, levels_err, widths, widths_err = [], [], [], []
    n_iter = 100
    for e in tqdm(energies, desc="Shower maximum vs energies"):
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
    #Fit the max energy depth
    def log(x, a, b):
        return a * np.log(x / b)
    popt, pcov = curve_fit(log, energies, levels, sigma=levels_err, absolute_sigma=True)
    def plot(ax, x, y, yerr, title, xlabel, ylabel, style, color):
        ax.plot(x, y, "--", color="royalblue", linewidth=2, zorder=1)
        ax.errorbar(
            x, y, yerr=yerr, fmt="o", 
            capsize=6, elinewidth=2, 
            ecolor="darkblue", color="darkblue", 
            markersize=2, zorder=2
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
    perr = np.sqrt(np.diag(pcov))
    x_fit = np.linspace(min(energies), max(energies), 200)
    y_fit = log(x_fit, *popt)
    residuals = levels - log(energies, *popt)
    chi2 = np.sum((residuals / levels_err) ** 2)
    ndof = len(energies) - len(popt) 
    chi2_red = chi2 / ndof

    fig, ax1 = plt.subplots(1, figsize=(8, 6)) 
    plot(ax1, energies, levels, levels_err, 
        "Shower maximum vs. Initial energy", 
        "Initial energy (MeV)", 
        "Shower maximum (steps)", 
        'o-', "orange")
    ax1.plot(x_fit, y_fit, "-", color="gray", linewidth=2, zorder=3)
    textstr = '\n'.join([
        f'{name} = {val:.2f} ± {err:.2f}' 
        for name, val, err in zip(['a','b'], popt, perr)
    ])
    textstr += f"\nχ²/ndof =  {chi2_red:.2f}"
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes,
        fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.tight_layout()
    plt.savefig("plots/maximum_vs_energy_fit.pdf")
    plt.show()
    plt.close()

#studies the number of nodes, the number of edges, the depth over different energies
def study_properties(initial_energy, final_energy, times):
    def mean_err(data):
        return np.mean(data), np.std(data, ddof=1) / np.sqrt(len(data))
    energies = np.linspace(initial_energy, final_energy, times)
    depth, depth_err, node, node_err, edges, edges_err = [], [], [], [], [], []
    n_iter = 100
    for e in tqdm(energies, desc="Network properties"):
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
        ax.plot(
            x, y, style, 
            color="royalblue", linewidth=2, zorder=1
        )
        ax.errorbar(
            x, y, yerr=yerr, fmt="o",
            capsize=6, elinewidth=2,
            ecolor=color, color=color,
            markersize=4, zorder=2
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
    plot(
        ax, energies, depth, depth_err, 
        "Shower depth vs. Initial energy", 
        "Initial energy (MeV)", 
        "Depth (steps)"
    )
    plt.tight_layout()
    plt.savefig("plots/depth_vs_energy.pdf")
    plt.show()
    plt.close()
    
#Studies tree properties 
def network_degree(initial_energy, n_iter=100):
    avg_clustering_list = []
    deg_assortativity_list = []
    avg_degree_list = []
    branching_factor_list = []
    assort_kind_list = []
    diameter_list = []
    for i in tqdm(range(n_iter), "Treee properties"):
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


# ================================
# Analysis & plotting for showers
# ================================
from typing import Callable, Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


def analyze_markov_models(
    markov_models: List[str],
    N: int = 100,
    depth: int = 40,
    initial_energy: float = 1000.0,
    Z: int = 40,
    initial_particle: str = "electron",
    attribute_name: Optional[str] = None,
    bar_plot_labels: Tuple[str, str] = ("Before max width", "After max width"),
    save_dir: Optional[str] = "plots",
) -> Dict[str, dict]:
    """
    - Runs N showers per markov model.
    - For each model, aggregates counts of nodes with out-degree 0/1/2 (mean & std).
    - Plots:
        1) Bar plot (your exact "split at max width" style) for *two* models.
        2) Metrics plot (solid line + dots, your 'plot()' style).
    - Computes per-model:
        Average clustering, Degree assortativity, Branching factor, Attribute assortativity.

    Returns a dict with the aggregated values.
    """

    # ---------- helpers ----------
    def _is_digraph(G) -> bool:
        return isinstance(G, (nx.DiGraph, nx.MultiDiGraph))

    def _out_degrees(G):
        # treat undirected degree as "out-degree" if G is undirected
        return dict(G.out_degree()) if _is_digraph(G) else dict(G.degree())

    def _count_outdeg_categories(G, cats=(0, 1, 2)) -> Dict[int, int]:
        degs = _out_degrees(G).values()
        counts = {c: 0 for c in cats}
        for d in degs:
            if d in counts:
                counts[d] += 1
        return counts

    def _avg_clustering(G):
        UG = G.to_undirected() if _is_digraph(G) else G
        if len(UG) <= 1:
            return np.nan
        return nx.average_clustering(UG)

    def _degree_assortativity(G):
        # Your requested function
        return nx.degree_assortativity_coefficient(G)

    def _branching_factor(G):
        # average out-degree among nodes with out-degree > 0
        degs = _out_degrees(G)
        positives = [d for d in degs.values() if d > 0]
        if not positives:
            return 0.0
        return float(np.mean(positives))

    def _attribute_assortativity(G, attr: Optional[str]):
        return nx.attribute_assortativity_coefficient(G, "kind")

    # ---------- run ----------
    outdeg_counts_by_model = {m: [] for m in markov_models}
    metrics_by_model = {
        m: {
            "avg_clustering": [],
            "degree_assortativity": [],
            "branching_factor": [],
            "attribute_assortativity": [],
        }
        for m in markov_models
    }

    for model in markov_models:
        for _ in range(N):
            G, energy_deposed, norm = generate_shower(
                depth=depth,
                initial_energy=initial_energy,
                Z=Z,
                initial_particle=initial_particle,
                markov_model=model,
            )

            outdeg_counts_by_model[model].append(_count_outdeg_categories(G, cats=(0, 1, 2)))

            metrics_by_model[model]["avg_clustering"].append(_avg_clustering(G))
            metrics_by_model[model]["degree_assortativity"].append(_degree_assortativity(G))
            metrics_by_model[model]["branching_factor"].append(_branching_factor(G))
            metrics_by_model[model]["attribute_assortativity"].append(_attribute_assortativity(G, attribute_name))

    # ---------- aggregate ----------
    outdeg_stats = {}
    for model, cnt_list in outdeg_counts_by_model.items():
        arr0 = np.array([c[0] for c in cnt_list], dtype=float)
        arr1 = np.array([c[1] for c in cnt_list], dtype=float)
        arr2 = np.array([c[2] for c in cnt_list], dtype=float)
        outdeg_stats[model] = {
            "mean": {0: float(np.nanmean(arr0)), 1: float(np.nanmean(arr1)), 2: float(np.nanmean(arr2))},
            "std":  {0: float(np.nanstd(arr0, ddof=1)), 1: float(np.nanstd(arr1, ddof=1)), 2: float(np.nanstd(arr2, ddof=1))},
        }

    metrics_summary = {}
    for model, mdict in metrics_by_model.items():
        summary = {}
        for k, vals in mdict.items():
            a = np.array(vals, dtype=float)
            if np.all(np.isnan(a)):
                summary[k] = (np.nan, np.nan)
            else:
                summary[k] = (float(np.nanmean(a)), float(np.nanstd(a, ddof=1)))
        metrics_summary[model] = summary

    # ==================================================
    # PLOT 1 — BAR: your exact “split at max width” style
    # (works when there are exactly TWO models)
    # ==================================================
    if len(markov_models) == 2:
        labels = ["0", "1", "2"]
        x = np.arange(len(labels))
        width = 0.35

        m1, m2 = markov_models
        degree_means_first = np.array([outdeg_stats[m1]["mean"][k] for k in [0, 1, 2]], dtype=float)
        degree_errors_first = np.array([outdeg_stats[m1]["std"][k] for k in [0, 1, 2]], dtype=float)
        degree_means_second = np.array([outdeg_stats[m2]["mean"][k] for k in [0, 1, 2]], dtype=float)
        degree_errors_second = np.array([outdeg_stats[m2]["std"][k] for k in [0, 1, 2]], dtype=float)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.bar(
            x - width/2, degree_means_first, width,
            yerr=degree_errors_first, label='_nolegend_',
            color='#4C72B0', edgecolor='black', alpha=0.8, capsize=5,
            error_kw={'elinewidth': 1.2, 'ecolor': 'black'}
        )
        ax.bar(
            x + width/2, degree_means_second, width,
            yerr=degree_errors_second, label='_nolegend_',
            color='#55A868', edgecolor='black', alpha=0.8, capsize=5,
            error_kw={'elinewidth': 1.2, 'ecolor': 'black'}
        )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel("Occurrence", fontsize=12)
        ax.set.xlabel("Out-Degree", fontsize=12)
        ax.set_title("Out-Degree Distribution split at max width", fontsize=14)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)

        proxy1 = mpatches.Patch(color='#4C72B0', label=bar_plot_labels[0])
        proxy2 = mpatches.Patch(color='#55A868', label=bar_plot_labels[1])
        ax.legend(handles=[proxy1, proxy2], framealpha=0.8, edgecolor='gray')

        textstr_first = '\n'.join([f'Degree {i}: {degree_means_first[i]:.2f} ± {degree_errors_first[i]:.2f}' for i in range(3)])
        ax.text(0.02, 0.95, textstr_first, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#4C72B0'))

        textstr_second = '\n'.join([f'Degree {i}: {degree_means_second[i]:.2f} ± {degree_errors_second[i]:.2f}' for i in range(3)])
        ax.text(0.02, 0.82, textstr_second, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#55A868'))

        plt.tight_layout()
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/out_degree_comparison.pdf")
        plt.show()
        plt.close()
    else:
        # Fallback: grouped bars across many models (kept simple)
        categories = [0, 1, 2]
        x = np.arange(len(categories))
        width = 0.8 / max(1, len(markov_models))
        fig1, ax1 = plt.subplots(figsize=(9, 6), facecolor="white")
        palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#64B5CD", "#CCB974"]
        for i, model in enumerate(markov_models):
            means = [outdeg_stats[model]["mean"][c] for c in categories]
            stds  = [outdeg_stats[model]["std"][c] for c in categories]
            xshift = x + i * width - (len(markov_models) - 1) * width / 2
            ax1.bar(xshift, means, width, yerr=stds, capsize=5,
                    color=palette[i % len(palette)], edgecolor='black',
                    alpha=0.85, error_kw={'elinewidth':1.2, 'ecolor':'black'}, label=model)
        ax1.set_xticks(x)
        ax1.set_xticklabels(["0", "1", "2"], fontsize=12)
        ax1.set_ylabel("Occurrence", fontsize=12)
        ax1.set_xlabel("Out-Degree", fontsize=12)
        ax1.set_title("Out-Degree Distribution by Model", fontsize=14)
        ax1.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(framealpha=0.85, edgecolor='gray')
        plt.tight_layout()
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/out_degree_by_model.pdf")
        plt.show()
        plt.close()

    # ==================================================
    # PLOT 2 — METRICS: solid lines + dots (same color)
    # ==================================================
    # Your plotting helper (solid line)
    def plot(ax, x, y, yerr, title, xlabel, ylabel, color="darkblue"):
        ax.plot(x, y, "-", color=color, linewidth=2, zorder=1)  # SOLID line
        ax.errorbar(
            x, y, yerr=yerr, fmt="o",
            capsize=6, elinewidth=2,
            ecolor=color, color=color,
            markersize=4, zorder=2
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

    model_x = np.arange(len(markov_models))

    def _mvals(key):
        means = [metrics_summary[m][key][0] for m in markov_models]
        stds  = [metrics_summary[m][key][1] for m in markov_models]
        return np.array(means, float), np.array(stds, float)

    ac_mean, ac_std = _mvals("avg_clustering")
    da_mean, da_std = _mvals("degree_assortativity")
    bf_mean, bf_std = _mvals("branching_factor")
    aa_mean, aa_std = _mvals("attribute_assortativity")

    fig2, ax2 = plt.subplots(figsize=(8, 6), facecolor='white')
    #plot(ax2, model_x, ac_mean, ac_std, "Graph properties vs. Markov model", "Model", "Value", color="red")
    plot(ax2, model_x, da_mean, da_std, "Graph properties vs. Markov model", "Model", "Value", color="green")
    plot(ax2, model_x, bf_mean, bf_std, "Graph properties vs. Markov model", "Model", "Value", color="gold")  # yellow lines can be hard to see; 'gold' reads better
    plot(ax2, model_x, aa_mean, aa_std, "Graph properties vs. Markov model", "Model", "Value", color="blue")

    ax2.set_xticks(model_x)
    ax2.set_xticklabels(markov_models, rotation=0)

    # Legend with matching colors
    handles = [
        #plt.Line2D([0], [0], color="red",  marker="o", linestyle="-", linewidth=2, markersize=4, label="Average clustering"),
        plt.Line2D([0], [0], color="green",marker="o", linestyle="-", linewidth=2, markersize=4, label="Degree assortativity"),
        plt.Line2D([0], [0], color="gold", marker="o", linestyle="-", linewidth=2, markersize=4, label="Branching factor"),
        plt.Line2D([0], [0], color="blue", marker="o", linestyle="-", linewidth=2, markersize=4, label="Attribute assortativity"),
    ]
    leg2 = ax2.legend(handles=handles, framealpha=0.85, edgecolor='gray', loc="best")

    plt.tight_layout()
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/metrics_vs_model.pdf")
    plt.show()
    plt.close()

    return {"outdeg_stats": outdeg_stats, "metrics": metrics_summary}


def network_prop(initial_energy, final_energy, times, n_iter=100, save_dir="plots"):
    """
    Study degree assortativity, branching factor, and attribute assortativity 
    of showers as a function of the initial energy.

    Parameters
    ----------
    initial_energy : float
        Starting value for the energy range (MeV).
    final_energy : float
        Final value for the energy range (MeV).
    times : int
        Number of energy levels to evaluate between initial and final energy.
    n_iter : int, optional
        Number of shower realizations per energy level (default=100).
    save_dir : str, optional
        Directory to save plots (default='plots').
    """

    def mean_err(data):
        """Compute mean and standard error of the mean."""
        data = np.array(data, dtype=float)
        return np.mean(data), np.std(data, ddof=1) / np.sqrt(len(data))

    def branching_factor(G):
        """Compute average branching factor = mean(out-degree of non-leaf nodes)."""
        out_degrees = [d for _, d in G.out_degree() if d > 0]
        return np.mean(out_degrees) if out_degrees else 0.0

    def attribute_assortativity(G, attr="step"):
        """Compute assortativity based on a given node attribute."""
        try:
            return nx.attribute_assortativity_coefficient(G, attr)
        except Exception:
            return np.nan

    # Arrays for energies and metrics
    energies = np.linspace(initial_energy, final_energy, times)
    da_mean, da_err = [], []
    bf_mean, bf_err = [], []
    aa_mean, aa_err = [], []

    for e in tqdm(energies, desc="Computing network metrics"):
        deg_assort, branch_fact, attr_assort = [], [], []

        for _ in range(n_iter):
            shower, _, _ = generate_shower(
                depth=40, initial_energy=e, Z=10, initial_particle="electron"
            )

            # Compute metrics safely
            try:
                da = nx.degree_assortativity_coefficient(shower)
            except Exception:
                da = np.nan

            bf = branching_factor(shower)
            aa = attribute_assortativity(shower, "step")

            deg_assort.append(da)
            branch_fact.append(bf)
            attr_assort.append(aa)

        # Mean and error for each metric
        m, err = mean_err(deg_assort)
        da_mean.append(m); da_err.append(err)
        m, err = mean_err(branch_fact)
        bf_mean.append(m); bf_err.append(err)
        m, err = mean_err(attr_assort)
        aa_mean.append(m); aa_err.append(err)

    # Plot helper
    def plot(ax, x, y, yerr, title, xlabel, ylabel, color="darkblue"):
        ax.plot(x, y, "-", color=color, linewidth=2, zorder=1)
        ax.errorbar(
            x, y, yerr=yerr, fmt="o",
            capsize=6, elinewidth=2,
            ecolor=color, color=color,
            markersize=4, zorder=2
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.6)

    # --- Plot all metrics together ---
    fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')

    plot(ax, energies, da_mean, da_err, "", "Initial energy (MeV)", "Metric value", color="green")
    plot(ax, energies, bf_mean, bf_err, "", "Initial energy (MeV)", "Metric value", color="gold")
    plot(ax, energies, aa_mean, aa_err, "", "Initial energy (MeV)", "Metric value", color="blue")

    handles = [
        plt.Line2D([0], [0], color="green", marker="o", linestyle="-", linewidth=2, markersize=4, label="Degree assortativity"),
        plt.Line2D([0], [0], color="gold", marker="o", linestyle="-", linewidth=2, markersize=4, label="Branching factor"),
        plt.Line2D([0], [0], color="blue", marker="o", linestyle="-", linewidth=2, markersize=4, label="Attribute assortativity"),
    ]
    ax.legend(handles=handles, framealpha=0.85, edgecolor='gray', loc="best")
    ax.set_title("Network metrics vs. Initial energy", fontsize=14)

    plt.tight_layout()
    import os
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(f"{save_dir}/metrics_vs_energy.pdf")
    plt.show()
    plt.close()

    return {
        "energies": energies,
        "degree_assortativity": (np.array(da_mean), np.array(da_err)),
        "branching_factor": (np.array(bf_mean), np.array(bf_err)),
        "attribute_assortativity": (np.array(aa_mean), np.array(aa_err)),
    }