import networkx as nx
import random
import matplotlib.pyplot as plt
from build_shower.em_utils import *

#costanti fisiche#####
E_cut=2.5 #MeV     below this energy value, the particle is absorbed by the material
##################################

#generate the shower
def generate_shower(depth, initial_energy, Z, initial_particle, markov_model="model"):
    """
    depth = int; it is the maximum depth that the shower can reach
    intial_energy = int; the initial energy in MeV of the cascade
    Z = int; atomic number of the material
    initial_particle = string: "electron" or "photon"
    """

    nodes, edges, history, energy_deposit= [], [], [], []
    neg_buffer, pos_buffer = [], []
    step = 0

    # STEP 0: starting particle
    if (initial_particle=="electron"):
        first_inter = interaction(kind="brems", step=step, substep=0, charge=-1, energy=initial_energy) #interaction object
    elif (initial_particle=="photon"): 
        first_inter = interaction(kind="pp", step=step, substep=0, charge=2, energy=initial_energy)  
    else:
        print("Invalid first particle")   
    first = f"{first_inter.step}_{first_inter.kind}_{first_inter.substep}"
    nodes.append(first)
    history.append([first_inter])
    step += 1
    counter_int=0
    markov_array=[]
    while step < depth:
        old_interactions = history[step - 1]
        if len(old_interactions)==0:
            break
        state = []  # iteration of the step, it is then appended in the history
        energy_state = []
        substep = 0 
        create_buffer(old_interactions, pos_buffer, neg_buffer) #how many electron or positron
        for old_inter in old_interactions:
            if old_inter.energy<E_cut:
                energy_state.append(old_inter.energy)
            elif old_inter.kind == "brems":
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "brems", markov_model)
                if old_inter.charge == +1: #positron
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep,state, old_inter.charge, energy[0])
                    substep += 1
                    photon_decay(nodes, edges, prob, old_inter, state, step, substep, energy[1])
                    substep += 1
                elif old_inter.charge == -1: #electron
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep,state, old_inter.charge, energy[0])
                    substep += 1
                    photon_decay(nodes, edges, prob, old_inter, state, step, substep, energy[1])
                    substep += 1
            elif old_inter.kind == "pp": 
                charge=old_inter.charge
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "pp", markov_model)
                if charge==2:
                    #positron
                    new_charge=+1
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step,substep, state, new_charge, energy[0])
                    substep += 1
                    #electron
                    new_charge=-1 #electron 
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step,substep, state, new_charge,energy[1])
                    substep += 1   
                elif charge==+1: #only positron left
                    new_charge=+1
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep,state, new_charge, energy[0])
                    substep += 1
                elif charge==-1: #only electron left
                    new_charge=-1
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep, state, new_charge, energy[1])
            elif old_inter.kind == "ann":
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "ann", markov_model)
                photon_decay(nodes, edges, prob, old_inter, state, step, substep, energy[0])
                substep += 1
                photon_decay(nodes, edges, prob, old_inter, state, step, substep, energy[1])
            elif old_inter.kind=="stay_p": 
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "stay_p", markov_model)
                photon_decay(nodes, edges, prob, old_inter, state, step, substep, old_inter.energy*0.8) 
                substep += 1
            elif old_inter.kind == "stay_e": 
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "stay_e", markov_model)
                charge=old_inter.charge
                if charge==+1:
                    #positron
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep, state, charge, old_inter.energy*0.8)
                    substep += 1
                elif charge==-1:
                    #electron
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep,state, charge, old_inter.energy*0.8)
                    substep += 1
            #clean the buffers     
            for p in pos_buffer:
                if p.charge !=1 and p.charge !=2:
                    pos_buffer.remove(p)
            for n in neg_buffer:
                if n.charge !=-1 and n.charge !=2:
                    neg_buffer.remove(n)

        history.append(state)
        energy_deposit.append(energy_state)
        step += 1   
        counter_int=counter_int+len(state) 
        markov_array.append(prob)
    energy_for_step = [sum(sub) for sub in energy_deposit]
    shower = nx.DiGraph()
    shower.add_nodes_from(nodes)
    shower.add_edges_from(edges)  
    states = list(markov_array[0].keys())
    sum_matrix={s: {t: 0.0 for t in states} for s in states}

    #compute the average transition matrix: used in the analysis
    for matrix in markov_array:
        for s in states:
            for t in states:
                sum_matrix[s][t] += matrix[s][t]
    keys_to_sum = {
        "brems": ["brems", "ann", "stay_e"],
        "pp": ["brems", "ann", "stay_e"],
        "stay_e": ["brems", "ann", "stay_e"],
        "ann": ["pp", "stay_p"],
        "stay_p": ["pp", "stay_p"]
    }
    norm = {}
    for s, t in sum_matrix.items():
        norm[s] = {}
        group_sum = sum(t[k] for k in keys_to_sum[s] if k in t)
        other_keys = [k for k in t if k not in keys_to_sum[s]]
        other_sum = sum(t[k] for k in other_keys)

        for k, v in t.items():
            if k in keys_to_sum[s]:
                norm[s][k] = v / group_sum if group_sum > 0 else 0.0
            else:
                norm[s][k] = v / other_sum if other_sum > 0 else 0.0
    return shower, energy_for_step, norm

