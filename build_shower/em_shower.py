import networkx as nx
import random
import matplotlib.pyplot as plt
from build_shower.em_utils import *

#costanti fisiche#####
E_cut=2 #MeV, sotto questa soglia l'energia si deposita
##################################

#dizionario dei decadimenti
decays = {
    #"brems": [photon_decay, lepton_decay],
    #"pp": [lepton_decay, lepton_decay],
    "ann": [photon_decay],
    #"stay_e": [lepton_decay],
    "stay_p": [photon_decay],
}


#genera la shower, partendo da un bremsstralung
def generate_shower(depth, initial_energy, Z):
    # Lista di nodi e archi da costruire alla fine
    nodes = []
    edges = []
    step = 0
    history = []  # array che contiene tutte le interazioni nello step n-esimo
    neg_buffer = [] #elettroni
    pos_buffer=[] #positroni
    #used_nodes =set() #evitare  ripetizioni

    # STEP 0: electron iniziale che fa bremsstrahlung
    first_inter = interaction(kind="brems", step=step, substep=0, charge=-1, energy=initial_energy)
    first = f"{first_inter.step}_{first_inter.kind}_{first_inter.substep}"
    nodes.append(first)
    history.append([first_inter])
    step += 1
    energy_deposit=[]

    while step < depth:
        old_interactions = history[step - 1]
        if len(old_interactions)==0:
            break
        state = []  # Lista di nuove interazioni per questo step
        energy_state = []
        substep = 0  # Conta il numero di decadimenti per step
        create_buffer(old_interactions, pos_buffer, neg_buffer) #riempie i due buffer con le interazioni in quello step
        for old_inter in old_interactions:
            if old_inter.energy<E_cut:
                energy_state.append(old_inter.energy)
            elif old_inter.kind == "brems":
                if old_inter.charge == +1: #positrone
                    prob_p, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "brems")
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob_p, nodes, edges, step, substep,state, old_inter.charge, energy[0])
                    substep += 1
                    photon_decay(nodes, edges, prob_p, old_inter, state, step, substep, energy[1])
                    substep += 1
                elif old_inter.charge == -1: #electron
                    prob_e, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "brems")
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob_e, nodes, edges, step, substep,state, old_inter.charge, energy[0])
                    substep += 1
                    photon_decay(nodes, edges, prob_e, old_inter, state, step, substep, energy[1])
                    substep += 1
            elif old_inter.kind == "pp": 
                charge=old_inter.charge
                if charge==2:
                    #positrone
                    prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "pp")
                    new_charge=+1
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step,substep, state, new_charge, energy[0])
                    substep += 1
                    #electron
                    new_charge=-1 #electron 
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step,substep, state, new_charge,energy[1])
                    substep += 1   
                elif charge==+1: #rimasto solo il positrone
                    prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "pp")
                    new_charge=+1
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep,state, new_charge, energy[0])
                    substep += 1
                elif charge==-1: #rimasto solo l'electron
                    prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "pp")
                    new_charge=-1
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep, state, new_charge, energy[1])
                    substep += 1
            elif old_inter.kind == "ann":
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "ann")
                photon_decay(nodes, edges, prob, old_inter, state, step, substep, energy[0])
                substep += 1
                photon_decay(nodes, edges, prob, old_inter, state, step, substep, energy[1])
            elif old_inter.kind=="stay_p":
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "stay_p")
                photon_decay(nodes, edges, prob, old_inter, state, step, substep, old_inter.energy*0.8) 
                substep += 1
            elif old_inter.kind == "stay_e":
                prob, energy= energy_division(old_inter.energy, Z, old_interactions, neg_buffer, "stay_e")
                charge=old_inter.charge
                if charge==+1:
                    #positrone
                    positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep, state, charge, old_inter.energy*0.8)
                    substep += 1
                elif charge==-1:
                    #electron
                    electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep,state, charge, old_inter.energy*0.8)
                    substep += 1
            #pulisce i due buffer da interazioni già usate        
            for p in pos_buffer:
                if p.charge !=1 and p.charge !=2:
                    pos_buffer.remove(p)
            for n in neg_buffer:
                if n.charge !=-1 and n.charge !=2:
                    neg_buffer.remove(n)
        history.append(state)
        energy_deposit.append(energy_state)
        step += 1
    # Una volta costruite liste di nodi e archi, creo il grafo (più veloce rispetto a aggiungere nodi di volta in volta)
    energy_for_step = [sum(sub) for sub in energy_deposit]
    shower = nx.DiGraph()
    shower.add_nodes_from(nodes)
    shower.add_edges_from(edges)

    return shower, energy_for_step

