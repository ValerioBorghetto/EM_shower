import networkx as nx
import random
import numpy as np
from network_utils import *

####Physical constants
me_c2=0.511 #MeV
########################


#classe che definisce l'interazione univocamente
class interaction():
    def __init__(self, kind, step, substep, energy, charge=None):
        #kind=il tipo di interazione; step=il numero di passi della doccia; substep=in ogni passo, traccia i vari decadimenti
        self.kind = kind
        self.step=step
        self.substep = substep
        self.energy=energy
        self.charge= charge #+1=positrone, 0=fotone, -1=elettrone, +2 elettrone e positrone insieme (pp)
    def print_inter(self):
        print(f"Type: {self.kind}, Charge: {self.charge}")
    def int_name(self):
        return f"{self.step}_{self.kind}_{self.substep}"

#Probabilities of interactions (fatte abbastanza a caso)
#Bremsstralhung
def bremss_probab(E0, Z):
    return 0.98 / (1 + np.exp(- (np.log10(E0) - np.log10(3)) * 40)) * (E0 >= 1)
#Pair production
def pairprod_probab(E0, Z):

    #print(1 / (1 + np.exp(- (np.log10(E0) - np.log10(3)) * 10)) * (E0 >= 1))
    return 1 / (1 + np.exp(- (np.log10(E0) - np.log10(3)) * 40)) * (E0 >= 1)
#Annihilation
def ann_probab(E0, Z, buffer, old_interactions):
    # Gaussiana centrata a 50 MeV, larghezza 0.5 (log10 scale)
    #peak = np.exp(-((np.log10(E0) - np.log10(50))**2) / (2 * 0.3**2)) *(E0>=1)
    #print(np.clip(len(buffer)/len(old_interactions), 0, 1))
    return np.clip(len(buffer)/len(old_interactions), 0, 1)

#genera la matrice con le varie probabilità (dovrà essere cambiata perchè le prob. devono dipendere dalle energie, devono quindi essere delle funzioni)
def build_markov(lepton_energy, photon_energy, Z, buffer, old_interactions):
    p_eb = bremss_probab(lepton_energy, Z)   # elettrone → brems
    #print("prob brems:",p_eb)
    p_ea = ann_probab(lepton_energy, Z, buffer, old_interactions)   # elettrone → annichilazione
    p_es = 1-p_eb - p_ea   # elettrone → stay_e
    p_pp = pairprod_probab(photon_energy, Z)   # fotone → pair production
    p_ps = 1-p_pp   # fotone → stay_p
    prob = { 
        'brems': {'brems': p_eb, 'pp': p_pp, 'ann':p_ea, 'stay_e': p_es, 'stay_p':0},
        'pp': {'brems': p_eb, 'pp': 0, 'ann':p_ea, 'stay_e': p_es, 'stay_p':0}, 
        'ann': {'brems': 0, 'pp': p_pp, 'ann':0, 'stay_e': 0, 'stay_p':p_ps},
        'stay_e':{'brems': p_eb, 'pp': p_pp, 'ann':0, 'stay_e': p_es, 'stay_p':0},
        'stay_p':{'brems': 0, 'pp': p_pp, 'ann':0, 'stay_e': 0, 'stay_p':p_ps}
    }
    #print(prob)
    return prob

def draw_markov(energy, tree=True, adj_matrix=True):
    prob=build_markov(energy)
    states = ['brems', 'pp', 'ann', 'stay_e', 'stay_p']
    G = nx.DiGraph()
    G.add_nodes_from(states)
    for state_from in states:
        for state_to in states:
            weight = prob[state_from].get(state_to, 0) #se non lo trova da 0
            if weight>0:
                G.add_edge(state_from, state_to, weight=weight)
    if tree:
        nx.draw(G, with_labels=True)
    if adj_matrix:
        transition_matrix = nx.to_numpy_array(G, nodelist=states)
        plot_adjacency_matrix(adj_matrix=transition_matrix, title="Transition matrix", labels=states)

def create_buffer(old_interactions, pos_buffer, neg_buffer): #crea i due buffer, ovvero gli array contenti tutti gli elettroni (o positroni) liberi
    for old_inter in old_interactions:
            if old_inter.charge== -1:
                neg_buffer.append(old_inter)
            elif old_inter.charge == +1:
                pos_buffer.append(old_inter)
            elif old_inter.charge == +2:
                neg_buffer.append(old_inter)
                pos_buffer.append(old_inter)

#interactions generation
def generate_interaction(nodes, edges, new_inter, old_inter, state): #la particle mi dice il tipo dell'edge, così che tengo traccia delle particelle
    new = f"{new_inter.step}_{new_inter.kind}_{new_inter.substep}" #NB IL SUBSTEP è ORRENDO DA VEDERE, TIENILO PERCHè PUò SERVIRE PER CAPIRE SE CI SONO ERRORI NELLA CATENA, MA ALLA FINE NON FARLO VISUALIZZARE
    previous = f"{old_inter.step}_{old_inter.kind}_{old_inter.substep}"
    nodes.append((new,{"charge": new_inter.charge, "kind": new_inter.kind}))
    edges.append((previous, new, {"charge":new_inter.charge})) #serve negli edges???? Per ora no
    state.append(new_inter)

def photon_decay(nodes, edges, prob, old_inter, state, step, substep, energy):
    r1 = random.random()
    if r1 < prob['brems']['pp'] and energy > 2 * me_c2:
        new_inter = interaction(kind="pp", step=step, substep=substep, charge=+2, energy=energy) 
    else:
        new_inter = interaction(kind="stay_p", step=step, substep=substep, charge=0, energy=energy)
    generate_interaction(nodes, edges, new_inter, old_inter, state)

def positron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep, state, charge, energy): #da abbellire... è cosceno cosi
    #estrarre cosa esce
    r2 = random.random()
    if r2 < prob['brems']['brems']:
        new_kind = "brems" 
    elif r2 < prob['brems']['brems']+ prob['brems']['ann'] and len(neg_buffer) != 0:
        new_kind = "ann"
    else:
        new_kind = "stay_e"

    if new_kind=="brems":
        new_inter=interaction(kind=new_kind, step=step, substep=substep, charge=charge, energy=energy)
        generate_interaction(nodes, edges, new_inter, old_inter, state)                              
    elif new_kind=="stay_e": 
        new_inter=interaction(kind=new_kind, step=step, substep=substep, charge=charge, energy=energy)
        generate_interaction(nodes, edges, new_inter, old_inter, state)                    
    elif new_kind=="ann":
        new_inter=interaction(kind=new_kind, step=step, substep=substep, charge=0, energy=energy)
        generate_interaction(nodes, edges, new_inter, old_inter, state)
        number=random.randint(0, len(neg_buffer)-1)
        other_old=neg_buffer[number]
        energy_e=0.5*other_old.energy              #
        other_old.energy=other_old.energy-energy_e #non sarebbe giustissimo, ma inp rima approssimazione va bene
        new_inter.energy=new_inter.energy+energy_e
        edges.append((other_old.int_name(), new_inter.int_name(), {"charge":new_inter.charge})) 
        neg_buffer[number].charge=neg_buffer[number].charge-3 #così che se l'altra interazione è una brems o uno stay si disattiva, perchè va a carica troppo bassa, se è un pp invece, va a -1 e quindi poi avrà solo da aggiungere un legame con l'elettrone
    pos_buffer.remove(old_inter)

def electron_decay(neg_buffer, pos_buffer, old_inter, old_interactions, prob, nodes, edges, step, substep, state, charge, energy):
    #estrarre cosa esce
    r2 = random.random()
    if r2 < prob['brems']['brems']:
        new_kind = "brems" 
    elif r2 < prob['brems']['brems'] +  prob['brems']['ann'] and len(pos_buffer) != 0:
        new_kind = "ann"
    else:
        new_kind = "stay_e"
############## - se esce brems, lepton_decay semplice
    if new_kind=="brems":
        new_inter=interaction(kind=new_kind, step=step, substep=substep, charge=charge, energy=energy)
        generate_interaction(nodes, edges, new_inter, old_inter, state)     
############## - se esce stay, lepton_decay semplice                    
    elif new_kind=="stay_e":
        new_inter=interaction(kind=new_kind, step=step, substep=substep, charge=charge, energy=energy)
        generate_interaction(nodes, edges, new_inter, old_inter, state)                      
############## - se esce ann, deve essere in grado di: creare l'interazione (nodo+edge), scegliere uno dei possibili nodi precedenti, collegarlo con un altro edge al nuovo nodo, eliminarlo sia dal buffer che da old_interactions (o aggiungerlo al used_nodes, o modificare la carica a un valore assurdo per cui se c'è quel valore so già che è stato usato...questo mi piace di più di tutti) (particolare attenzione al pp, che invece dovrai toglierlo dalla rispettiva ( o aggiungerlo a used_nodes ma poi casini mi sa), e togliergli anche 1 di carica (se gli porti via l'elettrone) e 3 di carica (se gli porti via il positrone), così che si ritrovi sulla giusta carica restante. No, coi degree è più acile
    elif new_kind=="ann": 
        new_inter=interaction(kind=new_kind, step=step, substep=substep, charge=0, energy=energy) #la carica dovrebbe diventare zero poi no?
        generate_interaction(nodes, edges, new_inter, old_inter, state)
        number=random.randint(0, len(pos_buffer)-1)
        other_old=pos_buffer[number]
        energy_p=0.5*other_old.energy              #
        other_old.energy=other_old.energy-energy_p #non sarebbe giustissimo, ma inp rima approssimazione va bene
        new_inter.energy=new_inter.energy+energy_p
        edges.append((other_old.int_name(), new_inter.int_name(), {"charge":new_inter.charge})) 
        pos_buffer[number].charge=pos_buffer[number].charge-1 #così che se l'altra interazione è una brems o uno stay si disattiva, perchè va a carica troppo bassa, se è un pp invece, va a +1 e quindi poi avrà solo da aggiungere un legame con l'elettrone
    neg_buffer.remove(old_inter) #elimina dal buffer pure la prima interazione, ma dal nehative


def lepton_decay(type, neg_buffer, pos_buffer, old_inter, prob, nodes, edges, step, substep, state, charge, energy):
    r2 = random.random()
    p_brems = prob['brems']['brems']
    p_ann = prob['brems']['ann']

    # Scelta tipo interazione
    if r2 < p_brems:
        new_kind = "brems"
    elif r2 < p_brems + p_ann and pos_buffer:
        new_kind = "ann"
    else:
        new_kind = "stay_e"

    def create_and_generate(kind, chg, en):
        new = interaction(kind=kind, step=step, substep=substep, charge=chg, energy=en)
        generate_interaction(nodes, edges, new, old_inter, state)
        return new

    if new_kind in {"brems", "stay_e"}:
        create_and_generate(new_kind, charge, energy)

    elif new_kind == "ann" and type == "e-":
        new_inter = create_and_generate(new_kind, 0, energy)

        # Prendi positrone
        pos_idx = random.randint(0, len(pos_buffer)-1)
        pos = pos_buffer[pos_idx]
        energy_p = 0.5 * pos.energy
        pos.energy -= energy_p
        new_inter.energy += energy_p
        edges.append((pos.int_name(), new_inter.int_name(), {"charge": new_inter.charge}))
        pos.charge -= 1

        # Prendi elettrone
        new_inter2 = create_and_generate(new_kind, 0, energy)
        neg_idx = random.randint(0, len(neg_buffer)-1)
        neg = neg_buffer[neg_idx]
        energy_e = 0.5 * neg.energy
        neg.energy -= energy_e
        new_inter2.energy += energy_e
        edges.append((neg.int_name(), new_inter2.int_name(), {"charge": new_inter2.charge}))
        neg.charge -= 3

    # Rimuovi l'interazione vecchia dal buffer
    buffer = neg_buffer if type == "e-" else pos_buffer
    buffer.remove(old_inter)







def energy_division(total_energy, Z, interactions, buffer, daughters): 
    #daughters: "brems" (lepton e photon), "ann" (two photons), "pp" (two leptons)
    #buffer: negative for positron, positive for electrons
    energy=total_energy #l'energia a disposizione 
    if daughters=="brems":
        foton_energy = np.random.exponential(scale=0.5*energy) #fotone si prende una distribuzione esponenziale dell'energia, con costante 10%dell'energia totale
        foton_energy = min(foton_energy, energy * 0.99) #così che sicuro non sfora l'energia )anche se poco probabile)
        lepton_energy=energy-foton_energy
        prob = build_markov(lepton_energy, foton_energy, Z, buffer, interactions)
        energy = [lepton_energy, foton_energy]
        return prob, energy
    if daughters=="pp":
        if total_energy < 2 * me_c2:
            raise ValueError("Cannot have pair production if energy < 2*electron mass")
        K_total = total_energy - 2 * me_c2
        fraction = np.clip(np.random.normal(loc=0.5, scale=0.05), 0.1, 0.9)
        K_electron = K_total * fraction
        K_positron = K_total - K_electron
        prob = build_markov(0, K_electron, Z, buffer, interactions)
        energy = [ K_positron, K_electron]
        return prob, energy
    if daughters=="ann":
        E_tot = total_energy + 2 * me_c2  # energia totale nel centro di massa
        # Supponiamo distribuzione simmetrica ±10% attorno a metà energia
        E1_fraction = np.clip(np.random.normal(loc=0.5, scale=0.1), 0.1, 0.9)
        E_gamma1 = E_tot * E1_fraction
        E_gamma2 = E_tot - E_gamma1
        prob = build_markov(0, E_gamma1, Z, buffer, interactions)
        return prob, [E_gamma1, E_gamma2]
    if daughters=="stay_e":
        prob = build_markov(total_energy, 0, Z, buffer, interactions)
        return prob, total_energy
    if daughters=="stay_p":
        prob=build_markov(0, total_energy, Z, buffer, interactions)
        return prob, total_energy