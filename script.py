import networkx as nx
import random
from build_shower.em_utils import *
from build_shower.em_shower import *
from analysys import *
import matplotlib.pyplot as plt

import time

###settings###########
initial_energy=300  #(MeV), >10 for relativistic limit
depth=40            #Maximum depth of the material
material_Z=40       #work in progress
####################
"""
#disegna la rete che governa la cascata considerando l'energia iniziale (N.B. la probabilità nella shower si aggiorna di volta in volta
#questa è invece a probabilità fissa. Per la relazione quindi avrebbe senso modificare la porbabilità così che rispecchi mediamente il comportamento
#generale della shower
graph_m=draw_markov(initial_energy, tree=True, adj_matrix=True)

#centrality measures of the markov graph (bisogna però mettere le giuste probabilità sulla markov)
measures=["eigenvector", "betweenness", "in_degree", "out_degree", "flow betweenness"]
for m in measures:
    meas=centrality_meas(graph_m, kind=m)

start=time.time()
#generate the shower
shower, energy_deposed=generate_shower(depth=depth, initial_energy=initial_energy, Z=material_Z, initial_particle="electron") #30--->2 seconds


#execution time
end_time = time.time()
execution_time = end_time - start
print(f"Execution time: {execution_time} seconds")

#plot the shower
plot_shower(shower, tree=True, color=True) #tree decide se vuoi la raffigurazione ad albero, color se vuoi gli edges colorati

#study of the adjacency matrix of the shower
adj_matrix_study(shower)

#study of the shower properties
#plot deposited energy
plot_energy(energy_deposed)

plot_kinds(shower)

plot_width(shower)

#study the mean values over different initial energy values
shower_study(10, 1000, 10, energy=True, width=True)
"""

