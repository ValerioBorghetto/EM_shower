import networkx as nx
import random
from build_shower.em_utils import *
from build_shower.em_shower import *
from shower_analysis import *
from markov_analysis import *
import matplotlib.pyplot as plt

import time

###settings###########
initial_energy=2000  #(MeV), >10 for relativistic limit
depth=40            #Maximum depth of the material
material_Z=40       #work in progress
####################

#markov chain: nodo=stato, link= transizione possibile tra stati. ad ogni stato è associata una probabilità di transizione 
"""
start=time.time()
#generate the shower
shower, energy_deposed, markov_array=generate_shower(depth=depth, initial_energy=initial_energy, Z=material_Z, initial_particle="electron") #30--->2 seconds

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
#plot different interaction kinds occurrence
plot_kinds(shower)
#plot the width of the shower 
plot_width(shower)
#plot the occurrence of each interaction per shower level
level_count(shower)


#study the mean values over different initial energy values
shower_study(10, 1000, 10, energy=True, width=True)

#markov analysis
#fa la media di tutte le markov della shower, e ne studia le misure di centralità
avg_matrix=average_markov(markov_array)
print(avg_matrix)

avg_graph=draw_markov(avg_matrix)
measures=["random walk", "eigenvector", "betweenness", "in_degree", "out_degree", "flow betweenness"]
for m in measures:
    meas=centrality_meas(avg_graph, kind=m)

#study_properties(10, 1000, 10)

#network_degree(300)"""

interaction_show()
