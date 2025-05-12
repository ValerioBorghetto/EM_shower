import networkx as nx
import random
from build_shower.em_utils import *
from build_shower.em_shower import *
import matplotlib.pyplot as plt

import time

###settings###########
initial_energy=100 #(MeV), >10 for relativistic limit
depth=40
material_Z=40 #per ora inutile
####################

#disegna la rete che governa la cascata considerando l'energia iniziale (da sistemare)
#draw_markov(initial_energy, tree=True, adj_matrix=True)

start=time.time()
#genera la cascata
shower, energy_deposed=generate_shower(depth=depth, initial_energy=initial_energy, Z=material_Z) #30--->2 seconds

#tempo di esecuzione
end_time = time.time()
execution_time = end_time - start
print(f"Execution time: {execution_time} seconds")

#plotta la shower
plot_shower(shower, tree=True, color=True) #tree decide se vuoi la raffigurazione ad albero, color se vuoi gli edges colorati
#plotta l'energy depositata
plot_energy(energy_deposed)







