import networkx as nx
import random
from build_shower.em_utils import *
from build_shower.em_shower import *
from shower_analysis import *
from markov_analysis import *
import matplotlib.pyplot as plt
import time

###Settings###########
initial_energy=300  #(MeV), >10 for relativistic limit
depth=40            #Maximum depth of the material
material_Z=40       #work in progress
####################

#execution time
start=time.time()
#generate the shower
shower, energy_deposed, norm=generate_shower(depth=depth, initial_energy=initial_energy, Z=material_Z, initial_particle="electron") #30--->2 seconds
end_time = time.time()
execution_time = end_time - start
print(f"Execution time: {execution_time} seconds")

#plot the shower
plot_shower(shower, tree=True, color=True)

#study of the out-degrees
#out_degree_study(300, 50)
out_degree_study_max_width(300, 50)
#degree vs energy values
plot_degree_vs_energy_with_error()

#study of the shower properties:
#plot deposited energy
plot_energy(energy_deposed)
#plot different interaction kinds occurrence
plot_kinds(300, 50)
#plot the width of the shower 
plot_width(shower)
#plot the occurrence of each interaction per shower level
level_count(shower)

#draw the mean transition matrix
draw_markov(norm)
markov_plot(norm)

#study the mean values over different initial energy values
shower_study(10, 1000, 10, energy=True, width=True)

#markov transition and network analysis
study_properties(10, 1000, 10)

#study tree properties
network_degree(300)

#average markov simulation vs shower simulation
analyze_markov_vs_shower()

