# EM_shower
This repository contains a model to build and study electromagnetic showers using complex network theory. 

# Logic of the Cascade Creation
The model i based on 5 possible interactions:
- Bremsstralhung
- Annihilation
- Stay an electron (or positron)
* Pair production
* Stay a photon


The first three are the possible processes for an electron (or positron), while the remaining two are for a photon. At each step, for every particle, one of the possible processes is selected (the underlying idea is a Markov process). The choice depends on various probabilities, which are calculated at each step based on the particle’s energy (the dependence on the material has not yet been implemented). At each step, the energy is transferred from the parent particles to the daughter particles, halving at each iteration. When the energy falls below a certain threshold, it is absorbed by the target.
The network is built using interactions as nodes and particles as links, in order to better simulate the propagation of the cascade in the material.

# Analysis of the simulation

Work in progress

## Requirements

To run this project, simply execute the `script.py` file. You then need the following Python packages:

- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [NetworkX](https://networkx.org/)
- [SciPy](https://scipy.org/)
- [PyGraphviz](https://pygraphviz.github.io/)
  - Install system dependencies:  
    ```bash
    sudo apt-get install graphviz graphviz-dev
    ```
  - Then install PyGraphviz with pip:  
    ```bash
    pip install pygraphviz
    ```
- [tqdm](https://github.com/tqdm/tqdm)

