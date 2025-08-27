# EM_shower
This repository contains a model to build and study electromagnetic showers using complex network theory. 

# Logica della creazione della cascata.
Il modello si basa su 5 possibili interazioni:
- Bremsstralhung
- Annihilation
- Rimanere un elettrone (o positrone)
* Pair production
* Rimanere un fotone


Le prime tre sono le tre possibilità che un elettrone (o positrone) ha, le restanti due di un fotone. Ad ogni step, per ogni particella viene scelto uno dei possibili processi (l'idea alla base è un Markov process). La scelta dipende dalle vaire probabilità, che sono calcolate di volta in volta a seconda dell'energia della particella (la dipendenza dal materiale ancora non è stata inserita). Ad ogni step, l'energia viene trasferita dalle particelle madri alle figlie, dimezzandosi di volta in volta. Quando l'energia va sotto una certa soglia, questa viene assorbita dal bersaglio.
La rete viene costruita usando come nodi le interazioni, e come link le particelle, così da simulare meglio la propagazione della cascata nel materiale. 

# Analisi della simulazione

Work in progress

# Pacchetti necessari per il funzionamento:
Per far andare il codice sono necessari i seguenti pacchetti:
- Numpy
- Matplotlib
- Networkx
