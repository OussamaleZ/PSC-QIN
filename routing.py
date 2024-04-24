#préparation des modules

import pandas as pd
import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
import math



#préparation des données
villes = []
edges_with_weights =  #doit etre une liste de triplets (villeA,villeB, log(probabilité))

#Création du graphe
G=nx.Graph()


G.add_nodes_from (["villeA",...])
G.add_weighted_edges_from (edges_with_weights) 

#Given two cities 'A' and 'B'. To find the shortest path : we do this
shortest_path = nx.shortest_path(G, source='A', target='D') #sous la forme ['A', villes intérmediaires, 'B']
#The we can make a list where we put all the shortest paths between every two edges.

# 