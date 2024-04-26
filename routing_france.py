import geopy
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
import math

#préparation des données. probas est fonction de la distance


dist_lim = 18
print(distances_used)
edges_with_weights= [] #doit etre une liste de triplets (villeA,villeB, log(probabilité))
for t in distances_used :
    edges_with_weights.append((t[0],t[1],minus_log_probas[t[2]]))


G=nx.Graph()

G.add_nodes_from (villes)
G.add_weighted_edges_from (edges_with_weights) 

shortest_paths = []
for villeA in villes:
    for villeB in villes:
        if villeA != villeB :
            shortest_paths.append(nx.shortest_path(G, source=villeA, target=villeB))

print("les plus courts chemins en ne conisédernat que des distances inférieures à", dist_lim, "sont" , shortest_paths)

# Afficher le graphe avec les flux et les poids des arêtes
plt.figure(figsize = (12,12))

pos = nx.spring_layout(G)

def find_distance(u,v):
    for tuple in distances_used :
        if tuple[0]==u and tuple[1]==v:
            return tuple[2]
        
edge_labels = {(u, v): find_distance(u,v) for u, v, d in G.edges(data=True)}

nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, font_size=12, arrows=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

plt.title('Routgae en ile de France : on n\'utilise que les distances inférieures à 18km')
plt.show()

