import geopy
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import numpy as np
from numpy import linalg
import networkx as nx
import matplotlib.pyplot as plt
import math

#préparation des données. probas est fonction de la distance

villes = ["Lille","Tourcoing","Roubaix","Villeneuve d'Ascq"]
nombre_villes = len(villes)
#préparation des données
probas = [1,0.8284518828451883, 0.824, 0.6942148760330579, 0.7412280701754386, 0.6779661016949152, 0.6979591836734694, 0.7129629629629629, 0.625531914893617, 0.5714285714285714,0.5779467680608364, 0.5078125, 0.43568464730290457, 0.48616600790513836, 0.49795918367346936, 0.50199203187251, 0.4253731343283582, 0.4744186046511628, 0.3795918367346939, 0.35772357723577236, 0.3605150214592275, 0.3707865168539326, 0.3661417322834646, 0.33047210300429186, 0.30833333333333335, 0.311787072243346, 0.2911392405063291, 0.2981651376146789, 0.26359832635983266, 0.25, 0.23651452282157676, 0.1953125, 0.22270742358078602, 0.23247232472324722, 0.20434782608695654, 0.20155038759689922, 0.2025862068965517, 0.19540229885057472, 0.21367521367521367, 0.14096916299559473, 0.1889763779527559, 0.11851851851851852, 0.1450381679389313, 0.14227642276422764, 0.17355371900826447, 0.08583690987124463, 0.09787234042553192, 0.12719298245614036, 0.13157894736842105, 0.08482142857142858, 0.14102564102564102, 0.08898305084745763, 0.1111111111111111, 0.10212765957446808, 0.09448818897637795, 0.102880658436214, 0.10300429184549356, 0.07468879668049792, 0.10344827586206896, 0.09090909090909091, 0.06172839506172839, 0.07114624505928854, 0.04291845493562232, 0.060240963855421686, 0.045871559633027525, 0.06511627906976744, 0.05907172995780591, 0.04460966542750929, 0.05363984674329502, 0.058333333333333334, 0.04310344827586207, 0.03508771929824561, 0.024896265560165973, 0.05, 0.04824561403508772, 0.036734693877551024, 0.015151515151515152, 0.02074688796680498, 0.03734439834024896, 0.028925619834710745]
minus_log_probas = [-math.log(x) for x in probas]

distances = [('Paris, France', 'Boulogne-Billancourt, France', 6), ('Paris, France', 'Saint-Denis, France', 9), ('Paris, France', 'Versailles, France', 15), ('Paris, France', 'Palaiseau, France', 17), ('Paris, France', 'Evry-Courcouronnes, France', 27), ('Boulogne-Billancourt, France', 'Saint-Denis, France', 16), ('Boulogne-Billancourt, France', 'Versailles, France', 9), ('Boulogne-Billancourt, France', 'Palaiseau, France', 13), ('Boulogne-Billancourt, France', 'Evry-Courcouronnes, France', 27), ('Saint-Denis, France', 'Boulogne-Billancourt, France', 14), ('Saint-Denis, France', 'Versailles, France', 22), ('Saint-Denis, France', 'Palaiseau, France', 26), ('Saint-Denis, France', 'Evry-Courcouronnes, France', 35), ('Versailles, France', 'Palaiseau, France', 13), ('Versailles, France', 'Evry-Courcouronnes, France', 30),  ('Palaiseau, France', 'Evry-Courcouronnes, France', 17) ]#on ne garde que les villes entre lesquelles il y a une distance inférieure à 10km

dist_lim = 18
distances_used = [t for t in distances if t[2]<dist_lim]

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

