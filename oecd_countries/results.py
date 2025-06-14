# %% [markdown]
# #Libraries upload

# %%
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

file_base = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_paesi\\dati_oecd_per_anni\\"
# %%
#liste per le coordinate dei plot finali

ita_coord = []
chn_coord = []
usa_coord = []
brn_coord = []
jpn_coord = []
ind_coord = []
ukr_coord = []
# %%
#comando che mostra tutte le righe degli output

pd.set_option("display.max_rows", None)

np.random.seed(42)

# %% [markdown]
# #1995

# %%
#dati normalizzati (\sum_{i,j=1}^N a_{ij}=1, dove A=[a_{ij}] è la matrice di adiacenza pesata che descrive la rete)
G95 = nx.read_weighted_edgelist(f'{file_base}1995_normalizzato.txt', create_using=nx.DiGraph)


#dati invertiti (a'_{ij}=a_{ij}^-1) per calcolare la closeness, in modo che sugli archi i pesi siano invertiti,
#visto che due paesi sono tanto più "vicini" (economicamente parlando) quanto più commerciano fra loro
G95_c = nx.read_weighted_edgelist(f'{file_base}1995_closeness.txt', create_using=nx.DiGraph)


#dati iniziali senza modifiche, visto che con dati normalizzati l'algoritmo che estrae l'eigenvector centrality ha problemi di convergenza.
#con dati iniziali non ci sono comunque problemi per quanto riguarda l'ordine di grandezza della eigenvector estratta dai nodi
G95_e = nx.read_weighted_edgelist(f'{file_base}1995_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
#codice per rimuovere gli autoanelli dalla rete con dati normalizzati e salvarli in selfloop_column, che va poi aggiunto al set delle feature

aa = list(sorted(nx.selfloop_edges(G95)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G95.edges[aa[i]]['weight'])
G95.remove_edges_from(nx.selfloop_edges(G95))

# %%
#codice per rimuovere gli autoanelli dalla rete con dati invertiti

ab = list(sorted(nx.selfloop_edges(G95_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G95_c.edges[ab[i]]['weight'])
G95_c.remove_edges_from(nx.selfloop_edges(G95_c))

# %%
#codice per rimuovere gli autoanelli dalla rete con dati iniziali

ac = list(sorted(nx.selfloop_edges(G95_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G95_e.edges[ac[i]]['weight'])
G95_e.remove_edges_from(nx.selfloop_edges(G95_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G95.in_degree(weight='weight')

#out_strength
out_str = G95.out_degree(weight='weight')

#total_strength
tot_str = G95.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G95)

#betweenness
bet = nx.betweenness_centrality(G95, weight='weight')

#clustering coefficient
clc = nx.clustering(G95, weight='weight')

#pagerank
pag = nx.pagerank(G95, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G95_c, distance="weight")
out_clo = nx.closeness_centrality(G95_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G95_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
#%pip install fuzzy-c-means

# %%
from fcmeans import FCM

# %%
#fuzzy c-means e risultati (matrice U)

feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)

# %%
#matrice V (centri dei cluster)

V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print("Matrice V originale:")
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]
# Rename the columns
roles_percentages_sorted.columns = ['Role_0', 'Role_1', 'Role_2']



print("\nPercentuali di ruolo originarie:")
print(roles_percentages)
print("\nPercentuali di ruolo ordinate:")
print(roles_percentages_sorted)



# %%
#codice per salvare i ruoli con valore massimo di appartenenza per ogni nodo. Serve per calcolare il Rand score

labels_1995 = list(nodes_roles)
for i in range(len(labels_1995)):
    labels_1995[i] = int(labels_1995[i][-1])

# %%
print(len(labels_1995))

# %%
#codice per appendere le coordinate nei tre ruoli degli stati in analisi


coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)

# Supponiamo di voler scambiare i valori di Role_0 tra 'Node1' e 'Node2'
temp = roles_percentages_sorted.loc['JPN', 'Role_2']
roles_percentages_sorted.loc['JPN', 'Role_2'] = roles_percentages_sorted.loc['JPN', 'Role_1']
roles_percentages_sorted.loc['JPN', 'Role_1'] = temp
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)
# %%
#export su gephi per plot della rete con colori dei nodi che rappresentano i ruoli con valore massimo di appartenenza

label = {}

j = 0

for node in G95.nodes:
    label[node] = nodes_roles.loc[node]
    j = j + 1

nx.set_node_attributes(G95, label, 'fcm')

nx.write_graphml(G95, '1995.graphml')

# Colori consistenti per i ruoli
role_colors = {'Role_0': 'blue', 'Role_1': 'red', 'Role_2': 'green'}

# Posizione dei nodi
pos = nx.spring_layout(G95,k=0.08)  # o una posizione predefinita se disponibile

# Disegno dei nodi come torte
fig, ax = plt.subplots(figsize=(10,10))
for node in G95.nodes():
    percentages = roles_percentages_sorted.loc[node, ['Role_0', 'Role_1', 'Role_2']].values
    wedges, texts = ax.pie(percentages, colors=[role_colors['Role_0'], role_colors['Role_1'], role_colors['Role_2']], radius=0.1, center=pos[node], wedgeprops=dict(width=0.03, edgecolor='w'))
    ax.text(pos[node][0], pos[node][1], node, horizontalalignment='center', verticalalignment='center', fontweight='bold', color='black')


# Disegno degli archi
edges = nx.draw_networkx_edges(G95, pos, ax=ax, alpha=0.05, edge_color='gray',width=0.5)  # riduci l'alpha per trasparenza, cambia il colore per meno visibilità
# Aggiunta della legenda
ax.legend(wedges, ['Role_0', 'Role_1', 'Role_2'], title="Roles", loc="best")

# Aggiustamenti grafici
ax.set_aspect('equal')
plt.xlim([min(x[0] for x in pos.values()) - 0.1, max(x[0] for x in pos.values()) + 0.1])
plt.ylim([min(x[1] for x in pos.values()) - 0.1, max(x[1] for x in pos.values()) + 0.1])

plt.axis('off')

#plt.show()
plt.savefig('g95.png',bbox_inches='tight')
plt.close()

# %% [markdown]
# #1996


# %%
G96 = nx.read_weighted_edgelist(f'{file_base}1996_normalizzato.txt', create_using=nx.DiGraph)
G96_c = nx.read_weighted_edgelist(f'{file_base}1996_closeness.txt', create_using=nx.DiGraph)
G96_e = nx.read_weighted_edgelist(f'{file_base}1996_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G96)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G96.edges[aa[i]]['weight'])
G96.remove_edges_from(nx.selfloop_edges(G96))

# %%
ab = list(sorted(nx.selfloop_edges(G96_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G96_c.edges[ab[i]]['weight'])
G96_c.remove_edges_from(nx.selfloop_edges(G96_c))

# %%
ac = list(sorted(nx.selfloop_edges(G96_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G96_e.edges[ac[i]]['weight'])
G96_e.remove_edges_from(nx.selfloop_edges(G96_e))

# %% [markdown]
# ##Feature extraction


# %%
#in_strength
in_str = G96.in_degree(weight='weight')

#out_strength
out_str = G96.out_degree(weight='weight')

#total_strength
tot_str = G96.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G96)

#betweenness
bet = nx.betweenness_centrality(G96, weight='weight')

#clustering coefficient
clc = nx.clustering(G96, weight='weight')

#pagerank
pag = nx.pagerank(G96, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G96_c, distance="weight")
out_clo = nx.closeness_centrality(G96_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G96_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
print("roles percentages:",roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print("1996:",V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]


# %%
labels_1996 = list(nodes_roles)
for i in range(len(labels_1996)):
    labels_1996[i] = int(labels_1996[i][-1])

# %%
print(len(labels_1996))

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #1997

# %%
G97 = nx.read_weighted_edgelist(f'{file_base}1997_normalizzato.txt', create_using=nx.DiGraph)
G97_c = nx.read_weighted_edgelist(f'{file_base}1997_closeness.txt', create_using=nx.DiGraph)
G97_e = nx.read_weighted_edgelist(f'{file_base}1997_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G97)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G97.edges[aa[i]]['weight'])
G97.remove_edges_from(nx.selfloop_edges(G97))

# %%
ab = list(sorted(nx.selfloop_edges(G97_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G97_c.edges[ab[i]]['weight'])
G97_c.remove_edges_from(nx.selfloop_edges(G97_c))

# %%
ac = list(sorted(nx.selfloop_edges(G97_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G97_e.edges[ac[i]]['weight'])
G97_e.remove_edges_from(nx.selfloop_edges(G97_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G97.in_degree(weight='weight')

#out_strength
out_str = G97.out_degree(weight='weight')

#total_strength
tot_str = G97.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G97)

#betweenness
bet = nx.betweenness_centrality(G97, weight='weight')

#clustering coefficient
clc = nx.clustering(G97, weight='weight')

#pagerank
pag = nx.pagerank(G97, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G97_c, distance="weight")
out_clo = nx.closeness_centrality(G97_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G97_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print("1997:",V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



# %%
labels_1997 = list(nodes_roles)
for i in range(len(labels_1997)):
    labels_1997[i] = int(labels_1997[i][-1])

# %%
print(len(labels_1997))

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #1998

# %%
G98 = nx.read_weighted_edgelist(f'{file_base}1998_normalizzato.txt', create_using=nx.DiGraph)
G98_c = nx.read_weighted_edgelist(f'{file_base}1998_closeness.txt', create_using=nx.DiGraph)
G98_e = nx.read_weighted_edgelist(f'{file_base}1998_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G98)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G98.edges[aa[i]]['weight'])
G98.remove_edges_from(nx.selfloop_edges(G98))

# %%
ab = list(sorted(nx.selfloop_edges(G98_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G98_c.edges[ab[i]]['weight'])
G98_c.remove_edges_from(nx.selfloop_edges(G98_c))

# %%
ac = list(sorted(nx.selfloop_edges(G98_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G98_e.edges[ac[i]]['weight'])
G98_e.remove_edges_from(nx.selfloop_edges(G98_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G98.in_degree(weight='weight')

#out_strength
out_str = G98.out_degree(weight='weight')

#total_strength
tot_str = G98.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G98)

#betweenness
bet = nx.betweenness_centrality(G98, weight='weight')

#clustering coefficient
clc = nx.clustering(G98, weight='weight')

#pagerank
pag = nx.pagerank(G98, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G98_c, distance="weight")
out_clo = nx.closeness_centrality(G98_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G98_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1998:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]


# %%
labels_1998 = list(nodes_roles)
for i in range(len(labels_1998)):
    labels_1998[i] = int(labels_1998[i][-1])

# %%
#codice utile in caso, analizzando V, si noti come i ruoli restituiti dal fcm
#siano sfasati rispetto agli anni precedenti (role_0 in realtà è role_2 e viceversa) e quindi serva invertire le coordinate

coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)

coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #1999

# %%
G99 = nx.read_weighted_edgelist(f'{file_base}1999_normalizzato.txt', create_using=nx.DiGraph)
G99_c = nx.read_weighted_edgelist(f'{file_base}1999_closeness.txt', create_using=nx.DiGraph)
G99_e = nx.read_weighted_edgelist(f'{file_base}1999_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G99)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G99.edges[aa[i]]['weight'])
G99.remove_edges_from(nx.selfloop_edges(G99))

# %%
ab = list(sorted(nx.selfloop_edges(G99_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G99_c.edges[ab[i]]['weight'])
G99_c.remove_edges_from(nx.selfloop_edges(G99_c))

# %%
ac = list(sorted(nx.selfloop_edges(G99_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G99_e.edges[ac[i]]['weight'])
G99_e.remove_edges_from(nx.selfloop_edges(G99_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G99.in_degree(weight='weight')

#out_strength
out_str = G99.out_degree(weight='weight')

#total_strength
tot_str = G99.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G99)

#betweenness
bet = nx.betweenness_centrality(G99, weight='weight')

#clustering coefficient
clc = nx.clustering(G99, weight='weight')

#pagerank
pag = nx.pagerank(G99, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G99_c, distance="weight")
out_clo = nx.closeness_centrality(G99_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G99_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print("1999:",V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_1999 = list(nodes_roles)
for i in range(len(labels_1999)):
    labels_1999[i] = int(labels_1999[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)



coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2000

# %%
G00 = nx.read_weighted_edgelist(f'{file_base}2000_normalizzato.txt', create_using=nx.DiGraph)
G00_c = nx.read_weighted_edgelist(f'{file_base}2000_closeness.txt', create_using=nx.DiGraph)
G00_e = nx.read_weighted_edgelist(f'{file_base}2000_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G00)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G00.edges[aa[i]]['weight'])
G00.remove_edges_from(nx.selfloop_edges(G00))

# %%
ab = list(sorted(nx.selfloop_edges(G00_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G00_c.edges[ab[i]]['weight'])
G00_c.remove_edges_from(nx.selfloop_edges(G00_c))

# %%
ac = list(sorted(nx.selfloop_edges(G00_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G00_e.edges[ac[i]]['weight'])
G00_e.remove_edges_from(nx.selfloop_edges(G00_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G00.in_degree(weight='weight')

#out_strength
out_str = G00.out_degree(weight='weight')

#total_strength
tot_str = G00.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G00)

#betweenness
bet = nx.betweenness_centrality(G00, weight='weight')

#clustering coefficient
clc = nx.clustering(G00, weight='weight')

#pagerank
pag = nx.pagerank(G00, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G00_c, distance="weight")
out_clo = nx.closeness_centrality(G00_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G00_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2000 = list(nodes_roles)
for i in range(len(labels_2000)):
    labels_2000[i] = int(labels_2000[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)


# %% [markdown]
# #2001

# %%
G01 = nx.read_weighted_edgelist(f'{file_base}2001_normalizzato.txt', create_using=nx.DiGraph)
G01_c = nx.read_weighted_edgelist(f'{file_base}2001_closeness.txt', create_using=nx.DiGraph)
G01_e = nx.read_weighted_edgelist(f'{file_base}2001_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G01)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G01.edges[aa[i]]['weight'])
G01.remove_edges_from(nx.selfloop_edges(G01))

# %%
ab = list(sorted(nx.selfloop_edges(G01_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G01_c.edges[ab[i]]['weight'])
G01_c.remove_edges_from(nx.selfloop_edges(G01_c))

# %%
ac = list(sorted(nx.selfloop_edges(G01_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G01_e.edges[ac[i]]['weight'])
G01_e.remove_edges_from(nx.selfloop_edges(G01_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G01.in_degree(weight='weight')

#out_strength
out_str = G01.out_degree(weight='weight')

#total_strength
tot_str = G01.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G01)

#betweenness
bet = nx.betweenness_centrality(G01, weight='weight')

#clustering coefficient
clc = nx.clustering(G01, weight='weight')

#pagerank
pag = nx.pagerank(G01, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G01_c, distance="weight")
out_clo = nx.closeness_centrality(G01_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G01_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2001 = list(nodes_roles)
for i in range(len(labels_2001)):
    labels_2001[i] = int(labels_2001[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2002

# %%
G02 = nx.read_weighted_edgelist(f'{file_base}2002_normalizzato.txt', create_using=nx.DiGraph)
G02_c = nx.read_weighted_edgelist(f'{file_base}2002_closeness.txt', create_using=nx.DiGraph)
G02_e = nx.read_weighted_edgelist(f'{file_base}2002_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G02)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G02.edges[aa[i]]['weight'])
G02.remove_edges_from(nx.selfloop_edges(G02))

# %%
ab = list(sorted(nx.selfloop_edges(G02_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G02_c.edges[ab[i]]['weight'])
G02_c.remove_edges_from(nx.selfloop_edges(G02_c))

# %%
ac = list(sorted(nx.selfloop_edges(G02_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G02_e.edges[ac[i]]['weight'])
G02_e.remove_edges_from(nx.selfloop_edges(G02_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G02.in_degree(weight='weight')

#out_strength
out_str = G02.out_degree(weight='weight')

#total_strength
tot_str = G02.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G02)

#betweenness
bet = nx.betweenness_centrality(G02, weight='weight')

#clustering coefficient
clc = nx.clustering(G02, weight='weight')

#pagerank
pag = nx.pagerank(G02, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G02_c, distance="weight")
out_clo = nx.closeness_centrality(G02_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G02_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2002 = list(nodes_roles)
for i in range(len(labels_2002)):
    labels_2002[i] = int(labels_2002[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2003

# %%
G03 = nx.read_weighted_edgelist(f'{file_base}2003_normalizzato.txt', create_using=nx.DiGraph)
G03_c = nx.read_weighted_edgelist(f'{file_base}2003_closeness.txt', create_using=nx.DiGraph)
G03_e = nx.read_weighted_edgelist(f'{file_base}2003_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G03)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G03.edges[aa[i]]['weight'])
G03.remove_edges_from(nx.selfloop_edges(G03))

# %%
ab = list(sorted(nx.selfloop_edges(G03_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G03_c.edges[ab[i]]['weight'])
G03_c.remove_edges_from(nx.selfloop_edges(G03_c))

# %%
ac = list(sorted(nx.selfloop_edges(G03_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G03_e.edges[ac[i]]['weight'])
G03_e.remove_edges_from(nx.selfloop_edges(G03_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G03.in_degree(weight='weight')

#out_strength
out_str = G03.out_degree(weight='weight')

#total_strength
tot_str = G03.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G03)

#betweenness
bet = nx.betweenness_centrality(G03, weight='weight')

#clustering coefficient
clc = nx.clustering(G03, weight='weight')

#pagerank
pag = nx.pagerank(G03, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G03_c, distance="weight")
out_clo = nx.closeness_centrality(G03_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G03_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2003 = list(nodes_roles)
for i in range(len(labels_2003)):
    labels_2003[i] = int(labels_2003[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2004

# %%
G04 = nx.read_weighted_edgelist(f'{file_base}2004_normalizzato.txt', create_using=nx.DiGraph)
G04_c = nx.read_weighted_edgelist(f'{file_base}2004_closeness.txt', create_using=nx.DiGraph)
G04_e = nx.read_weighted_edgelist(f'{file_base}2004_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G04)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G04.edges[aa[i]]['weight'])
G04.remove_edges_from(nx.selfloop_edges(G04))

# %%
ab = list(sorted(nx.selfloop_edges(G04_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G04_c.edges[ab[i]]['weight'])
G04_c.remove_edges_from(nx.selfloop_edges(G04_c))

# %%
ac = list(sorted(nx.selfloop_edges(G04_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G04_e.edges[ac[i]]['weight'])
G04_e.remove_edges_from(nx.selfloop_edges(G04_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G04.in_degree(weight='weight')

#out_strength
out_str = G04.out_degree(weight='weight')

#total_strength
tot_str = G04.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G04)

#betweenness
bet = nx.betweenness_centrality(G04, weight='weight')

#clustering coefficient
clc = nx.clustering(G04, weight='weight')

#pagerank
pag = nx.pagerank(G04, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G04_c, distance="weight")
out_clo = nx.closeness_centrality(G04_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G04_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2004 = list(nodes_roles)
for i in range(len(labels_2004)):
    labels_2004[i] = int(labels_2004[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2005

# %%
G05 = nx.read_weighted_edgelist(f'{file_base}2005_normalizzato.txt', create_using=nx.DiGraph)
G05_c = nx.read_weighted_edgelist(f'{file_base}2005_closeness.txt', create_using=nx.DiGraph)
G05_e = nx.read_weighted_edgelist(f'{file_base}2005_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G05)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G05.edges[aa[i]]['weight'])
G05.remove_edges_from(nx.selfloop_edges(G05))

# %%
ab = list(sorted(nx.selfloop_edges(G05_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G05_c.edges[ab[i]]['weight'])
G05_c.remove_edges_from(nx.selfloop_edges(G05_c))

# %%
ac = list(sorted(nx.selfloop_edges(G05_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G05_e.edges[ac[i]]['weight'])
G05_e.remove_edges_from(nx.selfloop_edges(G05_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G05.in_degree(weight='weight')

#out_strength
out_str = G05.out_degree(weight='weight')

#total_strength
tot_str = G05.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G05)

#betweenness
bet = nx.betweenness_centrality(G05, weight='weight')

#clustering coefficient
clc = nx.clustering(G05, weight='weight')

#pagerank
pag = nx.pagerank(G05, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G05_c, distance="weight")
out_clo = nx.closeness_centrality(G05_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G05_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2005 = list(nodes_roles)
for i in range(len(labels_2001)):
    labels_2005[i] = int(labels_2005[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2006

# %%
G06 = nx.read_weighted_edgelist(f'{file_base}2006_normalizzato.txt', create_using=nx.DiGraph)
G06_c = nx.read_weighted_edgelist(f'{file_base}2006_closeness.txt', create_using=nx.DiGraph)
G06_e = nx.read_weighted_edgelist(f'{file_base}2006_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G06)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G06.edges[aa[i]]['weight'])
G06.remove_edges_from(nx.selfloop_edges(G06))

# %%
ab = list(sorted(nx.selfloop_edges(G06_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G06_c.edges[ab[i]]['weight'])
G06_c.remove_edges_from(nx.selfloop_edges(G06_c))

# %%
ac = list(sorted(nx.selfloop_edges(G06_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G06_e.edges[ac[i]]['weight'])
G06_e.remove_edges_from(nx.selfloop_edges(G06_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G06.in_degree(weight='weight')

#out_strength
out_str = G06.out_degree(weight='weight')

#total_strength
tot_str = G06.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G06)

#betweenness
bet = nx.betweenness_centrality(G06, weight='weight')

#clustering coefficient
clc = nx.clustering(G06, weight='weight')

#pagerank
pag = nx.pagerank(G06, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G06_c, distance="weight")
out_clo = nx.closeness_centrality(G06_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G06_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2006 = list(nodes_roles)
for i in range(len(labels_2006)):
    labels_2006[i] = int(labels_2006[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2007
# %%
G07 = nx.read_weighted_edgelist(f'{file_base}2007_normalizzato.txt', create_using=nx.DiGraph)
G07_c = nx.read_weighted_edgelist(f'{file_base}2007_closeness.txt', create_using=nx.DiGraph)
G07_e = nx.read_weighted_edgelist(f'{file_base}2007_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G07)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G07.edges[aa[i]]['weight'])
G07.remove_edges_from(nx.selfloop_edges(G07))

# %%
ab = list(sorted(nx.selfloop_edges(G07_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G07_c.edges[ab[i]]['weight'])
G07_c.remove_edges_from(nx.selfloop_edges(G07_c))

# %%
ac = list(sorted(nx.selfloop_edges(G07_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G07_e.edges[ac[i]]['weight'])
G07_e.remove_edges_from(nx.selfloop_edges(G07_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G07.in_degree(weight='weight')

#out_strength
out_str = G07.out_degree(weight='weight')

#total_strength
tot_str = G07.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G07)

#betweenness
bet = nx.betweenness_centrality(G07, weight='weight')

#clustering coefficient
clc = nx.clustering(G07, weight='weight')

#pagerank
pag = nx.pagerank(G07, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G07_c, distance="weight")
out_clo = nx.closeness_centrality(G07_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G07_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2007 = list(nodes_roles)
for i in range(len(labels_2007)):
    labels_2007[i] = int(labels_2007[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2008
# %%
G08 = nx.read_weighted_edgelist(f'{file_base}2008_normalizzato.txt', create_using=nx.DiGraph)
G08_c = nx.read_weighted_edgelist(f'{file_base}2008_closeness.txt', create_using=nx.DiGraph)
G08_e = nx.read_weighted_edgelist(f'{file_base}2008_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G08)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G08.edges[aa[i]]['weight'])
G08.remove_edges_from(nx.selfloop_edges(G08))

# %%
ab = list(sorted(nx.selfloop_edges(G08_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G08_c.edges[ab[i]]['weight'])
G08_c.remove_edges_from(nx.selfloop_edges(G08_c))

# %%
ac = list(sorted(nx.selfloop_edges(G08_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G08_e.edges[ac[i]]['weight'])
G08_e.remove_edges_from(nx.selfloop_edges(G08_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G08.in_degree(weight='weight')

#out_strength
out_str = G08.out_degree(weight='weight')

#total_strength
tot_str = G08.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G08)

#betweenness
bet = nx.betweenness_centrality(G08, weight='weight')

#clustering coefficient
clc = nx.clustering(G08, weight='weight')

#pagerank
pag = nx.pagerank(G08, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G08_c, distance="weight")
out_clo = nx.closeness_centrality(G08_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G08_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2008 = list(nodes_roles)
for i in range(len(labels_2008)):
    labels_2008[i] = int(labels_2008[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2009
# %%

G09 = nx.read_weighted_edgelist(f'{file_base}2009_normalizzato.txt', create_using=nx.DiGraph)
G09_c = nx.read_weighted_edgelist(f'{file_base}2009_closeness.txt', create_using=nx.DiGraph)
G09_e = nx.read_weighted_edgelist(f'{file_base}2009_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G09)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G09.edges[aa[i]]['weight'])
G09.remove_edges_from(nx.selfloop_edges(G09))

# %%
ab = list(sorted(nx.selfloop_edges(G09_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G09_c.edges[ab[i]]['weight'])
G09_c.remove_edges_from(nx.selfloop_edges(G09_c))

# %%
ac = list(sorted(nx.selfloop_edges(G09_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G09_e.edges[ac[i]]['weight'])
G09_e.remove_edges_from(nx.selfloop_edges(G09_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G09.in_degree(weight='weight')

#out_strength
out_str = G09.out_degree(weight='weight')

#total_strength
tot_str = G09.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G09)

#betweenness
bet = nx.betweenness_centrality(G09, weight='weight')

#clustering coefficient
clc = nx.clustering(G09, weight='weight')

#pagerank
pag = nx.pagerank(G09, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G09_c, distance="weight")
out_clo = nx.closeness_centrality(G09_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G09_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2009 = list(nodes_roles)
for i in range(len(labels_2009)):
    labels_2009[i] = int(labels_2009[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2010

# %%
G10 = nx.read_weighted_edgelist(f'{file_base}2010_normalizzato.txt', create_using=nx.DiGraph)
G10_c = nx.read_weighted_edgelist(f'{file_base}2010_closeness.txt', create_using=nx.DiGraph)
G10_e = nx.read_weighted_edgelist(f'{file_base}2010_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G10)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G10.edges[aa[i]]['weight'])
G10.remove_edges_from(nx.selfloop_edges(G10))

# %%
ab = list(sorted(nx.selfloop_edges(G10_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G10_c.edges[ab[i]]['weight'])
G10_c.remove_edges_from(nx.selfloop_edges(G10_c))

# %%
ac = list(sorted(nx.selfloop_edges(G10_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G10_e.edges[ac[i]]['weight'])
G10_e.remove_edges_from(nx.selfloop_edges(G10_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G10.in_degree(weight='weight')

#out_strength
out_str = G10.out_degree(weight='weight')

#total_strength
tot_str = G10.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G10)

#betweenness
bet = nx.betweenness_centrality(G10, weight='weight')

#clustering coefficient
clc = nx.clustering(G10, weight='weight')

#pagerank
pag = nx.pagerank(G10, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G10_c, distance="weight")
out_clo = nx.closeness_centrality(G10_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G10_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2010 = list(nodes_roles)
for i in range(len(labels_2010)):
    labels_2010[i] = int(labels_2010[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2011

# %%
G11 = nx.read_weighted_edgelist(f'{file_base}2011_normalizzato.txt', create_using=nx.DiGraph)
G11_c = nx.read_weighted_edgelist(f'{file_base}2011_closeness.txt', create_using=nx.DiGraph)
G11_e = nx.read_weighted_edgelist(f'{file_base}2011_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G11)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G11.edges[aa[i]]['weight'])
G11.remove_edges_from(nx.selfloop_edges(G11))

# %%
ab = list(sorted(nx.selfloop_edges(G11_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G11_c.edges[ab[i]]['weight'])
G11_c.remove_edges_from(nx.selfloop_edges(G11_c))

# %%
ac = list(sorted(nx.selfloop_edges(G11_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G11_e.edges[ac[i]]['weight'])
G11_e.remove_edges_from(nx.selfloop_edges(G11_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G11.in_degree(weight='weight')

#out_strength
out_str = G11.out_degree(weight='weight')

#total_strength
tot_str = G11.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G11)

#betweenness
bet = nx.betweenness_centrality(G11, weight='weight')

#clustering coefficient
clc = nx.clustering(G11, weight='weight')

#pagerank
pag = nx.pagerank(G11, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G11_c, distance="weight")
out_clo = nx.closeness_centrality(G11_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G11_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]


# %%
labels_2011 = list(nodes_roles)
for i in range(len(labels_2011)):
    labels_2011[i] = int(labels_2011[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2012

# %%
G12 = nx.read_weighted_edgelist(f'{file_base}2012_normalizzato.txt', create_using=nx.DiGraph)
G12_c = nx.read_weighted_edgelist(f'{file_base}2012_closeness.txt', create_using=nx.DiGraph)
G12_e = nx.read_weighted_edgelist(f'{file_base}2012_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G12)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G12.edges[aa[i]]['weight'])
G12.remove_edges_from(nx.selfloop_edges(G12))

# %%
ab = list(sorted(nx.selfloop_edges(G12_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G12_c.edges[ab[i]]['weight'])
G12_c.remove_edges_from(nx.selfloop_edges(G12_c))

# %%
ac = list(sorted(nx.selfloop_edges(G12_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G12_e.edges[ac[i]]['weight'])
G12_e.remove_edges_from(nx.selfloop_edges(G12_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G12.in_degree(weight='weight')

#out_strength
out_str = G12.out_degree(weight='weight')

#total_strength
tot_str = G12.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G12)

#betweenness
bet = nx.betweenness_centrality(G12, weight='weight')

#clustering coefficient
clc = nx.clustering(G12, weight='weight')

#pagerank
pag = nx.pagerank(G12, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G12_c, distance="weight")
out_clo = nx.closeness_centrality(G12_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G12_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=1.5)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]

# %%
labels_2012 = list(nodes_roles)
for i in range(len(labels_2012)):
    labels_2012[i] = int(labels_2012[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2013

# %%
G13 = nx.read_weighted_edgelist(f'{file_base}2013_normalizzato.txt', create_using=nx.DiGraph)
G13_c = nx.read_weighted_edgelist(f'{file_base}2013_closeness.txt', create_using=nx.DiGraph)
G13_e = nx.read_weighted_edgelist(f'{file_base}2013_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G13)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G13.edges[aa[i]]['weight'])
G13.remove_edges_from(nx.selfloop_edges(G13))

# %%
ab = list(sorted(nx.selfloop_edges(G13_c)))
selfloop_column = []
for i in range(len(ab)):
    selfloop_column.append(G13_c.edges[ab[i]]['weight'])
G13_c.remove_edges_from(nx.selfloop_edges(G13_c))

# %%
ac = list(sorted(nx.selfloop_edges(G13_e)))
selfloop_column = []
for i in range(len(ac)):
    selfloop_column.append(G13_e.edges[ac[i]]['weight'])
G13_e.remove_edges_from(nx.selfloop_edges(G13_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G13.in_degree(weight='weight')

#out_strength
out_str = G13.out_degree(weight='weight')

#total_strength
tot_str = G13.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G13)

#betweenness
bet = nx.betweenness_centrality(G13, weight='weight')

#clustering coefficient
clc = nx.clustering(G13, weight='weight')

#pagerank
pag = nx.pagerank(G13, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G13_c, distance="weight")
out_clo = nx.closeness_centrality(G13_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G13_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



    
    

    
    

    




# %%
labels_2013 = list(nodes_roles)
for i in range(len(labels_2013)):
    labels_2013[i] = int(labels_2013[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)



coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2014

# %%
G14 = nx.read_weighted_edgelist(f'{file_base}2014_normalizzato.txt', create_using=nx.DiGraph)
G14_c = nx.read_weighted_edgelist(f'{file_base}2014_closeness.txt', create_using=nx.DiGraph)
G14_e = nx.read_weighted_edgelist(f'{file_base}2014_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G14)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G14.edges[aa[i]]['weight'])
G14.remove_edges_from(nx.selfloop_edges(G14))

# %%
ab = list(sorted(nx.selfloop_edges(G14_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G14_c.edges[ab[i]]['weight'])
G14_c.remove_edges_from(nx.selfloop_edges(G14_c))

# %%
ac = list(sorted(nx.selfloop_edges(G14_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G14_e.edges[ac[i]]['weight'])
G14_e.remove_edges_from(nx.selfloop_edges(G14_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G14.in_degree(weight='weight')

#out_strength
out_str = G14.out_degree(weight='weight')

#total_strength
tot_str = G14.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G14)

#betweenness
bet = nx.betweenness_centrality(G14, weight='weight')

#clustering coefficient
clc = nx.clustering(G14, weight='weight')

#pagerank
pag = nx.pagerank(G14, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G14_c, distance="weight")
out_clo = nx.closeness_centrality(G14_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G14_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



    
    

    
    

    



# %%
labels_2014 = list(nodes_roles)
for i in range(len(labels_2014)):
    labels_2014[i] = int(labels_2014[i][-1])



# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)


coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)

c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2015

# %%
G15 = nx.read_weighted_edgelist(f'{file_base}2015_normalizzato.txt', create_using=nx.DiGraph)
G15_c = nx.read_weighted_edgelist(f'{file_base}2015_closeness.txt', create_using=nx.DiGraph)
G15_e = nx.read_weighted_edgelist(f'{file_base}2015_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G15)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G15.edges[aa[i]]['weight'])
G15.remove_edges_from(nx.selfloop_edges(G15))

# %%
ab = list(sorted(nx.selfloop_edges(G15_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G15_c.edges[ab[i]]['weight'])
G15_c.remove_edges_from(nx.selfloop_edges(G15_c))

# %%
ac = list(sorted(nx.selfloop_edges(G15_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G15_e.edges[ac[i]]['weight'])
G15_e.remove_edges_from(nx.selfloop_edges(G15_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G15.in_degree(weight='weight')

#out_strength
out_str = G15.out_degree(weight='weight')

#total_strength
tot_str = G15.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G15)

#betweenness
bet = nx.betweenness_centrality(G15, weight='weight')

#clustering coefficient
clc = nx.clustering(G15, weight='weight')

#pagerank
pag = nx.pagerank(G15, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G15_c, distance="weight")
out_clo = nx.closeness_centrality(G15_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G15_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



    
    

    
    

    




# %%
labels_2015 = list(nodes_roles)
for i in range(len(labels_2015)):
    labels_2015[i] = int(labels_2015[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)

coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2016

# %%
G16 = nx.read_weighted_edgelist(f'{file_base}2016_normalizzato.txt', create_using=nx.DiGraph)
G16_c = nx.read_weighted_edgelist(f'{file_base}2016_closeness.txt', create_using=nx.DiGraph)
G16_e = nx.read_weighted_edgelist(f'{file_base}2016_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G16)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G16.edges[aa[i]]['weight'])
G16.remove_edges_from(nx.selfloop_edges(G16))

# %%
ab = list(sorted(nx.selfloop_edges(G16_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G16_c.edges[ab[i]]['weight'])
G16_c.remove_edges_from(nx.selfloop_edges(G16_c))

# %%
ac = list(sorted(nx.selfloop_edges(G16_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G16_e.edges[ac[i]]['weight'])
G16_e.remove_edges_from(nx.selfloop_edges(G16_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G16.in_degree(weight='weight')

#out_strength
out_str = G16.out_degree(weight='weight')

#total_strength
tot_str = G16.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G16)

#betweenness
bet = nx.betweenness_centrality(G16, weight='weight')

#clustering coefficient
clc = nx.clustering(G16, weight='weight')

#pagerank
pag = nx.pagerank(G16, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G16_c, distance="weight")
out_clo = nx.closeness_centrality(G16_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G16_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



    
    

    
    

    




# %%
labels_2016 = list(nodes_roles)
for i in range(len(labels_2016)):
    labels_2016[i] = int(labels_2016[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)


coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %% [markdown]
# #2017

# %%
G17 = nx.read_weighted_edgelist(f'{file_base}2017_normalizzato.txt', create_using=nx.DiGraph)
G17_c = nx.read_weighted_edgelist(f'{file_base}2017_closeness.txt', create_using=nx.DiGraph)
G17_e = nx.read_weighted_edgelist(f'{file_base}2017_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G17)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G17.edges[aa[i]]['weight'])
G17.remove_edges_from(nx.selfloop_edges(G17))

# %%
ab = list(sorted(nx.selfloop_edges(G17_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G17_c.edges[ab[i]]['weight'])
G17_c.remove_edges_from(nx.selfloop_edges(G17_c))

# %%
ac = list(sorted(nx.selfloop_edges(G17_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G17_e.edges[ac[i]]['weight'])
G17_e.remove_edges_from(nx.selfloop_edges(G17_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G17.in_degree(weight='weight')

#out_strength
out_str = G17.out_degree(weight='weight')

#total_strength
tot_str = G17.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G17)

#betweenness
bet = nx.betweenness_centrality(G17, weight='weight')

#clustering coefficient
clc = nx.clustering(G17, weight='weight')

#pagerank
pag = nx.pagerank(G17, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G17_c, distance="weight")
out_clo = nx.closeness_centrality(G17_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G17_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n1995:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



    
    

    
    

    




# %%
labels_2017 = list(nodes_roles)
for i in range(len(labels_2017)):
    labels_2017[i] = int(labels_2017[i][-1])

# %%
coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)


coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %%
print(len(labels_2017))

# %%
label = {}

j = 0

for node in G17.nodes:
    label[node] = nodes_roles.loc[node]
    j = j + 1

nx.set_node_attributes(G17, label, 'fcm')

nx.write_graphml(G17, '2017.graphml')
# %% [markdown]
# #2018

# %%
G18 = nx.read_weighted_edgelist(f'{file_base}2018_normalizzato.txt', create_using=nx.DiGraph)
G18_c = nx.read_weighted_edgelist(f'{file_base}2018_closeness.txt', create_using=nx.DiGraph)
G18_e = nx.read_weighted_edgelist(f'{file_base}2018_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
aa = list(sorted(nx.selfloop_edges(G18)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G18.edges[aa[i]]['weight'])
G18.remove_edges_from(nx.selfloop_edges(G18))

# %%
ab = list(sorted(nx.selfloop_edges(G18_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G18_c.edges[ab[i]]['weight'])
G18_c.remove_edges_from(nx.selfloop_edges(G18_c))

# %%
ac = list(sorted(nx.selfloop_edges(G18_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G18_e.edges[ac[i]]['weight'])
G18_e.remove_edges_from(nx.selfloop_edges(G18_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G18.in_degree(weight='weight')

#out_strength
out_str = G18.out_degree(weight='weight')

#total_strength
tot_str = G18.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G18)

#betweenness
bet = nx.betweenness_centrality(G18, weight='weight')

#clustering coefficient
clc = nx.clustering(G18, weight='weight')

#pagerank
pag = nx.pagerank(G18, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G18_c, distance="weight")
out_clo = nx.closeness_centrality(G18_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G18_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n2018:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



    
    

    
    

    





# %%
labels_2018 = list(nodes_roles)
for i in range(len(labels_2018)):
    labels_2018[i] = int(labels_2018[i][-1])

# %%
#codice per appendere le coordinate nei tre ruoli degli stati in analisi

coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)



coord = roles_percentages_sorted.loc['CHN'].tolist()

chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()

brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()

usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()

jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()

ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %%
label = {}

j = 0

for node in G18.nodes:
    label[node] = nodes_roles.loc[node]
    j = j + 1

nx.set_node_attributes(G18, label, 'fcm')

nx.write_graphml(G18, '2018.graphml')

# #2019

# %%
#dati normalizzati (\sum_{i,j=1}^N a_{ij}=1, dove A=[a_{ij}] è la matrice di adiacenza pesata che descrive la rete)
G19 = nx.read_weighted_edgelist(f'{file_base}2019_normalizzato.txt', create_using=nx.DiGraph)


#dati invertiti (a'_{ij}=a_{ij}^-1) per calcolare la closeness, in modo che sugli archi i pesi siano invertiti,
#visto che due paesi sono tanto più "vicini" (economicamente parlando) quanto più commerciano fra loro
G19_c = nx.read_weighted_edgelist(f'{file_base}2019_closeness.txt', create_using=nx.DiGraph)


#dati iniziali senza modifiche, visto che con dati normalizzati l'algoritmo che estrae l'eigenvector centrality ha problemi di convergenza.
#con dati iniziali non ci sono comunque problemi per quanto riguarda l'ordine di grandezza della eigenvector estratta dai nodi
G19_e = nx.read_weighted_edgelist(f'{file_base}2019_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
#codice per rimuovere gli autoanelli dalla rete con dati normalizzati e salvarli in selfloop_column, che va poi aggiunto al set delle feature

aa = list(sorted(nx.selfloop_edges(G19)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G19.edges[aa[i]]['weight'])
G19.remove_edges_from(nx.selfloop_edges(G19))

# %%
#codice per rimuovere gli autoanelli dalla rete con dati invertiti

ab = list(sorted(nx.selfloop_edges(G19_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G19_c.edges[ab[i]]['weight'])
G19_c.remove_edges_from(nx.selfloop_edges(G19_c))

# %%
#codice per rimuovere gli autoanelli dalla rete con dati iniziali

ac = list(sorted(nx.selfloop_edges(G19_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G19_e.edges[ac[i]]['weight'])
G19_e.remove_edges_from(nx.selfloop_edges(G19_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G19.in_degree(weight='weight')

#out_strength
out_str = G19.out_degree(weight='weight')

#total_strength
tot_str = G19.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G19)

#betweenness
bet = nx.betweenness_centrality(G19, weight='weight')

#clustering coefficient
clc = nx.clustering(G19, weight='weight')

#pagerank
pag = nx.pagerank(G19, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G19_c, distance="weight")
out_clo = nx.closeness_centrality(G19_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G19_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
#print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
#%pip install fuzzy-c-means

# %%
from fcmeans import FCM

# %%
#fuzzy c-means e risultati (matrice U)

feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n2019:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]



    
    

    
    

    


# %%
#matrice V (centri dei cluster)




# %%
#codice per salvare i ruoli con valore massimo di appartenenza per ogni nodo. Serve per calcolare il Rand score

labels_2019 = list(nodes_roles)
for i in range(len(labels_2019)):
    labels_2019[i] = int(labels_2019[i][-1])

# %%
#codice per appendere le coordinate nei tre ruoli degli stati in analisi

coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)

coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)

# %%
print(len(labels_2019))
# %%
label = {}

j = 0

for node in G19.nodes:
    label[node] = nodes_roles.loc[node]
    j = j + 1

nx.set_node_attributes(G19, label, 'fcm')

nx.write_graphml(G19, '2019.graphml')


# #2020

# %%
#dati normalizzati (\sum_{i,j=1}^N a_{ij}=1, dove A=[a_{ij}] è la matrice di adiacenza pesata che descrive la rete)
G20 = nx.read_weighted_edgelist(f'{file_base}2020_normalizzato.txt', create_using=nx.DiGraph)


#dati invertiti (a'_{ij}=a_{ij}^-1) per calcolare la closeness, in modo che sugli archi i pesi siano invertiti,
#visto che due paesi sono tanto più "vicini" (economicamente parlando) quanto più commerciano fra loro
G20_c = nx.read_weighted_edgelist(f'{file_base}2020_closeness.txt', create_using=nx.DiGraph)


#dati iniziali senza modifiche, visto che con dati normalizzati l'algoritmo che estrae l'eigenvector centrality ha problemi di convergenza.
#con dati iniziali non ci sono comunque problemi per quanto riguarda l'ordine di grandezza della eigenvector estratta dai nodi
G20_e = nx.read_weighted_edgelist(f'{file_base}2020_paesi_edgelist.txt', create_using=nx.DiGraph)

# %%
#codice per rimuovere gli autoanelli dalla rete con dati normalizzati e salvarli in selfloop_column, che va poi aggiunto al set delle feature

aa = list(sorted(nx.selfloop_edges(G20)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G20.edges[aa[i]]['weight'])
G20.remove_edges_from(nx.selfloop_edges(G20))

# %%
#codice per rimuovere gli autoanelli dalla rete con dati invertiti

ab = list(sorted(nx.selfloop_edges(G20_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G20_c.edges[ab[i]]['weight'])
G20_c.remove_edges_from(nx.selfloop_edges(G20_c))

# %%
#codice per rimuovere gli autoanelli dalla rete con dati iniziali

ac = list(sorted(nx.selfloop_edges(G20_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G20_e.edges[ac[i]]['weight'])
G20_e.remove_edges_from(nx.selfloop_edges(G20_e))

# %% [markdown]
# ##Feature extraction

# %%
#in_strength
in_str = G20.in_degree(weight='weight')

#out_strength
out_str = G20.out_degree(weight='weight')

#total_strength
tot_str = G20.degree(weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G20)

#betweenness
bet = nx.betweenness_centrality(G20, weight='weight')

#clustering coefficient
clc = nx.clustering(G20, weight='weight')

#pagerank
pag = nx.pagerank(G20, alpha=1, weight='weight')

# %%
#closeness
in_clo = nx.closeness_centrality(G20_c, distance="weight")
out_clo = nx.closeness_centrality(G20_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G20_e, weight='weight')

# %%
in_strength = [x[1] for x in in_str]

# %%
out_strength = [x[1] for x in out_str]

# %%
tot_strength = [x[1] for x in tot_str]

# %%
features=pd.DataFrame({'in_strength': in_strength})

# %%
features.index=bet.keys()

# %%
features['out_strength']=out_strength
features['total_strength']=tot_strength
features['selfloop']=selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# %%
##print(features)

# %%
row_names = features.index


# %% [markdown]
# ##Feature selection + FCM

# %%
#%pip install fuzzy-c-means

# %%
from fcmeans import FCM

# %%
#fuzzy c-means e risultati (matrice U)

feat = features.drop(columns=['betweenness', 'selfloop'], axis=1)
fcm = FCM(n_clusters=3, m=2)


fcm.fit(feat.values)

roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
roles_percentages.index = row_names
#print('\n2020:')
#print(roles_percentages)
#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)

print("Indici roles_percentages originale:", roles_percentages.index)
#print(nodes_roles)
V = pd.DataFrame(fcm.centers, columns = ['in_strength','out_strength','total_strength', 'eigenvector', 'in_closeness', 'out_closeness', 'clustering_coefficient', 'pagerank', 'hubs', 'authorities'])
print("Indici V originale:", V.index)
print(V)
V['sum'] = V.sum(axis=1)
sorted_indices = V['sum'].argsort().values
print("Ordine dei cluster basato sulla somma:", sorted_indices)
V_sorted = V.loc[sorted_indices]
print("V ordinata:")
print(V_sorted)
roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]
print("roles percentages before rename:",roles_percentages_sorted)
# Rename the columns
roles_percentages_sorted.columns = ['Role_0', 'Role_1', 'Role_2']
print("roles percentages after rename:",roles_percentages_sorted)
# %%
#codice per salvare i ruoli con valore massimo di appartenenza per ogni nodo. Serve per calcolare il Rand score

labels_2020 = list(nodes_roles)
for i in range(len(labels_2020)):
    labels_2020[i] = int(labels_2020[i][-1])


# %%
#codice per appendere le coordinate nei tre ruoli degli stati in analisi

coords = roles_percentages_sorted.loc['ITA'].tolist()
ita_coord.append(coords)
coord = roles_percentages_sorted.loc['CHN'].tolist()
chn_coord.append(coord)
coor = roles_percentages_sorted.loc['BRN'].tolist()
brn_coord.append(coor)
c_u = roles_percentages_sorted.loc['USA'].tolist()
usa_coord.append(c_u)
c_j = roles_percentages_sorted.loc['JPN'].tolist()
jpn_coord.append(c_j)
c_i = roles_percentages_sorted.loc['IND'].tolist()
ind_coord.append(c_i)
c_k = roles_percentages_sorted.loc['UKR'].tolist()
ukr_coord.append(c_k)


# %%
print(len(labels_2020))

# %%
label = {}

j = 0

for node in G20.nodes:
    label[node] = nodes_roles.loc[node]
    j = j + 1

nx.set_node_attributes(G20, label, 'fcm')

nx.write_graphml(G20, '2020.graphml')

# Colori consistenti per i ruoli
role_colors = {'Role_0': 'blue', 'Role_1': 'red', 'Role_2': 'green'}

# Posizione dei nodi
pos = nx.spring_layout(G20,k=0.08)  # o una posizione predefinita se disponibile

# Disegno dei nodi come torte
fig, ax = plt.subplots(figsize=(10,10))
for node in G20.nodes():
    percentages = roles_percentages_sorted.loc[node, ['Role_0', 'Role_1', 'Role_2']].values
    wedges, texts = ax.pie(percentages, colors=[role_colors['Role_0'], role_colors['Role_1'], role_colors['Role_2']], radius=0.1, center=pos[node], wedgeprops=dict(width=0.03, edgecolor='w'))
    ax.text(pos[node][0], pos[node][1], node, horizontalalignment='center', verticalalignment='center', fontweight='bold', color='black')


# Disegno degli archi
edges = nx.draw_networkx_edges(G20, pos, ax=ax, alpha=0.05, edge_color='gray',width=0.5)  # riduci l'alpha per trasparenza, cambia il colore per meno visibilità
# Aggiunta della legenda
ax.legend(wedges, ['Role_0', 'Role_1', 'Role_2'], title="Roles", loc="best")

# Aggiustamenti grafici
ax.set_aspect('equal')
plt.xlim([min(x[0] for x in pos.values()) - 0.1, max(x[0] for x in pos.values()) + 0.1])
plt.ylim([min(x[1] for x in pos.values()) - 0.1, max(x[1] for x in pos.values()) + 0.1])

plt.axis('off')

#plt.show()
plt.savefig('g20.png',bbox_inches='tight')
plt.close()
# %%
#inizializzazione liste per le coordinate nei tre ruoli di ogni paese

role_0_ita = []
role_1_ita = []
role_2_ita = []
#role_3_ita = []
role_0_chn = []
role_1_chn = []
role_2_chn = []
#role_3_chn = []
role_0_brn = []
role_1_brn = []
role_2_brn = []
#role_3_brn = []
role_0_usa = []
role_1_usa = []
role_2_usa = []
#role_3_usa = []
role_0_jpn = []
role_1_jpn = []
role_2_jpn = []
#role_3_jpn = []
role_0_ind = []
role_1_ind = []
role_2_ind = []
#role_3_ind = []
role_0_ukr = []
role_1_ukr = []
role_2_ukr = []
#role_3_ukr = []

# %%
#coordinate nei tre ruoli dei paesi
#print("ita:",ita_coord)
#print("chn:",chn_coord)
#print("usa:",usa_coord)
#print("brn:",brn_coord)
#print("jpn:",jpn_coord)
#print("ind:",ind_coord)
#print("ukr:",ukr_coord)
for i in range(len(ita_coord)):
    role_0_ita.append(ita_coord[i][0])
    role_1_ita.append(ita_coord[i][1])
    role_2_ita.append(ita_coord[i][2])
    #role_3_ita.append(ita_coord[i][3])
    role_0_chn.append(chn_coord[i][0])
    role_1_chn.append(chn_coord[i][1])
    role_2_chn.append(chn_coord[i][2])
    #role_3_chn.append(chn_coord[i][3])
    role_0_brn.append(brn_coord[i][0])
    role_1_brn.append(brn_coord[i][1])
    role_2_brn.append(brn_coord[i][2])
    #role_3_brn.append(brn_coord[i][3])
    role_0_usa.append(usa_coord[i][0])
    role_1_usa.append(usa_coord[i][1])
    role_2_usa.append(usa_coord[i][2])
    #role_3_usa.append(usa_coord[i][3])
    role_0_jpn.append(jpn_coord[i][0])
    role_1_jpn.append(jpn_coord[i][1])
    role_2_jpn.append(jpn_coord[i][2])
    #role_3_jpn.append(jpn_coord[i][3])
    role_0_ind.append(ind_coord[i][0])
    role_1_ind.append(ind_coord[i][1])
    role_2_ind.append(ind_coord[i][2])
    #role_3_ind.append(ind_coord[i][3])
    role_0_ukr.append(ukr_coord[i][0])
    role_1_ukr.append(ukr_coord[i][1])
    role_2_ukr.append(ukr_coord[i][2])
    #role_3_ukr.append(ukr_coord[i][3])

print("role ita 0:",role_0_ita)
print("role ita 1:",role_1_ita)
print("role ita 2:",role_2_ita)
# %%
x = [1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,2019,2020]
#x = [1995, 1998, 2001, 2004, 2007, 2010, 2013, 2015, 2018]

# %%
plt.plot(x, role_0_ita, label="role_0")
plt.plot(x, role_1_ita, label="role_1")
plt.plot(x, role_2_ita, label="role_2")
#plt.plot(x, role_3_ita, label="role_3")
plt.legend(bbox_to_anchor =(0.0,-0.17, 1, -0.027), mode= 'expand', loc='lower center', borderaxespad=0., ncols=3)
plt.title('Italy')
#plt.show()
plt.savefig('ITA.png',bbox_inches='tight')
plt.close()

# %%
plt.plot(x, role_0_chn, label="role_0")
plt.plot(x, role_1_chn, label="role_1")
plt.plot(x, role_2_chn, label="role_2")
#plt.plot(x, role_3_chn, label="role_3")
plt.legend(bbox_to_anchor =(0.0,-0.17, 1, -0.027), mode= 'expand', loc='lower center',
           borderaxespad=0., ncols=3)
plt.title('China')
#plt.show()
plt.savefig('CHI.png',bbox_inches='tight')
plt.close()

# %%
plt.plot(x, role_0_usa, label="role_0")
plt.plot(x, role_1_usa, label="role_1")
plt.plot(x, role_2_usa, label="role_2")
#plt.plot(x, role_3_usa, label="role_3")
plt.legend(bbox_to_anchor =(0.0,-0.17, 1, -0.027), mode= 'expand', loc='lower center', borderaxespad=0., ncols=3)
plt.title('USA')
#plt.show()
plt.savefig('USA.png',bbox_inches='tight')
plt.close()

# %%
plt.plot(x, role_0_brn, label="role_0")
plt.plot(x, role_1_brn, label="role_1")
plt.plot(x, role_2_brn, label="role_2")
#plt.plot(x, role_3_brn, label="role_3")
plt.legend(bbox_to_anchor =(0.0,-0.17, 1, -0.027), mode= 'expand', loc='lower center', borderaxespad=0., ncols=3)
plt.title('Brunei')
#plt.show()
plt.savefig('BRU.png',bbox_inches='tight')
plt.close()

# %%
plt.plot(x, role_0_jpn, label="role_0")
plt.plot(x, role_1_jpn, label="role_1")
plt.plot(x, role_2_jpn, label="role_2")
#plt.plot(x, role_3_jpn, label="role_3")
plt.legend(bbox_to_anchor =(0.0,-0.17, 1, -0.027), mode= 'expand', loc='lower center', borderaxespad=0., ncols=3)
plt.title('Japan')
#plt.show()
plt.savefig('JPN.png',bbox_inches='tight')
plt.close()

# %%
plt.plot(x, role_0_ind, label="role_0")
plt.plot(x, role_1_ind, label="role_1")
plt.plot(x, role_2_ind, label="role_2")
#plt.plot(x, role_3_ind, label="role_3")
plt.legend(bbox_to_anchor =(0.0,-0.17, 1, -0.027), mode= 'expand', loc='lower center', borderaxespad=0., ncols=3)
plt.title('India')
#plt.show()
plt.savefig('IND.png',bbox_inches='tight')
plt.close()

plt.plot(x, role_0_ukr, label="role_0")
plt.plot(x, role_1_ukr, label="role_1")
plt.plot(x, role_2_ukr, label="role_2")
#plt.plot(x, role_3_ukr, label="role_3")
plt.legend(bbox_to_anchor =(0.0,-0.17, 1, -0.027), mode= 'expand', loc='lower center', borderaxespad=0., ncols=3)
plt.title('Ucraina')
#plt.show()
plt.savefig('UKR.png',bbox_inches='tight')
plt.close()
# %% [markdown]
# ##Rand score

# %%
from sklearn.metrics import rand_score

# %%
r1 = rand_score(labels_1995, labels_1996)
r2 = rand_score(labels_1996, labels_1997)
r3 = rand_score(labels_1997, labels_1998)
r4 = rand_score(labels_1998, labels_1999)
r5 = rand_score(labels_1999, labels_2000)
r6 = rand_score(labels_2001, labels_2002)
r7 = rand_score(labels_2002, labels_2003)
r8 = rand_score(labels_2003, labels_2004)
r9 = rand_score(labels_2004, labels_2005)
r10 = rand_score(labels_2005, labels_2006)
r11 = rand_score(labels_2006, labels_2007)
r12 = rand_score(labels_2007, labels_2008)
r13 = rand_score(labels_2008, labels_2009)
r14 = rand_score(labels_2009, labels_2010)
r15 = rand_score(labels_2010, labels_2011)
r16 = rand_score(labels_2011, labels_2012)
r17 = rand_score(labels_2012, labels_2013)
r18 = rand_score(labels_2013, labels_2014)
r19 = rand_score(labels_2014, labels_2015)
r20 = rand_score(labels_2015, labels_2016)
r21 = rand_score(labels_2016, labels_2017)
r22 = rand_score(labels_2017, labels_2018)
r23 = rand_score(labels_2018, labels_2019)
r24 = rand_score(labels_2019, labels_2020)

# %%
rand_score=[r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20,r21,r22,r23,r24]

# %%
print('rand score')
print(rand_score)

# %%
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # main axes
ax.plot(rand_score)
ax.set_title('Rand_Score')
lab = ['1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017','2018','2019']

positions = list(range(len(lab)))
# Set the positions and labels of the ticks
ax.set_xticks(positions)
ax.set_xticklabels(lab,rotation=45)
#plt.show()
plt.savefig('rand_score.png',bbox_inches='tight')
plt.close()

#plt.plot(rand_score, label="rand_score")
#fig, ax = plt.subplots()
#lab = ['1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017']
#ax.set_xticklabels(lab)
#plt.title('Rand_Score')
#plt.show()


