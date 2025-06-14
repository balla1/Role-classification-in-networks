
#Data libraries and upload

# 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)
# 
base_path = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\"
G95 = nx.read_weighted_edgelist(f'{base_path}1995_normalizzato.txt', create_using=nx.DiGraph)
G95_c = nx.read_weighted_edgelist(f'{base_path}1995_closeness.txt', create_using=nx.DiGraph)
G95_e = nx.read_weighted_edgelist(f'{base_path}1995_aziende_edgelist.txt', create_using=nx.DiGraph)

# 
aa = list(sorted(nx.selfloop_edges(G95)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G95.edges[aa[i]]['weight'])
G95.remove_edges_from(nx.selfloop_edges(G95)) 

# 
ab = list(sorted(nx.selfloop_edges(G95_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G95_c.edges[ab[i]]['weight'])
G95_c.remove_edges_from(nx.selfloop_edges(G95_c))

# 
ac = list(sorted(nx.selfloop_edges(G95_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G95_e.edges[ac[i]]['weight'])
G95_e.remove_edges_from(nx.selfloop_edges(G95_e))


#Feature extraction

# 
#in_strength
in_str = G95.in_degree(weight='weight')

#out_strength
out_str = G95.out_degree(weight='weight')

#total_strength
tot_str = G95.degree(weight='weight')

#betweenness
bet = nx.betweenness_centrality(G95, weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G95)

#clustering coefficient
clc = nx.clustering(G95, weight='weight')

#pagerank
pag = nx.pagerank(G95, alpha=1, weight='weight')

# 
#closeness
in_clo = nx.closeness_centrality(G95_c, distance="weight")
out_clo = nx.closeness_centrality(G95_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G95_e, weight='weight')

# 
in_strength = [x[1] for x in in_str]

# 
out_strength = [x[1] for x in out_str]

# 
tot_strength = [x[1] for x in tot_str]

# 
features=pd.DataFrame({'in_strength': in_strength})

# 
features.index=bet.keys()

# 
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

# 
#print(features)


#FCM Clustering pre-feature selection

# 


# 
from fcmeans import FCM

# 
row_names=features.index

# 
pd.set_option("display.max_rows", None)

# 
from sklearn.metrics import silhouette_score


# Analisi su silhouette score e partition coefficient per la scelta dei parametri

# 
n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
pc = list()
silhouette_avg = list()
for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=1.5)
    fcm.fit(features.values)
    pc.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))

# 
print("partition with m=1.5:",pc)

# 
print("silhouette with m=1.5:",silhouette_avg)

# 
plt.subplot(1, 2, 1)
plt.plot(n_clusters_list, silhouette_avg)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=1.5')

plt.subplot(1, 2, 2)
plt.plot(n_clusters_list, pc)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=1.5')
plt.show()


n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
pc = list()
silhouette_avg = list()
# fare un altro plot affianco a questo con partition coefficient  metterli uno affianco all'altro

for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=2.0)
    fcm.fit(features.values)
    pc.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))


# 
print("partition with m=2.0:",pc)

# 
print("silhouette with m=2.0:",silhouette_avg)

# 
#plot con silhouette score e partition coefficient

plt.subplot(1, 2, 1)
plt.plot(n_clusters_list, silhouette_avg)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=2.0')

plt.subplot(1, 2, 2)
plt.plot(n_clusters_list, pc)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=2.0')
plt.show()



n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
pc = list()
silhouette_avg = list()
for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=2.5)
    fcm.fit(features.values)
    pc.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))

# 
print("partition with m=2.5:", pc)

# 
print("silhouette with m=2.5:",silhouette_avg)

# 
plt.subplot(1, 2, 1)
plt.plot(n_clusters_list, silhouette_avg)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=2.5')

plt.subplot(1, 2, 2)
plt.plot(n_clusters_list, pc)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=2.5')
plt.show()

n_clusters=len(n_clusters_list)


rows = int(np.ceil(np.sqrt(n_clusters)))
cols = int(np.ceil(n_clusters / rows))
f, axes = plt.subplots(rows, cols, figsize=(10,16))
for n_clusters, model, axe in zip(n_clusters_list, models, axes.ravel()):
    # get validation metrics
    pc = model.partition_coefficient
    fcm_centers = model.centers
    fcm_labels = model.predict(features.values)
    # plot result
    axe.scatter(features.values[:,0], features.values[:,1], c=fcm_labels, alpha=.1)
    axe.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='black')
    axe.set_title(f'n_cluster = {n_clusters}, PC = {pc:.3f}')
plt.show()




# 
#codice per plottare Partition coefficient e Partition entropy al variare di m per scegliere il valore di m
n_clusters = 3

m_values=[1.5,2,2.5]
models = []
pec = []
silhouette_avg = []

# Create and evaluate models for different fuzziness factors
for m_value in m_values:
    fcm = FCM(n_clusters=n_clusters, m=m_value)
    fcm.fit(features.values)
    pec.append(fcm.partition_entropy_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))

print("partition entropy coefficient:",pec)
num_m=len(m_values)

rows = int(np.ceil(np.sqrt(num_m)))
cols = int(np.ceil(num_m / rows))
f, axes = plt.subplots(rows, cols, figsize=(10,16))
for m_value, model, axe in zip(m_values, models, axes.ravel()):
    # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient
    fcm_centers = model.centers
    fcm_labels = model.predict(features.values)
    # plot result
    axe.scatter(features.values[:,0], features.values[:,1], c=fcm_labels, alpha=.1)
    axe.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=500, c='black')
    axe.set_title(f'm = {m_value}, PC = {pc:.3f}, PEC = {pec:.3f}')
plt.show()


# Chosen number of clusters, looking at PEC, PC and FSI is 3, m is 1.5
# scatter plot per vedere i risultati del clustering con 3 cluster e m=1.5


# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(features.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
print('\nroles percentages:')
#print(roles_percentages)

print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print(nodes_roles)


#Feature selection

# 
#valuto average silhouette score, partition coefficient e partition entropy
#eliminando una sola feature per volta dal feature set

temp_feat=features
models=list()
silhouette_avg=list()

for i in range(len(features.columns)):
  temp_feat=temp_feat.drop(temp_feat.columns[[i]], axis=1)
  fcm = FCM(n_clusters=3, m=1.5)
  fcm.fit(temp_feat.values)
  models.append(fcm)
  roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
  nodes_roles = roles_percentages.idxmax(axis=1)
  silhouette_avg.append(silhouette_score(temp_feat.values, nodes_roles))
  temp_feat=features

# 
print("silhouette average score:",silhouette_avg)


# I risultati del silhouette score indicano che eliminando selfloop o betweenness i risultati migliorano sensibilmente

# 
for model in models:
    # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient
    print("Partition Coefficient:", pc, "Partition Entropy Coefficient:", pec)


# I risultati del silhouette score sono confermati da partition coefficient e partition entropy

# 
#libreria per il laplacian score



# 
#from scipy.optimize import linear_sum_assignment as la

# 
from skfeature.function.similarity_based import SPEC
from skfeature.utility import unsupervised_evaluation

# 
features_score=features.to_numpy()

# 
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(features, **kwargs_W)

# obtain the scores of features
score_l = lap_score.lap_score(features_score, W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score_l)


# Creazione dell'istogramma
plt.bar(features.columns, score_l) # Creazione dell'istogramma con i valori di score_l e le etichette delle colonne

# Impostazione dell'angolo per le etichette sull'asse X
plt.xticks(rotation=45)  # Rotazione di 45 gradi
# Impostazione dei limiti dell'asse Y

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Mostra il grafico
plt.show()

print('Average silhouette score:')
print(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))
all_sil=silhouette_score(features.values, roles_percentages.idxmax(axis=1)) #salvo il valore per confrontarlo con i successivi, 
# %%
#in ordine dalla più rilevante alla meno rilevante
print('Partition coefficient:')
print(fcm.partition_coefficient)
all_pc=fcm.partition_coefficient #salvo il valore per confrontarlo con i successivi


# 
#in ordine dalla più rilevante alla meno rilevante

print("ordine features dalla più rilevante alla meno rilevante:",idx)
print("ordine di rilevanza delle features:", features.columns[idx])
# 
print("score features:",score_l)


# I risultati del Laplacian score confermano la poca rilevanza del selfloop, mentre sono più ambigui per quanto riguarda la betweenness



fcm_full = FCM(n_clusters=3, m=1.5)
fcm_full.fit(features.values)
full_labels = fcm_full.u.argmax(axis=1)

plt.figure(figsize=(6, 6))
plt.subplot(1, 3, 1)
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=full_labels, cmap='viridis')
plt.title('Con tutte le features')

fcm_reduced = FCM(n_clusters=3, m=1.5)
features_reduced = features.drop(features.columns[[6, 3]], axis=1)
fcm_reduced.fit(features_reduced.values)
reduced_labels = fcm_reduced.u.argmax(axis=1)

plt.subplot(1, 3, 2)
plt.scatter(features_reduced.iloc[:, 0], features_reduced.iloc[:, 1], c=reduced_labels, cmap='viridis')
plt.title('Senza feature out_closeness e selfloop')

fcm_reduced_2 = FCM(n_clusters=3, m=1.5)
features_reduced_2 = features.drop(features.columns[[6, 3, 8]], axis=1)
fcm_reduced_2.fit(features_reduced_2.values)
reduced_labels_2 = fcm_reduced_2.u.argmax(axis=1)

plt.subplot(1, 3, 3)
plt.scatter(features_reduced_2.iloc[:, 0], features_reduced_2.iloc[:, 1], c=reduced_labels_2, cmap='viridis')
plt.title('Senza feature out_closeness, selfloop e betweenness')
plt.show()
print("Columns eliminated: ", features.columns[[6, 3, 8]])

#I cluster appaiono più compatti e meglio separati.
# Questo suggerisce che l'eliminazione di queste feature ha potenzialmente migliorato la qualità del clustering, rendendo i cluster più definiti e distinti.
# eliminare le feature out_closeness e selfloop sia stata una scelta benefica per il  modello di clustering. 
#Le feature eliminate non solo avevano punteggi Laplacian elevati, indicando una minore rilevanza, 
#ma la loro rimozione ha anche migliorato le metriche di valutazione del clustering, il che è evidente sia numericamente sia visivamente.

# 
feat_self = features.drop(columns='out_closeness')

# 
#print(feat_self)

# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(feat_self.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
print('\nroles percentages:')
#print(roles_percentages)

print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print(nodes_roles)


# Feature selection su queste features

# 
feat_self_spec=feat_self.to_numpy()

# 
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(feat_self, **kwargs_W)

# obtain the scores of features
score_l = lap_score.lap_score(feat_self_spec, W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score_l)

# 
print("laplacian score:",score_l)

# 
print("ordine di rilevanza:",idx)

# 
#average silhouette score con feature selection su feat_self (senza out_closeness)

print("average silhouette score con feature selection su feat_self (senza out_closeness):",silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))

print('Average silhouette score:')
print(silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))
no_outclos_sil=silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)) # no out_closeness 
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_outclos_pc=fcm.partition_coefficient # no 

# without both out_closeness and selfloop

# 
feat = features.drop(columns=['out_closeness', 'selfloop'])

# 
# average silhouette score con feature selection su feat (senza out_closeness e selfloop)
#print(feat)

# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(feat.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
print('\nroles percentages:')
print(roles_percentages)



print('cluster centers:')
cluster_centers = pd.DataFrame(fcm.centers, columns=feat.columns) # cluster centers
print(cluster_centers) # cluster centers as pandas dataframe
print(fcm.centers) # cluster centers as numpy array

#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print("node_roles:",nodes_roles)


# Feature selection su queste features

# 
feat_spec=feat.to_numpy()

# 
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(feat, **kwargs_W)

# obtain the scores of features
score_l = lap_score.lap_score(feat_spec, W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score_l)

# 
print("laplacian score senza out_closeness e selfloop:",score_l)


# Nessun valore svetta sugli altri, dunque non c'è evidenza (anche considerando l'average silhouette score calcolato precedentemente) per eliminare ulteriori feature dal feature set

# 
print("ordine di importanza:",idx)

# 
#average silhouette score con feature selection su feat (senza out_closeness e selfloop)
print("average silhouette score con feature selection su feat (senza out_closeness e selfloop):",silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))

print('Average silhouette score:')
print(silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))
no_self_outclos_sil=silhouette_score(feat.values, roles_percentages.idxmax(axis=1)) # no selfloop silhouette score
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_self_outclos_pc=fcm.partition_coefficient # no selfloop partition coefficient

# Il risultato è ottimo per quanto riguarda l'average silhouette score, dunque è sensato rimuovere entrambe dal feature set


# without both out_closeness and selfloop

#  proviamo anche senza betweenness
feat = features.drop(columns=['betweenness','out_closeness', 'selfloop'])

# 
# average silhouette score con feature selection su feat (senza out_closeness e selfloop)
#print(feat)

# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(feat.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
print('\nroles percentages:')
print(roles_percentages)


#print('cluster centers:')
cluster_centers = pd.DataFrame(fcm.centers, columns=feat.columns) # cluster centers
#print(cluster_centers) # cluster centers as pandas dataframe
#print(fcm.centers) # cluster centers as numpy array

#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print("node_roles:",nodes_roles)


# Feature selection su queste features

# 
feat_spec=feat.to_numpy()

# 
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
W = construct_W.construct_W(feat, **kwargs_W)

# obtain the scores of features
score_l = lap_score.lap_score(feat_spec, W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score_l)

print("laplacian score senza out_closeness, selfloop e betweenness:",score_l)


# Nessun valore svetta sugli altri, dunque non c'è evidenza (anche considerando l'average silhouette score calcolato precedentemente) per eliminare ulteriori feature dal feature set

# 
print("ordine di importanza:",idx)

# 
#average silhouette score con feature selection su feat (senza out_closeness e selfloop)
print("average silhouette score con feature selection su feat (senza out_closeness, selfloop e betweenness):",silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))
print('Average silhouette score:')
print(silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))
no_self_outclos_bet_sil=silhouette_score(feat.values, roles_percentages.idxmax(axis=1)) # no selfloop silhouette score
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_self_outclos_bet_pc=fcm.partition_coefficient # no selfloop partition coefficient

# Il risultato è ottimo per quanto riguarda l'average silhouette score, dunque è sensato rimuovere tutte e 3 dal feature set


# without both out_closeness and selfloop and betweenness




sil= [all_sil, no_outclos_sil, no_self_outclos_sil, no_self_outclos_bet_sil]
pc=[all_pc, no_outclos_pc, no_self_outclos_pc, no_self_outclos_bet_pc]
sil_names=['All features', 'Without selfloop', 'Without selfloop and out_closeness', 'Without selfloop, betweenness and out_closeness']
pc_names=['All features', 'Without selfloop', 'Without selfloop and out_closeness', 'Without selfloop, betweenness and out_closeness']

# Creazione dell'istogramma
plt.bar(sil_names,sil) # Creazione dell'istogramma con i valori di score_l e le etichette delle colonne

# Impostazione dell'angolo per le etichette sull'asse X
plt.xticks(rotation=45)  # Rotazione di 45 gradi e allineamento a destra
# Impostazione dei limiti dell'asse Y
min_y = min(sil) - 0.05  # Un po' sopra il minimo dei valori
max_y = max(sil) + 0.05  # Un po' sopra il massimo dei valori
plt.ylim(min_y, max_y)
# Etichette degli assi e titolo

plt.title('Average silhouette score')
plt.tight_layout() 
# Mostra il grafico
plt.show()

# Creazione dell'istogramma
plt.bar(pc_names,pc) # Creazione dell'istogramma con i valori di score_l e le etichette delle colonne

# Impostazione dell'angolo per le etichette sull'asse X
plt.xticks(rotation=45)  # Rotazione di 45 gradi
min_y = min(pc) - 0.05 # Un po' sopra il minimo dei valori
max_y = max(pc) + 0.05  # Un po' sopra il massimo dei valori
plt.ylim(min_y, max_y)
# Etichette degli assi e titolo

plt.title('Partition coefficient')
plt.tight_layout() 
# Mostra il grafico
plt.show()