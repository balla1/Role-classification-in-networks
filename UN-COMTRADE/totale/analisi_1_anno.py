
#Data libraries and upload

# 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(42)
# 
base_path = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\dati\\2022\\"
G22 = nx.read_weighted_edgelist(f'{base_path}2022_normalizzato.txt', create_using=nx.DiGraph)
G22_c = nx.read_weighted_edgelist(f'{base_path}2022_closeness.txt', create_using=nx.DiGraph)
G22_e = nx.read_weighted_edgelist(f'{base_path}2022_paesi_edgelist.txt', create_using=nx.DiGraph)

# 
aa = list(sorted(nx.selfloop_edges(G22)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G22.edges[aa[i]]['weight'])
G22.remove_edges_from(nx.selfloop_edges(G22)) 

# 
ab = list(sorted(nx.selfloop_edges(G22_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G22_c.edges[ab[i]]['weight'])
G22_c.remove_edges_from(nx.selfloop_edges(G22_c))

# 
ac = list(sorted(nx.selfloop_edges(G22_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G22_e.edges[ac[i]]['weight'])
G22_e.remove_edges_from(nx.selfloop_edges(G22_e))


#Feature extraction

# 
#in_strength
in_str = G22.in_degree(weight='weight')

#out_strength
out_str = G22.out_degree(weight='weight')

#total_strength
tot_str = G22.degree(weight='weight')

#betweenness
bet = nx.betweenness_centrality(G22, weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G22)

#clustering coefficient
clc = nx.clustering(G22, weight='weight')

#pagerank
pag = nx.pagerank(G22, alpha=1, weight='weight')

# 
#closeness
in_clo = nx.closeness_centrality(G22_c, distance="weight")
out_clo = nx.closeness_centrality(G22_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G22_e, weight='weight')

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
# Aggiungi la colonna selfloop solo se selfloop_column non è vuoto
if selfloop_column:
    features['selfloop'] = selfloop_column
features['eigenvector']=features.index.map(eig)
features['in_closeness']=features.index.map(in_clo)
features['out_closeness']=features.index.map(out_clo)
features['clustering coefficient']=features.index.map(clc)
features['betweenness']=features.index.map(bet)
features['pagerank']=features.index.map(pag)
features['hubs']=features.index.map(hubs)
features['authorities']=features.index.map(auths)

# 
print("features:",features)


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
n_clusters_list = [2, 3, 4, 5, 6, 7,8,9]
models = list()
pc_1 = list()
silhouette_avg_1 = list()
for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=1.5)
    fcm.fit(features.values)
    pc_1.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg_1.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))

print("partition with m=1.5:",pc_1)
print("silhouette with m=1.5:",silhouette_avg_1)
# voglio scambiare i primi due valori di silhoutte_avg_1 tra di loro
silhouette_avg_1[0], silhouette_avg_1[1] = silhouette_avg_1[1], silhouette_avg_1[0]

n_clusters_list = [2, 3, 4, 5, 6, 7,8,9]
models = list()
pc_2 = list()
silhouette_avg_2 = list()
# fare un altro plot affianco a questo con partition coefficient  metterli uno affianco all'altro

for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=2.0)
    fcm.fit(features.values)
    pc_2.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg_2.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))


print("partition with m=2.0:",pc_2)
print("silhouette with m=2.0:",silhouette_avg_2)
silhouette_avg_2[0], silhouette_avg_2[1] = silhouette_avg_2[1], silhouette_avg_2[0]


n_clusters_list = [2, 3, 4, 5, 6, 7,8,9]
models = list()
pc_3 = list()
silhouette_avg_3 = list()
for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=2.5)
    fcm.fit(features.values)
    pc_3.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg_3.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))


print("partition with m=2.5:", pc_3)
print("silhouette with m=2.5:",silhouette_avg_3)
silhouette_avg_3[0], silhouette_avg_3[1] = silhouette_avg_3[1], silhouette_avg_3[0]
#subplot(3,2,5)  e (3,2,6) per fare un plot con silhouette score e partition coefficient per m=2.5
#3,2,5 è giusto per completare la terza riga?
plt.subplot(3, 2, 1)
plt.plot(n_clusters_list, silhouette_avg_1)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=1.5')

plt.subplot(3, 2, 2)
plt.plot(n_clusters_list, pc_1)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=1.5')

plt.subplot(3, 2, 3)
plt.plot(n_clusters_list, silhouette_avg_2)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=2.0')

plt.subplot(3, 2, 4)
plt.plot(n_clusters_list, pc_2)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=2.0')


plt.subplot(3, 2, 5)
plt.plot(n_clusters_list, silhouette_avg_3)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=2.5')

plt.subplot(3, 2, 6)
plt.plot(n_clusters_list, pc_3)
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
n_clusters = 2

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

print("partition entropy coefficient with 2 cluster:",pec)
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


# Chosen number of clusters, looking at PEC, PC and FSI is 2, m is 1.5
# scatter plot per vedere i risultati del clustering con 2 cluster e m=1.5


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
print("silhouette average score eliminando ogni volta una feature diversa nell'ordine iniziale:",silhouette_avg)


# I risultati del silhouette score indicano che eliminando selfloop o betweenness i risultati migliorano sensibilmente

i=0
for model in models:
    # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient
    eliminated_feature = features.columns[i]
    i=i+1
    print("eliminating:",eliminated_feature," Partition Coefficient :", pc, "Partition Entropy Coefficient:", pec)


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
idx = np.argsort(score_l)[::-1]

# print the names of the features in the order of their scores
print("ordine features dalla più rilevante alla meno rilevante:")
print(features.columns[idx])

# 
#in ordine dalla più rilevante alla meno rilevante

print("ordine features dalla più rilevante alla meno rilevante:",idx)

# 
print(" laplacian score features :",score_l)

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


# Eliminiamo le feature con score Laplacian più basso


fcm_full = FCM(n_clusters=3, m=1.5)
fcm_full.fit(features.values)
full_labels = fcm_full.u.argmax(axis=1)

plt.figure(figsize=(6, 6))
plt.subplot(1, 4, 1)
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=full_labels, cmap='viridis')
plt.title('Con tutte le features')

fcm_reduced = FCM(n_clusters=3, m=1.5)
features_reduced = features.drop(features.columns[[6]], axis=1)
fcm_reduced.fit(features_reduced.values)
reduced_labels = fcm_reduced.u.argmax(axis=1)

plt.subplot(1, 4, 2)
plt.scatter(features_reduced.iloc[:, 0], features_reduced.iloc[:, 1], c=reduced_labels, cmap='viridis')
plt.title('clustering_coefficient')

fcm_reduced_2 = FCM(n_clusters=3, m=1.5)
features_reduced_2 = features.drop(features.columns[[6, 7]], axis=1)
fcm_reduced_2.fit(features_reduced_2.values)
reduced_labels_2 = fcm_reduced_2.u.argmax(axis=1)

plt.subplot(1, 4, 3)
plt.scatter(features_reduced_2.iloc[:, 0], features_reduced_2.iloc[:, 1], c=reduced_labels_2, cmap='viridis')
plt.title('clust coeff, betweenness')
print("Columns eliminated: ", features.columns[[6, 7]])

fcm_reduced_3 = FCM(n_clusters=3, m=1.5)
features_reduced_3 = features.drop(features.columns[[6, 2, 7]], axis=1)
fcm_reduced_3.fit(features_reduced_3.values)
reduced_labels_3 = fcm_reduced_3.u.argmax(axis=1)

plt.subplot(1, 4, 4)
plt.scatter(features_reduced_3.iloc[:, 0], features_reduced_3.iloc[:, 1], c=reduced_labels_3, cmap='viridis')
plt.title('clust coeff,tot_stre,betweenness ')
plt.show()
print("Columns eliminated: ", features.columns[[6, 2, 7]])
#I cluster appaiono più compatti e meglio separati.
# Questo suggerisce che l'eliminazione di queste feature ha potenzialmente migliorato la qualità del clustering, rendendo i cluster più definiti e distinti.
# eliminare le feature out_closeness e selfloop sia stata una scelta benefica per il  modello di clustering. 
#Le feature eliminate non solo avevano punteggi Laplacian elevati, indicando una minore rilevanza, 
#ma la loro rimozione ha anche migliorato le metriche di valutazione del clustering, il che è evidente sia numericamente sia visivamente.

# 
feat_self = features.drop(columns=['in_closeness','out_closeness'])

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
print("laplacian score senza clustering coeff:",score_l)

# 
print("ordine di rilevanza:",idx)
plt.bar(feat_self.columns, score_l) # Creazione dell'istogramma con i valori di score_l e le etichette delle colonne

# Impostazione dell'angolo per le etichette sull'asse X
plt.xticks(rotation=45)  # Rotazione di 45 gradi
# Impostazione dei limiti dell'asse Y

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Mostra il grafico
plt.show()
# 
#average silhouette score con feature selection su feat_self (senza out_closeness)

print("average silhouette score con feature selection su feat_self (senza clustering coefficient):",silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))

print('Average silhouette score:')
print(silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))
no_closs_sil=silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)) #salvo il valore per confrontarlo con i successivi, 
# %%
#in ordine dalla più rilevante alla meno rilevante
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_closs_pc=fcm.partition_coefficient #salvo il valore per confrontarlo con i successivi

# without both out_closeness and selfloop

# 
feat = features.drop(columns=['in_closeness','out_closeness','authorities'])

# 
# average silhouette score con feature selection su feat (senza out_closeness e selfloop)
#print(feat)

# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(feat.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
print('\nroles percentages:')
#print(roles_percentages)

print('\nNodes x Roles:')
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
print("laplacian score senza clustering coefficient e betweenness:",score_l)


# Nessun valore svetta sugli altri, dunque non c'è evidenza (anche considerando l'average silhouette score calcolato precedentemente) per eliminare ulteriori feature dal feature set

# 
print("ordine di importanza:",idx)
plt.bar(feat.columns, score_l) # Creazione dell'istogramma con i valori di score_l e le etichette delle colonne

# Impostazione dell'angolo per le etichette sull'asse X
plt.xticks(rotation=45)  # Rotazione di 45 gradi
# Impostazione dei limiti dell'asse Y

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Mostra il grafico
plt.show()

print('roles percentages veri:')
print(roles_percentages)

# vorrei stampare i valori dei centri dei cluster per ogni feature e avere una tabella con i nomi delle feature e i valori del centro del cluster
print('cluster centers veri:')
cluster_centers = pd.DataFrame(fcm.centers, columns=feat.columns) # cluster centers
print(cluster_centers) # cluster centers as pandas dataframe
print(fcm.centers) # cluster centers as numpy array
# 
#average silhouette score con feature selection su feat (senza out_closeness e selfloop)
#print("average silhouette score con feature selection su feat (senza clustering coefficient e betweenness):",silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))

print('Average silhouette score:')
print(silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))
no_closs_autho_sill=silhouette_score(feat.values, roles_percentages.idxmax(axis=1)) #salvo il valore per confrontarlo con i successivi, 
# %%
#in ordine dalla più rilevante alla meno rilevante
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_closs_autho_pc=fcm.partition_coefficient #salvo il valore per confrontarlo con i successivi


# Il risultato è ottimo per quanto riguarda l'average silhouette score, dunque è sensato rimuovere entrambe dal feature set


# without both out_closeness and selfloop

#  proviamo anche senza betweenness
feat = features.drop(columns=['in_closeness','authorities','out_closeness','clustering coefficient'])

# 
# average silhouette score con feature selection su feat (senza out_closeness e selfloop)
#print(feat)

# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(feat.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
print('\nroles percentages:')
#print(roles_percentages)

print('\nNodes x Roles:')
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

print("laplacian score senza clustering_coeff, authorities,betweenness,out_closeness:",score_l)


# Nessun valore svetta sugli altri, dunque non c'è evidenza (anche considerando l'average silhouette score calcolato precedentemente) per eliminare ulteriori feature dal feature set

# 
print("ordine di importanza:",idx)
plt.bar(feat.columns, score_l) # Creazione dell'istogramma con i valori di score_l e le etichette delle colonne

# Impostazione dell'angolo per le etichette sull'asse X
plt.xticks(rotation=45)  # Rotazione di 45 gradi
# Impostazione dei limiti dell'asse Y

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Mostra il grafico
plt.show()
print('roles percentages:')
print(roles_percentages)

# vorrei stampare i valori dei centri dei cluster per ogni feature e avere una tabella con i nomi delle feature e i valori del centro del cluster
print('cluster centers:')
cluster_centers = pd.DataFrame(fcm.centers, columns=feat.columns) # cluster centers
print(cluster_centers) # cluster centers as pandas dataframe
print(fcm.centers) # cluster centers as numpy array
# 
#average silhouette score con feature selection su feat (senza out_closeness e selfloop)
#print("average silhouette score con feature selection su feat (clustering_coeff, authorities,betweenness,out_closeness):",silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))

print('Average silhouette score:')
print(silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))
no_closs_auth_clust_sil=silhouette_score(feat.values, roles_percentages.idxmax(axis=1)) #salvo il valore per confrontarlo con i successivi, 
# %%
#in ordine dalla più rilevante alla meno rilevante
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_closs_auth_clust_pc=fcm.partition_coefficient #salvo il valore per confrontarlo con i successivi


# Il risultato è ottimo per quanto riguarda l'average silhouette score, dunque è sensato rimuovere tutte e 3 dal feature set


# without both out_closeness and selfloop and betweenness



sil= [all_sil, no_closs_sil, no_closs_autho_sill, no_closs_auth_clust_sil]
pc=[all_pc, no_closs_pc, no_closs_autho_pc, no_closs_auth_clust_pc]
sil_names=['All features', 'no closeness', 'no auth, closeness', 'no closeness, auth and clust coeff']
pc_names=['All features', 'no closeness', 'no auth, closeness', 'no closeness, auth and clust coeff']
# Creazione dell'istogramma
plt.bar(sil_names,sil) # Creazione dell'istogramma con i valori di score_l e le etichette delle colonne

# Impostazione dell'angolo per le etichette sull'asse X
plt.xticks(rotation=45)  # Rotazione di 45 gradi e allineamento a destra
# Impostazione dei limiti dell'asse Y
min_y = min(sil) - 0.01  # Un po' sopra il minimo dei valori
max_y = max(sil) + 0.01  # Un po' sopra il massimo dei valori
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
min_y = min(pc) - 0.01 # Un po' sopra il minimo dei valori
max_y = max(pc) + 0.01  # Un po' sopra il massimo dei valori
plt.ylim(min_y, max_y)
# Etichette degli assi e titolo

plt.title('Partition coefficient')
plt.tight_layout() 
# Mostra il grafico
plt.show()