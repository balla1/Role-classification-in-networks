
#Data libraries and upload

# 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.sparse import csr_matrix
from skfeature.utility import construct_W

from sklearn.neighbors import NearestNeighbors

np.random.seed(42)

base_path = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\social\\txt\\"

G22_09_02 = nx.read_weighted_edgelist(f'{base_path}2022-09-02_normalized_network.txt', create_using=nx.DiGraph)
G22_09_02_c = nx.read_weighted_edgelist(f'{base_path}2022-09-02_inverted_network.txt', create_using=nx.DiGraph)
G22_09_02_e = nx.read_weighted_edgelist(f'{base_path}2022-09-02_retweet_weighted_network.txt', create_using=nx.DiGraph)

# 
aa = list(sorted(nx.selfloop_edges(G22_09_02)))
selfloop_column = []
for i in range(len(aa)):
    selfloop_column.append(G22_09_02.edges[aa[i]]['weight'])
G22_09_02.remove_edges_from(nx.selfloop_edges(G22_09_02)) 

# 
ab = list(sorted(nx.selfloop_edges(G22_09_02_c)))
selfloop_column1 = []
for i in range(len(ab)):
    selfloop_column1.append(G22_09_02_c.edges[ab[i]]['weight'])
G22_09_02_c.remove_edges_from(nx.selfloop_edges(G22_09_02_c))

# 
ac = list(sorted(nx.selfloop_edges(G22_09_02_e)))
selfloop_column2 = []
for i in range(len(ac)):
    selfloop_column2.append(G22_09_02_e.edges[ac[i]]['weight'])
G22_09_02_e.remove_edges_from(nx.selfloop_edges(G22_09_02_e))

# Verifica la presenza di self-loops
print(f"Numero di self-loops in G22_09_02: {len(aa)}")
# Verifica la presenza di self-loops
print(f"Numero di self-loops in G22_09_02_c: {len(ab)}")
# Verifica la presenza di self-loops
print(f"Numero di self-loops in G22_09_02_e: {len(ac)}")

#Feature extraction

# 
#in_strength
in_str = G22_09_02.in_degree(weight='weight')

#out_strength
out_str = G22_09_02.out_degree(weight='weight')

#total_strength
tot_str = G22_09_02.degree(weight='weight')

#betweenness
bet = nx.betweenness_centrality(G22_09_02, weight='weight')

#hubs and authorities
hubs, auths = nx.hits(G22_09_02)

#clustering coefficient
clc = nx.clustering(G22_09_02, weight='weight')

#pagerank
pag = nx.pagerank(G22_09_02, alpha=1, weight='weight')

# 
#closeness
in_clo = nx.closeness_centrality(G22_09_02_c, distance="weight")
out_clo = nx.closeness_centrality(G22_09_02_c.reverse(), distance="weight")

#eigenvector
eig = nx.eigenvector_centrality(G22_09_02_e, weight='weight')

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
# Controlla la presenza di self-loops
#if len(selfloop_column) == 0:
    # Nessun self-loop, assegna una colonna di zeri
#    features['selfloop'] = 0
#else:
    # Aggiungi i valori dei self-loop
#    features['selfloop'] = selfloop_column

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





# 
#from scipy.optimize import linear_sum_assignment as la
from scipy.optimize import linear_sum_assignment
from scipy.sparse import lil_matrix

# Codice che utilizza linear_sum_assignment

# 
from skfeature.function.similarity_based import SPEC
from skfeature.utility import unsupervised_evaluation

# 
features_score=features.to_numpy()

# 

from skfeature.function.similarity_based import lap_score

kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
# Converte la matrice delle feature in formato sparso
features_sparse = csr_matrix(features.values)

# Costruzione manuale della matrice W usando NearestNeighbors
k_neighbors = 5  # Puoi cambiare il numero di vicini se necessario

# Usa NearestNeighbors per trovare i k vicini più vicini
nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', algorithm='auto')
nn.fit(features_sparse)

# Trova i vicini per ogni nodo
distances, indices = nn.kneighbors(features_sparse)

# Costruisci la matrice sparsa W
W = lil_matrix((features_sparse.shape[0], features_sparse.shape[0]))


for i, neighbors in enumerate(indices):
    for j, neighbor in enumerate(neighbors):
        W[i, neighbor] = np.exp(-distances[i][j])



# obtain the scores of features
score_l = lap_score.lap_score(features_sparse.toarray(), W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score_l) # ordine di rilevanza delle features secondo il Laplacian score , più è basso più è rilevante la feature 
# ascending order, the smaller the value, the more relevant the feature is 

# 
#in order from the most relevant to the least relevant

print("order of features from most relevant to least relevant:", idx)
print("order of feature relevance:", features.columns[idx]) 
# 
print("feature scores:", score_l) 

# Create the histogram
plt.bar(features.columns, score_l) # Create the histogram with score_l values and column labels

# Set the angle for the labels on the X axis
plt.xticks(rotation=45)  # Rotate by 45 degrees
# Set the limits of the Y axis

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Show the plot
plt.show()

print('Average silhouette score:')
print(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))
all_sil = silhouette_score(features.values, roles_percentages.idxmax(axis=1)) #save the value to compare with the next ones, 
# %%
#in order from the most relevant to the least relevant
print('Partition coefficient:')
print(fcm.partition_coefficient)
all_pc = fcm.partition_coefficient #save the value to compare with the next ones

# The results of the Laplacian score confirm the low relevance of the selfloop, while they are more ambiguous regarding the betweenness



fcm_full = FCM(n_clusters=2, m=1.5)
fcm_full.fit(features.values)
full_labels = fcm_full.u.argmax(axis=1)

plt.figure(figsize=(6, 6))
plt.subplot(1, 3, 1)
plt.scatter(features.iloc[:, 0], features.iloc[:, 1], c=full_labels, cmap='viridis')
plt.title('With all features')

fcm_reduced = FCM(n_clusters=2, m=1.5)
features_reduced = features.drop(features.columns[[3, 6, 7, 8]], axis=1)
fcm_reduced.fit(features_reduced.values)
reduced_labels = fcm_reduced.u.argmax(axis=1)

plt.subplot(1, 3, 2)
plt.scatter(features_reduced.iloc[:, 0], features_reduced.iloc[:, 1], c=reduced_labels, cmap='viridis')
plt.title('Without features out_closeness, selfloop, clustering coefficient and betweenness')

fcm_reduced_2 = FCM(n_clusters=2, m=1.5)
features_reduced_2 = features.drop(features.columns[[3, 6, 7, 8, 10]], axis=1)
fcm_reduced_2.fit(features_reduced_2.values)
reduced_labels_2 = fcm_reduced_2.u.argmax(axis=1)

plt.subplot(1, 3, 3)
plt.scatter(features_reduced_2.iloc[:, 0], features_reduced_2.iloc[:, 1], c=reduced_labels_2, cmap='viridis')
plt.title('Without features out_closeness, selfloop, betweenness, clustering coefficient and authorities')
plt.show()
print("Columns eliminated: ", features.columns[[3, 6, 7, 8, 10]])

#The clusters appear more compact and better separated.
# This suggests that removing these features has potentially improved the quality of the clustering, making the clusters more defined and distinct.
# removing the features out_closeness and selfloop was a beneficial choice for the clustering model. 
#The removed features not only had high Laplacian scores, indicating lower relevance, 
#but their removal also improved the clustering evaluation metrics, which is evident both numerically and visually.



# 
feat_self = features.drop(columns='betweenness')

# 
#print(feat_self)

# 
fcm = FCM(n_clusters=2, m=1.5)
fcm.fit(feat_self.values)
num_clusters = fcm.u.shape[1]
role_columns = [f'Role_{i}' for i in range(num_clusters)]
roles_percentages = pd.DataFrame(fcm.u, columns=role_columns)
roles_percentages.index = row_names
print('\nroles percentages:')
#print(roles_percentages)

print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print(nodes_roles)


# Feature selection on these features

# 
feat_self_spec = feat_self.to_numpy()

# 
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
# Convert the feature matrix to sparse format
features_sparse = csr_matrix(feat_self.values)

# Manually construct the W matrix using NearestNeighbors
k_neighbors = 5  # You can change the number of neighbors if necessary

# Use NearestNeighbors to find the k nearest neighbors
nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', algorithm='auto')
nn.fit(features_sparse)

# Find the neighbors for each node
distances, indices = nn.kneighbors(features_sparse)

# Construct the sparse W matrix
W = lil_matrix((features_sparse.shape[0], features_sparse.shape[0]))


for i, neighbors in enumerate(indices):
    for j, neighbor in enumerate(neighbors):
        W[i, neighbor] = np.exp(-distances[i][j])



# obtain the scores of features
score_l = lap_score.lap_score(features_sparse.toarray(), W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score_l)

# 
print("laplacian score:", score_l)

# 
print("order of relevance:", idx)

# 
#average silhouette score with feature selection on feat_self (without in_strength)

print("average silhouette score with feature selection on feat_self (without betweenness):", silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))

plt.bar(feat_self.columns, score_l) # Create the histogram with score_l values and column labels

# Set the angle for the labels on the X axis
plt.xticks(rotation=45)  # Rotate by 45 degrees
# Set the limits of the Y axis

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Show the plot
plt.show()
# 
#average silhouette score with feature selection on feat_self (without out_closeness)

print("average silhouette score with feature selection on feat_self (without clustering coefficient):", silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))

print('Average silhouette score:')
print(silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))
no_betw_sil = silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)) #save the value to compare with the next ones, 
# %%
#in order from the most relevant to the least relevant
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_betw_pc = fcm.partition_coefficient #save the value to compare with the next ones

# without both in_strength and selfloop

# 
feat = features.drop(columns=['betweenness', 'clustering coefficient'])

# 
# average silhouette score with feature selection on feat (without betweenness and clustering coefficient)
#print(feat)

# 
fcm = FCM(n_clusters=2, m=1.5)
fcm.fit(feat.values)
num_clusters = fcm.u.shape[1]
role_columns = [f'Role_{i}' for i in range(num_clusters)]
roles_percentages = pd.DataFrame(fcm.u, columns=role_columns)
roles_percentages.index = row_names
print('\nroles percentages:')
#print(roles_percentages)

print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print("node_roles:",nodes_roles)


# Feature selection on these features

# 
feat_spec = feat.to_numpy()

# 
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score

kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
# Convert the feature matrix to sparse format
features_sparse = csr_matrix(feat.values)

# Manually construct the W matrix using NearestNeighbors
k_neighbors = 5  # You can change the number of neighbors if necessary

# Use NearestNeighbors to find the k nearest neighbors
nn = NearestNeighbors(n_neighbors=k_neighbors, metric='euclidean', algorithm='auto')
nn.fit(features_sparse)

# Find the neighbors for each node
distances, indices = nn.kneighbors(features_sparse)

# Construct the sparse W matrix
W = lil_matrix((features_sparse.shape[0], features_sparse.shape[0]))


for i, neighbors in enumerate(indices):
    for j, neighbor in enumerate(neighbors):
        W[i, neighbor] = np.exp(-distances[i][j])



# obtain the scores of features
score_l = lap_score.lap_score(features_sparse.toarray(), W=W)

# sort the feature scores in an ascending order according to the feature scores
idx = lap_score.feature_ranking(score_l)

# 
print("laplacian score without betweenness and clustering coefficient:", score_l)


# No value stands out from the others, so there is no evidence (also considering the average silhouette score calculated previously) to eliminate further features from the feature set

# 
print("order of importance:", idx)

# 
#average silhouette score with feature selection on feat (without betweenness and clustering coefficient)
print("average silhouette score with feature selection on feat (without betweenness and clustering coefficient):", silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))


# The result is excellent regarding the average silhouette score, so it makes sense to remove both from the feature set
plt.bar(feat.columns, score_l) # Create the histogram with score_l values and column labels

# Set the angle for the labels on the X axis
plt.xticks(rotation=45)  # Rotate by 45 degrees
# Set the limits of the Y axis

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Show the plot
plt.show()

print('true roles percentages:')
print(roles_percentages)

# I would like to print the values of the cluster centers for each feature and have a table with the feature names and the cluster center values
print('true cluster centers:')
cluster_centers = pd.DataFrame(fcm.centers, columns=feat.columns) # cluster centers
print(cluster_centers) # cluster centers as pandas dataframe
print(fcm.centers) # cluster centers as numpy array
# 
#average silhouette score with feature selection on feat (without out_closeness and selfloop)
#print("average silhouette score with feature selection on feat (without clustering coefficient and betweenness):", silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))

print('Average silhouette score:')
print(silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))
no_betw_clust_sill = silhouette_score(feat.values, roles_percentages.idxmax(axis=1)) #save the value to compare with the next ones, 
# %%
#in order from the most relevant to the least relevant
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_betw_clust_pc = fcm.partition_coefficient #save the value to compare with the next ones


roles_percentages.index = row_names

# Assign the main role to each node
nodes_roles = roles_percentages.idxmax(axis=1)

# Map of roles
role_mapping = nodes_roles.to_dict()


# Define the roles
roles = ['Role_0', 'Role_1']
interaction_matrix = pd.DataFrame(0, index=roles, columns=roles)

import seaborn as sns

# Calculate the interactions between roles
for source, target, data in G22_09_02.edges(data=True):
    source_role = role_mapping.get(source, None)
    target_role = role_mapping.get(target, None)
    
    if source_role in roles and target_role in roles:
        interaction_matrix.loc[source_role, target_role] += 1

print("Matrix of interactions between roles:")
print(interaction_matrix)

# Create the heatmap
sns.set(style="white")
plt.figure(figsize=(8, 6))
sns.heatmap(interaction_matrix, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Matrix of interactions between roles")
plt.xlabel("Role target")
plt.ylabel("Role source")
plt.show()




sil = [all_sil, no_betw_clust_sill]
pc = [all_pc, no_betw_clust_pc]
sil_names = ['All features', 'no betweenness, clustering coefficient']
pc_names = ['All features', 'no betweenness, clustering coefficient']
# Create the histogram
plt.bar(sil_names, sil) # Create the histogram with score_l values and column labels

# Set the angle for the labels on the X axis
plt.xticks(rotation=45)  # Rotate by 45 degrees and align to the right
# Set the limits of the Y axis
min_y = min(sil) - 0.05  # A bit above the minimum values
max_y = max(sil) + 0.05  # A bit above the maximum values
plt.ylim(min_y, max_y)
# Axis labels and title

plt.title('Average silhouette score')
plt.tight_layout() 
# Show the plot
plt.show()

# Create the histogram
plt.bar(pc_names, pc) # Create the histogram with score_l values and column labels

# Set the angle for the labels on the X axis
plt.xticks(rotation=45)  # Rotate by 45 degrees
min_y = min(pc) - 0.1 # A bit above the minimum values
max_y = max(pc) + 0.1 # A bit above the maximum values
plt.ylim(min_y, max_y)
# Axis labels and title

plt.title('Partition coefficient')
plt.tight_layout() 
# Show the plot
plt.show()
