
# #Data libraries and upload

# 
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)
# 
base_path = ""
G95 = nx.read_weighted_edgelist(f'{base_path}1995_normalizzato.txt', create_using=nx.DiGraph)
G95_c = nx.read_weighted_edgelist(f'{base_path}1995_closeness.txt', create_using=nx.DiGraph)
G95_e = nx.read_weighted_edgelist(f'{base_path}1995_paesi_edgelist.txt', create_using=nx.DiGraph)

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

# 
# #Feature extraction

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
print(features)

# 
# #FCM Clustering pre-feature selection

# 


# 
from fcmeans import FCM

# 
row_names=features.index

# 
pd.set_option("display.max_rows", None)

# 
from sklearn.metrics import silhouette_score

# 
# ##Analisi su silhouette score e partition coefficient per la scelta dei parametri

# 
n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
pec = list()
pc = list()
silhouette_avg = list()
for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=1.5)
    fcm.fit(features.values)
    pec.append(fcm.partition_entropy_coefficient)
    pc.append(fcm.partition_coefficient) 
    
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))

# 
print(pec)

# 
print(silhouette_avg)

# 
plt.plot(n_clusters_list, silhouette_avg)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=1.5')

plt.show()


plt.plot(n_clusters_list, pc)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=1.5')

plt.show()


n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
pec = list()
pc = list()
silhouette_avg = list()
for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=2.0)
    fcm.fit(features.values)
    pec.append(fcm.partition_entropy_coefficient)
    pc.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))

# 
print(pec)

# 
print(silhouette_avg)

# 
plt.plot(n_clusters_list, silhouette_avg)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=2.0')

plt.show()


plt.plot(n_clusters_list, pc)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=2.0')

plt.show()

n_clusters_list = [2, 3, 4, 5, 6, 7]
models = list()
pec = list()
pc = list()
silhouette_avg = list()
for n in n_clusters_list:
    fcm = FCM(n_clusters=n, m=2.5)
    fcm.fit(features.values)
    pec.append(fcm.partition_entropy_coefficient)
    pc.append(fcm.partition_coefficient)
    models.append(fcm)
    roles_percentages = pd.DataFrame(fcm.u)
    silhouette_avg.append(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))

# 
print(pec)

# 
print(silhouette_avg)

# 
plt.plot(n_clusters_list, silhouette_avg)
plt.xlabel("n_clusters")
plt.ylabel("silhouette_avg_score")
plt.title('m=2.5')

plt.show()


plt.plot(n_clusters_list, pc)
plt.xlabel("n_clusters")
plt.ylabel("partition_coefficient")
plt.title('m=2.5')

plt.show()

n_clusters=len(n_clusters_list)


rows = int(np.ceil(np.sqrt(n_clusters))) 
cols = int(np.ceil(n_clusters / rows)) 
f, axes = plt.subplots(rows, cols, figsize=(11,16)) 
for n_clusters, model, ax in zip(n_clusters_list, models, axes.ravel()):  
    # get validation metrics
    pc = model.partition_coefficient
    fcm_centers = model.centers # cluster centers 
    fcm_labels = model.predict(features.values)
    # plot result
    ax.scatter(features.values[:,0], features.values[:,1], c=fcm_labels, alpha=.1)
    ax.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=501, c='black')
    ax.set_title(f'n_cluster = {n_clusters}, PC = {pc:.3f}')
plt.show()


######### cluster chosen 3

# 
#plotting partition coefficient and average silhouette score for different values of m

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


num_m=len(m_values)

rows = int(np.ceil(np.sqrt(num_m)))
cols = int(np.ceil(num_m / rows))
f, axes = plt.subplots(rows, cols, figsize=(11,16))
for m_value, model, axe in zip(m_values, models, axes.ravel()):
    # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient
    fcm_centers = model.centers
    fcm_labels = model.predict(features.values)
    # plot result
    axe.scatter(features.values[:,0], features.values[:,1], c=fcm_labels, alpha=.1)
    axe.scatter(fcm_centers[:,0], fcm_centers[:,1], marker="+", s=501, c='black')
    axe.set_title(f'm = {m_value}, PC = {pc:.3f}, PEC = {pec:.3f}')
plt.show()



# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(features.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
print('\nroles percentages:')
print(roles_percentages)

print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
print(nodes_roles)



# Feature selection with average silhouette score
# eliminating one feature at a time

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
print('Silhouette score:')
print(silhouette_avg)

# 
# results indicate that selfloop and betweenness are the least relevant features

# 
for model in models:
    # get validation metrics
    pc = model.partition_coefficient
    pec = model.partition_entropy_coefficient
    print(pc, pec)


# results are consistent with the silhouette score
# 
#laplacian score



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






# histogram of the Laplacian score
plt.bar(features.columns, score_l) 


plt.xticks(rotation=45) 

plt.ylabel('Laplacian Score')
plt.title('Laplacian Score for Each Feature')
plt.tight_layout() 
# Mostra il grafico
plt.show()

print('Average silhouette score:')
print(silhouette_score(features.values, roles_percentages.idxmax(axis=1)))
all_sil=silhouette_score(features.values, roles_percentages.idxmax(axis=1))
print('Partition coefficient:')
print(fcm.partition_coefficient)
all_pc=fcm.partition_coefficient 

print(idx)

# 
print(score_l)


# results confirm the low relevance of selfloop, while they are more ambiguous for betweenness
# 
# #FCM without betweenness and selfloop

# 
# ##without selfloop

# 
feat_self = features.drop(columns='selfloop')

# 
#print(feat_self)

# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(feat_self.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
#print('\nroles percentages:')
#print(roles_percentages)

#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print(nodes_roles)

# 
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
print(score_l)

# 
print(idx)

# 
print('Average silhouette score:')
print(silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)))
no_self_sil=silhouette_score(feat_self.values, roles_percentages.idxmax(axis=1)) # no selfloop silhouette score
print('Partition coefficient:')
print(fcm.partition_coefficient)
no_self_pc=fcm.partition_coefficient # no selfloop partition coefficient
# 
# ##without both betweenness and selfloop

# 
feat = features.drop(columns=['betweenness', 'selfloop'])

# 
#print(feat)

# 
fcm = FCM(n_clusters=3, m=1.5)
fcm.fit(feat.values)
roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
roles_percentages.index = row_names
#print('\nroles percentages:')
#print(roles_percentages)

#print('\nNodes x Roles:')
nodes_roles = roles_percentages.idxmax(axis=1)
#print(nodes_roles)

# 
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

print('roles percentages:')
print(roles_percentages)


print('cluster centers:')
cluster_centers = pd.DataFrame(fcm.centers, columns=feat.columns) # cluster centers
print(cluster_centers) # cluster centers as pandas dataframe
print(fcm.centers) # cluster centers as numpy array

# 
print('Laplacian score:')
print(score_l)

# 
# no values stand out, hence there is no evidence (also considering the average silhouette score calculated previously) to eliminate further features from the feature set
# 
print('Feature ranking:')
print(idx)

# 
print('Average silhouette score:')
print(silhouette_score(feat.values, roles_percentages.idxmax(axis=1)))
no_bet_self_sil=silhouette_score(feat.values, roles_percentages.idxmax(axis=1))

# 
# good result for the average silhouette score, hence it is reasonable to remove both from the feature set

print('Partition coefficient:')
print(fcm.partition_coefficient)
no_bet_self_pc=fcm.partition_coefficient # no betweenness and selfloop partition coefficient


#############################################



##########################################

sil= [all_sil, no_self_sil, no_bet_self_sil]
pc=[all_pc, no_self_pc, no_bet_self_pc]
sil_names=['All features', 'Without selfloop', 'Without selfloop and betweenness']
pc_names=['All features', 'Without selfloop', 'Without selfloop and betweenness']


plt.bar(sil_names,sil) 


plt.xticks(rotation=45)  
min_y = min(sil) - 0.10  
max_y = max(sil) + 0.10  
plt.ylim(min_y, max_y)


plt.title('Average silhouette score')
plt.tight_layout() 

plt.show()


plt.bar(pc_names,pc) 

plt.xticks(rotation=45)
min_y = min(pc) - 0.05 
max_y = max(pc) + 0.05  
plt.ylim(min_y, max_y)


plt.title('Partition coefficient')
plt.tight_layout() 

plt.show()
