# Import delle librerie necessarie
import pandas as pd
import networkx as nx
import numpy as np
from fcmeans import FCM
from skfeature.utility import construct_W
from skfeature.function.similarity_based import lap_score
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Imposta il seed per la riproducibilità
np.random.seed(42)

# Anni per i quali eseguire l'analisi
years = list(range(1995, 2023))  # Supponiamo di avere i dati dal 1995 al 2022
n_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9]
m_values = [1.3, 1.4, 1.5, 1.6, 1.7, 2.0, 2.1, 2.5, 2.6, 2.7]
all_results = {}

def debug_print_year_data(year, features):
    print(f"\nYear: {year}")
    print(features)
    print("Number of NaN values in each column:")
    print(features.isna().sum())

# Ciclo principale per iterare su tutti gli anni
for year in years:
    print(f"Processing year: {year}")
    base_path = f"D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\dati\\{year}\\{year}"
    salvataggio = f"D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\section_XI\\"

    # Caricamento dei grafi
    G = nx.read_weighted_edgelist(f'{base_path}_normalizzato_SECTION_XI.txt', create_using=nx.DiGraph)
    G_c = nx.read_weighted_edgelist(f'{base_path}_closeness_SECTION_XI.txt', create_using=nx.DiGraph)
    G_e = nx.read_weighted_edgelist(f'{base_path}_paesi_edgelist_SECTION_XI.txt', create_using=nx.DiGraph)
    
    # Gestione dei selfloop
    aa = list(sorted(nx.selfloop_edges(G)))
    selfloop_column = [G.edges[edge]['weight'] for edge in aa]
    G.remove_edges_from(nx.selfloop_edges(G))
    
    ab = list(sorted(nx.selfloop_edges(G_c)))
    selfloop_column1 = [G_c.edges[edge]['weight'] for edge in ab]
    G_c.remove_edges_from(nx.selfloop_edges(G_c))
    
    ac = list(sorted(nx.selfloop_edges(G_e)))
    selfloop_column2 = [G_e.edges[edge]['weight'] for edge in ac]
    G_e.remove_edges_from(nx.selfloop_edges(G_e))
    
    # Estrazione delle caratteristiche
    in_str = G.in_degree(weight='weight')
    out_str = G.out_degree(weight='weight')
    tot_str = G.degree(weight='weight')
    bet = nx.betweenness_centrality(G, weight='weight')
    hubs, auths = nx.hits(G)
    clc = nx.clustering(G, weight='weight')
    
    # Gestione dell'errore di convergenza per PageRank nel 2006
    
    try:
        pag = nx.pagerank(G, alpha=1, weight='weight', max_iter=1000, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        
        print(f"PageRank failed to converge for year {year}")
        pag = {node: float('nan') for node in G.nodes}
    
    
    in_clo = nx.closeness_centrality(G_c, distance="weight")
    out_clo = nx.closeness_centrality(G_c.reverse(), distance="weight")
    
    # Gestione dell'errore di convergenza per la centralità autovettoriale nel 1997
    
    try:
        eig = nx.eigenvector_centrality(G_e, weight='weight', max_iter=1000, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        print(f"Eigenvector centrality failed to converge for year {year}")
        eig = {node: float('nan') for node in G_e.nodes}

    
    in_strength = [x[1] for x in in_str]
    out_strength = [x[1] for x in out_str]
    tot_strength = [x[1] for x in tot_str]
    
    features = pd.DataFrame({'in_strength': in_strength})
    features.index = bet.keys()
    features['out_strength'] = out_strength
    features['total_strength'] = tot_strength
    if selfloop_column:
        features['selfloop'] = selfloop_column
    features['eigenvector'] = features.index.map(eig)
    features['in_closeness'] = features.index.map(in_clo)
    features['out_closeness'] = features.index.map(out_clo)
    features['clustering coefficient'] = features.index.map(clc)
    features['betweenness'] = features.index.map(bet)
    features['pagerank'] = features.index.map(pag)
    features['hubs'] = features.index.map(hubs)
    features['authorities'] = features.index.map(auths)
    
    # Debug print per controllare i valori NaN
    debug_print_year_data(year, features)
    
    # Riempie i valori NaN con 0
    features = features.fillna(0)
    
    # Esecuzione del clustering
    results = []
    for m in m_values:
        for n_clusters in n_clusters_list:
            fcm = FCM(n_clusters=n_clusters, m=m)
            fcm.fit(features.values)
            pc = fcm.partition_coefficient
            if len(set(fcm.u.argmax(axis=1))) > 1:  # Assicurarsi che ci siano almeno 2 cluster
                silhouette_avg = silhouette_score(features.values, fcm.u.argmax(axis=1))
            else:
                silhouette_avg = -1  # Valore dummy se c'è un solo cluster
            results.append((n_clusters, m, pc, silhouette_avg))
    
    all_results[year] = results

    # Stampa i risultati per l'anno corrente
    print(f"Anno: {year}")
    for r in results:
        print(f"  Clusters: {r[0]}, Fuzziness: {r[1]}, Partition Coefficient: {r[2]}, Silhouette Score: {r[3]}")

    # Feature selection: valutazione eliminando una feature alla volta
    temp_feat = features.copy()
    models = list()
    silhouette_avg = list()
    for i in range(len(features.columns)):
        temp_feat = temp_feat.drop(temp_feat.columns[[i]], axis=1)
        fcm = FCM(n_clusters=3, m=1.5)
        fcm.fit(temp_feat.values)
        models.append(fcm)
        roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
        nodes_roles = roles_percentages.idxmax(axis=1)
        try:
            silhouette_avg.append(silhouette_score(temp_feat.values, nodes_roles))
        except ValueError as e:
            print(f"Error calculating silhouette score for year {year}, feature index {i}: {e}")
            silhouette_avg.append(float('nan'))
        temp_feat = features.copy()

    # Stampa i risultati della feature selection
    print(f"Silhouette average score eliminando ogni volta una feature diversa nell'anno {year}: {silhouette_avg}")
    i = 0
    for model in models:
        pc = model.partition_coefficient
        pec = model.partition_entropy_coefficient
        eliminated_feature = features.columns[i]
        i += 1
        print(f"Eliminating: {eliminated_feature}, Partition Coefficient: {pc}, Partition Entropy Coefficient: {pec}")

    # Calcolo del Laplacian score
    features_score = features.to_numpy()
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(features, **kwargs_W)
    score_l = lap_score.lap_score(features_score, W=W)
    idx = np.argsort(score_l)[::-1]
    print("Ordine features dalla più rilevante alla meno rilevante:", features.columns[idx])
    print("Laplacian score features:", score_l)
    # Aggiungi Laplacian score ai risultati
    laplacian_scores = {feature: score for feature, score in zip(features.columns, score_l)}
    all_results[year] = {
        'clustering_results': results,
        'laplacian_scores': laplacian_scores
    }

import os
# Verifica se la cartella esiste, altrimenti la crea
if not os.path.exists(salvataggio):
    os.makedirs(salvataggio)

# Salva i risultati di clustering in un file CSV
clustering_results = [
    {'year': year, 'n_clusters': r[0], 'fuzziness': r[1], 'partition_coefficient': r[2], 'silhouette_score': r[3]}
    for year, res in all_results.items()
    for r in res['clustering_results']
]
results_df = pd.DataFrame(clustering_results)
results_df.to_csv(f'{salvataggio}clustering_results_all_years_section_XI.txt', sep='\t', index=False)

# Salva i risultati dei Laplacian score in un file di testo
laplacian_scores = [
    {'year': year, **res['laplacian_scores']}
    for year, res in all_results.items()
]
laplacian_scores_df = pd.DataFrame(laplacian_scores)
laplacian_scores_df.to_csv(f'{salvataggio}laplacian_scores_all_years_section_XI.txt', sep='\t', index=False)

# Numero totale di subplot necessari
num_plots = len(m_values) * 2  # Due subplot per ciascun m_value

# Calcolo delle dimensioni della griglia
rows = (num_plots // 2) + (num_plots % 2)

# Visualizzazione dei risultati per ogni anno
for year, res in all_results.items():
    clustering_results = res['clustering_results']
    
    plt.figure(figsize=(18, rows * 5))
    plt.suptitle(f'Year: {year}', fontsize=16)

    for i, m_value in enumerate(m_values):
        pc_values = [r[2] for r in clustering_results if r[1] == m_value]
        silhouette_values = [r[3] for r in clustering_results if r[1] == m_value]

        if len(silhouette_values) == len(n_clusters_list):
            plt.subplot(rows, 2, 2*i+1)
            plt.plot(n_clusters_list, silhouette_values, marker='o')
            plt.xlabel("n_clusters")
            plt.ylabel("silhouette_avg_score")
            plt.title(f'm={m_value}')

            plt.subplot(rows, 2, 2*i+2)
            plt.plot(n_clusters_list, pc_values, marker='o')
            plt.xlabel("n_clusters")
            plt.ylabel("partition_coefficient")
            plt.title(f'm={m_value}')
            plt.close()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
