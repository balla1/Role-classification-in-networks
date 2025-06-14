import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import logging  # Import the logging module


from fcmeans import FCM
from sklearn.metrics import rand_score

file_base = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\dati\\"
salvataggio = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\section_II\\"

# Parametri specifici per ogni anno
m_values = {
    1995: 1.5,
    1996: 1.5,
    1997: 1.5,
    1998: 2.0,
    1999: 1.5,
    2000: 2.0,
    2001: 2.0,
    2002: 1.5,
    2003: 1.5,
    2004: 2.0,
    2005: 2.5,
    2006: 2.0,
    2007: 2.0,
    2008: 1.5,
    2009: 2.0,
    2010: 1.5,
    2011: 1.5,
    2012: 1.5,
    2013: 2.0,
    2014: 2.0,
    2015: 2.5,
    2016: 2.0,
    2017: 2.5,
    2018: 2.0,
    2019: 2.0,
    2020: 2.0,
    2021: 2.0,
    2022: 1.5
}

years = list(range(1995, 2023))
all_features = []
all_labels = []
# Liste per le coordinate dei plot finali
ita_coord = []
chn_coord = []
usa_coord = []
qat_coord = []
jpn_coord = []
ind_coord = []
ukr_coord = []

# Function to read and preprocess graphs
def preprocess_graphs(year):
    G = nx.read_weighted_edgelist(f'{file_base}{year}\\{year}_normalizzato_SECTION_II.txt', create_using=nx.DiGraph)
    G_c = nx.read_weighted_edgelist(f'{file_base}{year}\\{year}_closeness_SECTION_II.txt', create_using=nx.DiGraph)
    G_e = nx.read_weighted_edgelist(f'{file_base}{year}\\{year}_paesi_edgelist_SECTION_II.txt', create_using=nx.DiGraph)
    
    # Remove self-loops and store them
    selfloop_column = remove_self_loops(G)
    remove_self_loops(G_c)
    remove_self_loops(G_e)
    
    return G, G_c, G_e, selfloop_column

def remove_self_loops(G):
    selfloops = list(nx.selfloop_edges(G, data='weight'))
    G.remove_edges_from(nx.selfloop_edges(G))
    return [weight for u, v, weight in selfloops]

def extract_features(G, G_c, G_e, selfloop_column):
    logging.info("Extracting features")
    in_str = G.in_degree(weight='weight')
    out_str = G.out_degree(weight='weight')
    tot_str = G.degree(weight='weight')
    hubs, auths = nx.hits(G)
    bet = nx.betweenness_centrality(G, weight='weight')
    clc = nx.clustering(G, weight='weight')
    in_clo = nx.closeness_centrality(G_c, distance="weight")
    out_clo = nx.closeness_centrality(G_c.reverse(), distance="weight")
    
    # Handling eigenvector centrality calculation
    try:
        eig = nx.eigenvector_centrality(G_e, weight='weight', max_iter=1000, tol=1e-4)  # Increased max_iter
    except nx.PowerIterationFailedConvergence:
        logging.warning("Eigenvector centrality did not converge, using a fallback method.")
        # Handle non-convergence case, e.g., use a different centrality measure
        eig = {node: 0 for node in G_e.nodes()}  # Fallback to zero centrality
    try:
        pag = nx.pagerank(G, alpha=1, weight='weight', max_iter=1000, tol=1e-4)
    except nx.PowerIterationFailedConvergence:
        print(f"PageRank failed to converge for year {year}")
        pag = {node: float('nan') for node in G.nodes}

    in_strength = [x[1] for x in in_str]
    out_strength = [x[1] for x in out_str]
    tot_strength = [x[1] for x in tot_str]
    features = pd.DataFrame({'in_strength': in_strength})
    features.index = bet.keys()
    features['out_strength'] = out_strength
    features['total_strength'] = tot_strength
    features['eigenvector'] = features.index.map(eig)
    features['in_closeness'] = features.index.map(in_clo)
    features['out_closeness'] = features.index.map(out_clo)
    features['clustering coefficient'] = features.index.map(clc)
    features['betweenness'] = features.index.map(bet)
    features['pagerank'] = features.index.map(pag)
    features['hubs'] = features.index.map(hubs)
    features['authorities'] = features.index.map(auths)
    
    if selfloop_column:
        features['selfloop'] = selfloop_column
    
    return features

from scipy.optimize import linear_sum_assignment

def compute_similarity_matrix(V1, V2):
    max_length = max(len(V1.columns), len(V2.columns))
    V1_padded = np.zeros((V1.shape[0], max_length))
    V2_padded = np.zeros((V2.shape[0], max_length))
    
    V1_padded[:, :V1.shape[1]] = V1.values
    V2_padded[:, :V2.shape[1]] = V2.values
    
    similarity_matrix = np.zeros((V1.shape[0], V2.shape[0]))
    for i, vec1 in enumerate(V1_padded):
        for j, vec2 in enumerate(V2_padded):
            similarity_matrix[i, j] = np.linalg.norm(vec1 - vec2)
    return similarity_matrix

def reorder_clusters(V_prev, V_current):
    similarity_matrix = compute_similarity_matrix(V_prev, V_current)
    row_ind, col_ind = linear_sum_assignment(similarity_matrix)
    return col_ind

def perform_fcm(features, m, V_prev=None):
    feat = features.drop(columns=[ 'in_closeness','hubs','authorities','clustering coefficient','out_closeness'], axis=1)
    fcm = FCM(n_clusters=3, m=m)
    fcm.fit(feat.values)
    
    roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1','Role_2'])
    roles_percentages.index = features.index
    nodes_roles = roles_percentages.idxmax(axis=1)
    
    V = pd.DataFrame(fcm.centers, columns=feat.columns)
    
    if V_prev is not None:
        sorted_indices = reorder_clusters(V_prev, V)
    else:
        V['sum'] = V.sum(axis=1)
        sorted_indices = V['sum'].argsort().values
    
    V_sorted = V.iloc[sorted_indices]
    
    roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]
    roles_percentages_sorted.columns = ['Role_0', 'Role_1','Role_2']
    
    labels = list(nodes_roles)
    for i in range(len(labels)):
        labels[i] = int(labels[i][-1])
    
    return features, labels, roles_percentages_sorted, nodes_roles, V_sorted

# Funzione per salvare i grafici
def save_graphs(year, G, roles_percentages_sorted, nodes_roles):
    label = {}
    for node in G.nodes:
        label[node] = nodes_roles.loc[node]

    nx.set_node_attributes(G, label, 'fcm')
    nx.write_graphml(G, f'{salvataggio}{year}_section_II.graphml')

    role_colors = {'Role_0': 'blue', 'Role_1': 'red','Role_2': 'green'}
    pos = nx.spring_layout(G, k=0.022) # k è la distanza tra i nodi del grafo, più è piccolo più i nodi sono vicini 
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for node in G.nodes():
        percentages = roles_percentages_sorted.loc[node, ['Role_0', 'Role_1','Role_2']].values
        wedges, _ = ax.pie(percentages, colors=[role_colors['Role_0'], role_colors['Role_1'], role_colors['Role_2']],
                           radius=0.1, center=pos[node], wedgeprops=dict(width=0.03, edgecolor='w'))
        ax.text(pos[node][0], pos[node][1], node, horizontalalignment='center', verticalalignment='center',
                fontweight='bold', color='black')
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.01, edge_color='gray', width=0.5)
    ax.legend(wedges, ['Role_0', 'Role_1','Role_2'], title="Roles", loc="best")
    ax.set_aspect('equal')
    plt.xlim([min(x[0] for x in pos.values()) - 0.1, max(x[0] for x in pos.values()) + 0.1])
    plt.ylim([min(x[1] for x in pos.values()) - 0.1, max(x[1] for x in pos.values()) + 0.1])
    plt.axis('off')
    plt.savefig(f'{salvataggio}g{year}_section_II.png', bbox_inches='tight')
    plt.close()


def standardize_features(features, common_columns):
    for col in common_columns:
        if col not in features.columns:
            features[col] = 0
    return features[common_columns]

common_columns = ['in_strength', 'out_strength', 'total_strength', 'eigenvector', 'in_closeness', 
                  'out_closeness', 'clustering coefficient', 'betweenness', 'pagerank', 'hubs', 'authorities']

# 
#
V_prev = None
for year in years:
    G, G_c, G_e, selfloop_column = preprocess_graphs(year)
    features = extract_features(G, G_c, G_e, selfloop_column)
    features = standardize_features(features, common_columns)  # Standardizza le caratteristiche
    m = m_values.get(year, 1.5)
    features, labels, roles_percentages_sorted, nodes_roles, V_prev = perform_fcm(features, m, V_prev)
    
    all_features.append(features)
    all_labels.append(labels)
    
    # Aggiornare le coordinate
    coords = roles_percentages_sorted.loc['ITA'].tolist()
    ita_coord.append(coords)
    coord = roles_percentages_sorted.loc['CHN'].tolist()
    chn_coord.append(coord)
    coor = roles_percentages_sorted.loc['QAT'].tolist()
    qat_coord.append(coor)
    c_u = roles_percentages_sorted.loc['USA'].tolist()
    usa_coord.append(c_u)
    c_j = roles_percentages_sorted.loc['JPN'].tolist()
    jpn_coord.append(c_j)
    c_i = roles_percentages_sorted.loc['IND'].tolist()
    ind_coord.append(c_i)
    c_k = roles_percentages_sorted.loc['UKR'].tolist()
    ukr_coord.append(c_k)
    
    if year in {1995, 2020, 2022}:
        save_graphs(year, G, roles_percentages_sorted, nodes_roles)


# Stampa i risultati
print("ita coord: ", ita_coord)
print("chn coord: ", chn_coord)
print("usa coord: ", usa_coord)
print("qat coord: ", qat_coord)
print("jpn coord: ", jpn_coord)
print("ind coord: ", ind_coord)
print("ukr coord: ", ukr_coord)


# Verifica delle etichette per ogni anno
#def print_label_info(label_list, year):
    #print(f"Year {year}:")
    #print(f"Number of labels: {len(label_list)}")
    #print(f"Labels: {label_list[:5]}...")

# Stampa informazioni sulle etichette per ogni anno
#for year, labels in zip(years, all_labels):
#    print_label_info(labels, year)

# Calcolo dei Rand Scores sui paesi comuni
common_countries = set(all_features[0].index)
for features in all_features[1:]:
    common_countries.intersection_update(set(features.index))

if not common_countries:
    raise ValueError("Non ci sono paesi comuni tra tutti gli anni.")

print(f"Paesi coinvolti nello studio: {common_countries}")

filtered_labels = []
for year_labels, features in zip(all_labels, all_features):
    country_to_label = dict(zip(features.index, year_labels))
    filtered_labels.append([country_to_label[country] for country in common_countries if country in country_to_label])

# Calcolo dei Rand Scores filtrati
filtered_rand_scores = [
    rand_score(filtered_labels[i], filtered_labels[i + 1])
    for i in range(len(filtered_labels) - 1)
]

# Stampa dei Rand Scores filtrati
print('Filtered Rand scores:')
print(filtered_rand_scores)

# Plot dei Rand Scores filtrati
fig = plt.figure(figsize=(12, 6))
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.plot(filtered_rand_scores)
ax.set_title('Filtered Rand_Score')
lab = [str(year) for year in range(1996, 2023)]
positions = list(range(len(lab)))
ax.set_xticks(positions)
ax.set_xticklabels(lab, rotation=45)
plt.savefig(f'{salvataggio}filtered_rand_score_section_II.png', bbox_inches='tight')
plt.close()


# Analisi delle coordinate dei paesi selezionati
role_0_ita, role_1_ita , role_2_ita = [], [],[]
role_0_chn, role_1_chn ,role_2_chn= [], [],[]
role_0_qat, role_1_qat ,role_2_qat= [], [],[]
role_0_usa, role_1_usa ,role_2_usa= [], [],[]
role_0_jpn, role_1_jpn,role_2_jpn = [], [],[]
role_0_ind, role_1_ind ,role_2_ind= [], [],[]
role_0_ukr, role_1_ukr,role_2_ukr = [], [],[]

for i in range(len(ita_coord)):
    role_0_ita.append(ita_coord[i][0])
    role_1_ita.append(ita_coord[i][1])
    role_2_ita.append(ita_coord[i][2])
    role_0_chn.append(chn_coord[i][0])
    role_1_chn.append(chn_coord[i][1])
    role_2_chn.append(chn_coord[i][2])
    role_0_qat.append(qat_coord[i][0])
    role_1_qat.append(qat_coord[i][1])
    role_2_qat.append(qat_coord[i][2])
    
    role_0_usa.append(usa_coord[i][0])
    role_1_usa.append(usa_coord[i][1])
    role_2_usa.append(usa_coord[i][2])
    
    role_0_jpn.append(jpn_coord[i][0])
    role_1_jpn.append(jpn_coord[i][1])
    role_2_jpn.append(jpn_coord[i][2])
    
    role_0_ind.append(ind_coord[i][0])
    role_1_ind.append(ind_coord[i][1])
    role_2_ind.append(ind_coord[i][2])
    
    role_0_ukr.append(ukr_coord[i][0])
    role_1_ukr.append(ukr_coord[i][1])
    role_2_ukr.append(ukr_coord[i][2])
    

x = list(range(1995, 2023))

# Funzione per salvare i grafici di evoluzione dei ruoli
def plot_role_evolution(x, role_0, role_1,role_2, country, filename):
    plt.plot(x, role_0, label="role_0")
    plt.plot(x, role_1, label="role_1")
    plt.plot(x, role_2, label="role_2")
    
    plt.legend(bbox_to_anchor=(0.0, -0.17, 1, -0.027), mode='expand', loc='lower center', borderaxespad=0., ncols=2)
    plt.title(country)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Salvare i grafici per ciascun paese
plot_role_evolution(x, role_0_ita, role_1_ita,role_2_ita, 'Italy', f'{salvataggio}ITA_section_II.png')
plot_role_evolution(x, role_0_chn, role_1_chn, role_2_chn,'China', f'{salvataggio}CHN_section_II.png')
plot_role_evolution(x, role_0_usa, role_1_usa, role_2_usa,'USA', f'{salvataggio}USA_section_II.png')
plot_role_evolution(x, role_0_qat, role_1_qat,role_2_qat, 'Qatar', f'{salvataggio}QAT_section_II.png')
plot_role_evolution(x, role_0_jpn, role_1_jpn,role_2_jpn, 'Japan', f'{salvataggio}JPN_section_II.png')
plot_role_evolution(x, role_0_ind, role_1_ind,role_2_ind, 'India', f'{salvataggio}IND_section_II.png')
plot_role_evolution(x, role_0_ukr, role_1_ukr,role_2_ukr, 'Ukraine', f'{salvataggio}UKR_section_II.png')

print("Analisi completata.")
