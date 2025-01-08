import pandas as pd
import networkx as nx
import igraph as ig
import matplotlib.pyplot as plt
import numpy as np
from fcmeans import FCM
from sklearn.metrics import rand_score
import multiprocessing as mp
from joblib import Parallel, delayed
import cProfile
import plotly.graph_objects as go

file_base = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\social\\txt\\"
salvataggio = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\social\\results_3\\"

# Specific parameters for each day
m_values = {
    '2022-09-02': 1.5,
    '2022-09-03': 1.5,
    '2022-09-04': 1.5,
    '2022-09-05': 1.5,
    '2022-09-06': 1.5,
    '2022-09-07': 1.5,
    '2022-09-08': 1.5,
    '2022-09-09': 1.5,
    '2022-09-10': 1.5,
    '2022-09-11': 1.5,
    '2022-09-12': 1.5,
    '2022-09-13': 1.5,
    '2022-09-14': 1.5,
    '2022-09-15': 1.5,
    '2022-09-16': 1.5,
    '2022-09-17': 1.5,
    '2022-09-18': 1.5,
    '2022-09-19': 1.5,
    '2022-09-20': 1.5,
    '2022-09-21': 1.5,
    '2022-09-22': 1.5,
    '2022-09-28': 1.5,
    '2022-09-29': 1.5,
    '2022-09-30': 1.5,
    '2022-10-01': 1.5,
    '2022-10-02': 1.5,
    '2022-10-03': 1.5,
    '2022-10-04': 1.5,
    '2022-10-05': 1.5,
    '2022-10-06': 1.5,
    '2022-10-07': 1.5,
    '2022-10-08': 1.5,
    '2022-10-09': 1.5,
    '2022-10-10': 1.5,
    '2022-10-11': 1.5,
    '2022-10-12': 1.5,
    '2022-10-13': 1.5,
    '2022-10-14': 1.5,
    '2022-10-15': 1.5,
    '2022-10-16': 1.5,
    '2022-10-17': 1.5,
    '2022-10-18': 1.5,
    '2022-10-19': 1.5,
    '2022-10-20': 1.5
}

days = pd.date_range(start='2022-09-02', end='2022-10-20').strftime('%Y-%m-%d').tolist()
days = [day for day in days if day not in ['2022-09-23', '2022-09-24', '2022-09-25', '2022-09-26', '2022-09-27']]

# Function to read and preprocess graphs
def preprocess_graphs(day):
    G = nx.read_weighted_edgelist(f'{file_base}{day}_normalized.txt', create_using=nx.DiGraph)
    G_c = nx.read_weighted_edgelist(f'{file_base}{day}_inverted.txt', create_using=nx.DiGraph)
    G_e = nx.read_weighted_edgelist(f'{file_base}{day}_original.txt', create_using=nx.DiGraph)
    selfloop_column = remove_self_loops(G)
    remove_self_loops(G_c)
    remove_self_loops(G_e)
    return G, G_c, G_e, selfloop_column

def remove_self_loops(G):
    selfloops = list(nx.selfloop_edges(G, data='weight'))
    G.remove_edges_from(nx.selfloop_edges(G))
    return [weight for u, v, weight in selfloops]

# Function to extract features using igraph to improve speed
def extract_features(G, G_c, G_e, selfloop_column):
    ig_graph = ig.Graph.from_networkx(G)
    
    in_str = G.in_degree(weight='weight')
    out_str = G.out_degree(weight='weight')
    tot_str = G.degree(weight='weight')
    
    # Optimization: reduction of calculated centralities
    bet = ig_graph.betweenness(directed=True, cutoff=50)  # Approximated to improve speed
    clc = nx.clustering(G, weight='weight')
    pag = nx.pagerank(G, alpha=0.85, tol=1e-4)
    in_clo = nx.closeness_centrality(G_c, distance="weight")
    out_clo = nx.closeness_centrality(G_c.reverse(), distance="weight")
    eig = nx.eigenvector_centrality(G_e, weight='weight', max_iter=50)  # Iteration limit
    hubs, auths = nx.hits(G, max_iter=50)

    in_strength = np.array([x[1] for x in in_str])
    out_strength = np.array([x[1] for x in out_str])
    tot_strength = np.array([x[1] for x in tot_str])
    
    features = pd.DataFrame({'in_strength': in_strength}) 
    features.index = G.nodes
    features['out_strength'] = out_strength
    features['total_strength'] = tot_strength
    features['eigenvector'] = features.index.map(eig)
    features['in_closeness'] = features.index.map(in_clo)
    features['out_closeness'] = features.index.map(out_clo)
    features['clustering coefficient'] = features.index.map(clc)
    features['betweenness'] = bet  # Betweenness calculated with igraph
    features['pagerank'] = features.index.map(pag)
    features['hubs'] = features.index.map(hubs)
    features['authorities'] = features.index.map(auths)
    
    if selfloop_column:
        features['selfloop'] = selfloop_column
    
    return features

# Function to perform FCM clustering with 3 roles
def perform_fcm(feat, features, m_v):
    fcm = FCM(n_clusters=3, m=m_v)  # Changed to 3 clusters
    fcm.fit(feat.values)
    
    roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])  # Added Role_2
    roles_percentages.index = features.index
    nodes_roles = roles_percentages.idxmax(axis=1)
    
    V = pd.DataFrame(fcm.centers, columns=feat.columns)
    V['sum'] = V.sum(axis=1)
    sorted_indices = V['sum'].argsort().values
    roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]
    roles_percentages_sorted.columns = ['Role_0', 'Role_1', 'Role_2']  # Added Role_2
    
    labels = [int(nodes_roles.iloc[i][-1]) for i in range(len(nodes_roles))]
    
    return features, labels, roles_percentages_sorted, nodes_roles

# Function to save graphs
def save_graphs(day, G, roles_percentages_sorted, nodes_roles):
    label = {node: nodes_roles.loc[node] for node in G.nodes}
    nx.set_node_attributes(G, label, 'fcm')
    nx.write_graphml(G, f'{salvataggio}{day}_tot.graphml')

    role_colors = {'Role_0': 'blue', 'Role_1': 'red', 'Role_2': 'green'}  # Added color for Role_2
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    fig, ax = plt.subplots(figsize=(30, 30))
    for node in G.nodes():
        percentages = roles_percentages_sorted.loc[node, ['Role_0', 'Role_1', 'Role_2']].values  # Added Role_2
        wedges, _ = ax.pie(percentages, colors=[role_colors['Role_0'], role_colors['Role_1'], role_colors['Role_2']],  # Added Role_2
                           radius=0.05, center=pos[node], wedgeprops=dict(width=0.02, edgecolor='w'))
        ax.text(pos[node][0], pos[node][1], node, horizontalalignment='center', verticalalignment='center',
                fontweight='bold', color='black')

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.001, edge_color='gray', width=0.5)
    # Reduce the size of node texts
    for node, (x, y) in pos.items():
        ax.text(x, y, str(node), fontsize=5, horizontalalignment='center', verticalalignment='center')

    ax.legend(wedges, ['Role_0', 'Role_1', 'Role_2'], title="Roles", loc="best")  # Added Role_2
    ax.set_aspect('equal')
    plt.savefig(f'{salvataggio}g{day}_tot.png', bbox_inches='tight', dpi=300)
    plt.close()

# Profiling for each day
def process_day(day):
    profiler = cProfile.Profile()
    profiler.enable()

    feature_removal_groups = {
        'group_1': ['2022-09-02', '2022-09-04',  '2022-09-06', '2022-09-07', '2022-09-08', '2022-09-10', '2022-09-11', '2022-09-12', '2022-09-13', '2022-09-16', '2022-09-17', '2022-09-18', '2022-09-19', '2022-09-20', '2022-09-21', '2022-09-22', '2022-09-30', '2022-10-07', '2022-10-11', '2022-10-16', '2022-10-18'],
        'group_2': ['2022-09-03', '2022-09-05', '2022-09-09', '2022-10-13', '2022-10-20'],
        'group_3': ['2022-09-14'],
        'group_4': ['2022-09-15', '2022-10-01', '2022-10-02', '2022-10-03', '2022-10-05', '2022-10-08','2022-10-09', '2022-10-10','2022-10-14','2022-10-17'],
        'group_5': ['2022-09-28','2022-09-29'],
        'group_6': ['2022-10-04'],
        'group_7': ['2022-10-06'],
        'group_8': ['2022-10-12', '2022-10-15', '2022-10-19']
    }
    features_to_remove = {
        'group_1': ['out_closeness', 'clustering coefficient', 'betweenness', 'hubs', 'authorities'],
        'group_2': ['out_closeness', 'clustering coefficient', 'betweenness', 'authorities'],
        'group_3': ['out_closeness', 'clustering coefficient', 'betweenness', 'hubs'],
        'group_4': ['clustering coefficient', 'betweenness', 'hubs', 'authorities'],
        'group_5': ['out_closeness', 'clustering coefficient', 'betweenness', 'pagerank', 'hubs', 'authorities'],
        'group_6': ['out_strength', 'clustering coefficient', 'betweenness', 'hubs', 'authorities'],
        'group_7': ['clustering coefficient', 'betweenness', 'authorities'],
        'group_8': ['out_strength', 'out_closeness', 'clustering coefficient', 'betweenness', 'hubs', 'authorities']
    }

    G, G_c, G_e, selfloop_column = preprocess_graphs(day)
    features = extract_features(G, G_c, G_e, selfloop_column)

    for group, days_list in feature_removal_groups.items():
        if day in days_list:
            feat = features.drop(columns=features_to_remove[group], axis=1)
            break

    m = m_values.get(day, 1.5)  # Default value of m if not specified
    features, labels, roles_percentages_sorted, nodes_roles = perform_fcm(feat, features, m)

    # Save graphs only for certain days
    if day in {'2022-09-02', '2022-10-20'}:
        save_graphs(day, G, roles_percentages_sorted, nodes_roles)

    profiler.disable()
    profiler.print_stats(sort='cumtime')  # Show profiling statistics

    return features, labels

# Parallelization with joblib
def run_parallel():
    return Parallel(n_jobs=2)(delayed(process_day)(day) for day in days)

# Code execution
if __name__ == '__main__':
    all_features_labels = run_parallel()
    all_features, all_labels = zip(*all_features_labels)

    # Print information for each day
    for day, labels in zip(days, all_labels):
        print(f"Day {day}: Number of labels: {len(labels)}")

    print("Analysis completed.")

# Function to create the Sankey diagram for 3 roles
def create_sankey(all_labels, days):
    # Create a list of links between consecutive days for each node
    source = []
    target = []
    value = []

    # Iterate over pairs of consecutive days
    for i in range(len(all_labels) - 1):
        labels_day1 = all_labels[i]
        labels_day2 = all_labels[i + 1]

        # Find the minimum length between labels_day1 and labels_day2
        min_length = min(len(labels_day1), len(labels_day2))

        # Create a mapping between roles on consecutive days
        for j in range(min_length):  # Iterate only up to the minimum length
            source.append(f'{days[i]}_role_{labels_day1[j]}')
            target.append(f'{days[i + 1]}_role_{labels_day2[j]}')
            value.append(1)

    # Collect all unique labels
    labels = list(set(source + target))

    # Map labels to their indices
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # Convert source and target to their numeric indices
    source_indices = [label_to_index[s] for s in source]
    target_indices = [label_to_index[t] for t in target]

    # Create the Sankey diagram with padding and smaller font
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=30,  # Increase padding to better separate nodes
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[label.split('_')[0] for label in labels],  # Show only dates, hiding roles
            color="blue"
        ),
        link=dict(
            source=source_indices,
            target=target_indices,
            value=value,
            color="rgba(63, 81, 181, 0.5)"  # Make links more transparent
        )
    ))

    
    fig.update_layout(
        title_text="Role Evolution between Days",
        font_size=10,
        height=600, 
        margin=dict(l=0, r=0, t=50, b=50)  
    )

    
    fig.show()


create_sankey(all_labels, days)
