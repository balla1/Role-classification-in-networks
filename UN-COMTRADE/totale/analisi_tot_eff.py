import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from fcmeans import FCM
from sklearn.metrics import rand_score
import ternary 
import matplotlib.colors as mcolors
import plotly.express as px

from pandas.api.types import CategoricalDtype

def to_cartesian(point, scale=1):
    """
    Converte le coordinate ternarie in coordinate cartesiane.

    Parametri:
    point: tuple o lista di tre float (a, b, c)
    scale: float, scala del triangolo ternario (default=1)

    Ritorna:
    tuple di due float (x, y)
    """
    a, b, c = point
    s = a + b + c
    x = (b + 0.5 * c) / s
    y = (np.sqrt(3) / 2) * c / s
    return (x * scale, y * scale)


file_base = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\dati\\"
salvataggio = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\totale\\"

# Parametri specifici per ogni anno
m_values = {
    1995: 1.5,
    1996: 1.5,
    1997: 2.0,
    1998: 1.5,
    1999: 2.0,
    2000: 2.0,
    2001: 1.5,
    2002: 2.0,
    2003: 1.5,
    2004: 2.0,
    2005: 1.5,
    2006: 1.5,
    2007: 2.0,
    2008: 1.5,
    2009: 2.0,
    2010: 2.0,
    2011: 1.5,
    2012: 2.0,
    2013: 1.5,
    2014: 2.0,
    2015: 1.5,
    2016: 2.0,
    2017: 1.5,
    2018: 2.0,
    2019: 2.0,
    2020: 1.5,
    2021: 1.5,
    2022: 2.0
}
np.random.seed(42)
years = list(range(1995, 2023))
all_features = []
all_labels = []
all_nodes_roles = []
# Liste per le coordinate dei plot finali
ita_coord = []
chn_coord = []
usa_coord = []
qat_coord = []
jpn_coord = []
ind_coord = []
ukr_coord = []

selected_countries = ['ITA', 'CHN', 'JPN', 'USA', 'QAT', 'IND', 'UKR']


def normalize_coords(coords_list):
    normalized_coords = []
    for coords in coords_list:
        total = sum(coords)
        if total == 0:
            normalized_coords.append([0, 0, 0])
        else:
            normalized_coords.append([c / total for c in coords])
    return normalized_coords

def plot_country_ternary(country_coords, country_name, years, filename):
    scale = 1.0  # Le appartenenze ai cluster sommano a 1

    # Creazione della figura
    fig, ax = plt.subplots(figsize=(8, 7))
    tax = ternary.TernaryAxesSubplot(ax=ax, scale=scale)
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="gray", multiple=0.1)
    tax.set_title(f'Evoluzione dei Ruoli per {country_name}', fontsize=15)

    # Impostazione delle etichette degli assi
    fontsize = 12
    tax.left_axis_label("Role 2", fontsize=fontsize)
    tax.right_axis_label("Role 1", fontsize=fontsize)
    tax.bottom_axis_label("Role 0", fontsize=fontsize)
    tax.ticks(axis='lbr', multiple=0.1, linewidth=1, fontsize=10)
    tax.clear_matplotlib_ticks()

    # Creazione di una colormap
    cmap = plt.get_cmap('viridis')
    norm = mcolors.Normalize(vmin=years[0], vmax=years[-1])

    # Plot dei punti
    for i in range(len(country_coords)):
        point = country_coords[i]
        year = years[i]
        color = cmap(norm(year))

        # Plot del punto
        tax.scatter([point], marker='o', facecolor=color, s=50, vmin=None, vmax=None)


        # Annotazione dell'anno ogni 5 anni
        if year % 5 == 0 or i == 0 or i == len(country_coords) - 1:
            tax.annotate(text=f'{year}', position=point, fontsize=8, horizontalalignment='center')

    # Creazione di una barra dei colori
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1)
    cbar.set_label('Anno', fontsize=12)

    tax._redraw_labels()
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Function to read and preprocess graphs
def preprocess_graphs(year):
    G = nx.read_weighted_edgelist(f'{file_base}{year}\\{year}_normalizzato.txt', create_using=nx.DiGraph)
    G_c = nx.read_weighted_edgelist(f'{file_base}{year}\\{year}_closeness.txt', create_using=nx.DiGraph)
    G_e = nx.read_weighted_edgelist(f'{file_base}{year}\\{year}_paesi_edgelist.txt', create_using=nx.DiGraph)
    
    # Remove self-loops and store them
    selfloop_column = remove_self_loops(G)
    remove_self_loops(G_c)
    remove_self_loops(G_e)
    
    return G, G_c, G_e, selfloop_column

def remove_self_loops(G):
    selfloops = list(nx.selfloop_edges(G, data='weight'))
    G.remove_edges_from(nx.selfloop_edges(G))
    return [weight for u, v, weight in selfloops]

# Function to extract features
def extract_features(G, G_c, G_e, selfloop_column):
    in_str = G.in_degree(weight='weight')
    out_str = G.out_degree(weight='weight')
    tot_str = G.degree(weight='weight')
    hubs, auths = nx.hits(G)
    bet = nx.betweenness_centrality(G, weight='weight')
    clc = nx.clustering(G, weight='weight')
    pag = nx.pagerank(G, alpha=1, weight='weight')
    in_clo = nx.closeness_centrality(G_c, distance="weight")
    out_clo = nx.closeness_centrality(G_c.reverse(), distance="weight")
    eig = nx.eigenvector_centrality(G_e, weight='weight')

    in_strength = [x[1] for x in in_str]
    out_strength = [x[1] for x in out_str]
    tot_strength = [x[1] for x in tot_str]
    features=pd.DataFrame({'in_strength': in_strength}) 
    features.index=bet.keys()
    features['out_strength']=out_strength
    features['total_strength']=tot_strength
    features['eigenvector']=features.index.map(eig)
    features['in_closeness']=features.index.map(in_clo)
    features['out_closeness']=features.index.map(out_clo)
    features['clustering coefficient']=features.index.map(clc)
    features['betweenness']=features.index.map(bet)
    features['pagerank']=features.index.map(pag)
    features['hubs']=features.index.map(hubs)
    features['authorities']=features.index.map(auths)
    
    if selfloop_column:
        features['selfloop'] = selfloop_column
    
    return features

# Funzione per eseguire FCM
def perform_fcm(features, m):
    feat = features.drop(columns=['clustering coefficient', 'authorities', 'betweenness', 'out_closeness'], axis=1)
    fcm = FCM(n_clusters=3, m=m)
    fcm.fit(feat.values)
    
    roles_percentages = pd.DataFrame(fcm.u, columns=['Role_0', 'Role_1', 'Role_2'])
    roles_percentages.index = features.index
    nodes_roles = roles_percentages.idxmax(axis=1)
    
    V = pd.DataFrame(fcm.centers, columns=feat.columns)
    V['sum'] = V.sum(axis=1)
    sorted_indices = V['sum'].argsort().values
    V_sorted = V.loc[sorted_indices]
    
    roles_percentages_sorted = roles_percentages.iloc[:, sorted_indices]
    roles_percentages_sorted.columns = ['Role_0', 'Role_1', 'Role_2']
    
    labels = list(nodes_roles)
    for i in range(len(labels)):
        labels[i] = int(labels[i][-1])
    
    return features, labels, roles_percentages_sorted, nodes_roles

# Funzione per salvare i grafici
def save_graphs(year, G, roles_percentages_sorted, nodes_roles):
    label = {}
    for node in G.nodes:
        label[node] = nodes_roles.loc[node]

    nx.set_node_attributes(G, label, 'fcm')
    nx.write_graphml(G, f'{salvataggio}{year}_tot.graphml')

    role_colors = {'Role_0': 'blue', 'Role_1': 'red', 'Role_2': 'green'}
    pos = nx.spring_layout(G, k=0.12)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    for node in G.nodes():
        percentages = roles_percentages_sorted.loc[node, ['Role_0', 'Role_1', 'Role_2']].values
        wedges, _ = ax.pie(percentages, colors=[role_colors['Role_0'], role_colors['Role_1'], role_colors['Role_2']],
                           radius=0.1, center=pos[node], wedgeprops=dict(width=0.03, edgecolor='w'))
        ax.text(pos[node][0], pos[node][1], node, horizontalalignment='center', verticalalignment='center',
                fontweight='bold', color='black')
    
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.01, edge_color='gray', width=0.5)
    ax.legend(wedges, ['Role_0', 'Role_1', 'Role_2'], title="Roles", loc="best")
    ax.set_aspect('equal')
    plt.xlim([min(x[0] for x in pos.values()) - 0.1, max(x[0] for x in pos.values()) + 0.1])
    plt.ylim([min(x[1] for x in pos.values()) - 0.1, max(x[1] for x in pos.values()) + 0.1])
    plt.axis('off')
    plt.savefig(f'{salvataggio}g{year}_tot.png', bbox_inches='tight')
    plt.close()

# Loop per ogni anno
for year in years:
    G, G_c, G_e, selfloop_column = preprocess_graphs(year)
    features = extract_features(G, G_c, G_e, selfloop_column)
    m = m_values.get(year, 1.5)  # Valore di default di m se non specificato
    features, labels, roles_percentages_sorted, nodes_roles = perform_fcm(features, m)
    
    all_features.append(features)
    all_labels.append(labels)
    all_nodes_roles.append(nodes_roles) 

    # Aggiornare le coordinate
    coords = roles_percentages_sorted.loc['ITA'].tolist()
    ita_coord.append(coords)
    coord = roles_percentages_sorted.loc['CHN'].tolist()
    chn_coord.append(coord)
    coor = roles_percentages_sorted.loc['BRN'].tolist()
    qat_coord.append(coor)
    c_u = roles_percentages_sorted.loc['USA'].tolist()
    usa_coord.append(c_u)
    if year==1995 or year==1996:
        temp = roles_percentages_sorted.loc['JPN', 'Role_2']
        roles_percentages_sorted.loc['JPN', 'Role_2'] = roles_percentages_sorted.loc['JPN', 'Role_1']
        roles_percentages_sorted.loc['JPN', 'Role_1'] = temp
    c_j = roles_percentages_sorted.loc['JPN'].tolist()
    jpn_coord.append(c_j)
    c_i = roles_percentages_sorted.loc['IND'].tolist()
    ind_coord.append(c_i)
    c_k = roles_percentages_sorted.loc['UKR'].tolist()
    ukr_coord.append(c_k)
    
    if year in {1995, 2020, 2022}:
        save_graphs(year, G, roles_percentages_sorted, nodes_roles)

# Verifica delle etichette per ogni anno
def print_label_info(label_list, year):
    print(f"Year {year}:")
    print(f"Number of labels: {len(label_list)}")
    #print(f"Labels: {label_list[:5]}...")

# Stampa informazioni sulle etichette per ogni anno
for year, labels in zip(years, all_labels):
    print_label_info(labels, year)

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
plt.savefig(f'{salvataggio}filtered_rand_score_tot.png', bbox_inches='tight')
plt.close()

# Stampa dei paesi mancanti anno per anno
for i in range(len(years) - 1):
    year1, year2 = years[i], years[i + 1]
    countries1, countries2 = set(all_features[i].index), set(all_features[i + 1].index)
    missing_in_year2 = countries1 - countries2
    missing_in_year1 = countries2 - countries1
    print(f"Paesi mancanti in {year2} rispetto a {year1}: {missing_in_year2}")
    print(f"Paesi mancanti in {year1} rispetto a {year2}: {missing_in_year1}")

# Analisi delle coordinate dei paesi selezionati
role_0_ita, role_1_ita, role_2_ita = [], [], []
role_0_chn, role_1_chn, role_2_chn = [], [], []
role_0_qat, role_1_qat, role_2_qat = [], [], []
role_0_usa, role_1_usa, role_2_usa = [], [], []
role_0_jpn, role_1_jpn, role_2_jpn = [], [], []
role_0_ind, role_1_ind, role_2_ind = [], [], []
role_0_ukr, role_1_ukr, role_2_ukr = [], [], []

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
def plot_role_evolution(x, role_0, role_1, role_2, country, filename):
    plt.plot(x, role_0, label="role_0")
    plt.plot(x, role_1, label="role_1")
    plt.plot(x, role_2, label="role_2")
    plt.legend(bbox_to_anchor=(0.0, -0.17, 1, -0.027), mode='expand', loc='lower center', borderaxespad=0., ncols=3)
    plt.title(country)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Salvare i grafici per ciascun paese
plot_role_evolution(x, role_0_ita, role_1_ita, role_2_ita, 'Italy', f'{salvataggio}ITA_tot.png')
plot_role_evolution(x, role_0_chn, role_1_chn, role_2_chn, 'China', f'{salvataggio}CHI_tot.png')
plot_role_evolution(x, role_0_usa, role_1_usa, role_2_usa, 'USA', f'{salvataggio}USA_tot.png')
plot_role_evolution(x, role_0_qat, role_1_qat, role_2_qat, 'Qatar', f'{salvataggio}QAT_tot.png')
plot_role_evolution(x, role_0_jpn, role_1_jpn, role_2_jpn, 'Japan', f'{salvataggio}JPN_tot.png')
plot_role_evolution(x, role_0_ind, role_1_ind, role_2_ind, 'India', f'{salvataggio}IND_tot.png')
plot_role_evolution(x, role_0_ukr, role_1_ukr, role_2_ukr, 'Ukraine', f'{salvataggio}UKR_tot.png')


ita_coords_normalized = normalize_coords(ita_coord)
chn_coords_normalized = normalize_coords(chn_coord)
usa_coords_normalized = normalize_coords(usa_coord)
qat_coords_normalized = normalize_coords(qat_coord)
jpn_coords_normalized = normalize_coords(jpn_coord)
ind_coords_normalized = normalize_coords(ind_coord)
ukr_coords_normalized = normalize_coords(ukr_coord)

# Lista degli anni
years = list(range(1995, 2023))

# Creazione dei grafici per ciascun paese
plot_country_ternary(ita_coords_normalized, 'Italy', years, f'{salvataggio}ITA_ternary.png')
plot_country_ternary(chn_coords_normalized, 'China', years, f'{salvataggio}CHN_ternary.png')
plot_country_ternary(usa_coords_normalized, 'USA', years, f'{salvataggio}USA_ternary.png')
plot_country_ternary(qat_coords_normalized, 'Qatar', years, f'{salvataggio}QAT_ternary.png')
plot_country_ternary(jpn_coords_normalized, 'Japan', years, f'{salvataggio}JPN_ternary.png')
plot_country_ternary(ind_coords_normalized, 'India', years, f'{salvataggio}IND_ternary.png')
plot_country_ternary(ukr_coords_normalized, 'Ukraine', years, f'{salvataggio}UKR_ternary.png')

# Dopo aver completato l'analisi per tutti gli anni

# Generazione del diagramma di Sankey
import plotly.graph_objects as go

# Lista degli anni per i quali vuoi creare il diagramma di Sankey




# Supponiamo di avere le seguenti variabili già definite:
# - years: lista degli anni
# - selected_countries: lista dei paesi selezionati
# - all_nodes_roles: lista dei ruoli dominanti per ciascun anno (output di perform_fcm)
# - salvataggio: percorso di salvataggio dei file
# Mappare i codici dei paesi ai nomi completi
country_names = {
    'ITA': 'Italy',
    'CHN': 'China',
    'JPN': 'Japan',
    'USA': 'USA',
    'QAT': 'Qatar',
    'IND': 'India',
    'UKR': 'Ukraine'
}

# Preparare i dati
data = []

for i, year in enumerate(years):
    nodes_roles = all_nodes_roles[i]
    for country in selected_countries:
        if country in nodes_roles.index:
            role = nodes_roles.loc[country]
            data.append({'country': country, 'year': year, 'role': role})

df = pd.DataFrame(data)

# Creare le etichette dei nodi
df['node_label'] = df['country'] + ' ' + df['year'].astype(str)
labels = df['node_label'].unique()
label_to_index = {label: idx for idx, label in enumerate(labels)}

# Definire i colori dei ruoli
role_colors = {'Role_0': 'blue', 'Role_1': 'red', 'Role_2': 'green'}

# Creare la lista dei colori dei nodi
node_colors = [role_colors[df[df['node_label'] == label]['role'].values[0]] for label in labels]

# Mappature per le posizioni dei nodi
years_list = sorted(df['year'].unique())
year_to_x = {int(year): i / (len(years_list) - 1) for i, year in enumerate(years_list)}

countries_list = sorted(selected_countries)
country_to_y = {country: i / (len(countries_list) - 1) for i, country in enumerate(countries_list)}

# Posizioni dei nodi
node_x = []
node_y = []

for label in labels:
    country, year = label.split()
    x = year_to_x[int(year)]
    y = country_to_y[country]
    node_x.append(x)
    node_y.append(y)

# Costruire le liste source, target, value, link_roles e link_countries
sources = []
targets = []
values = []
link_roles = []
link_countries = []

for country in selected_countries:
    country_data = df[df['country'] == country].sort_values('year')
    num_entries = len(country_data)
    for i in range(num_entries - 1):
        source_label = country_data.iloc[i]['node_label']
        target_label = country_data.iloc[i + 1]['node_label']
        source_idx = label_to_index[source_label]
        target_idx = label_to_index[target_label]
        sources.append(source_idx)
        targets.append(target_idx)
        values.append(1)
        link_roles.append(country_data.iloc[i + 1]['role'])
        link_countries.append(country)

# Colori dei link (in base ai ruoli)
link_colors = [role_colors[role] for role in link_roles]

# Creare il diagramma di Sankey
fig = go.Figure(data=[go.Sankey(
    arrangement='fixed',
    node=dict(
        pad=10,
        thickness=10,
        line=dict(color='black', width=0.5),
        label=['' for _ in labels],
        color=node_colors,
        x=node_x,
        y=node_y,
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors,
        hovertemplate='Paese: %{customdata}<br>Anno: %{source.label.split(" ")[1]}<br>Ruolo: %{target.label}<extra></extra>',
        customdata=[country_names.get(country, country) for country in link_countries]
    )
)])

# Aggiungere annotazioni per gli anni sull'asse x
for year, x_pos in year_to_x.items():
    fig.add_annotation(
        x=x_pos,
        y=1.05,
        xref='paper',
        yref='paper',
        text=str(year),
        showarrow=False,
        font=dict(size=10),
        xanchor='center',
        yanchor='bottom'
    )

# Aggiungere annotazioni per i paesi sull'asse y
for country, y_pos in country_to_y.items():
    fig.add_annotation(
        x=-0.02,
        y=y_pos,
        xref='paper',
        yref='paper',
        text=country_names.get(country, country),
        showarrow=False,
        font=dict(size=10),
        xanchor='right',
        yanchor='middle'
    )

# Aggiornare il layout
fig.update_layout(
    title_text='Transizioni dei Ruoli Dominanti per i Paesi Selezionati nel Tempo',
    font_size=10,
    width=1200,
    height=600,
    margin=dict(l=100, r=100, t=100, b=100),
)

# Salvare il grafico
output_path = f"{salvataggio}sankey_diagram_selected_countries.png"
fig.write_image(output_path)

data = []

for i, year in enumerate(years):
    nodes_roles = all_nodes_roles[i]
    for country in selected_countries:
        if country in nodes_roles.index:
            role = nodes_roles.loc[country]
            data.append({'country': country, 'year': year, 'role': role})

df = pd.DataFrame(data)

# Convertire 'year' e 'role' in stringhe
df['year'] = df['year'].astype(str)

# Definire l'ordine delle categorie per 'role'
role_categories = ['Role_0', 'Role_1', 'Role_2']
role_cat_type = CategoricalDtype(categories=role_categories, ordered=True)
df['role'] = df['role'].astype(role_cat_type)

# Mappare i codici dei paesi ai nomi completi
country_names = {
    'ITA': 'Italy',
    'CHN': 'China',
    'JPN': 'Japan',
    'USA': 'USA',
    'QAT': 'Qatar',
    'IND': 'India',
    'UKR': 'Ukraine'
}

df['country_name'] = df['country'].map(country_names)
# Mappare i ruoli a colori
role_colors = {'Role_0': 'blue', 'Role_1': 'red', 'Role_2': 'green'}
df['role_color'] = df['role'].map(role_colors)

dimensions = [
    dict(values=df['year'], label='Anno'),
    dict(values=df['country_name'], label='Paese'),
    dict(values=df['role'], label='Ruolo Dominante'),
]

color = df['role_color'].tolist()

fig = go.Figure(data=[go.Parcats(
    dimensions=dimensions,
    line={'color': color, 'shape': 'hspline'},
)])

fig.update_layout(
    title="Transizioni dei Ruoli Dominanti dei Paesi Selezionati nel Tempo",
    font=dict(size=12),
    width=1200,
    height=600,
)

# Assicurati di avere kaleido installato
# pip install kaleido

output_path = f"{salvataggio}alluvial_diagram_selected_countries.png"
fig.write_image(output_path)

# Creare un grafico a linee per l'evoluzione dei ruoli
data = []

role_mapping = {'Role_0': 0, 'Role_1': 1, 'Role_2': 2}

for i, year in enumerate(years):
    nodes_roles = all_nodes_roles[i]
    for country in selected_countries:
        if country in nodes_roles.index:
            role = nodes_roles.loc[country]
            role_num = role_mapping[role]
            data.append({'country': country, 'year': year, 'role': role, 'role_num': role_num})

df = pd.DataFrame(data)

country_names = {
    'ITA': 'Italy',
    'CHN': 'China',
    'JPN': 'Japan',
    'USA': 'USA',
    'QAT': 'Qatar',
    'IND': 'India',
    'UKR': 'Ukraine'
}

df['country_name'] = df['country'].map(country_names)


import plotly.graph_objects as go



# Definire una mappa di colori per i ruoli
role_colors = {0: 'blue', 1: 'red', 2: 'green'}

import plotly.graph_objects as go

# Preparare i dati come in precedenza
# ... (codice per creare df con 'country', 'year', 'role', 'role_num', 'country_name') ...

# Funzione per creare il grafico per un gruppo di paesi
def create_role_evolution_plot(countries, filename):
    fig = go.Figure()

    # Aggiungere una traccia per ciascun paese
    for country in countries:
        country_data = df[df['country'] == country]
        fig.add_trace(go.Scatter(
            x=country_data['year'],
            y=country_data['role_num'],
            mode='lines+markers',
            name=country_names[country],
            line=dict(width=2),
            marker=dict(size=8),
            text=country_data['role'],
            hovertemplate='Paese: %{text}<br>Anno: %{x}<br>Ruolo: %{text}<extra></extra>',
        ))

    # Personalizzare l'asse y per mostrare i ruoli
    fig.update_yaxes(
        tickvals=[0, 1, 2],
        ticktext=['Role_0', 'Role_1', 'Role_2'],
        title_text='Ruolo Dominante',
    )

    # Personalizzare l'asse x
    fig.update_xaxes(
        title_text='Anno',
        tickmode='linear',
        tick0=int(df['year'].min()),
        dtick=1,
    )

    # Aggiornare il layout
    fig.update_layout(
        title='Evoluzione dei Ruoli Dominanti per i Paesi Selezionati nel Tempo',
        legend_title_text='Paesi',
        width=1000,
        height=600,
    )

    # Salvare il grafico
    fig.write_image(filename)

# Dividere i paesi in due gruppi
selected_countries_list = list(selected_countries)
mid_index = len(selected_countries_list) // 2
countries_group1 = selected_countries_list[:mid_index]
countries_group2 = selected_countries_list[mid_index:]

# Creare i grafici per ciascun gruppo
output_path1 = f"{salvataggio}role_evolution_group1.png"
create_role_evolution_plot(countries_group1, output_path1)

output_path2 = f"{salvataggio}role_evolution_group2.png"
create_role_evolution_plot(countries_group2, output_path2)



# heatmap per l'evoluzione dei ruoli

# Mappare i ruoli a valori numerici
role_mapping = {'Role_0': 0, 'Role_1': 1, 'Role_2': 2}
df['role_num'] = df['role'].map(role_mapping)

# Creare una tabella pivot
pivot_df = df.pivot(index='country_name', columns='year', values='role_num')

# Creare una heatmap
fig = px.imshow(
    pivot_df,
    labels=dict(x="Anno", y="Paese", color="Ruolo Dominante"),
    x=sorted(df['year'].unique()),
    y=sorted(df['country_name'].unique()),
    color_continuous_scale=['blue', 'red', 'green'],
    aspect='auto'
)

# Aggiornare la barra dei colori
fig.update_coloraxes(
    colorbar=dict(
        tickvals=[0, 1, 2],
        ticktext=['Role_0', 'Role_1', 'Role_2']
    )
)

# Personalizzare il layout
fig.update_layout(
    title="Evoluzione dei Ruoli Dominanti per i Paesi Selezionati nel Tempo",
    width=1200,
    height=600,
)

# Salvare il grafico
output_path = f"{salvataggio}role_heatmap.png"
fig.write_image(output_path)

print("Analisi completata.")
