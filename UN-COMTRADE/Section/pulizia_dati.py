import pandas as pd
import numpy as np
import csv
import os

# Percorsi di base per i file di input e le sottocartelle di output
base_input_path = 'D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\dati\\'
base_output_path = 'D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\dati\\'
country_codes_path = base_input_path + 'country_codes_V202401b.csv'
product_codes_path = base_input_path + 'product_codes_HS92_V202401b.csv'

# Carica i dati globali una sola volta per evitare ripetizioni
countries = pd.read_csv(country_codes_path)
pd.set_option('display.max_rows', None)
print("countries:", countries)
country_map = countries.set_index('country_code')['country_iso3'].to_dict()
products = pd.read_csv(product_codes_path)
products['code'] = products['code'].astype(str)
section_map = {'6': 'SECTION_II', '7': 'SECTION_II', '8': 'SECTION_II', '9': 'SECTION_II', '10': 'SECTION_II', '11': 'SECTION_II',
               '12': 'SECTION_II', '13': 'SECTION_II', '14': 'SECTION_II', '25': 'SECTION_V', '26': 'SECTION_V', '27': 'SECTION_V',
               '28': 'SECTION_VI', '29': 'SECTION_VI', '30': 'SECTION_VI', '31': 'SECTION_VI', '32': 'SECTION_VI', '33': 'SECTION_VI',
               '34': 'SECTION_VI', '35': 'SECTION_VI', '36': 'SECTION_VI', '37': 'SECTION_VI', '38': 'SECTION_VI', '50': 'SECTION_XI',
               '51': 'SECTION_XI', '52': 'SECTION_XI', '53': 'SECTION_XI', '54': 'SECTION_XI', '55': 'SECTION_XI', '56': 'SECTION_XI',
               '57': 'SECTION_XI', '58': 'SECTION_XI', '59': 'SECTION_XI', '60': 'SECTION_XI', '61': 'SECTION_XI', '62': 'SECTION_XI',
               '63': 'SECTION_XI', '84': 'SECTION_XVI', '85': 'SECTION_XVI', '93': 'SECTION_XIX'}

codici_escludere = {'R20', 'ZA1', 'S19', 'PUS', 'ANT', 'SCG', 'DD', 'DE', 'CS', 'SU', 'YU', 'TLS', 'IOT', 'PCN', 'LUX', 'ZAF', 'SWZ',
                    'GUM', 'TLS', 'BWA', 'SXM', 'MYT', 'ASM', 'NAM', 'LSO', 'PSE', 'SMR', 'MNE', 'SRB', 'CCK', 'CUW', 'SSD', 'BLM', 'BES', 'MYT', 'ATF', 'SHN'}

# Processa i dati per un dato anno
def process_year(year):
    input_file = f'{base_input_path}BACI_HS92_Y{year}_V202401b.csv'
    df = pd.read_csv(input_file, na_values=['NA'])
    df.replace({'NA': np.nan}, inplace=True)
    
    # Pulizia dei dati per la colonna 'q'
    df['q'] = df['q'].astype(str).str.strip()  # Rimuove gli spazi bianchi e converte in stringa per sicurezza
    df['q'] = df['q'].replace('NA', np.nan)  # Sostituisce 'NA' con NaN
    df['q'] = pd.to_numeric(df['q'], errors='coerce')  # Converte in float, coercendo i valori non numerici a NaN

    df['v'] = df['v'].astype(float)
    df['k'] = df['k'].astype(str)
    df['i'] = df['i'].map(country_map)
    df['j'] = df['j'].map(country_map)
    df = df[~df['i'].isin(codici_escludere) & ~df['j'].isin(codici_escludere)]
    df = df.merge(products, left_on='k', right_on='code', how='left')
    df['Section'] = df['k'].apply(lambda x: section_map.get(x[:2], 'Other'))
    df['Section'] = df['Section'].fillna('Other')

    sections_of_interest = ['SECTION_II', 'SECTION_V', 'SECTION_VI', 'SECTION_XI', 'SECTION_XVI', 'SECTION_XIX']
    df = df[df['Section'].isin(sections_of_interest)]
    df = df.dropna(subset=['q', 'v'])  # Rimuove le righe con valori NaN in 'q' o 'v'

    if df[['i', 'j', 'v']].isnull().any().any():
        print(f"Attenzione: ci sono valori nulli nei dati principali per l'anno {year}")
    output_dir = os.path.join(base_output_path, str(year))
    os.makedirs(output_dir, exist_ok=True)

    for section in sections_of_interest:
        section_df = df[df['Section'] == section]
        if section_df[['i', 'j', 'v']].isnull().any().any():
            print(f"Attenzione: ci sono valori nulli in {section} per l'anno {year}")
        else:
            # Salva il dataframe per export standard
            section_df[['i', 'j', 'v']].to_csv(os.path.join(output_dir, f'{year}_paesi_edgelist_{section}.csv'), index=False, header=False)

            # Calcola e salva il dataframe per i valori normalizzati
            normalized_df = section_df.copy()
            normalized_df['Normalized'] = normalized_df['v'] / normalized_df['v'].sum()
            normalized_df[['i', 'j', 'Normalized']].to_csv(os.path.join(output_dir, f'{year}_normalizzato_{section}.csv'), index=False, header=False)

            # Calcola e salva il dataframe per closeness
            closeness_df = normalized_df.copy()
            closeness_df['Closeness'] = 1 / closeness_df['Normalized']
            closeness_df[['i', 'j', 'Closeness']].to_csv(os.path.join(output_dir, f'{year}_closeness_{section}.csv'), index=False, header=False)

# Ciclo sugli anni dal 1995 al 2022
for year in range(1995, 2023):
    process_year(year)
    print(f'Dati per l\'anno {year} elaborati e salvati.')

# Funzione per caricare e aggregare i dati
def carica_e_aggrega_anno(anno):
    input_file = f'{base_input_path}BACI_HS92_Y{anno}_V202401b.csv'
    chunksize = 100000  # Definisci la dimensione del chunk
    aggregated_data = []

    for chunk in pd.read_csv(input_file, na_values=['NA'], chunksize=chunksize):
        chunk['k'] = chunk['k'].astype(str)
        chunk = chunk[~chunk['k'].str.startswith(('98', '99'))]
        
        # Mappatura dei codici paese
        chunk['i'] = chunk['i'].map(country_map)
        chunk['j'] = chunk['j'].map(country_map)
        chunk = chunk[~chunk['i'].isin(codici_escludere) & ~chunk['j'].isin(codici_escludere)]
        chunk['q'] = chunk['q'].astype(str).str.strip()
        chunk['q'] = chunk['q'].replace('NA', np.nan)
        chunk['q'] = pd.to_numeric(chunk['q'], errors='coerce')
        chunk = chunk.dropna(subset=['q', 'v'])

        if chunk[['i', 'j', 'v']].isnull().any().any():
            print(f"Attenzione: ci sono valori nulli nei dati principali per l'anno {anno}")

        aggregated_chunk = chunk.groupby(['i', 'j'])[['v', 'q']].sum().reset_index()
        aggregated_data.append(aggregated_chunk)
    
    df_aggregato = pd.concat(aggregated_data)
    df_aggregato = df_aggregato.groupby(['i', 'j'])[['v', 'q']].sum().reset_index()
    
    # Calcolo delle nuove metriche
    df_aggregato['Value_per_Unit'] = df_aggregato['v'] / df_aggregato['q']
    df_aggregato['Total_Value'] = df_aggregato['v'] * df_aggregato['q']
    
    # Crea la sottocartella se non esiste
    year_output_path = f'{base_output_path}{anno}'
    if not os.path.exists(year_output_path):
        os.makedirs(year_output_path)
    
    # Salvataggio dei file di base
    df_aggregato[['i', 'j', 'v']].to_csv(f'{year_output_path}\\{anno}_paesi_edgelist_value.csv', index=False, header=False)
    df_aggregato[['i', 'j', 'q']].to_csv(f'{year_output_path}\\{anno}_paesi_edgelist_quantity.csv', index=False, header=False)
    df_aggregato[['i', 'j', 'Value_per_Unit']].to_csv(f'{year_output_path}\\{anno}_paesi_edgelist_value_per_unit.csv', index=False, header=False)
    df_aggregato[['i', 'j', 'Total_Value']].to_csv(f'{year_output_path}\\{anno}_paesi_edgelist_total_value.csv', index=False, header=False)
    
    # Calcolo e salvataggio dei file normalizzati e closeness per tutte le metriche
    metrics = ['v', 'q', 'Value_per_Unit', 'Total_Value']
    for metric in metrics:
        normalized_df = df_aggregato[['i', 'j', metric]].copy()
        normalized_df['Normalized'] = normalized_df[metric] / normalized_df[metric].sum()
        normalized_df[['i', 'j', 'Normalized']].to_csv(f'{year_output_path}\\{anno}_normalizzato_{metric}.csv', index=False, header=False)
        
        closeness_df = normalized_df.copy()
        closeness_df['Closeness'] = 1 / closeness_df['Normalized']
        closeness_df[['i', 'j', 'Closeness']].to_csv(f'{year_output_path}\\{anno}_closeness_{metric}.csv', index=False, header=False)
    
    print(f"Dati per l'anno {anno} aggregati e salvati correttamente.")

# Ciclo per tutti gli anni dal 1995 al 2023
for anno in range(1995, 2023):
    carica_e_aggrega_anno(anno)

def csv_to_txt(input_dir):
    # Scansiona ogni cartella e file dentro la directory specificata
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                txt_file_path = csv_file_path.replace('.csv', '.txt')
                
                # Leggi il file CSV e scrivi i contenuti in un file TXT
                with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
                    reader = csv.reader(csv_file)
                    with open(txt_file_path, mode='w', encoding='utf-8') as txt_file:
                        for row in reader:
                            txt_file.write(' '.join(row) + '\n')
                print(f'Convertito {csv_file_path} in {txt_file_path}')

# Percorso della cartella contenente i file CSV
base_output_path = 'D:\\Mattia Ballardini\\TESI\\role_network_analysis\\UN-comtrade\\dati'
csv_to_txt(base_output_path)
