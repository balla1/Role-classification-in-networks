
import pandas as pd
import numpy as np

def carica_dati(file_path):
    # Carica i dati dal file CSV
    df = pd.read_csv(file_path)
    print("Colonne disponibili nel DataFrame:", df.columns)
    
    # Definisci i suffissi delle colonne da rimuovere
    suffissi_da_rimuovere = ['HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR', 'OUT','T']
    suffissi_righe=['TLS','VA','OUT','T']
    # Costruisci la lista di colonne da rimuovere basata sui suffissi
    columns_to_remove = [col for col in df.columns if any(col.endswith(suffix) for suffix in suffissi_da_rimuovere)]
    

    # Elimina le colonne dal DataFrame
    df.drop(columns=columns_to_remove, errors='ignore', inplace=True)
    
    print("Colonne rimanenti nel DataFrame:", df.columns)
    if (df.select_dtypes(include=[np.number]) < 0).any().any():
        print("Attenzione: ci sono valori negativi nel DataFrame originale.")
    
    # Costruisci la lista di righe da rimuovere basata sui suffissi
    rows_to_remove = df[df['V1'].apply(lambda x: any(x.endswith(suffix) for suffix in suffissi_righe))].index
    
    # Elimina le righe dal DataFrame
    df.drop(index=rows_to_remove, errors='ignore', inplace=True)
    
    if (df.select_dtypes(include=[np.number]) < 0).any().any():
        print("Attenzione: ci sono valori negativi nel DataFrame originale.")
    
    return df

def trasforma_e_aggrega(df):
    agg_data = []
    for idx, row in df.iterrows():
        settore_origine = row['V1'].split('_')[1]  # Assume che V1 sia nel formato PAESE_SETTORE
        for col in df.columns[1:]:  # Salta la prima colonna 'V1'
            settore_destinazione = col.split('_')[1]  # Assume che anche le colonne siano nel formato PAESE_SETTORE
            valore = row[col]
            agg_data.append({
                'Origine': settore_origine, 
                'Destinazione': settore_destinazione, 
                'Valore': valore
            })

    # Crea un DataFrame dai dati aggregati
    agg_df = pd.DataFrame(agg_data)
    # Aggrega i dati per origine e destinazione sommando i valori
    result_df = agg_df.groupby(['Origine', 'Destinazione']).sum().reset_index()
    if (result_df['Valore'] < 0).any():
        print("Attenzione: ci sono valori negativi nei dati aggregati.")
    return result_df



def crea_files(df, base_path):
    print("DataFrame prima di creare i file:", df.head())
    
    if 'Valore' not in df.columns:
        print("Errore: Colonna 'Valore' non presente nel DataFrame.")
        return
    
    df['Valore'] = pd.to_numeric(df['Valore'], errors='coerce')
    df.dropna(subset=['Valore'], inplace=True)
    
    if df['Valore'].min() < 0:
        print("Attenzione: presenti valori negativi nel campo 'Valore'.")
    df['Valore'] = df['Valore'].apply(lambda x: x if x > 0 else 1e-10)
    df['Valore'] = df['Valore'].astype(np.float64)

    total_volume = df['Valore'].sum()
    print("Volume totale calcolato per la normalizzazione:", total_volume)

    if total_volume <= 0:
        print("Errore: Il volume totale è zero, impossibile normalizzare i pesi.")
        return

    df['PesoNormalizzato'] = df['Valore'] / total_volume
    print("Normalized weights stats:", df['PesoNormalizzato'].describe())

    if not np.isclose(df['PesoNormalizzato'].sum(), 1.0, atol=0.01):
        print("Attenzione: la somma dei pesi normalizzati non è uguale a 1.")

    normalized_edgelist = pd.DataFrame({
        'Source': df['Origine'],
        'Target': df['Destinazione'],
        'Weight': df['PesoNormalizzato'].round(12)
    })
    normalized_edgelist.to_csv(f'{base_path}1995_normalizzato.csv', index=False, header=False)

    closeness_weights = 1.0 / df['PesoNormalizzato']
    closeness_edgelist = pd.DataFrame({
        'Source': df['Origine'],
        'Target': df['Destinazione'],
        'Weight': closeness_weights.apply(lambda x: np.format_float_scientific(x, precision=4) if x > 10000 else x)
    })
    closeness_edgelist.to_csv(f'{base_path}1995_closeness.csv', index=False, header=False)

    edgelist = pd.DataFrame({
        'Source': df['Origine'],
        'Target': df['Destinazione'],
        'Weight': df['Valore'].round(6)
    })
    edgelist.to_csv(f'{base_path}1995_aziende_edgelist.csv', index=False, header=False)

    print("Esempio di dati normalizzati:", normalized_edgelist.head())
    print("Esempio di dati closeness:", closeness_edgelist.head())
    print("Esempio di edgelist:", edgelist.head())

# Implementa questi cambiamenti e vedi se risolvono i problemi che stai incontrando.


# Percorso dei file
file_input = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_paesi\\dati_oecd_per_anni\\1995.csv"
file_base_output = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\"
file_new = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\1995_aziende_edgelist.txt"
# Carica e processa i dati


df = carica_dati(file_input)


df_flussi = trasforma_e_aggrega(df)
print("Dopo trasforma e aggrega:", df_flussi.head())

if (df_flussi.select_dtypes(include=[np.number]) < 0).any().any():
    print("Attenzione: ci sono valori negativi nel DataFrame originale.")
    
crea_files(df_flussi, file_base_output)

## paesi in pù rispetto a prima
## BGD,BLR,CIV,CMR,EGY,JOR,NGA,PAK,SEN,UKR


import csv

# Sostituisci 'input.csv' con il percorso del tuo file CSV
input_file = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\1995_normalizzato.csv"
# Sostituisci 'output.txt' con il percorso del file di testo di destinazione
output_file = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\1995_normalizzato.txt"

# Apri il file CSV per leggerlo
with open(input_file, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Apri il file di testo per scrivere
    with open(output_file, mode='w', encoding='utf-8') as txt_file:
        # Per ogni riga nel file CSV, scrivi quella riga nel file di testo
        for row in csv_reader:
            txt_file.write(' '.join(row) + '\n')  # Separare le colonne con spazi



# Sostituisci 'input.csv' con il percorso del tuo file CSV
input_file = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\1995_closeness.csv"
# Sostituisci 'output.txt' con il percorso del file di testo di destinazione
output_file = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\1995_closeness.txt"


# Apri il file CSV per leggerlo
with open(input_file, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Apri il file di testo per scrivere
    with open(output_file, mode='w', encoding='utf-8') as txt_file:
        # Per ogni riga nel file CSV, scrivi quella riga nel file di testo
        for row in csv_reader:
            txt_file.write(' '.join(row) + '\n')  # Separare le colonne con spazi



# Sostituisci 'input.csv' con il percorso del tuo file CSV
input_file = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\1995_aziende_edgelist.csv"
# Sostituisci 'output.txt' con il percorso del file di testo di destinazione
output_file = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_aziende\\oecd_per_anni_aziende\\1995_aziende_edgelist.txt"

# Apri il file CSV per leggerlo
with open(input_file, mode='r', newline='', encoding='utf-8') as csv_file:
    csv_reader = csv.reader(csv_file)
    
    # Apri il file di testo per scrivere
    with open(output_file, mode='w', encoding='utf-8') as txt_file:
        # Per ogni riga nel file CSV, scrivi quella riga nel file di testo
        for row in csv_reader:
            txt_file.write(' '.join(row) + '\n')  # Separare le colonne con spazi



