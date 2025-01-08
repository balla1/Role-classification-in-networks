import pandas as pd
import os

# Percorsi di base per i file di input e le sottocartelle di output
cartella_csv = 'csv'
cartella_txt = 'txt'
cartella_grafici = 'grafici'

# Crea le cartelle di destinazione se non esistono
if not os.path.exists(cartella_txt):
    os.makedirs(cartella_txt)
if not os.path.exists(cartella_grafici):
    os.makedirs(cartella_grafici)

# Funzione per normalizzare i pesi
def normalize_weights(df):
    df['weight_normalized'] = df['weight'] / df['weight'].sum()
    return df

# Funzione per invertire i pesi
def invert_weights(df):
    df['weight_inverted'] = 1 / df['weight_normalized']
    return df

# Itera attraverso tutti i file nella cartella CSV
for filename in os.listdir(cartella_csv):
    if filename.endswith('.csv'):
        # Percorso completo del file CSV
        csv_file_path = os.path.join(cartella_csv, filename)
        # Legge il file CSV
        df = pd.read_csv(csv_file_path)
        
        # Filtra solo i retweet
        df = df[df['type'] == 'retweet']
        
        # Stampa il nome del file e informazioni preliminari
        print(f"Analisi preliminare del file: {filename}")
        print(df.head())
        
        # Rimozione di duplicati (se presenti)
        df = df.drop_duplicates()

        # Gestione dei valori mancanti (se presenti)
        df = df.dropna()

        # Conta i retweet fra coppie di utenti
        df['weight'] = df.groupby(['author_id', 'to_user_id'])['type'].transform('count')

        # Somma i pesi per ogni coppia di utenti
        df_summed = df.groupby(['author_id', 'to_user_id'], as_index=False)['weight'].sum()

        # Salvataggio del file originale
        txt_file_path_original = os.path.join(cartella_txt, filename.replace('.csv', '_retweet_weighted_network.txt'))
        df_summed[['author_id', 'to_user_id', 'weight']].to_csv(txt_file_path_original, index=False, sep='\t', header=False)

        # Normalizzazione dei pesi
        df_normalized = normalize_weights(df_summed.copy())
        txt_file_path_normalized = os.path.join(cartella_txt, filename.replace('.csv', '_normalized_network.txt'))
        df_normalized[['author_id', 'to_user_id', 'weight_normalized']].to_csv(txt_file_path_normalized, index=False, sep='\t', header=False)
        
        # Inversione dei pesi per closeness
        df_inverted = invert_weights(df_normalized.copy())
        txt_file_path_inverted = os.path.join(cartella_txt, filename.replace('.csv', '_inverted_network.txt'))
        df_inverted[['author_id', 'to_user_id', 'weight_inverted']].to_csv(txt_file_path_inverted, index=False, sep='\t', header=False)

print("Conversione e analisi preliminare completate!")
