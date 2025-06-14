import pandas as pd
import numpy as np

def load_data(file_path):
    # Load data from the CSV file
    df = pd.read_csv(file_path)
    print("Available columns in the DataFrame:", df.columns)

    # Define column suffixes to remove
    suffixes_to_remove = ['HFCE', 'NPISH', 'GGFC', 'GFCF', 'INVNT', 'DPABR', 'OUT']
    row_suffixes = ['TLS', 'VA', 'OUT']
    
    # Build a list of columns to remove based on suffixes
    columns_to_remove = [col for col in df.columns if any(col.endswith(suffix) for suffix in suffixes_to_remove)]
    
    # Drop the columns from the DataFrame
    df.drop(columns=columns_to_remove, errors='ignore', inplace=True)
    
    print("Remaining columns in the DataFrame:", df.columns)
    if (df.select_dtypes(include=[np.number]) < 0).any().any():
        print("Warning: there are negative values in the original DataFrame.")

    # Build a list of rows to remove based on suffixes
    rows_to_remove = df[df['V1'].apply(lambda x: any(x.endswith(suffix) for suffix in row_suffixes))].index
    
    # Drop the rows from the DataFrame
    df.drop(index=rows_to_remove, errors='ignore', inplace=True)
    
    if (df.select_dtypes(include=[np.number]) < 0).any().any():
        print("Warning: there are negative values in the original DataFrame.")
    
    return df

def transform_and_aggregate(df):
    agg_data = []

    for idx, row in df.iterrows():
        parts = row['V1'].split('_')
        source_country = parts[0]  # Takes the first element as the country
        source_sector = '_'.join(parts[1:])  # Joins the rest as the sector

        for col in df.columns[1:]:  # Skip the first column 'V1'
            parts = col.split('_')
            target_country = parts[0]
            target_sector = '_'.join(parts[1:])

            value = row[col]
            agg_data.append({'Source': source_country, 'Target': target_country, 'Value': value})

    # Create a DataFrame from aggregated data
    agg_df = pd.DataFrame(agg_data)
    # Aggregate data by source and target, summing values
    result_df = agg_df.groupby(['Source', 'Target']).sum().reset_index()
    if (result_df['Value'] < 0).any():
        print("Warning: there are negative values in the aggregated data.")
    return result_df

def consolidate_china_mexico(df):
    # Map each country code to its simplified form
    country_mappings = {
        'CHN': ['CN1', 'CN2'],
        'MEX': ['MX1', 'MX2']
    }

    # Iterate over each country and its old codes
    for new_name, old_codes in country_mappings.items():
        # For each old code, find matching 'Source' and 'Target', sum and replace
        for code in old_codes:
            source_mask = (df['Source'] == code)
            target_mask = (df['Target'] == code)

            # Aggregate rows where either 'Source' or 'Target' match the old code
            df.loc[source_mask, 'Source'] = new_name
            df.loc[target_mask, 'Target'] = new_name

    # After adjusting names, sum duplicate rows that might have been created by name consolidation
    df = df.groupby(['Source', 'Target']).sum().reset_index()

    return df

def create_files(df, base_path):
    print("DataFrame before creating files:", df.head())
    
    # Ensure all weights are numeric, handle errors with 'coerce'
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df.dropna(subset=['Value'], inplace=True)  # Drop rows where value is not convertible

    # Calculate total volume for normalization
    total_volume = df['Value'].sum()
    print("Total volume calculated for normalization:", total_volume)

    if total_volume > 0:
        df['NormalizedWeight'] = df['Value'] / total_volume
        print("Normalized weights stats:", df['NormalizedWeight'].describe())
    else:
        print("Error: Total volume is zero, unable to normalize weights.")
        return  # Exit if total volume is zero to avoid division by zero

    # Save normalized data
    normalized_edgelist = pd.DataFrame({
        'Source': df['Source'],
        'Target': df['Target'],
        'Weight': df['NormalizedWeight'].round(12)
    })
    normalized_edgelist.to_csv(f'{base_path}2020_normalized.csv', index=False, header=False)

    # Calculate and save weights for closeness
    closeness_weights = 1.0 / df['NormalizedWeight']
    closeness_edgelist = pd.DataFrame({
        'Source': df['Source'],
        'Target': df['Target'],
        'Weight': closeness_weights.apply(lambda x: np.format_float_scientific(x, precision=4) if x > 10000 else x)
    })
    closeness_edgelist.to_csv(f'{base_path}2020_closeness.csv', index=False, header=False)

    # Save the original edgelist with rounded weights
    edgelist = pd.DataFrame({
        'Source': df['Source'],
        'Target': df['Target'],
        'Weight': df['Value'].round(6)
    })
    edgelist.to_csv(f'{base_path}2020_countries_edgelist.csv', index=False, header=False)

    # Control output
    print("Example of normalized data:", normalized_edgelist.head())
    print("Example of closeness data:", closeness_edgelist.head())
    print("Example of edgelist:", edgelist.head())



# Define file paths
file_input = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_paesi\\dati_oecd_per_anni\\2020.csv"
file_base_output = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_paesi\\dati_oecd_per_anni\\"
file_new = "D:\\Mattia Ballardini\\TESI\\role_network_analysis\\oecd_paesi\\dati_oecd_per_anni\\2020_paesi_edgelist.txt"

# Load and process data
df = load_data(file_input)
print(df.head())

df_flussi = transform_and_aggregate(df)
print("After transform and aggregate:", df_flussi.head())

df_aggrega = consolidate_china_mexico(df_flussi)
print("After consolidating China and Mexico:", df_aggrega.head())

create_files(df_aggrega, file_base_output)
