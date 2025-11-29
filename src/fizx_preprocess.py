import pandas as pd

df_main = pd.read_parquet("../data/processed/dataset_tft_completo.parquet")

df_topology = pd.read_csv("../data/processed/static_features_tft.csv")

df_main['geocode'] = df_main['geocode'].astype(str)
df_topology['geocode'] = df_topology['geocode'].astype(str)

cols_to_merge = ['geocode', 'num_neighbors']

if 'num_neighbors' in df_main.columns:
    df_main = df_main.drop(columns=['num_neighbors'])

df_merged = df_main.merge(df_topology[cols_to_merge], on='geocode', how='left')

df_merged['num_neighbors'] = df_merged['num_neighbors'].fillna(0)

df_merged.to_parquet("../data/processed/dataset_tft_completo.parquet", index=False)
print("Saved")