import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

BASE_PATH = "../data/raw/data_sprint_2025"

FILES = {
    "dengue": "dengue.csv",
    "climate": "climate.csv",
    "environ": "environ_vars.csv",
    "forecast": "forecasting_climate.csv",
    "ocean": "ocean_climate_oscillations.csv",
    "pop": "datasus_population_2001_2024.csv",
    "health": "map_regional_health.csv",
    "episcanner": "dados_episcanner.csv",
    "topology": "static_features_tft.csv"
}


def load_raw_data(base_path, file_map):
    print(f"Loading file: {base_path}")
    loaded_data = {}

    for key, filename in file_map.items():
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            print(f"Opening {filename}...")
            loaded_data[key] = pd.read_csv(path)
        else:
            print(f"File not foun: {filename}")
            loaded_data[key] = None

    return loaded_data


# ==========================================
# 3. FUNÇÃO DE GERAÇÃO (Lógica de Merge)
# ==========================================
def generate_inference_dataset(raw_data):
    print("Generating inference dataset..")

    df = raw_data["dengue"].copy()
    df_climate = raw_data["climate"]
    df_environ = raw_data["environ"]
    df_forecast_climate = raw_data["forecast"]
    df_ocean = raw_data["ocean"]
    df_pop = raw_data["pop"]
    df_reg_health = raw_data["health"]
    df_episcanner = raw_data["episcanner"]
    df_topo = raw_data["topology"]

    df['date'] = pd.to_datetime(df['date'])
    df_climate['date'] = pd.to_datetime(df_climate['date'])
    df_ocean['date'] = pd.to_datetime(df_ocean['date'])

    df = df.sort_values(['geocode', 'date']).reset_index(drop=True)
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    min_date = df['date'].min()
    df['time_idx'] = ((df['date'] - min_date).dt.days / 7).astype(int)

    if 'epiweek' in df.columns:
        df['week_of_year'] = df['epiweek'].astype(str).str[-2:].astype(int)
    else:
        df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    df['week_cycle'] = df['week_of_year'].apply(lambda x: x - 40 if x >= 41 else x + 12)
    df['sin_week_cycle'] = np.sin(2 * np.pi * df['week_cycle'] / 52)
    df['cos_week_cycle'] = np.cos(2 * np.pi * df['week_cycle'] / 52)

    print("Adding inputs")

    cols_clima = [c for c in df_climate.columns if c not in ['epiweek']]
    df = pd.merge(df, df_climate[cols_clima], on=['geocode', 'date'], how='left')

    clim_cols = [c for c in df_climate.columns if c not in ['geocode', 'date', 'epiweek']]
    df[clim_cols] = df.groupby('geocode')[clim_cols].ffill()

    df = pd.merge(df, df_ocean, on='date', how='left')
    df[['enso', 'iod', 'pdo']] = df[['enso', 'iod', 'pdo']].ffill()

    df = pd.merge(df, df_pop, on=['geocode', 'year'], how='left')
    df['log_pop'] = np.log1p(df['population'])
    df['log_pop'] = df.groupby('geocode')['log_pop'].ffill()

    if 'uf_code' in df.columns and 'uf_code' in df_environ.columns:
        df = df.drop(columns=['uf_code'])
    df = pd.merge(df, df_environ, on='geocode', how='left')

    cols_reg = ['geocode', 'macroregion_name', 'regional_name']
    cols_reg_exist = [c for c in cols_reg if c in df_reg_health.columns]
    df = pd.merge(df, df_reg_health[cols_reg_exist], on='geocode', how='left')

    # 4. PREVISÃO CLIMÁTICA
    if df_forecast_climate is not None:
        print("Climate forecast")
        df['forecast_temp_med'] = df['temp_med']
        df['forecast_precip_tot'] = df['precip_med'] * 7  #

    # 5. TARGETS (Sem Drop)
    print("Targets stage")
    target_cols = ['geocode', 'year', 'R0', 'peak_week', 'total_cases', 'alpha', 'beta']
    df_epi = df_episcanner[target_cols].copy()
    df_epi['log_total_cases'] = np.log1p(df_epi['total_cases'])

    df = pd.merge(df, df_epi, on=['geocode', 'year'], how='left')

    cols_target_final = ["R0", "peak_week", "log_total_cases", "alpha", "beta"]
    for col in cols_target_final:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    # 6. TDA / TOPOLOGIA
    if 'tda_entropy_H1' not in df.columns:
        print("TDA not found!")
        df['tda_entropy_H1'] = 0.0
        df['tda_amplitude_H1'] = 0.0

    if df_topo is not None:
        print("Topology stage")
        df_topo['geocode'] = df_topo['geocode'].astype(str)
        df['geocode'] = df['geocode'].astype(str)

        if 'num_neighbors' not in df.columns:
            df = df.merge(df_topo[['geocode', 'num_neighbors']], on='geocode', how='left')
            df['num_neighbors'] = df['num_neighbors'].fillna(0)

    df['casos'] = df['casos'].fillna(0)
    df['incidence'] = (df['casos'] / df['population']) * 100000
    df['incidence'] = df['incidence'].fillna(0)

    df['geocode'] = df['geocode'].astype(str)
    df['time_idx'] = df['time_idx'].astype(int)

    for col in ["uf", "koppen", "biome", "macroregion_name"]:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)

    print(f"Inference dataset: {len(df)}")
    return df


raw_data = load_raw_data(BASE_PATH, FILES)

if raw_data["dengue"] is not None:
    inference_df = generate_inference_dataset(raw_data)
    output_file = "../data/processed/dataset_inference.parquet"
    inference_df.to_parquet(output_file, index=False)
    print(f"💾 Arquivo salvo com sucesso: {output_file}")
else:
    print("❌ Erro: Arquivo 'dengue.csv' não encontrado. Verifique o caminho.")