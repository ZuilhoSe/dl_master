import os
import glob
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
import warnings

warnings.filterwarnings("ignore")

original_load = torch.load
def patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load

# --- Configurações ---
DATA_PATH = "../data/processed/dataset_tft_completo.parquet"
CHECKPOINT_BASE_DIR = "models/checkpoints/"
RESULTS_OUTPUT = "avaliacao_final_modelos.csv"

TARGETS = ["R0", "peak_week", "log_total_cases", "alpha", "beta"]
TARGET_IDX_R0 = 0
TARGET_IDX_CASES = 2

# Mesma lista mestre do treino
ALL_FEATURES = {
    "time_varying_known": ["time_idx", "week_cycle", "sin_week_cycle", "cos_week_cycle", "log_pop", "forecast_temp_med",
                           "forecast_precip_tot"],
    "time_varying_unknown": ["casos", "incidence", "temp_med", "precip_med", "rel_humid_med", "enso", "iod",
                             "tda_entropy_H1", "tda_amplitude_H1"],
    "static_categoricals": ["uf", "koppen", "biome", "macroregion_name"],
    "static_reals": ["num_neighbors"]
}

# (Copie o seu ABLATION_CONFIGS do script anterior para cá)
ABLATION_CONFIGS = {
    "1_Full_Model": [],
    "2_No_TDA": ["tda_entropy_H1", "tda_amplitude_H1"],
    "3_No_Climate": ["temp_med", "precip_med", "rel_humid_med", "enso", "iod", "forecast_temp_med",
                     "forecast_precip_tot"],
    "4_No_Spatial_Identity": ["uf", "biome", "macroregion_name", "koppen"],
    "5_No_Connectivity": ["num_neighbors"],
    "6_No_Static_Context": ["uf", "biome", "macroregion_name", "koppen", "num_neighbors", "beta_center", "alpha_center",
                            "log_total_cases_center"]
}


def filter_features(exclude_list):
    return {k: [x for x in v if x not in exclude_list] for k, v in ALL_FEATURES.items()}


def load_data():
    print("⏳ Carregando dados...")
    data = pd.read_parquet(DATA_PATH)
    data["time_idx"] = data["time_idx"].astype(int)
    data["geocode"] = data["geocode"].astype(str)
    for col in ALL_FEATURES["static_categoricals"]:
        if col in data.columns:
            data[col] = data[col].astype(str)
    return data


def evaluate_experiment(exp_name, exclude_list, data):
    print(f"\n📊 Avaliando: {exp_name}")

    # 1. Encontrar o checkpoint
    ckpt_dir = os.path.join(CHECKPOINT_BASE_DIR, exp_name)
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))

    if not ckpts:
        print(f"⚠️ Nenhum modelo encontrado para {exp_name}. Pulando...")
        return None

    best_ckpt = ckpts[0]  # Pega o primeiro (e único) checkpoint salvo

    # 2. Filtrar dados e preparar Dataloader
    feats = filter_features(exclude_list)
    required_cols = TARGETS + feats["time_varying_unknown"] + feats["time_varying_known"] + feats[
        "static_categoricals"] + feats["static_reals"]
    data_clean = data.dropna(subset=required_cols).copy()

    max_prediction_length = 1
    max_encoder_length = 52
    training_cutoff = data_clean["time_idx"].max() - max_prediction_length

    target_normalizer = MultiNormalizer([
        GroupNormalizer(groups=["geocode"], transformation="softplus"),
        GroupNormalizer(groups=["geocode"], transformation="softplus"),
        GroupNormalizer(groups=["geocode"], transformation=None),
        GroupNormalizer(groups=["geocode"], transformation="logit"),
        GroupNormalizer(groups=["geocode"], transformation="softplus")
    ])

    # Cria apenas o dataset de treinamento para servir de base para o val/test
    training_dataset = TimeSeriesDataSet(
        data_clean[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx", target=TARGETS, group_ids=["geocode"],
        min_encoder_length=20, max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length, max_prediction_length=max_prediction_length,
        static_categoricals=feats["static_categoricals"], static_reals=feats["static_reals"],
        time_varying_known_reals=feats["time_varying_known"], time_varying_unknown_reals=feats["time_varying_unknown"],
        allow_missing_timesteps=True, target_normalizer=target_normalizer,
        add_relative_time_idx=True, add_target_scales=True, add_encoder_length=True,
    )

    # Cria o dataloader de teste (focado apenas no último time_idx disponível)
    test_dataset = TimeSeriesDataSet.from_dataset(training_dataset, data_clean, predict=True, stop_randomization=True)
    test_dataloader = test_dataset.to_dataloader(train=False, batch_size=256, num_workers=0)

    # 3. Carregar modelo e prever
    model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    model.eval()

    # 3.1 Pede as previsões simples (sem flags extras para não bugar a biblioteca)
    raw_predictions = model.predict(test_dataloader, mode="prediction")

    # raw_predictions é uma lista de 5 tensores (um para cada target)
    # Usamos .flatten() no lugar de .squeeze() para evitar bugs com batches de tamanho 1
    pred_r0 = raw_predictions[TARGET_IDX_R0].flatten().numpy()
    pred_cases = raw_predictions[TARGET_IDX_CASES].flatten().numpy()

    # 3.2 Extrair o gabarito (valores reais) DIRETO do Dataloader
    true_r0_list = []
    true_cases_list = []

    # Iteramos no dataloader (exatamente na mesma ordem que o model.predict fez internamente)
    for batch_x, batch_y in test_dataloader:
        # Em alvos múltiplos (MultiTarget), batch_y é uma tupla: (alvos, pesos)
        # batch_y[0] contém uma lista com os 5 alvos na ordem que definimos
        targets = batch_y[0]

        true_r0_batch = targets[TARGET_IDX_R0].flatten().numpy()
        true_cases_batch = targets[TARGET_IDX_CASES].flatten().numpy()

        true_r0_list.append(true_r0_batch)
        true_cases_list.append(true_cases_batch)

    # Juntamos todos os batches em um array único
    true_r0 = np.concatenate(true_r0_list)
    true_cases = np.concatenate(true_cases_list)

    # 3.3 Verificação de segurança (Sanity Check)
    if len(pred_r0) != len(true_r0):
        print(f"⚠️ Erro de tamanho: Previsões ({len(pred_r0)}) vs Reais ({len(true_r0)})")
        return None

    # 3.4 Calcular as métricas finais
    mae_r0 = np.mean(np.abs(pred_r0 - true_r0))
    rmse_r0 = np.sqrt(np.mean(np.square(pred_r0 - true_r0)))

    mae_cases = np.mean(np.abs(pred_cases - true_cases))

    print(f"✅ MAE R0: {mae_r0:.3f} | RMSE R0: {rmse_r0:.3f} | MAE Casos(log): {mae_cases:.3f}")

    return {
        "Model": exp_name.split("_", 1)[1],  # Remove o número inicial do nome
        "MAE_R0": mae_r0,
        "RMSE_R0": rmse_r0,
        "MAE_LogCases": mae_cases
    }


def plot_results(df):
    """Gera um gráfico de barras comparativo elegante para o artigo."""
    # Ordenar pelos piores modelos até o melhor
    df = df.sort_values("MAE_R0", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Gráfico de barras
    ax = sns.barplot(x="MAE_R0", y="Model", data=df, palette="viridis")

    # Destacar o Full Model com uma cor diferente
    for i, patch in enumerate(ax.patches):
        if df.iloc[i]["Model"] == "Full_Model":
            patch.set_facecolor('crimson')

    plt.title("Estudo de Ablação: Erro Absoluto Médio (MAE) na previsão do $R_0$", fontsize=14, pad=15)
    plt.xlabel("MAE (Menor é melhor)", fontsize=12)
    plt.ylabel("Configuração do Modelo", fontsize=12)
    plt.tight_layout()
    plt.savefig("ablation_mae_comparison.png", dpi=300)
    print("\n📉 Gráfico salvo como 'ablation_mae_comparison.png'")


if __name__ == "__main__":
    data = load_data()
    results = []

    for exp_name, exclude_list in ABLATION_CONFIGS.items():
        res = evaluate_experiment(exp_name, exclude_list, data)
        if res:
            results.append(res)

    df_results = pd.DataFrame(results)
    df_results.to_csv(RESULTS_OUTPUT, index=False)
    print(f"\n📁 Resultados salvos em: {RESULTS_OUTPUT}")

    # Gera o gráfico
    plot_results(df_results)
    print("\nResumo Final:")
    print(df_results.to_markdown(index=False))