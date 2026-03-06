import warnings
import pandas as pd
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer, MultiNormalizer
from pytorch_forecasting.metrics import MultiLoss, QuantileLoss

# --- Configurações Globais ---
torch.set_float32_matmul_precision('medium')
warnings.filterwarnings("ignore")

DATA_PATH = "../data/processed/dataset_tft_completo.parquet"
RESULTS_FILE = "ablation_results.csv"

# Hiperparâmetros
BATCH_SIZE = 1024
MAX_EPOCHS = 25
LEARNING_RATE = 0.03
GRADIENT_CLIP_VAL = 0.1

TARGETS = ["R0", "peak_week", "log_total_cases", "alpha", "beta"]

ALL_FEATURES = {
    "time_varying_known": [
        "time_idx", "week_cycle", "sin_week_cycle", "cos_week_cycle", "log_pop",
        "forecast_temp_med", "forecast_precip_tot"
    ],
    "time_varying_unknown": [
        "casos", "incidence",
        "temp_med", "precip_med", "rel_humid_med",
        "enso", "iod",
        "tda_entropy_H1", "tda_amplitude_H1"
    ],
    "static_categoricals": ["uf", "koppen", "biome", "macroregion_name"],
    "static_reals": ["num_neighbors"]
}

# --- Configuração dos Experimentos de Ablação ---
ABLATION_CONFIGS = {
    "1_Full_Model": [],

    "2_No_TDA": [
        "tda_entropy_H1", "tda_amplitude_H1"
    ],

    "3_No_Climate": [
        "temp_med", "precip_med", "rel_humid_med", "enso", "iod",
        "forecast_temp_med", "forecast_precip_tot"
    ],

    "4_No_Spatial_Identity": [
        "uf",               # Estado
        "biome",            # Bioma (Amazônia, Cerrado, etc.)
        "macroregion_name", # Região (Norte, Sul...)
        "koppen"            # Clima geral
    ],

    "5_No_Connectivity": [
        "num_neighbors"
    ],

    "6_No_Static_Context": [
        "uf", "biome", "macroregion_name", "koppen", "num_neighbors", # Espaciais
        "beta_center", "alpha_center", "log_total_cases_center"       # Estatísticos
    ]
}


def filter_features(exclude_list):
    """Filtra as listas de features baseada na lista de exclusão."""
    features = {
        "time_varying_known": [x for x in ALL_FEATURES["time_varying_known"] if x not in exclude_list],
        "time_varying_unknown": [x for x in ALL_FEATURES["time_varying_unknown"] if x not in exclude_list],
        "static_categoricals": [x for x in ALL_FEATURES["static_categoricals"] if x not in exclude_list],
        "static_reals": [x for x in ALL_FEATURES["static_reals"] if x not in exclude_list],
    }
    return features


def load_data():
    print("⏳ Carregando dados brutos...")
    data = pd.read_parquet(DATA_PATH)
    data["time_idx"] = data["time_idx"].astype(int)
    data["geocode"] = data["geocode"].astype(str)

    # Converter categóricas para string
    for col in ALL_FEATURES["static_categoricals"]:
        if col in data.columns:
            data[col] = data[col].astype(str)

    return data


def train_ablation(experiment_name, exclude_list, data):
    print(f"\n{'=' * 40}")
    print(f"🚀 Iniciando Experimento: {experiment_name}")
    print(f"❌ Removendo variáveis: {exclude_list}")
    print(f"{'=' * 40}\n")

    # 1. Filtrar Features para este experimento
    feats = filter_features(exclude_list)

    # Verificar colunas necessárias para limpeza
    required_cols = TARGETS + feats["time_varying_unknown"] + feats["time_varying_known"] + \
                    feats["static_categoricals"] + feats["static_reals"]

    # Limpeza específica para este set de features
    data_clean = data.dropna(subset=required_cols).copy()
    print(f"Linhas após limpeza para {experiment_name}: {len(data_clean)}")

    # 2. Configurar Dataset
    max_prediction_length = 1
    max_encoder_length = 52
    training_cutoff = data_clean["time_idx"].max() - max_prediction_length

    # Normalizadores (Mantém a lógica original)
    target_normalizer = MultiNormalizer([
        GroupNormalizer(groups=["geocode"], transformation="softplus"),  # R0
        GroupNormalizer(groups=["geocode"], transformation="softplus"),  # peak
        GroupNormalizer(groups=["geocode"], transformation=None),  # log_cases
        GroupNormalizer(groups=["geocode"], transformation="logit"),  # alpha
        GroupNormalizer(groups=["geocode"], transformation="softplus")  # beta
    ])

    training_dataset = TimeSeriesDataSet(
        data_clean[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=TARGETS,
        group_ids=["geocode"],
        min_encoder_length=20,
        max_encoder_length=max_encoder_length,
        min_prediction_length=max_prediction_length,
        max_prediction_length=max_prediction_length,

        static_categoricals=feats["static_categoricals"],
        static_reals=feats["static_reals"],
        time_varying_known_reals=feats["time_varying_known"],
        time_varying_unknown_reals=feats["time_varying_unknown"],

        allow_missing_timesteps=True,
        target_normalizer=target_normalizer,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(
        training_dataset, data_clean, predict=True, stop_randomization=True
    )

    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True
    )
    val_dataloader = validation.to_dataloader(
        train=False,
        batch_size=BATCH_SIZE * 2,
        num_workers=4,
        persistent_workers=True
    )
    # 3. Configurar Modelo e Trainer
    # Nota: output_size fixo pois os TARGETS não mudam
    losses = [QuantileLoss(), QuantileLoss(), QuantileLoss(), QuantileLoss(), QuantileLoss()]

    # Print if using cuda
    if torch.cuda.is_available():
        print("GPU is available!")

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=LEARNING_RATE,
        hidden_size=32,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=16,
        output_size=[7, 7, 7, 7, 7],
        loss=MultiLoss(losses),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # Logger único para cada experimento
    logger = TensorBoardLogger("models/ablation_logs", name=experiment_name)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"models/checkpoints/{experiment_name}",
        filename="{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss", mode="min", save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=4, verbose=True, mode="min"
    )

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_model_summary=False,
        limit_train_batches=0.75,
        limit_val_batches=0.75,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=logger
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # 4. Coletar Métricas
    best_val_loss = trainer.checkpoint_callback.best_model_score.item()
    best_model_path = trainer.checkpoint_callback.best_model_path

    return {
        "Experiment": experiment_name,
        "Val_Loss": best_val_loss,
        "Best_Model_Path": best_model_path,
        "Features_Removed": str(exclude_list)
    }


if __name__ == "__main__":
    # Carrega dados uma vez
    full_data = load_data()

    results = []

    # Loop pelos experimentos
    for exp_name, exclude_list in ABLATION_CONFIGS.items():
        try:
            res = train_ablation(exp_name, exclude_list, full_data)
            results.append(res)

            # Salvar parcial a cada iteração (segurança)
            pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

        except Exception as e:
            print(f"❌ Erro no experimento {exp_name}: {str(e)}")

    print("\n✅ Ablation Study Concluído!")
    print(pd.DataFrame(results))