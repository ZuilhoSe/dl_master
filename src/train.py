import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet
import json
import os
import warnings

# Suprime avisos chatos do Pandas/PyTorch
warnings.filterwarnings("ignore")

from src.models import build_tft_model


def load_and_merge_data(data_path, graph_path, static_topo_path, config_path):
    """
    Carrega parquet, faz merge dos embeddings e LIMPA nulos residuais.
    """
    print("📂 Carregando dados...")

    # 1. Carregar Dataset Principal
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset principal não encontrado: {data_path}")
    df = pd.read_parquet(data_path)

    # 2. Merge: Topologia Estática
    if os.path.exists(static_topo_path):
        print("🗺️ Integrando contagem de vizinhos...")
        df_topo = pd.read_csv(static_topo_path)
        if 'num_neighbors' in df_topo.columns:
            df_topo['num_neighbors'] = df_topo['num_neighbors'].fillna(0).astype(int)
            df = df.merge(df_topo[['geocode', 'num_neighbors']], on='geocode', how='left')
            df['num_neighbors'] = df['num_neighbors'].fillna(0)
    else:
        df['num_neighbors'] = 0

    # 3. Merge: Embeddings do Grafo
    graph_cols = []
    if os.path.exists(graph_path):
        print("🕸️ Integrando Embeddings do Grafo...")
        df_graph = pd.read_csv(graph_path)
        graph_cols = [c for c in df_graph.columns if c.startswith('graph_emb')]
        df = df.merge(df_graph, on='geocode', how='left')
        df[graph_cols] = df[graph_cols].fillna(0)

    # 4. Carregar Configuração JSON
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- CORREÇÃO CRÍTICA DE NULOS (SANITIZAÇÃO) ---
    print("🧹 Sanitizando dados (removendo NaNs residuais)...")

    # Lista de todas as colunas numéricas que entram no modelo
    all_reals = (
            config['time_varying_known_reals'] +
            config['time_varying_unknown_reals'] +
            config.get('static_reals', []) +
            graph_cols
    )

    # Garantir unicidade
    all_reals = list(set(all_reals))

    for col in all_reals:
        if col in df.columns:
            # Estratégia 1: Preencher buracos no tempo (copia do vizinho temporal)
            # ffill pega o anterior, bfill pega o próximo (resolve o problema do início da série)
            df[col] = df.groupby('geocode')[col].ffill().bfill()

            # Estratégia 2: Se a cidade INTEIRA for NaN, preenche com a Mediana Global
            # Isso evita que o código quebre se uma cidade não tiver dados de clima
            if df[col].isnull().sum() > 0:
                median_val = df[col].median()
                # Se até a mediana for NaN (coluna vazia), usa 0
                if pd.isna(median_val):
                    median_val = 0

                df[col] = df[col].fillna(median_val)
                # print(f"   -> {col}: Preenchidos {df[col].isnull().sum()} N/As com mediana {median_val:.2f}")

    # Verificação Final
    nuls = df[all_reals].isnull().sum().sum()
    if nuls > 0:
        raise ValueError(f"Ainda existem {nuls} valores nulos no dataset! Verifique o pré-processamento.")

    return df, config, graph_cols


def train_dengue_network():
    # --- CAMINHOS ---
    DATA_PATH = "../data/processed/dataset_tft_completo.parquet"
    GRAPH_PATH = "../data/processed/graph_embeddings.csv"
    STATIC_TOPO_PATH = "../data/processed/static_features_tft.csv"  # <--- ARQUIVO QUE FALTAVA
    CONFIG_PATH = "../data/processed/tft_config.json"
    MODEL_OUT_DIR = "../models/checkpoints"

    # 1. Preparar Dados
    df, config, graph_cols = load_and_merge_data(DATA_PATH, GRAPH_PATH, STATIC_TOPO_PATH, CONFIG_PATH)

    # 2. Atualizar Lista de Variáveis Estáticas
    # O JSON tem a lista original. Adicionamos o grafo e garantimos que num_neighbors está lá.
    static_reals = config.get('static_reals', []) + graph_cols

    # Garantir que num_neighbors está na lista de estáticos se ele existir no DF
    if 'num_neighbors' in df.columns and 'num_neighbors' not in static_reals:
        static_reals.append('num_neighbors')

    # Remover duplicatastraining_dataset
    static_reals = list(set(static_reals))

    # 3. Definição do Corte de Tempo (Treino vs Validação)
    # Reservamos as últimas 52 semanas (1 ano) para validação
    max_time_idx = df['time_idx'].max()
    training_cutoff = max_time_idx - 52

    print("🛠️ Criando TimeSeriesDataSet...")

    # 4. Inicializar Dataset
    training_dataset = TimeSeriesDataSet(
        df[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=config['targets'],
        group_ids=["geocode"],

        # Janelas de Tempo (1 Ano Epidemiológico)
        min_encoder_length=20,
        max_encoder_length=53,
        min_prediction_length=1,
        max_prediction_length=1,

        # Features
        static_categoricals=config['static_categoricals'],
        static_reals=static_reals,  # Inclui grafo e num_neighbors

        time_varying_known_reals=config['time_varying_known_reals'],
        time_varying_unknown_reals=config['time_varying_unknown_reals'],

        # Configurações do TFT
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        # Allow missing timesteps (segurança para dados reais sujos)
        allow_missing_timesteps=True,
        add_nan = True
    )

    # 5. Validação
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, df, predict=True, stop_randomization=True
    )

    # 6. DataLoaders
    # Atenção: num_workers=0 é obrigatório no Windows para evitar erros de multiprocessamento
    batch_size = 64
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=batch_size, num_workers=0
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False, batch_size=batch_size * 2, num_workers=0
    )

    # 7. Construir Modelo
    tft = build_tft_model(
        training_dataset,
        params={
            "hidden_size": 64,
            "dropout": 0.1,
            "lstm_layers": 2,
            "attention_head_size": 4
        }
    )

    # 8. Configurar Treinador
    checkpoint_callback = ModelCheckpoint(
        dirpath=MODEL_OUT_DIR,
        filename="dengue_tft-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        patience=8,
        verbose=True,
        mode="min"
    )

    lr_logger = LearningRateMonitor()

    print("🚀 Iniciando Loop de Treinamento...")
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        gradient_clip_val=0.1,
        callbacks=[early_stop_callback, checkpoint_callback, lr_logger],
        enable_model_summary=True,
    )

    # 9. Fit
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    print(f"✅ Melhor modelo salvo em: {checkpoint_callback.best_model_path}")
    return tft, trainer


if __name__ == "__main__":
    model, trainer = train_dengue_network()