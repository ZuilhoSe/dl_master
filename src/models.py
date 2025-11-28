import torch
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss, MultiLoss


def build_tft_model(training_dataset, params=None):
    """
    Constructs the Temporal Fusion Transformer architecture for Multi-Target forecasting.
    """
    if params is None:
        params = {
            "hidden_size": 64,
            "lstm_layers": 1,
            "dropout": 0.1,
            "attention_head_size": 4
        }

    n_targets = len(training_dataset.target_names)

    loss_fn = MultiLoss([QuantileLoss() for _ in range(n_targets)])

    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,

        learning_rate=0.03,
        hidden_size=params["hidden_size"],
        attention_head_size=params["attention_head_size"],
        dropout=params["dropout"],
        hidden_continuous_size=params["hidden_size"],
        lstm_layers=params["lstm_layers"],

        output_size=7,
        loss=loss_fn,

        reduce_on_plateau_patience=4,
    )

    print(f"    TFT Model Built successfully!")
    print(f"    Targets: {training_dataset.target_names}")
    print(f"    Static Features (including Graph): {len(training_dataset.static_reals)}")

    return tft