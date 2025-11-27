import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)


def load_processed_data():
    """Loads the optimized parquet file and the configuration JSON."""
    print("Loading data")
    df = pd.read_parquet("../data/processed/dataset_tft_completo.parquet")

    with open("../data/processed/tft_config.json", "r") as f:
        config = json.load(f)

    print(f"Data loaded. Shape: {df.shape}")
    print(f"Targets defined: {config['targets']}")
    return df, config


def plot_target_distributions(df, targets):
    """Checks if the targets (R0, Alpha, Beta) follow a reasonable distribution."""
    print("\nPlotting Target Distributions")

    df_unique = df[['geocode', 'year'] + targets].drop_duplicates()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, target in enumerate(targets):
        if i < len(axes):
            sns.histplot(df_unique[target], kde=True, ax=axes[i], color='teal')
            axes[i].set_title(f"Distribution of {target} (Unique City-Years)")
            axes[i].set_xlabel(target)

    plt.tight_layout()
    plt.show()


def analyze_tda_features(df):
    """
    Investigates if Topological features correlate with epidemic intensity.
    Hypothesis: Higher TDA Amplitude (H1) should correlate with higher Total Cases.
    """
    print("\nAnalyzing Topology vs. Epidemic Size")

    df['tda_amplitude_H1'] = np.log1p(df['tda_amplitude_H1'])

    sample_df = df.sample(n=10000, random_state=42)

    fig, ax = plt.subplots(1, 2, figsize=(18, 6))

    sns.scatterplot(
        data=sample_df,
        x='tda_amplitude_H1',
        y='log_total_cases',
        alpha=0.3,
        ax=ax[0],
        hue='macroregion_name'
    )
    ax[0].set_title("TDA Amplitude (H1) vs. Annual Outbreak Size")
    ax[0].set_xlabel("Topological Amplitude (Loop Size)")
    ax[0].set_ylabel("Log Total Cases (Target)")

    sns.scatterplot(
        data=sample_df,
        x='tda_entropy_H1',
        y='R0',
        alpha=0.3,
        ax=ax[1],
        hue='macroregion_name'
    )
    ax[1].set_title("TDA Entropy (H1) vs. R0")

    plt.show()


def visualize_single_city_series(df, geocode=None):
    """Visualizes the time series of inputs and the broadcasted target for one city."""
    print("\nVisualizing Single City Dynamics")

    if not geocode:
        # Pick a random city with data
        geocode = df['geocode'].sample(1).values[0]

    city_data = df[df['geocode'] == geocode].sort_values('time_idx')

    fig, ax1 = plt.subplots(figsize=(15, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time Index (Weeks)')
    ax1.set_ylabel('Weekly Incidence (per 100k)', color=color)
    ax1.plot(city_data['time_idx'], city_data['incidence'], color=color, label='Weekly Incidence (Input)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Annual R0 (Target)', color=color)
    ax2.plot(city_data['time_idx'], city_data['R0'], color=color, linestyle='--', linewidth=2,
             label='Annual R0 (Target)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(f"Dynamics for City {geocode}: Weekly Input vs. Annual Parametric Target")
    fig.tight_layout()
    plt.show()


def plot_random_phase_space(df):
    """
    Randomly selects a city and year from the dataframe and visualizes
    its phase space reconstruction (Time-Delay Embedding).
    """
    sample = df.sample(1)
    geocode = sample['geocode'].values[0]
    year = sample['year'].values[0]

    print(f"Randomly Selected: City {geocode} - Year {year}")

    subset = df[(df['geocode'] == geocode) & (df['year'] == year)].sort_values('time_idx')
    subset['lag1'] = subset['incidence'].shift(1)
    subset['lag2'] = subset['incidence'].shift(2)

    plt.figure(figsize=(10, 8))

    sc = plt.scatter(
        subset['incidence'],
        subset['lag1'],
        c=subset['time_idx'],
        cmap='viridis',
        alpha=0.7,
        edgecolor='k',
        s=60
    )

    plt.title(f"Phase Space Trajectory (2D Embedding)\nCity: {geocode} | Year: {year}")
    plt.xlabel("Incidence $x(t)$")
    plt.ylabel("Lag 1 $x(t-1)$")
    plt.colorbar(sc, label='Time Progression (Weeks)')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.plot(subset['incidence'], subset['lag1'], alpha=0.3, color='gray')

    plt.show()


if __name__ == "__main__":
    df_final, config = load_processed_data()
    plot_target_distributions(df_final, config['targets'])
    analyze_tda_features(df_final)
    visualize_single_city_series(df_final)
    plot_random_phase_space(df_final)