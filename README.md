# Data Processing Pipeline: Deep Learning for Parametric Dengue Nowcasting

## 1. Context & Data Source
This project utilizes data from the **Infodengue-Mosqlimate Dengue Challenge 2025**. The challenge was a collaborative initiative centered around the question: *"What is the expected number of probable dengue cases in the upcoming season?"*.

The main objective was to support the development of predictive models in Brazil. To facilitate this, the Mosqlimate Platform compiled a comprehensive dataset integrating epidemiological, demographic, and climate data.

* **Source URL:** [https://sprint.mosqlimate.org/data/2025/](https://sprint.mosqlimate.org/data/2025/)
* **Target Definition:** Unlike traditional forecasting models that predict raw cases per week, this project employs a **Parametric Nowcasting** approach. We train a Temporal Fusion Transformer (TFT) to predict the parameters of the **Richards Growth Model** (used by the *Episcanner* tool) for the current epidemic year.

## 2. Methodology & Pipeline

The processing pipeline consolidates heterogeneous data sources into a unified structure suitable for Multi-Target Deep Learning.

### A. Data Integration
We merged the following datasets based on `geocode` (municipality ID) and `date` (weekly):
1.  **Epidemiological:** Weekly dengue cases (primary time series).
2.  **Climate (Observed):** ERA5 reanalysis data (temperature, precipitation, humidity).
3.  **Climate (Forecast):** Climate model predictions (used as *Known Future Inputs*).
4.  **Oceanic Indices:** Global oscillation indices like ENSO, IOD, and PDO (broadcasted to all cities).
5.  **Demographics:** Annual population estimates (interpolated weekly).
6.  **Static Data:** Biomes, climate zones (Köppen), and regional health administrative divisions.

### B. Topological Data Analysis (TDA)
To capture early warning signals of regime shifts (e.g., transition from endemic to epidemic), we applied **Persistent Homology** using the `giotto-tda` library.
* **Input:** Sliding window of normalized incidence (53 weeks).
* **Technique:** Takens Embedding $\rightarrow$ Vietoris-Rips Persistence.
* **Extracted Features:**
    * `tda_entropy_H1`: Complexity of cycles in phase space.
    * `tda_amplitude_H1`: Magnitude of the epidemic loop.
    * `tda_entropy_H0`: Complexity of the epidemic loop.
    * `tda_amplitude_H0`: Magnitude of the endemic loop.

### C. Spatial Topology Construction (Adjacency Matrix)
Beyond temporal topology, we modeled the **spatial connectivity** of the 5,570 Brazilian municipalities to understand contagion flow.
* **Source:** `municipios.gpkg` (GeoPackage containing municipality polygons).
* **Method:** We computed the **Adjacency Matrix** by identifying municipalities that share a physical boundary (using the geometric predicate `touches`).
* **Outputs:**
    * **Edge List:** A list of connected node pairs (source $\leftrightarrow$ target), allowing for future Graph Neural Network (GNN) implementations.
    * **Static Spatial Features:** Calculated **Node Degree** (number of neighbors) for each municipality. This feature (`num_neighbors`) is fed into the TFT as a static real variable to quantify spatial centrality/isolation.

### D. Parametric Target Broadcasting
The target variables are the annual epidemiological descriptors provided by the **Episcanner** dataset:
* $R_0$ (Basic Reproduction Number)
* Peak Week
* Total Cases (Log-transformed)
* Alpha & Beta (Shape parameters)

**Logic:** Since these are annual values, they are broadcasted to every week of the corresponding year. The model learns to estimate these annual "ground truths" based on the weekly data observed up to the current time step (*Nowcasting*).

### E. Data Filtering
We filtered the dataset to include only municipality-years that met the Episcanner criteria for an epidemic (at least 3 weeks with $R_t > 1$ and >50 cumulative cases).
* **Original Rows:** ~4.4 Million
* **Processed Rows:** ~1.3 Million (High-quality epidemic data)

## 3. Output Files
The processed data is stored in `data/processed/` using efficient formats:
* `dataset_tft_completo.parquet`: Main dataset (Parquet + Zstd compression).
* `tft_config.json`: Metadata defining which columns are *Known Inputs*, *Unknown Inputs*, *Static*, and *Targets*.
* `adjacencia_edges.csv`: The spatial adjacency list (Edge List).
* `static_features_tft.csv`: Pre-computed static spatial features (merged into the main parquet).