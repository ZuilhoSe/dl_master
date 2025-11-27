import pandas as pd
import numpy as np
import networkx as nx
from sklearn.manifold import SpectralEmbedding
import scipy.sparse as sp
import os


def generate_graph_embeddings(edges_path, embedding_dim=16, output_path="../data/processed/graph_embeddings.csv"):
    print("Generating graph embeddings")

    edges = pd.read_csv(edges_path)

    print("Building graph")
    G = nx.from_pandas_edgelist(edges, 'source_geocode', 'target_geocode')
    all_nodes = sorted(list(G.nodes()))

    print(f"Processing {len(all_nodes)} cities")
    adj_matrix = nx.to_scipy_sparse_array(G, nodelist=all_nodes, format='csr')

    adj_matrix.indices = adj_matrix.indices.astype(np.int32)
    adj_matrix.indptr = adj_matrix.indptr.astype(np.int32)

    print("Calculating embeddings")
    se = SpectralEmbedding(
        n_components=embedding_dim,
        affinity='precomputed',
        random_state=42,
        n_jobs=-1
    )
    embeddings = se.fit_transform(adj_matrix)

    cols = [f'graph_emb_{i}' for i in range(embedding_dim)]
    df_emb = pd.DataFrame(embeddings, columns=cols)
    df_emb['geocode'] = all_nodes
    df_emb = df_emb[['geocode'] + cols]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_emb.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    return df_emb, cols