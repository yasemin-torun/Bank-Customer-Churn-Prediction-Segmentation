# segmentation/segmentation.py
import os
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# ---------- Core Prep ----------
def prepare_for_clustering(
    df: pd.DataFrame,
    drop_cols: Optional[List[str]] = None
) -> Tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """
    y gibi hedef sütunlarını çıkar, sayısal özellikleri StandardScaler ile ölçekle.
    """
    drop_cols = drop_cols or []
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feat_df.values)
    return X_scaled, feat_df, scaler


# ---------- k selection ----------
def silhouette_scores(X: np.ndarray, k_min: int = 2, k_max: int = 10) -> Dict[int, float]:
    scores = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init="auto")
        lbl = km.fit_predict(X)
        scores[k] = silhouette_score(X, lbl)
    return scores

def pick_k_by_silhouette(scores: Dict[int, float]) -> int:
    return max(scores, key=scores.get)

def plot_silhouette(scores: Dict[int, float]) -> None:
    ks = list(scores.keys())
    vals = list(scores.values())
    plt.figure(figsize=(6,4))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Analysis")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- clustering ----------
def fit_kmeans(X: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, KMeans]:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    return labels, km


# ---------- visualization ----------
def pca_scatter(X: np.ndarray, labels: np.ndarray, title: str = "Customer Segmentation (PCA Visualization)") -> None:
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    sc = plt.scatter(Xp[:,0], Xp[:,1], c=labels, alpha=0.65)
    plt.xlabel(f"PCA 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var.)")
    plt.ylabel(f"PCA 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var.)")
    plt.title(title)
    cbar = plt.colorbar(sc)
    cbar.set_label("Cluster")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# ---------- reporting ----------
def attach_clusters(df: pd.DataFrame, labels: np.ndarray, colname: str = "Cluster") -> pd.DataFrame:
    out = df.copy()
    out[colname] = labels
    return out

def churn_by_cluster(df_with_clusters: pd.DataFrame, y_col: str = "y", cluster_col: str = "Cluster") -> pd.DataFrame:
    tbl = df_with_clusters.groupby(cluster_col)[y_col].agg(["mean", "count", "sum"])
    tbl = tbl.rename(columns={"mean": "churn_rate", "count": "n", "sum": "churners"})
    tbl["churn_rate_%"] = (tbl["churn_rate"] * 100).round(2)
    return tbl[["n", "churners", "churn_rate_%"]].sort_values("churn_rate_%", ascending=False)

def save_outputs(df_with_clusters: pd.DataFrame, churn_tbl: pd.DataFrame, outdir: str = "outputs") -> None:
    os.makedirs(outdir, exist_ok=True)
    df_with_clusters.to_csv(os.path.join(outdir, "segmented.csv"), index=False)
    churn_tbl.to_csv(os.path.join(outdir, "churn_by_cluster.csv"))

def cluster_profile(
    df_with_clusters: pd.DataFrame,
    cluster_col: str = "Cluster",
    categorical_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    - Sayısal sütunlar için kümeye göre ortalamalar
    - Kategorik sütunlar için kümeye göre yüzde dağılımı
    """
    num_cols = df_with_clusters.select_dtypes(include=[np.number]).columns.tolist()
    if cluster_col in num_cols:
        num_cols.remove(cluster_col)

    # Sayısal ortalamalar
    num_means = df_with_clusters.groupby(cluster_col)[num_cols].mean()

    # Kategorik yüzdeler
    cat_summary = pd.DataFrame()
    if categorical_cols:
        for col in categorical_cols:
            if col in df_with_clusters.columns:
                tmp = (
                    df_with_clusters.groupby([cluster_col, col]).size()
                    .groupby(level=0).apply(lambda s: s / s.sum())
                    .rename("ratio")
                    .reset_index()
                    .pivot(index=cluster_col, columns=col, values="ratio")
                    .fillna(0.0)
                )
                tmp.columns = [f"{col}={v}" for v in tmp.columns]
                cat_summary = pd.concat([cat_summary, tmp], axis=1)

    return pd.concat([num_means, cat_summary], axis=1).sort_index()