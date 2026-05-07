import os
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


RFM_COLUMNS = ["Recency", "Frequency", "Monetary"]


def build_rfm(transactions: pd.DataFrame, reference_date=None) -> pd.DataFrame:
    """Build customer-level Recency, Frequency and Monetary features."""
    if reference_date is None:
        reference_date = transactions["InvoiceDate"].max() + pd.Timedelta(days=1)
    else:
        reference_date = pd.to_datetime(reference_date)

    rfm = (
        transactions.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda value: (reference_date - value.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )
    rfm["AverageOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    return rfm


def fit_kmeans_segments(
    rfm: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42,
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """Cluster customers with KMeans using RFM features."""
    segmented = rfm.copy()

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(segmented[RFM_COLUMNS])

    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    segmented["Cluster"] = model.fit_predict(features_scaled)
    segmented["Segment"] = _name_segments(segmented)

    return segmented, model, scaler


def segment_summary(segmented_rfm: pd.DataFrame) -> pd.DataFrame:
    return (
        segmented_rfm.groupby("Segment")
        .agg(
            Customers=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean"),
            AvgOrderValue=("AverageOrderValue", "mean"),
        )
        .sort_values("AvgMonetary", ascending=False)
        .round(2)
    )


def elbow_scores(
    rfm: pd.DataFrame,
    min_clusters: int = 2,
    max_clusters: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(rfm[RFM_COLUMNS])

    scores = []
    for n_clusters in range(min_clusters, max_clusters + 1):
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        model.fit(features_scaled)
        scores.append({"Clusters": n_clusters, "Inertia": model.inertia_})

    return pd.DataFrame(scores)


def _name_segments(segmented: pd.DataFrame) -> pd.Series:
    cluster_profile = (
        segmented.groupby("Cluster")[RFM_COLUMNS]
        .mean()
        .assign(
            MonetaryRank=lambda df: df["Monetary"].rank(method="first"),
            FrequencyRank=lambda df: df["Frequency"].rank(method="first"),
            RecencyRank=lambda df: df["Recency"].rank(method="first", ascending=False),
        )
    )
    cluster_profile["Score"] = (
        cluster_profile["MonetaryRank"]
        + cluster_profile["FrequencyRank"]
        + cluster_profile["RecencyRank"]
    )

    ordered_clusters = cluster_profile.sort_values("Score", ascending=False).index
    default_names = ["High Value", "Loyal", "Occasional", "At Risk"]
    labels = {
        cluster: default_names[index] if index < len(default_names) else f"Segment {index + 1}"
        for index, cluster in enumerate(ordered_clusters)
    }

    return segmented["Cluster"].map(labels)
