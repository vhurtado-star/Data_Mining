from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


data_path = Path(__file__).resolve().parents[1] / "data" / "raw" / "warehouse_sku_team2.csv"
continuous_features = [
	"daily_picking_frequency",
	"unit_weight_kg",
	"storage_volume_m3",
	"stackability_index",
]

warehouse_data = pd.read_csv(data_path)
numeric_features = warehouse_data[continuous_features]

scaler = StandardScaler()
scaled_features = pd.DataFrame(
	scaler.fit_transform(numeric_features),
	columns=continuous_features,
	index=numeric_features.index,
)

scaled_means = scaled_features.mean()
scaled_variances = scaled_features.var(ddof=0)

print("Media después de StandardScaler:")
print(scaled_means)
print("\nVarianza después de StandardScaler:")
print(scaled_variances)

print("Métricas de modelos K-Means:")
print(f"{'K':>3} {'Inercia (WCSS)':>18} {'Silhouette':>12}")
print("-" * 37)
k_values = list(range(2, 11))
inertias = []
silhouette_scores = []

for k in range(2, 11):
	kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
	labels = kmeans.fit_predict(scaled_features)
	silhouette = silhouette_score(scaled_features, labels)
	inertias.append(kmeans.inertia_)
	silhouette_scores.append(silhouette)

	print(f"{k:>3} {kmeans.inertia_:>18.4f} {silhouette:>12.4f}")

output_directory = Path(__file__).resolve().parents[1] / "reports" / "figures"
output_directory.mkdir(parents=True, exist_ok=True)
output_path = output_directory / "warehouse_diagnostics.png"

figure, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(k_values, inertias, marker="o", linewidth=2)
axes[0].scatter(3, inertias[k_values.index(3)], color="red", s=90, zorder=3)
axes[0].annotate(
	"Selected K=3",
	xy=(3, inertias[k_values.index(3)]),
	xytext=(4, inertias[k_values.index(3)]),
	arrowprops={"arrowstyle": "->", "color": "red"},
	color="red",
)
axes[0].set_title("Elbow Curve")
axes[0].set_xlabel("Number of clusters (K)")
axes[0].set_ylabel("Inertia (WCSS)")
axes[0].set_xticks(k_values)
axes[0].grid(alpha=0.3)

axes[1].plot(k_values, silhouette_scores, marker="o", linewidth=2)
axes[1].scatter(3, silhouette_scores[k_values.index(3)], color="red", s=90, zorder=3)
axes[1].annotate(
	"Selected K=3",
	xy=(3, silhouette_scores[k_values.index(3)]),
	xytext=(4, silhouette_scores[k_values.index(3)]),
	arrowprops={"arrowstyle": "->", "color": "red"},
	color="red",
)
axes[1].set_title("Silhouette Curve")
axes[1].set_xlabel("Number of clusters (K)")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_xticks(k_values)
axes[1].grid(alpha=0.3)

figure.tight_layout()
figure.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(figure)
print(f"Diagnostic figure saved to: {output_path}")

final_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = final_kmeans.fit_predict(scaled_features)
warehouse_data["cluster"] = pd.Series(
	cluster_labels,
	index=warehouse_data.index,
)

centroids = warehouse_data.groupby("cluster")[continuous_features].mean()
print("\nCentroides del modelo final en unidades físicas:")
print(centroids)
