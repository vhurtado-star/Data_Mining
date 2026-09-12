from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


data_path = Path(__file__).resolve().parent / "../data/processed/solar_features_engineered.csv"
data = pd.read_csv(data_path)

metadata = data[["timestamp", "panel_id"]]

features = data[
	[
		"voltage_v",
		"current_a",
		"power_w",
		"temperature_c",
		"efficiency_pct",
		"solar_radiation_w_m2",
		"ambient_temp_c",
		"wind_speed_m_s",
	]
]
X = features

print("PCA pipeline: StandardScaler -> PCA (no log transformation)")

raw_variance = X.var()

print("Metadata data types:")
print(metadata.dtypes)

print("Features data types:")
print(features.dtypes)

print("Raw variance of each feature:")
print(raw_variance)
print(
	"Solar radiation variance exceeds 100,000:",
	raw_variance["solar_radiation_w_m2"] > 100_000,
)

pipeline = Pipeline([
	("scaler", StandardScaler()),
	("pca", PCA()),
])
components_matrix = pipeline.fit_transform(X)
X_standardized = pd.DataFrame(
	pipeline.named_steps["scaler"].transform(X),
	columns=X.columns,
	index=X.index,
)
caracteristicas_scaled = X_standardized

print("Standardized feature means:")
print(X_standardized.mean())
print("Standardized feature variances:")
print(X_standardized.var(ddof=0))

pca = pipeline.named_steps["pca"]

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

print("PCA explained variance:")
print("Component | Individual variance | Cumulative variance")
for component_number, (individual, cumulative) in enumerate(
	zip(explained_variance_ratio, cumulative_explained_variance),
	start=1,
):
	print(
		f"PC{component_number:<9} | "
		f"{individual * 100:>19.2f}% | "
		f"{cumulative * 100:>19.2f}%"
	)

print("PCA fitted successfully.")
print("Explained variance ratio:")
print(explained_variance_ratio)

component_names = [
	f"PC{component_number}"
	for component_number in range(1, pca.components_.shape[0] + 1)
]
component_weights = pd.DataFrame(
	pca.components_.T,
	index=features.columns,
	columns=component_names,
)
print("PCA component weights:")
print(component_weights)

component_labels = [
	f"PC{component_number}"
	for component_number in range(1, len(explained_variance_ratio) + 1)
]
component_positions = range(1, len(explained_variance_ratio) + 1)

figure, variance_axis = plt.subplots(figsize=(10, 6))
variance_axis.bar(
	component_positions,
	explained_variance_ratio,
	color="steelblue",
	label="Individual variance",
)
variance_axis.set_xlabel("Principal component")
variance_axis.set_ylabel("Individual explained variance ratio")
variance_axis.set_xticks(list(component_positions), component_labels)
variance_axis.set_ylim(0, 1)

cumulative_axis = variance_axis.twinx()
cumulative_axis.step(
	list(component_positions),
	cumulative_explained_variance,
	where="mid",
	color="darkorange",
	marker="o",
	label="Cumulative variance",
)
cumulative_axis.axhline(
	0.95,
	color="crimson",
	linestyle=":",
	label="95% variance",
)
cumulative_axis.set_ylabel("Cumulative explained variance ratio")
cumulative_axis.set_ylim(0, 1.05)

handles, labels = variance_axis.get_legend_handles_labels()
cumulative_handles, cumulative_labels = cumulative_axis.get_legend_handles_labels()
variance_axis.legend(
	handles + cumulative_handles,
	labels + cumulative_labels,
	loc="center right",
)
figure.suptitle("PCA Scree Plot")
figure.tight_layout()

plot_path = Path(__file__).resolve().parent / "../reports/figures/scree_plot.png"
plot_path.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(plot_path, dpi=300, bbox_inches="tight")
plt.close(figure)
print(f"Scree plot saved to: {plot_path}")

pca_three_components = PCA(n_components=3)
components_matrix = pca_three_components.fit_transform(caracteristicas_scaled)
components = pd.DataFrame(
	components_matrix,
	columns=["PC1", "PC2", "PC3"],
	index=caracteristicas_scaled.index,
)

compressed_data = pd.concat(
	[metadata.reset_index(drop=True), components.reset_index(drop=True)],
	axis=1,
)

print("Compressed PCA DataFrame shape:", compressed_data.shape)
print("First three rows of compressed PCA DataFrame:")
print(compressed_data.head(3))

compressed_path = Path(__file__).resolve().parent / "../data/processed/solar_compressed_pca.csv"
compressed_data.to_csv(compressed_path, index=False)
print(f"Compressed PCA data saved to: {compressed_path}")
