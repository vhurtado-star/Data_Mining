import os

import pandas as pd


script_directory = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(
	script_directory,
	"..",
	"data",
	"raw",
	"solar_telemetry_corrupted.csv",
)

dataset = pd.read_csv(dataset_path)

print("Shape:", dataset.shape)
print(dataset.head(3))

continuous_columns = [
	"voltage_v",
	"current_a",
	"power_w",
	"temperature_c",
	"efficiency_pct",
]
median_values = dataset[continuous_columns].median()
dataset[continuous_columns] = dataset[continuous_columns].fillna(median_values)

print("Median values used for imputation:")
print(median_values)
remaining_nulls = dataset[continuous_columns].isnull().sum()
print("Remaining nulls by column:")
print(remaining_nulls)
print("Any nulls remaining:", remaining_nulls.any())

categorical_columns = ["panel_id", "timestamp"]
dataset[categorical_columns] = dataset[categorical_columns].fillna("Unknown")

categorical_nulls = dataset[categorical_columns].isnull().sum()
print("Remaining nulls in categorical columns:")
print(categorical_nulls)
print("Any categorical nulls remaining:", categorical_nulls.any())

efficiency_out_of_bounds = dataset.index[
	(dataset["efficiency_pct"] < 0.0) | (dataset["efficiency_pct"] > 100.0)
]
dataset["efficiency_pct"] = dataset["efficiency_pct"].clip(lower=0.0, upper=100.0)

temperature_95th_percentile = dataset["temperature_c"].quantile(0.95)
temperature_above_limit = dataset["temperature_c"] > 45.0
dataset.loc[temperature_above_limit, "temperature_c"] = temperature_95th_percentile

print("Indices of rows capped for efficiency:", efficiency_out_of_bounds.tolist())
print("95th percentile temperature:", temperature_95th_percentile)
print("Rows capped for temperature:", temperature_above_limit.sum())

processed_directory = os.path.join(script_directory, "..", "data", "processed")
if not os.path.exists(processed_directory):
	os.makedirs(processed_directory)

cleaned_dataset_path = os.path.join(
	processed_directory,
	"solar_telemetry_cleaned.csv",
)
dataset.to_csv(cleaned_dataset_path, index=False)

print("Final cleaned dataset shape:", dataset.shape)
print("Final missing value counts:")
print(dataset.isnull().sum())
