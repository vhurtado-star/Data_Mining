import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_and_merge_data():
	script_directory = os.path.dirname(os.path.abspath(__file__))
	raw_data_directory = os.path.abspath(
		os.path.join(script_directory, "..", "data", "raw")
	)

	telemetry = pd.read_csv(
		os.path.join(raw_data_directory, "solar_telemetry_cleaned.csv")
	)
	metadata = pd.read_csv(os.path.join(raw_data_directory, "panel_metadata.csv"))
	weather = pd.read_csv(
		os.path.join(raw_data_directory, "weather_station_log.csv")
	)

	telemetry["timestamp"] = pd.to_datetime(telemetry["timestamp"])
	weather["timestamp"] = pd.to_datetime(weather["timestamp"])

	print("Shapes before merges:")
	print("Telemetry:", telemetry.shape)
	print("Metadata:", metadata.shape)
	print("Weather:", weather.shape)

	merged_data = telemetry.merge(metadata, on="panel_id", how="left")
	print("After telemetry-metadata merge:", merged_data.shape)

	merged_data = merged_data.merge(weather, on="timestamp", how="left")
	print("After telemetry-weather merge:", merged_data.shape)

	return merged_data


def encode_manufacturer(merged_data):
	manufacturer_dummies = pd.get_dummies(
		merged_data["manufacturer"],
		prefix="manufacturer",
		drop_first=True,
		dtype=int,
	)
	encoded_data = pd.concat(
		[merged_data.drop(columns=["manufacturer"]), manufacturer_dummies],
		axis=1,
	)

	print("Updated dataframe columns:")
	print(encoded_data.columns.tolist())

	return encoded_data


def scale_features(dataframe):
	continuous_columns = ["voltage_v", "ambient_temp_c"]
	standard_scaler = StandardScaler()
	dataframe[[f"{column}_scaled" for column in continuous_columns]] = (
		standard_scaler.fit_transform(dataframe[continuous_columns])
	)

	efficiency_scaler = MinMaxScaler(feature_range=(1e-6, 1 - 1e-6))
	dataframe["efficiency_pct_scaled"] = efficiency_scaler.fit_transform(
		dataframe[["efficiency_pct"]]
	)

	print("StandardScaler means:", standard_scaler.mean_)
	print("StandardScaler standard deviations:", standard_scaler.scale_)
	print("MinMaxScaler minimums:", efficiency_scaler.data_min_)
	print("MinMaxScaler maximums:", efficiency_scaler.data_max_)
	print("Scaled values for rows 0 and 500:")
	print(
		dataframe.loc[
			[0, 500],
			[
				"voltage_v_scaled",
				"ambient_temp_c_scaled",
				"efficiency_pct_scaled",
			],
		]
	)

	return dataframe


def engineer_features():
	engineered_data = load_and_merge_data()
	assert engineered_data.shape == (720, 14)
	engineered_data = encode_manufacturer(engineered_data)
	engineered_data = scale_features(engineered_data)

	script_directory = os.path.dirname(os.path.abspath(__file__))
	processed_directory = os.path.abspath(
		os.path.join(script_directory, "..", "data", "processed")
	)
	os.makedirs(processed_directory, exist_ok=True)
	output_path = os.path.join(
		processed_directory,
		"solar_features_engineered.csv",
	)
	engineered_data.to_csv(output_path, index=False)
	print("Engineered features saved to:", output_path)

	return engineered_data


if __name__ == "__main__":
	engineer_features()
