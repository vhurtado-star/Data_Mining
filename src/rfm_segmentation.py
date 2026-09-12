from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def calculate_pearson_skewness(data: pd.DataFrame) -> pd.DataFrame:
	"""Calculate Pearson's second skewness coefficient for numeric features."""
	numeric_data = data.select_dtypes(include="number")
	results = pd.DataFrame({
		"feature": numeric_data.columns,
		"mean": numeric_data.mean().to_numpy(),
		"median": numeric_data.median().to_numpy(),
		"std_dev": numeric_data.std(ddof=1).to_numpy(),
	})
	results["pearson_skewness"] = (
		3 * (results["mean"] - results["median"]) / results["std_dev"]
	)
	return results


def main() -> None:
	project_root = Path(__file__).resolve().parents[1]
	input_path = project_root / "data" / "raw" / "segmentation_team2_saas.csv"
	data = pd.read_csv(input_path)
	rfm_columns = [
		"recency_login_days",
		"frequency_api_calls",
		"monthly_spend_mxn",
		"active_user_seats",
	]
	rfm_data = data[rfm_columns]

	scaler = StandardScaler()
	scaled_rfm = pd.DataFrame(
		scaler.fit_transform(rfm_data),
		columns=rfm_columns,
		index=rfm_data.index,
	)

	# StandardScaler creates negative values, so shift each feature before np.log.
	positive_scaled_rfm = scaled_rfm - scaled_rfm.min() + 1e-9
	log_rfm = np.log(positive_scaled_rfm)
	log_rfm = pd.DataFrame(
		log_rfm,
		columns=rfm_columns,
		index=rfm_data.index,
	)

	print("RFM processing: StandardScaler -> positive shift -> np.log")
	print("Variance after StandardScaler:")
	print(scaled_rfm.var(ddof=0).to_string())
	print("\nVariance after logarithmic transformation:")
	print(log_rfm.var(ddof=0).to_string())
	print("\nResulting transformed RFM DataFrame:")
	print(log_rfm.to_string())

	results = calculate_pearson_skewness(log_rfm)

	print("Pearson skewness coefficient: 3 * (mean - median) / sample standard deviation")
	print(results.to_string(index=False, formatters={
		"mean": "{:,.4f}".format,
		"median": "{:,.4f}".format,
		"std_dev": "{:,.4f}".format,
		"pearson_skewness": "{:,.6f}".format,
	}))


if __name__ == "__main__":
	main()
