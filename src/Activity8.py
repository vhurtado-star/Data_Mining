from pathlib import Path

import numpy as np
import pandas as pd


project_root = Path(__file__).resolve().parents[1]
filename = "association_team2_saas.csv"
input_path = project_root / "data" / "raw" / filename

df = pd.read_csv(input_path)

print(f"Assigned filename: {filename}")
print(f"Raw rows ingested: {len(df)}")
print(f"Unique transactions (N): {df['account_id'].nunique()}")
print(f"Unique items: {df['feature_module'].nunique()}")

transaction_matrix = pd.crosstab(df['account_id'], df['feature_module']) > 0
print(f"Binary transactional matrix dimensions: {transaction_matrix.shape}")

basket = transaction_matrix.astype(int)
cooc_matrix = basket.T.dot(basket)
cooc_values = cooc_matrix.to_numpy(copy=True)
np.fill_diagonal(cooc_values, 0)
cooc_matrix.iloc[:, :] = cooc_values

number_of_transactions = basket.shape[0]
joint_support = cooc_matrix / number_of_transactions

upper_triangle = np.triu(np.ones(joint_support.shape, dtype=bool), k=1)
candidate_pairs = (
	joint_support.where(upper_triangle)
		.rename_axis(index="item_a", columns="item_b")
		.stack()
		.reset_index(name="joint_support")
)
candidate_pairs = candidate_pairs[
	candidate_pairs["joint_support"] >= 0.05
].sort_values("joint_support", ascending=False)

print(f"\nCandidate item pairs with joint support >= 0.05: {len(candidate_pairs)}")
print("Top 5 item pairs by joint support:")
print(candidate_pairs.head(5).to_string(index=False))

item_support = transaction_matrix.mean()
sorted_item_support = item_support.sort_values()

rules = []
for pair in candidate_pairs.itertuples(index=False):
	for antecedent, consequent in (
		(pair.item_a, pair.item_b),
		(pair.item_b, pair.item_a),
	):
		support = pair.joint_support
		confidence = support / item_support[antecedent]
		lift = support / (item_support[antecedent] * item_support[consequent])
		rules.append({
			"Antecedent": antecedent,
			"Consequent": consequent,
			"Support": support,
			"Confidence": confidence,
			"Lift": lift,
		})

high_interest_rules = pd.DataFrame(rules)
high_interest_rules = high_interest_rules[
	(high_interest_rules["Lift"] > 1.2)
	& (high_interest_rules["Confidence"] >= 0.40)
].sort_values("Lift", ascending=False)

print("\nTop 6 high-interest rules (Lift > 1.2 and Confidence >= 0.40):")
print(high_interest_rules.head(6).round(3).to_string(index=False))

print("\n3 items with the lowest support:")
print(sorted_item_support.head(3).to_string())

print("\n3 items with the highest support:")
print(sorted_item_support.tail(3).sort_values(ascending=False).to_string())
