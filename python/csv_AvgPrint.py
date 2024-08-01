import pandas as pd

# Load the CSV file
file_path = 'C:/kkt/2024_07_24_Colony/runs/detect/train/results.csv'
results_df = pd.read_csv(file_path)

# Strip whitespace from column names
results_df.columns = results_df.columns.str.strip()

# Calculate average precision, recall, and mAP over all epochs
average_precision = results_df['metrics/precision(B)'].mean()
average_recall = results_df['metrics/recall(B)'].mean()
average_map50 = results_df['metrics/mAP50(B)'].mean()
average_map50_95 = results_df['metrics/mAP50-95(B)'].mean()

# Print results
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average mAP@50: {average_map50:.4f}")
print(f"Average mAP@50-95: {average_map50_95:.4f}")
