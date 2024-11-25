import os
import pandas as pd

# Path to the test_model directory
test_model_dir = "/home/jay/new_yolo/test_model/"  # Update the path if needed

# Initialize a list to store summary data
summary_data = []

# Iterate through all subdirectories in test_model directory
for model_dir in os.listdir(test_model_dir):
    model_path = os.path.join(test_model_dir, model_dir)
    results_csv_path = os.path.join(model_path, "results.csv")

    # Check if the directory and results.csv exist
    if os.path.isdir(model_path) and os.path.exists(results_csv_path):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(results_csv_path)

        # Clean column headers by removing blank spaces
        df.columns = df.columns.str.replace(" ", "")

        # Extract the required metrics
        max_map50 = df["metrics/mAP50(B)"].max()
        max_map50_95 = df["metrics/mAP50-95(B)"].max()
        max_train_loss = df["train/box_loss"].iloc[-1]
        max_val_loss = df["val/box_loss"].iloc[-1]

        # Append the data to the summary list
        summary_data.append({
            "model": model_dir,
            "mAP50": max_map50,
            "mAP50-95": max_map50_95,
            "Train_loss": max_train_loss,
            "Val_loss": max_val_loss
        })

# Create a summary DataFrame
summary_df = pd.DataFrame(summary_data)

# Save the summary as a CSV file
summary_csv_path = os.path.join(test_model_dir, "summary.csv")
summary_df.to_csv(summary_csv_path, index=False)

print(f"Summary CSV generated: {summary_csv_path}")
