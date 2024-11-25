import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to your models directory
models_dir = "/home/jay/new_yolo/test_model/"
model_names = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]

# Directory to save plots
plots_dir = "/home/jay/new_yolo/plots/train"
os.makedirs(plots_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize dictionaries to store data
precision_data, recall_data, mAP50_data, mAP50_90_data, loss_data = {}, {}, {}, {}, {}

# Loop through each model directory and collect metrics data
for model_name in model_names:
    model_path = os.path.join(models_dir, model_name, "results.csv")
    if os.path.exists(model_path):
        # Load metrics data
        df = pd.read_csv(model_path)
        
        # Strip spaces from column names
        df.columns = df.columns.str.strip()
        
        # Extract metrics across epochs
        precision_data[model_name] = df["metrics/precision(B)"]
        recall_data[model_name] = df["metrics/recall(B)"]
        mAP50_data[model_name] = df["metrics/mAP50(B)"]
        mAP50_90_data[model_name] = df["metrics/mAP50-95(B)"]
        
        # Combine the loss columns for overall loss
        loss_data[model_name] = df["train/box_loss"] + df["train/cls_loss"] + df["train/dfl_loss"]

# Plotting function
def plot_metric(data_dict, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for model_name, metric_values in data_dict.items():
        plt.plot(metric_values, label=model_name)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)  # Save the plot to the specified directory
    plt.close()  # Close the figure to avoid memory issues

# Generate and save plots
plot_metric(precision_data, "Model Precision Over Epochs", "Precision", "precision.png")
plot_metric(recall_data, "Model Recall Over Epochs", "Recall", "recall.png")
plot_metric(mAP50_data, "Model mAP@0.5 Over Epochs", "mAP@0.5", "mAP50.png")
plot_metric(mAP50_90_data, "Model mAP@0.5-0.9 Over Epochs", "mAP@0.5-0.9", "mAP50_90.png")
plot_metric(loss_data, "Model Loss Over Epochs", "Loss", "loss.png")

print(f"Plots have been saved to: {plots_dir}")
