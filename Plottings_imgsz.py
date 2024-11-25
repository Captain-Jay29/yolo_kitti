'''
import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to your models directory
models_dir = "/home/jay/new_yolo/test_model/"
model_names = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]

# Directory to save plots
plots_dir = "/home/jay/new_yolo/plots/train3_imgsz"
os.makedirs(plots_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize dictionaries to store data by image size
precision_data, recall_data, mAP50_data, mAP50_90_data, loss_data = {}, {}, {}, {}, {}
model_groups = {"640": [], "512": [], "416": []}

# Function to determine image size based on model name
def get_imgsz_from_model_name(model_name):
    if '640' in model_name:
        return '640'
    elif '512' in model_name:
        return '512'
    elif '416' in model_name:
        return '416'
    return None

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
        
        # Categorize model by image size
        imgsz = get_imgsz_from_model_name(model_name)
        if imgsz:
            model_groups[imgsz].append(model_name)

# Plotting function with rolling average for smoothing and subplot options
def plot_metric(data_dict, title, ylabel, filename, rolling_window=3):
    plt.figure(figsize=(12, 8))
    
    # Create subplots grid (2 rows and 3 columns for example)
    num_plots = len(data_dict)
    rows = (num_plots + 2) // 3  # Automatically adjust number of rows based on models

    for i, (model_name, metric_values) in enumerate(data_dict.items()):
        ax = plt.subplot(rows, 3, i + 1)  # Create subplots
        
        # Smooth the data with a rolling average
        smoothed_values = pd.Series(metric_values).rolling(window=rolling_window).mean()
        
        ax.plot(smoothed_values, label=model_name)
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True)
        ax.legend(fontsize=6)
        
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top space for the suptitle
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)  # Save the plot to the specified directory
    plt.close()  # Close the figure to avoid memory issues

# Generate and save plots for each image size group
for imgsz, models_in_group in model_groups.items():
    if models_in_group:  # Only generate a plot if there are models for this imgsz
        filtered_precision = {name: precision_data[name] for name in models_in_group}
        filtered_recall = {name: recall_data[name] for name in models_in_group}
        filtered_mAP50 = {name: mAP50_data[name] for name in models_in_group}
        filtered_mAP50_90 = {name: mAP50_90_data[name] for name in models_in_group}
        filtered_loss = {name: loss_data[name] for name in models_in_group}
        
        # Plot for Precision
        plot_metric(filtered_precision, f"Model Precision Over Epochs (imgsz={imgsz})", "Precision", f"precision_{imgsz}.png")
        # Plot for Recall
        plot_metric(filtered_recall, f"Model Recall Over Epochs (imgsz={imgsz})", "Recall", f"recall_{imgsz}.png")
        # Plot for mAP50
        plot_metric(filtered_mAP50, f"Model mAP@0.5 Over Epochs (imgsz={imgsz})", "mAP@0.5", f"mAP50_{imgsz}.png")
        # Plot for mAP50-90
        plot_metric(filtered_mAP50_90, f"Model mAP@0.5-0.9 Over Epochs (imgsz={imgsz})", "mAP@0.5-0.9", f"mAP50_90_{imgsz}.png")
        # Plot for Loss
        plot_metric(filtered_loss, f"Model Loss Over Epochs (imgsz={imgsz})", "Loss", f"loss_{imgsz}.png")

print(f"Plots have been saved to: {plots_dir}")
'''

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your models directory
models_dir = "/home/jay/new_yolo/test_model/"
model_names = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]

# Directory to save plots
plots_dir = "/home/jay/new_yolo/plots/train4"
os.makedirs(plots_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize dictionaries to store data by image size
precision_data, recall_data, mAP50_data, mAP50_90_data, loss_data = {}, {}, {}, {}, {}
model_groups = {"640": [], "512": [], "416": []}

# Function to determine image size based on model name
def get_imgsz_from_model_name(model_name):
    if '640' in model_name:
        return '640'
    elif '512' in model_name:
        return '512'
    elif '416' in model_name:
        return '416'
    return None

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
        
        # Categorize model by image size
        imgsz = get_imgsz_from_model_name(model_name)
        if imgsz:
            model_groups[imgsz].append(model_name)

# Set Seaborn color palette for vibrant plots
sns.set_palette("Set2")  # You can experiment with different palettes like 'Set2', 'Paired', etc.

# Plotting function with rolling average for smoothing and subplot options
def plot_metric(data_dict, title, ylabel, filename, rolling_window=3):
    plt.figure(figsize=(12, 8))
    
    # Create subplots grid (2 rows and 3 columns for example)
    num_plots = len(data_dict)
    rows = (num_plots + 2) // 3  # Automatically adjust number of rows based on models

    for i, (model_name, metric_values) in enumerate(data_dict.items()):
        ax = plt.subplot(rows, 3, i + 1)  # Create subplots
        
        # Smooth the data with a rolling average
        smoothed_values = pd.Series(metric_values).rolling(window=rolling_window).mean()
        
        # Use different line styles and markers
        ax.plot(smoothed_values, label=model_name, marker='o', linestyle='-', markersize=6, linewidth=2)
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True)
        ax.legend(fontsize=6)
        
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top space for the suptitle
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)  # Save the plot to the specified directory
    plt.close()  # Close the figure to avoid memory issues

# Generate and save plots for each image size group
for imgsz, models_in_group in model_groups.items():
    if models_in_group:  # Only generate a plot if there are models for this imgsz
        filtered_precision = {name: precision_data[name] for name in models_in_group}
        filtered_recall = {name: recall_data[name] for name in models_in_group}
        filtered_mAP50 = {name: mAP50_data[name] for name in models_in_group}
        filtered_mAP50_90 = {name: mAP50_90_data[name] for name in models_in_group}
        filtered_loss = {name: loss_data[name] for name in models_in_group}
        
        # Plot for Precision
        plot_metric(filtered_precision, f"Model Precision Over Epochs (imgsz={imgsz})", "Precision", f"precision_{imgsz}.png")
        # Plot for Recall
        plot_metric(filtered_recall, f"Model Recall Over Epochs (imgsz={imgsz})", "Recall", f"recall_{imgsz}.png")
        # Plot for mAP50
        plot_metric(filtered_mAP50, f"Model mAP@0.5 Over Epochs (imgsz={imgsz})", "mAP@0.5", f"mAP50_{imgsz}.png")
        # Plot for mAP50-90
        plot_metric(filtered_mAP50_90, f"Model mAP@0.5-0.9 Over Epochs (imgsz={imgsz})", "mAP@0.5-0.9", f"mAP50_90_{imgsz}.png")
        # Plot for Loss
        plot_metric(filtered_loss, f"Model Loss Over Epochs (imgsz={imgsz})", "Loss", f"loss_{imgsz}.png")

print(f"Plots have been saved to: {plots_dir}")
