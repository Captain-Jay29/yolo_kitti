import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Path to your models directory
models_dir = "/home/jay/new_yolo/test_model/"
model_names = [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]

# Directory to save plots
plots_dir = "/home/jay/new_yolo/plots/train5"
os.makedirs(plots_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Initialize dictionaries to store data by image size and batch size
precision_data, recall_data, mAP50_data, mAP50_90_data, loss_data = {}, {}, {}, {}, {}
model_groups = {"640": {"8": [], "16": [], "32": []}, 
                "512": {"8": [], "16": [], "32": []}, 
                "416": {"8": [], "16": [], "32": []}}

# Function to extract image size and batch size from model name
def get_imgsz_and_batchsize_from_model_name(model_name):
    if '640' in model_name:
        imgsz = '640'
    elif '512' in model_name:
        imgsz = '512'
    elif '416' in model_name:
        imgsz = '416'
    else:
        imgsz = None

    if 'batch_8' in model_name:
        batchsize = '8'
    elif 'batch_16' in model_name:
        batchsize = '16'
    elif 'batch_32' in model_name:
        batchsize = '32'
    else:
        batchsize = None
    
    return imgsz, batchsize

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
        
        # Categorize model by image size and batch size
        imgsz, batchsize = get_imgsz_and_batchsize_from_model_name(model_name)
        if imgsz and batchsize:
            model_groups[imgsz][batchsize].append(model_name)

# Set Seaborn color palette with distinct colors for batch sizes
sns.set_palette("Set2")  # Use a distinct palette to differentiate batch sizes
color_cycle = sns.color_palette("Set2", n_colors=3)  # For batch sizes 8, 16, 32
batch_size_colors = {'8': color_cycle[0], '16': color_cycle[1], '32': color_cycle[2]}

# Plotting function for vibrant line-only plots
def plot_metric(data_dict, title, ylabel, filename, rolling_window=3):
    plt.figure(figsize=(12, 8))
    
    # Create subplots grid (2 rows and 3 columns for example)
    rows = 2  # Two rows for metrics (Precision, Recall, mAP, Loss)
    cols = 3  # Three columns for the different metrics
    
    # Loop through image sizes (416, 512, 640)
    for i, imgsz in enumerate(model_groups.keys()):
        # Get all models for this image size
        models_for_imgsz = model_groups[imgsz]
        
        # Create subplots for each metric (precision, recall, etc.) for each image size
        ax = plt.subplot(rows, cols, i + 1)  # Subplot for each image size (2x3 grid)
        
        for batchsize, models_in_group in models_for_imgsz.items():
            # Filter data for each batch size group (8, 16, 32)
            filtered_data = {model_name: data_dict.get(model_name, []) for model_name in models_in_group}
            
            # Plot each model line for the current batch size with the corresponding color
            for model_name, metric_values in filtered_data.items():
                if len(metric_values) > 0:  # Only plot if there are values to plot
                    smoothed_values = pd.Series(metric_values).rolling(window=rolling_window).mean()  # Smooth the data
                    label = f'{batchsize} - {model_name}'
                    ax.plot(smoothed_values, label=label, 
                            linestyle='-', color=batch_size_colors[batchsize], linewidth=2)

        ax.set_title(f"Image Size {imgsz}", fontsize=12)
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8)  # Show the legend
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)  # Adjust the top space for the suptitle
    save_path = os.path.join(plots_dir, filename)
    plt.savefig(save_path)  # Save the plot to the specified directory
    plt.close()  # Close the figure to avoid memory issues

# Generate and save plots for each image size group
for imgsz in model_groups.keys():
    filtered_precision = {name: precision_data.get(name, []) for batch_sizes in model_groups[imgsz].values() for name in batch_sizes}
    filtered_recall = {name: recall_data.get(name, []) for batch_sizes in model_groups[imgsz].values() for name in batch_sizes}
    filtered_mAP50 = {name: mAP50_data.get(name, []) for batch_sizes in model_groups[imgsz].values() for name in batch_sizes}
    filtered_mAP50_90 = {name: mAP50_90_data.get(name, []) for batch_sizes in model_groups[imgsz].values() for name in batch_sizes}
    filtered_loss = {name: loss_data.get(name, []) for batch_sizes in model_groups[imgsz].values() for name in batch_sizes}
    
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