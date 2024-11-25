import torch
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

# Specify the model directory (change model_name as needed)
model_name = "imgsz_416_batch_8_lr_0.01"
model_dir = Path(f"/home/jay/new_yolo/test_model/{model_name}")
weights_path = model_dir / "weights/best.pt"
metrics_dir = model_dir / "metrics"
metrics_dir.mkdir(exist_ok=True)

# Load the model
model = YOLO(weights_path)

# Run evaluation on the validation dataset and retrieve results
results = model.val()

# Gather additional metrics from the results object
# These metrics are likely available in `results.box`
additional_metrics = {
    "Average Precision (AP) @ 0.5": results.box.map50,           # mAP at IoU=0.5
    "Average Precision (AP) @ 0.5:0.95": results.box.map,        # mAP at IoU=0.5:0.95
    "Precision (Mean)": results.box.mp,                          # Mean Precision across all classes
    "Recall (Mean)": results.box.mr,                             # Mean Recall across all classes
}

# Save the additional metrics as a CSV file
metrics_file = metrics_dir / "additional_metrics.csv"
pd.DataFrame([additional_metrics]).to_csv(metrics_file, index=False)

print(f"Additional metrics calculated and saved in {metrics_file}")
