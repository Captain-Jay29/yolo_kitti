from ultralytics import YOLO
from pathlib import Path
import torch

# Set the model configuration you want to resume training for
model_name = "imgsz_640_batch_16_lr_0.01"  # Change this to the specific model name
output_dir = Path("test_model")
project_path = output_dir / model_name
last_checkpoint = project_path / 'weights' / 'last.pt'

# Verify that the checkpoint exists
if not last_checkpoint.exists():
    raise FileNotFoundError(f"Checkpoint {last_checkpoint} not found. Please check the model name or path.")

# Load the model from the checkpoint
print(f"Resuming training from checkpoint: {last_checkpoint}")
model = YOLO(last_checkpoint)

# Define the number of total and remaining epochs
total_epochs = 55  # Total epochs you initially intended for each model
completed_epochs = 26  # Adjust this based on the last saved epoch; here, it completed 41
remaining_epochs = total_epochs - completed_epochs

# Run YOLO training for the remaining epochs
model.train(
    data="kitti.yaml",
    imgsz=416,  # Match this with the image size used in this specific model
    batch=16,    # Match the batch size used in this specific model
    lr0=0.01,   # Match the learning rate used in this specific model
    epochs=remaining_epochs,
    project="test_model",
    name=model_name,
    resume=True  # Enable resume flag to continue training from checkpoint
)

# Save the final model weights directly
final_model_path = output_dir / f"yolo_{model_name}_resumed.pt"
torch.save(model.model.state_dict(), final_model_path)
print(f"Training completed. Resumed model saved to {final_model_path}")
