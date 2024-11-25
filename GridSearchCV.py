from sklearn.model_selection import ParameterGrid
from pathlib import Path
from ultralytics import YOLO
import torch
import os

# Ensure the test_model directory exists
output_dir = Path('test_model')
output_dir.mkdir(exist_ok=True)

class YOLOEstimator:
    def __init__(self, img_size, batch_size, lr):
        self.img_size = img_size
        self.batch_size = batch_size
        self.lr = lr
        self.model = YOLO('yolov8n.yaml')  # Initialize the YOLO model

    def fit(self):
        # Run YOLO training with the specified hyperparameters
        self.model.train(
            data='/home/jay/new_yolo/kitti.yaml',
            imgsz=self.img_size,
            batch=self.batch_size,
            lr0=self.lr,
            epochs=55,  # Adjust if you want more epochs in each test
            project='test_model',  # Custom directory for saving results
            name=f"imgsz_{self.img_size}_batch_{self.batch_size}_lr_{self.lr}"
        )
        
        # Save the model to a specific folder with a descriptive name
        model_name = f"yolo_imgsz_{self.img_size}_batch_{self.batch_size}_lr_{self.lr}.pt"
        model_path = output_dir / model_name
        torch.save(self.model.model.state_dict(), model_path)  # Save model weights to the specified path

# Define the parameter grid
param_grid = {
    'img_size': [416, 512, 640],         # 3 options for image size
    'batch_size': [8, 16, 32],           # 3 options for batch size
    'lr': [0.001, 0.01]                  # 2 options for learning rate
}

# Iterate over all parameter combinations in param_grid
ptr = 0
path = '/home/jay/new_yolo/test_model'
count = sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) and name.startswith("imgsz"))

print(f'total sub-folders: {count}')

print(f"Number of folders starting with 'imgsz': {count}")
for params in ParameterGrid(param_grid):

    print(f"imgsz_{params['img_size']}_batch_{params['batch_size']}_lr_{params['lr']}")
    if ptr <= count-1:
        ptr += 1
        continue
    # break

    # Instantiate the YOLOEstimator with current parameters
    estimator = YOLOEstimator(
        img_size=params['img_size'],
        batch_size=params['batch_size'],
        lr=params['lr']
    )
    
    # Train and save the model
    estimator.fit()
    ptr += 1

print("Training completed. All models are saved in the 'test_model' folder.")