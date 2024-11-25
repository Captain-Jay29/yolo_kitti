from ultralytics import YOLO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import shutil
from PIL import Image
import os


# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'

# base_dir = Path('/media/otter/kitti')
# img_path = base_dir / 'training' / 'image_2'
# label_path = Path('labels_with_dont_care')
# with open('classes.json','r') as f:
#     classes = json.load(f)

# ims = sorted(list(img_path.glob('*')))
# labels = sorted(list(label_path.glob('*')))
# pairs = list(zip(ims,labels))

# train, test = train_test_split(pairs, test_size=0.1, shuffle=True)

# train_path = Path('train').resolve()
# train_path.mkdir(exist_ok=True)
# valid_path = Path('valid').resolve()
# valid_path.mkdir(exist_ok=True)

# for t_img, t_lb in tqdm(train):
#     im_path = train_path / t_img.name
#     lb_path = train_path / t_lb.name
#     shutil.copy(t_img, im_path)
#     shutil.copy(t_lb, lb_path)

# for v_img, v_lb in tqdm(test):
#     im_path = valid_path / v_img.name
#     lb_path = valid_path / v_lb.name
#     shutil.copy(v_img, im_path)
#     shutil.copy(v_lb, lb_path)

# yaml_file = 'names:\n'
# yaml_file += '\n'.join(f'- {c}' for c in classes)
# yaml_file += f'\nnc: {len(classes)}'
# yaml_file += f'\ntrain: {str(train_path)}\nval: {str(valid_path)}'
# with open('kitti.yaml','w') as f:
#     f.write(yaml_file)

model = YOLO('yolov8n.yaml')

train_results = model.train(data='kitti.yaml', epochs=40)

store = []

metrics = model.val()
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps # a list contains map50-95 of each category

store.append(metrics.box.map)
store.append(metrics.box.map50)
store.append(metrics.box.map75)
store.append(metrics.box.maps)

print(store)