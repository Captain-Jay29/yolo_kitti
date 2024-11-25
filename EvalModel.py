from ultralytics import YOLO

model = YOLO("/home/jay/new_yolo/runs/detect/train/weights/best.pt")

# train_results = model.train(data='/home/jay/new_yolo/kitti.yaml')

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