from ultralytics import YOLO

# 1. Set path of model you want to inference.
model_path = "/home/ymhong/creatz/ultralytics/runs/pose/train22/weights/best.pt"
model = YOLO(model_path)

# 2. Set test data path and arguments.
test_data_path = "/home/ymhong/data/xolite_0618_labeled_yolo/test/*/*.jpg"
results = model(test_data_path, save=True, line_width=1, save_txt=True)