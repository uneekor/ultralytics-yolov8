from ultralytics import YOLO

# 1. Set path of model you want to export.
model_path = "/home/ymhong/creatz/ultralytics/runs/pose/train22/weights/best.pt"
model = YOLO(model_path)
results = model.export(format='onnx')