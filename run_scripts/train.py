from ultralytics import YOLO

# 1. Set pretrained model.
model = YOLO('yolov8n-pose.pt')  # load a pretrained model

# 2. set model arguments.
model.train(data='club_pose.yaml', epochs=100, batch=64)  # train the model