import onnxruntime
import torch
import numpy as np
import cv2
import os
import sys
from pathlib import Path

from ultralytics.utils import DEFAULT_CFG, LOGGER, ops, colorstr
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.checks import check_imgsz

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# set arguments
weights = "/home/ymhong/creatz/ultralytics/runs/pose/train22/weights/best.onnx"   # model path
test_path = "/home/ymhong/data/xolite_0620_pose_labeled_yolo/driver/data_20240620_0851_4910/b0_0005.jpg" # test image path
# test_path = "/home/ymhong/data/xolite_0620_pose_labeled_yolo/iron/data_20240620_0859_0884/b0_0005.jpg"
conf = 0.25
iou = 0.7
save_txt = True
save_img = True

# set up directories
save_dir = increment_path(Path(ROOT / 'runs/pose') / 'predict', exist_ok=False)   # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Set device
device=torch.device("cuda")
cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]

# Create an inference session using the ONNX model and specify execution providers
session = onnxruntime.InferenceSession(weights, providers=providers)

output_names = [x.name for x in session.get_outputs()]
metadata = session.get_modelmeta().custom_metadata_map

imgsz = check_imgsz(metadata['imgsz'], stride=metadata['stride'], min_dim=2)  # check image size

# read test image
init_im = cv2.imread(test_path)

# 1. Preprocess
not_tensor = not isinstance(init_im, torch.Tensor)
letterbox = LetterBox(imgsz, stride=metadata['stride'])
test_im = letterbox(image=init_im)
test_im = test_im[..., ::-1].transpose((2, 0, 1))
test_im = np.ascontiguousarray(test_im)  # contiguous
test_im = torch.from_numpy(test_im)

test_im = test_im.to(device)
test_im = test_im.half() if False else test_im.float()  # uint8 to fp16/32
if not_tensor:
    test_im /= 255  # 0 - 255 to 0.0 - 1.0
if len(test_im.shape) == 3:
    test_im = test_im[None]  # expand for batch dim

# 2. Inference
if torch.is_tensor(test_im):
    test_im = test_im.cpu().numpy()
elif isinstance(test_im, list):
    test_im = [im.cpu().numpy() if torch.is_tensor(im) else im for im in test_im]
else:
    test_im = np.array(test_im)

model_inputs = session.get_inputs()
outputs = session.run(output_names, {model_inputs[0].name: test_im}) # TODO : output names?
preds = torch.from_numpy(outputs[0]).to(device)    # assuming single image # TODO : check 

# 3. Postprocess
preds = ops.non_max_suppression(
            preds,
            conf,
            iou,
            agnostic=False,
            max_det=300,
            classes=None,
            nc=3, # model class num
        )

# this list is not used
results = []

# convert metadata['names'] from string to dict type
names = metadata['names'].replace("'", '"')
names = eval(names)

# convert metadata['names'] string to tuple type
if isinstance(metadata['kpt_shape'], str):
    kpt_shape_str = metadata['kpt_shape'].strip('()[]{}')
    kpt_shape = tuple(map(int, kpt_shape_str.split(',')))
else:
    kpt_shape = metadata['kpt_shape']

for i, pred in enumerate(preds):
    pred[:, :4] = ops.scale_boxes(test_im.shape[2:], pred[:, :4], init_im.shape).round()
    pred_kpts = pred[:, 6:].view(len(pred), *kpt_shape) if len(pred) else pred[:, 6:]
    pred_kpts = ops.scale_coords(test_im.shape[2:], pred_kpts, init_im.shape)
    img_path = test_path
    results.append(
        Results(init_im, path=img_path, names=names, boxes=pred[:, :6], keypoints=pred_kpts)
    )

# write result
path = Path(test_path)
save_path = save_dir / (path.parts[-2] + path.name)     # image.jpg
txt_path = str(save_dir / 'labels' / save_path.stem)
annotator = Annotator(init_im, line_width=1, example=str(names))

# initialize file
if save_txt:
    if os.path.isfile(f'{txt_path}.txt'):
        with open(f'{txt_path}.txt', 'w') as f:
            f.write('')

# write results
for x1, y1, x2, y2, conf, cls, *key_xyxy_conf in reversed(preds[0]):
    if save_txt:    # write to file
        # convert (x1, y1, x2, y2) to (x_cen, y_cen, width, height) and normalize it
        x_cen_n = round((x1.item() + x2.item()) / 2.0 / init_im.shape[1], 6)
        y_cen_n = round((y1.item() + y2.item()) / 2.0 / init_im.shape[0], 6)
        width_n = round((x2.item() - x1.item()) / init_im.shape[1], 6)
        height_n = round((y2.item() - y1.item()) / init_im.shape[0], 6)

        # normalize key points coordinates to 0~1
        key_xyxy_conf = [tensor.item() for tensor in key_xyxy_conf] # tensor to float
        key_xyxy_conf_line = []
        for i in range(0, len(key_xyxy_conf), 3):
            key_x = round(key_xyxy_conf[i] / init_im.shape[1], 6)
            key_y = round(key_xyxy_conf[i + 1] / init_im.shape[0], 6)
            key_conf = round(key_xyxy_conf[i + 2], 6)
            key_xyxy_conf_line.extend([key_x, key_y, key_conf])

        # write output to YOLO txt format
        line = (int(cls.item()), x_cen_n, y_cen_n, width_n, height_n, key_xyxy_conf_line)
        with open(f'{txt_path}.txt', 'a') as f:
            last_element = line[-1]
            if isinstance(last_element, list):
                line = ' '.join(map(str, line[:-1])) + ' ' + ' '.join(map(str, last_element)) + '\n'
            else:
                line = ' '.join(map(str, line)) + '\n'
            f.write(line)

    if save_img:    # add bbox to image
        c = int(cls.item())    # integer class
        label = f'{names[c]} {conf.item():.2f}'
        annotator.box_label((x1, y1, x2, y2), label, color=colors(c, True))
        for k in reversed(pred_kpts):
            annotator.kpts(k, init_im.shape, radius=2, kpt_line=1)

if save_img:
    cv2.imwrite(save_path, init_im)

if save_txt or save_img:
    s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")