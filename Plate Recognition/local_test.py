import torch
import numpy as np
import cv2
from time import time
import sys
from yolov5 import YOLOv5
import warnings
from torchsummary import summary

warnings.filterwarnings("ignore")

model_path = "best.pt"
device = "cpu"  # or "cpu"

# init yolov5 model
model = YOLOv5(model_path, device)

frame = cv2.imread("car.jpg")


# perform inference
results = model.predict(frame)
print("This is ouput",type(str(results)))
print([{"predictions": {results.pred[0]}}])
predictions = results.pred[0]
boxes = (predictions[:, :4]) # x1, y1, x2, y2
scores = predictions[:, 4]
categories = predictions[:, 5]
print(boxes.numpy().tolist())
print(scores)
print(categories)
#%% SAVING MODEL
print(model)
model.eval()
example_input = torch.rand(-1, 1, 200, 50)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("best_script.pt")

