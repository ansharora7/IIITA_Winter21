import os
import torch
from IPython.display import Image
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import random

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices("GPU"))

print("GPU Available: "+str(torch.cuda.is_available()))
os.system('sh bdd100k_test.sh')

detections_dir = "runs/detect/yolo_road_det/"
detection_images = [os.path.join(detections_dir, x) for x in os.listdir(detections_dir)]

random_detection_image = Image.open(random.choice(detection_images))
plt.imshow(np.array(random_detection_image))