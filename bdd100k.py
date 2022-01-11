import os
import torch
import random

random.seed(108)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.config.list_physical_devices("GPU"))

print("GPU Available: "+str(torch.cuda.is_available()))
os.system('sh bdd100k.sh')
