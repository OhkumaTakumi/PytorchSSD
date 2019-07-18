import json
import pickle

import cv2
import numpy as np
import os
import os.path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from utils.pycocotools.coco import COCO
from utils.pycocotools.cocoeval import COCOeval

from matplotlib import  pyplot as plt


with open('/home/takumi/research/PytorchSSD/result/100000000_box.binaryfile', 'rb') as f:
    data = pickle.load(f)

print(data[0])

itar_list=[]

for i in range(len(data[0])):
    itar_list.append(100*(i+1))



plt.plot(itar_list,data[0])
plt.plot(itar_list,data[1])
plt.plot(itar_list,data[2])

plt.show()
