import json
import pickle

import cv2
import numpy as np
import os
import os.path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

pseudo_label_path0 = "/home/takumi/data/YouTube-BB/pseudo/pseudo_label0.json"
pseudo_label_path1 = "/home/takumi/data/YouTube-BB/pseudo/pseudo_label1.json"

class Pseudoframes(data.Dataset):
    def __init__(self, class_list=None, preproc=None, target_transform=None):
        self.preproc = preproc
        self.target_transform = target_transform
        with open(pseudo_label_path0, 'r') as f:
            self.pseudo_label = json.load(f)
        with open(pseudo_label_path1, 'r') as f:
            self.pseudo_label.update(json.load(f))

        self.img_path = []
        self.annotations = []
        for image in self.pseudo_label:
            frame = self.pseudo_label[image]
            bbox = []
            for anno in frame:
                tensor = anno["bbox"] + [anno["class_id"]]
                bbox.append(tensor)
                path = anno["img_path"]
            if len(bbox) > 0:
                self.img_path.append(path)
                self.annotations.append(bbox)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img_id = self.img_path[index]
        target = self.annotations[index]
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        target = np.array(target)
        #print(target, img.shape,  index, img_id)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        img2 = img
        img2 = img2.numpy()
        #print(img[0][0][0])

        #print(3, type(target), target)

            # target = self.target_transform(target, width, height)
        # print(target.shape)

        return img, target




