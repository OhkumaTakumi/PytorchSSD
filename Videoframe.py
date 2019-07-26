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



class Videoframes(data.Dataset):
    def __init__(self, video_path, interval=1, transform=None):
        self.transform = transform
        self.video_path = video_path

        cap = cv2.VideoCapture(self.video_path)

        self.video_frames = []

        if cap.isOpened():

            digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

            n = 0

            while True:
                ret, frame = cap.read()
                if ret:
                    n += 1
                    if n % interval == 0:
                        self.video_frames.append(frame)
                else:
                    break


    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, index):
        img = self.video_frames[index]

        return img

    def pull_image(self, index):
        img = self.video_frames[index]

        return img

#path = "/home/takumi/data/YouTube-BB/videos/1/_1zmnFlrUwc+1+0.mp4"

#samplevideo = Videoframe2(path)



