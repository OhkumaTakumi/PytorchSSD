import pickle
import os
from Videoframe import Videoframes
import matplotlib.pyplot as plt

path1 = "/home/takumi/data/YouTube-BB"
path_result = "/detection_result/1/_1zmnFlrUwc+1+0"
class_list = ('__background__', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'bird', 'cat', 'dog', 'horse', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'umbrella', 'skateboard', 'knife', 'potted plant', 'toilet')
True_list = (0,9,1,8,5,15,13,10,17,21,12,3,20,4,19,6,7,16,22,11,14,18,0,2)

def hyouzi(class_num,path1, path_result):

    with open(path1 + path_result + ".pkl", 'rb') as f:
        data = pickle.load(f)

        #print(type(data))
        #print(len(data))
        #print(len(data[0]))
        #print(data[1][1])
        max_list=[]
        max_index=[-1]*len(data)
        for i in range(len(data)):
            #print(class_list[i])
            num=[]
            max_value = 0

            for j in range(len(data[0])):
                count = 0
                for box in data[i][j]:
                    if max_value < box[4]:
                        max_value = box[4]
                        max_index[i] = j
                    if box[4]>0.5:
                        count+=1
                num.append(count)
            max_list.append(max_value)
            #print(num)
        #print(max_list)

    a = sorted(range(len(max_list)), key = lambda k: max_list[k])

    print(class_list[a[-1]], max_list[a[-1]], class_list[True_list[class_num]], max_list[True_list[class_num]])
    if a[-1] != True_list[class_num]:
        img = testset[max_index[a[-1]]]
        plt.imshow(img)
        plt.show()



for class_num in range(1, 24):
    for i in range(50):
        if class_num != 23:
            continue
        path1 = "/home/takumi/data/YouTube-BB"
        path_classfolder = "/home/takumi/data/YouTube-BB/videos/{0}".format(class_num)

        video_list = os.listdir(path_classfolder)
        if len(video_list) > i:
            video = video_list[i]

            path_video = "/videos/{0}/".format(class_num) + video
            path_result = "/detection_result/{0}/".format(class_num) + video[:-4]
            testset = Videoframes(path1 + path_video)
            hyouzi(class_num,path1, path_result)


