import numpy as np
import cv2
import random
# img = cv2.imread('D:/graduation_project/workspace/dataset/HMDB51/train1/JTM_mc/10/0_0_1.jpg')
# print(img.shape)
# print(cv2.resize(img, (224, 224)).shape)
#

tmp = np.zeros((3, 2)).tolist()
for i in range(2):
    tmp2 = random.sample(tmp, 1)
    tmp.append(tmp2)
    print(tmp)
# tmp2.append(tmp[0])
# print(np.array(tmp2).shape)