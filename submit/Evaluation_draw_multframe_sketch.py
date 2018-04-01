import numpy as np
import os, cv2
import random, time
from matplotlib import pyplot as plt
import matplotlib as mpl
path = "D:/graduation_project/workspace/dataset/HMDB51/train2/JDM_ori/10/"

fig = plt.figure()


def subplot(fig, sub, frame, title, x=320, y=240):
    ax = fig.add_subplot(sub)
    plt.imshow(frame)
    ax.set_xlabel("(e)")
    ax.set_title(title)
    # ax.set_xlim([1, x])
    # ax.set_ylim([1, y])

# frame1 = cv2.imread(path + '611_1_6.jpg') #
# frame2 = cv2.imread(path + '1902_0_17.jpg') #
# frame3 = cv2.imread(path + '2409_1_21.jpg')    #catch
# frame4 = cv2.imread(path + '3084_9_26.jpg') #sit up
# frame5 = cv2.imread(path + '3454_0_30.jpg')    #dive
# frame6 = cv2.imread(path + '5471_0_45.jpg')   #clap
#
# subplot(fig, 231, frame1, '(a) climb stairs', x=224, y = 224)
# subplot(fig, 232, frame2, '(b) handstand', x=224, y = 224)
# subplot(fig, 233, frame3, '(c) kick ball', x=224, y = 224)
# subplot(fig, 234, frame4, '(d) pour', x=224, y = 224)
# subplot(fig, 235, frame5, '(e) push', x=224, y = 224)
# subplot(fig, 236, frame6, '(f) sword exercise', x=224, y = 224)

# root = 'D:/graduation_project/gp/paper_img/'
# frame1 = plt.imread(root + 'ori4.png') #
# frame2 = plt.imread(root + 'mc4.png') #
#
#
# subplot(fig, 121, frame1, '(a) situp -- original')
# subplot(fig, 122, frame2, '(b) situp -- interpolation')
#
# plt.show()


root = 'D:/graduation_project/gp/paper_img/'
frame1 = plt.imread("D:/graduation_project/gp/paper_img/211_0.png") #

subplot(fig, 111, frame1, 'Processed Image')

plt.show()