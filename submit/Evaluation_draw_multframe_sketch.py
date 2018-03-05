import numpy as np
import os, cv2
import random, time
from matplotlib import pyplot as plt
import matplotlib as mpl
path = "D:/graduation_project/workspace/dataset/HMDB51/train1/JTM_ori/25/"

frame1 = cv2.imread(path + '1059_24_10.jpg') #dribble
frame2 = cv2.imread(path + '1791_19_16.jpg') #golf
frame3 = cv2.imread(path + '222_5_3.jpg')    #catch
frame4 = cv2.imread(path + '4547_20_38.jpg') #sit up
frame5 = cv2.imread(path + '813_1_8.jpg')    #dive
frame6 = cv2.imread(path + '440_15_5.jpg')   #clap

fig = plt.figure()

def subplot(fig, sub, frame, title, x=320, y=240):
    ax = fig.add_subplot(sub)
    plt.imshow(frame)
    ax.set_xlabel(title)
    ax.set_xlim([1, x])
    ax.set_ylim([1, x])

subplot(fig, 231, frame1, '(a) dribble')
subplot(fig, 232, frame2, '(b) golf')
subplot(fig, 233, frame3, '(c) catch')
subplot(fig, 234, frame4, '(d) situp')
subplot(fig, 235, frame5, '(e) dive')
subplot(fig, 236, frame6, '(f) clap')

plt.show()