import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse
import os
from enum import Enum
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow.python.client import timeline

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}

root = "D:/graduation_project/workspace/dataset/" + args.dataset + '_train_test_splits/'

train = 'train'+str(args.split)
test = 'test'+str(args.split)

train_joints_root  = root + train + '_joints/'
test_joints_root = root + test + '_joints/'

train_x_path = train_joints_root + 'x/'
train_y_path = train_joints_root + 'y/'
test_x_path = test_joints_root + 'x/'
test_y_path = test_joints_root + 'y/'

train_image_path = root + train + '_spatial/'
test_image_path = root + test + '_spatial/'


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NMS_Threshold = 0.1
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4
Min_Subset_Score = 0.8
Max_Human = 96

def is_zero(x_lst, y_lst, index):
    if x_lst[index] == 0 and y_lst[index] == 0:
        return True
    else:
        return False


def draw_img(img, x_lst, y_lst):
    img_copied = np.copy(img)
    centers = {}
    for i in range(CocoPart.Background.value):
        if x_lst[i] == 0 and y_lst[i] == 0:
            continue
        center = (int(x_lst[i] + 0.5), int(y_lst[i] + 0.5))
        centers[i] = center
        cv2.circle(img_copied, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

    # draw line
    for pair_order, pair in enumerate(CocoPairsRender):
        if is_zero(x_lst, y_lst, pair[0]) or is_zero(x_lst, y_lst, pair[1]):
            continue

        img_copied = cv2.line(img_copied, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return img_copied

joints_keys = 18


def load_data(fn):
    file = open(fn, 'r')
    content = file.read()
    content = [float(i) for i in content.split(',')]
    tlen = len(content)
    content = np.array(content)
    return np.reshape(content,(tlen//joints_keys, joints_keys)), tlen//joints_keys

train_frame_id = 0
test_frame_id = 4
# file_list = os.listdir(train_x_path)
train_fn = '2_1.txt'
test_fn = '2_1.txt'
train_x, len_train = load_data(train_x_path+train_fn)
train_y, _ = load_data(train_y_path+train_fn)
test_x, len_test = load_data(test_x_path+test_fn)
test_y, _ = load_data(test_y_path+test_fn)

train_img_fn = train_fn.split('.')[0].split('_')[0] + '_' + str(train_frame_id) + '_' + train_fn.split('.')[0].split('_')[1]+ '.jpg'
test_img_fn = test_fn.split('.')[0].split('_')[0] + '_' + str(test_frame_id) + '_' + test_fn.split('.')[0].split('_')[1] + '.jpg'

train_img = draw_img(cv2.imread(train_image_path + train_img_fn), train_x[train_frame_id], train_y[train_frame_id])
test_img = draw_img(cv2.imread(test_image_path + test_img_fn), test_x[test_frame_id], test_y[test_frame_id])

image_h, image_w = train_img.shape[:2]
scale = 480.0 / image_w
newh1, neww1 = int(scale * image_h + 0.5), 480
train_img = cv2.resize(train_img, (neww1, newh1), interpolation=cv2.INTER_AREA)

image_h, image_w = test_img.shape[:2]
scale = 480.0 / image_w
newh2, neww2 = int(scale * image_h + 0.5), 480
test_img = cv2.resize(test_img, (neww2, newh2), interpolation=cv2.INTER_AREA)

convas = np.zeros([newh1+newh2, 480, 3], dtype=np.uint8)
convas[:newh1, :] = train_img
convas[newh1:, :] = test_img

cv2.imshow('result', convas)
cv2.waitKey(0)
