import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow.python.client import timeline

from common import estimate_pose, CocoPairsRender, preprocess, CocoColors, get_joints, draw_humans
from networks import get_network
from pose_dataset import CocoPoseLMDB


parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--scope', type=str, default='test1')
parser.add_argument('--input-width', type=int, default=320)
parser.add_argument('--input-height', type=int, default=240)
parser.add_argument('--stage-level', type=int, default=6)
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--model', type=str, default='cmu',
                    help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.88
config.gpu_options.allow_growth = True

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}
joints_keys = 18

root = "D:/graduation_project/workspace/dataset/" + args.dataset + '_train_test_splits/'
video_path = root + args.scope + '/'
detail_path = root + args.scope + '_spatial_detail/'
save_path_root  = root + args.scope + '_joints/'
x_path = save_path_root + 'x/'
y_path = save_path_root + 'y/'
file_finally_num = Dict[args.dataset][args.scope]


def run(image):
    pafMat, heatMat = sess.run(
        [
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
        ], feed_dict={'image:0': [image]}
    )
    heatMat, pafMat = heatMat[0], pafMat[0]

    humans = estimate_pose(heatMat, pafMat)

    joints = get_joints(image, humans)

    return joints

def process(lst):
    file_list = os.listdir(detail_path)
    test_time = time.time()

    startCount = 0
    for file in file_list:
        startCount += 1
        if startCount < args.start or file == '_frame_num.txt' or file not in lst:
            continue

        file_to_read_fid = open(detail_path + file, 'r')
        fid = file_to_read_fid.read()
        fid = [int(i) for i in fid.split(' ')]
        file_to_read_fid.close()

        file_to_write_x = open(x_path + file, 'w')
        file_to_write_y = open(y_path + file, 'w')

        video_to_capture = file.split('.')[0] + '.avi'
        vc = cv2.VideoCapture(video_path + video_to_capture)  # 读入视频文件
        if not vc.isOpened():
            print('Open failure! exit')
            exit(0)

        rval = True
        count = 0
        index = 0
        x_matrix = np.zeros((len(fid), joints_keys))
        y_matrix = np.zeros((len(fid), joints_keys))
        while rval and index < len(fid):  # 循环读取视频帧
            rval, frame = vc.read()
            if count == fid[index]:
                image = preprocess(frame, args.input_width, args.input_height)
                joints = run(image)

                for key in sorted(joints.keys()):
                    value = joints[key]
                    x_matrix[index][key] = value[0]
                    y_matrix[index][key] = value[1]
                index += 1
            count += 1
            cv2.waitKey(1)
        vc.release()

        file_to_write_x.write(','.join(str(i) for i in x_matrix.reshape(len(fid) * joints_keys).tolist()))
        file_to_write_y.write(','.join(str(i) for i in y_matrix.reshape(len(fid) * joints_keys).tolist()))

        if (startCount % 100 == 0):
            print(startCount, ": ", + time.time() - test_time)

        file_to_write_x.close()
        file_to_write_y.close()

    print('total time: ', time.time() - test_time)


def check_filesize(path):
    lst = []
    file_list = os.listdir(path)
    for file in file_list:
        filesize = os.path.getsize(path + file)
        if filesize == 0:
            lst.append(file)
    return lst


def fix():
    lst = check_filesize(x_path)
    while len(lst) != 0:
        print(lst)
        process(lst)
        lst = check_filesize(x_path)

if not os.path.exists(x_path):
    os.makedirs(x_path)
if not os.path.exists(y_path):
    os.makedirs(y_path)
assert os.path.exists(detail_path)


if __name__ == '__main__':
    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess)
        logging.info("loading model")

        file_num_list = [len(os.listdir(x_path)), len(os.listdir(y_path))]

        if os.path.exists(x_path) and os.path.exists(y_path):
            if min(file_num_list) == file_finally_num:
                fix()
                print(args.dataset, args.scope, "joints_xy has been generated! exit")
                exit(0)
            else:
                if args.start == 0:
                    args.start = max([0, min(file_num_list) - 1])
                    print(args.dataset, args.scope, 'start:', args.start)

        process(os.listdir(detail_path))
        fix()



