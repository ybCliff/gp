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

from openpose.common import estimate_pose, preprocess, get_best_joints
from openpose.networks import get_network

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.88
config.gpu_options.allow_growth = True
# source_folder = "D:/graduation_project/workspace/dataset/UCF10pa2-wedru 6 dri23y 4pr fqu 1_train_test_splits/train1/"
# save_joints_path = "D:/graduation_project/workspace/dataset/UCF101_train_test_splits/train1_joints/"

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

source_folder = "D:/graduation_project/workspace/dataset/HMDB51/video/"
save_joints_path = "D:/graduation_project/workspace/dataset/HMDB51/ori/"

if not os.path.exists(save_joints_path):
    os.makedirs(save_joints_path)

def run(image):
    pafMat, heatMat = sess.run(
        [
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
        ], feed_dict={'image:0': [image]}
    )
    heatMat, pafMat = heatMat[0], pafMat[0]

    humans = estimate_pose(heatMat, pafMat)

    joints = get_best_joints(image, humans)

    return joints

#omit[i] = type - 1
omit = [1, 3, 8, 10,
        11, 12, 14, 19,
        24, 27,
        32, 36, 38,
        40, 41, 42, 43, 47, 48, 49, 50]

if __name__ == '__main__':
    # while True:
    #     current_time = time.localtime(time.time())
    #     # print(current_time)
    #     if (current_time.tm_hour >= 2) and (current_time.tm_min >= 45):
    #         break
    #     time.sleep(10)

    file_list = os.listdir(source_folder)

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess)
        logging.debug('read image+')
        test_time = time.time()

        begin = args.start
        index = -1
        for file in file_list:
            index += 1
            if index < begin:
                continue
            type = int(file.split('.')[0].split('_')[1]) - 1
            if type in omit:
                continue
            round_time = time.time()

            save_file_name = file.split('.')[0] + '.txt'
            file_to_write = open(save_joints_path + save_file_name, 'w')
            vc = cv2.VideoCapture(source_folder + file)  # 读入视频文件
            if not vc.isOpened():
                print('Open failure! exit')
                exit(0)
            total = vc.get(cv2.CAP_PROP_FRAME_COUNT)

            record_count = [0] * 19
            record_sort = []
            body = ""
            rval = True
            while rval:  # 循环读取视频帧
                rval, frame = vc.read()
                joints = []
                if frame is not None:
                    image = preprocess(frame, args.input_width, args.input_height)
                    joints = run(image)
                    # print(len(joints), joints)
                    for key in sorted(joints.keys()):
                        value = joints[key]
                        body += (str(key) + ':')
                        body += (str(value[0]) + ',' + str(value[1]) + ' ')
                    body += '\n'
                record_count[len(joints)] += 1
                record_sort.append(len(joints))
                cv2.waitKey(1)
            vc.release()
            title = ' '.join(str(i) for i in record_count) + '\n'
            title += (' '.join(str(i) for i in record_sort) + '\n')
            file_to_write.write(title)
            file_to_write.write(body)
            file_to_write.close()

            print(index, ": ",  + time.time() - round_time)

            # if len(record_sort) != total:
            #     print('Frame number failure! exit')
            #     print(save_file_name)
            #     exit(0)



        print('total time: ', time.time() - test_time)


