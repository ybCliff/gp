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

source_folder = "D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/test1/"
save_joints_path = "D:/graduation_project/workspace/dataset/HMDB51_train_test_splits/test1_joints/ori/"

Dict = {'HMDB51':{'train1':3570, 'train2':3570, 'train3':3570, 'test1':3196, 'test2':3196, 'test3':3196},
        'UCF101':{'train1':9537, 'train2':9586, 'train3':9624, 'test1':3783, 'test2':3734, 'test3':3696}}
file_finally_num = Dict[args.dataset][args.scope]

if not os.path.exists(save_joints_path):
    os.makedirs(save_joints_path)
else:
    tlen = len(os.listdir(save_joints_path))
    if args.start == 0:
        args.start = max([0, tlen-1])

def display(image, humans, heatMat, pafMat):
    print("hahha")
    image_h, image_w = image.shape[:2]
    image = draw_humans(image, humans)

    scale = 480.0 / image_h
    newh, neww = 480, int(scale * image_w + 0.5)

    process_img = CocoPoseLMDB.display_image(image, heatMat, pafMat, as_numpy=True)
    image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

    convas = np.zeros([480, 640 + neww, 3], dtype=np.uint8)
    convas[:, :640] = process_img
    convas[:, 640:] = image

    cv2.imshow('result', convas)
    cv2.waitKey(0)

def run(image, show=False, trg_len=0):
    pafMat, heatMat = sess.run(
        [
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
            net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
        ], feed_dict={'image:0': [image]}
    )
    heatMat, pafMat = heatMat[0], pafMat[0]

    humans = estimate_pose(heatMat, pafMat)

    joints = get_joints(image, humans)
    if show and len(joints) == trg_len:
        display(image, humans, heatMat, pafMat)
    return joints

if __name__ == '__main__':
    while True:
        current_time = time.localtime(time.time())
        # print(current_time)
        if (current_time.tm_hour >= 2) and (current_time.tm_min >= 45):
            break
        time.sleep(10)

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
                    joints = run(image, show=False, trg_len=15)
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


