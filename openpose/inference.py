import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
from tensorflow.python.client import timeline

from common import estimate_pose, CocoPairsRender, read_imgfile, CocoColors, draw_humans
from networks import get_network
from pose_dataset import CocoPoseLMDB

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    test_time = time.time()
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/p2.jpg')
    parser.add_argument('--input-width', type=int, default=368)
    parser.add_argument('--input-height', type=int, default=368)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet / mobilenet_accurate / mobilenet_fast')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(1, args.input_height, args.input_width, 3), name='image')

    with tf.Session(config=config) as sess:
        net, _, last_layer = get_network(args.model, input_node, sess)
        print('-------------------')
        print('38')
        print('-------------------')
        logging.debug('read image+')
        image = read_imgfile(args.imgpath, args.input_width, args.input_height)
        vec = sess.run(net.get_output(name='concat_stage7'), feed_dict={'image:0': [image]})
        print('-------------------')
        print('44')
        print('-------------------')
        a = time.time()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        print('-------------------')
        print('50')
        print('-------------------')
        pafMat, heatMat = sess.run(
            [
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
            ], feed_dict={'image:0': [image]}, options=run_options, run_metadata=run_metadata
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))
        print('-------------------')
        print('57')
        print('-------------------')
        
        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open('timeline.json', 'w') as f:
            f.write(ctf)
        heatMat, pafMat = heatMat[0], pafMat[0]

        logging.debug('inference+')
        print('-------------------')
        print('68')
        print('-------------------')
        
        avg = 0
        for _ in range(10):
            a = time.time()
            sess.run(
                [
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=1)),
                    net.get_output(name=last_layer.format(stage=args.stage_level, aux=2))
                ], feed_dict={'image:0': [image]}
            )
            logging.info('inference- elapsed_time={}'.format(time.time() - a))
            avg += time.time() - a
        logging.info('prediction avg= %f' % (avg / 10))

        '''
        logging.info('pickle data')
        with open('person3.pickle', 'wb') as pickle_file:
            pickle.dump(image, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('heatmat.pickle', 'wb') as pickle_file:
            pickle.dump(heatMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('pafmat.pickle', 'wb') as pickle_file:
            pickle.dump(pafMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        '''

        logging.info('pose+')
        a = time.time()
        humans = estimate_pose(heatMat, pafMat)
        print(humans)
        print(humans[0])
        print(humans[0][0], humans[0][1])
        logging.info('pose- elapsed_time={}'.format(time.time() - a))

        logging.info('image={} heatMap={} pafMat={}'.format(image.shape, heatMat.shape, pafMat.shape))
        process_img = CocoPoseLMDB.display_image(image, heatMat, pafMat, as_numpy=True)

        # display
        image = cv2.imread(args.imgpath)
        image_h, image_w = image.shape[:2]
        image = draw_humans(image, humans)

        scale = 560.0 / image_h
        newh, neww = 560, int(scale * image_w + 0.5)

        image = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)

        convas = np.zeros([560, 640 + neww, 3], dtype=np.uint8)
        convas[:, :640] = process_img
        convas[:, 640:] = image

        cv2.imshow('result', convas)
        cv2.waitKey(0)

        tf.train.write_graph(sess.graph_def, '.', 'graph-tmp.pb', as_text=True)

        print('total time: ', time.time() - test_time)