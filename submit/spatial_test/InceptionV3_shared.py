from keras.applications.inception_v3 import InceptionV3
from Model_and_funcs import preprocess_file_list, My_InceptionV3
import argparse
import numpy as np
import os, time, cv2, random
import keras
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model
from scipy.stats import mode


parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--echo_begin', type=int, default=0)
parser.add_argument('--echo_end', type=int, default=1)
parser.add_argument('--spe', type=int, default=4)
parser.add_argument('--frame', type=int, default=10)
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--first', type=int, default=1)
parser.add_argument('--learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.0)

args = parser.parse_args()

root = 'D:/graduation_project/workspace/dataset/HMDB51/'
train = 'train'+str(args.split)+'/'
path_spatial = root + train + 'spatial_10/frame/'

num_classes = 51

save_model_root = 'D:/graduation_project/SPATIAL_training/InceptionV3/'
if not os.path.exists(save_model_root):
    os.makedirs(save_model_root)

def load_data(ori_file_list, batch_size, index, path):
    begin_time = time.time()
    begin = index * batch_size
    end = min([(index + 1) * batch_size, len(ori_file_list)])
    file_list = ori_file_list[begin:end]
    print(ori_file_list)
    x = []
    y = []
    for file in file_list:
        tmpy = int(file.split('.')[0].split('_')[2]) - 1
        tmpy = keras.utils.to_categorical(tmpy, num_classes)
        x.append(cv2.resize(cv2.imread(path + file), (224, 224)))
        # print(np.array(x_ori).shape, np.array(x_mc).shape)
        y.append(tmpy)
    print('\033[1;33;44m', 'Load data', index, 'done:', time.time()-begin_time, '\033[0m')
    return np.array(x), np.array(y)


def generate_batch_traindata_random(x_train, y_train, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    tlen = len(x_train)
    total = tlen // batch_size
    read_count = 0
    while (True):
        read_count += 1
        index = read_count % total
        begin, end = index * batch_size, min([(index + 1) * batch_size, tlen])
        yield x_train[begin:end], y_train[begin:end]


def get_model(first=True, model_path=''):
    print(first, model_path)

    return model

if __name__ == '__main__':
    print(args)
    load_model_path = save_model_root + args.model_name
    if args.first == 1:
        model = My_InceptionV3()
        for layer in model.layers:
            layer.trainable = True
    else:
        model = load_model(load_model_path)

    optimizer = SGD(lr=args.learning_rate, momentum=args.momentum)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')
    spe = args.spe
    epochs_per_round = 1000 #700
    round = 3
    batch_size = 7
    ori_file_list = preprocess_file_list(os.listdir(path_spatial))
    random.shuffle(ori_file_list)


    for e in range(args.echo_begin, args.echo_end):
        for i in range(round):
            x_train, y_train = load_data(ori_file_list, epochs_per_round * batch_size, i, path_spatial)
            print('Shape:', x_train.shape, y_train.shape)
            begin_time = time.time()
            history = model.fit_generator(generate_batch_traindata_random(x_train, y_train, batch_size),
                samples_per_epoch=spe, epochs=epochs_per_round,
                # validation_data=generate_batch_testdata_random(batch_size),
                # validation_steps=1,
                verbose=1)
            model_name = 'e' + str(e) + '_spe' + str(spe) + '_round' + str(i) + '.h5'
            model.save(save_model_root + model_name)
            print('\033[1;33;44m', 'echo:', e, 'round:', i, 'time:', time.time() - begin_time, '\033[0m')
            x_train = []
            y_train = []
