from keras.applications.inception_v3 import InceptionV3
from Model_and_funcs import preprocess_file_list, Two_input_shared_InceptionV3
import argparse
import numpy as np
import os, time, cv2, random
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model

parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--echo_begin', type=int, default=0)
parser.add_argument('--echo_end', type=int, default=1)
parser.add_argument('--frame', type=int, default=10)
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--first', type=bool, default=True)
args = parser.parse_args()

root = 'D:/graduation_project/workspace/dataset/HMDB51/'
train = 'train'+str(args.split)+'/'
folder_ori = 'JDM_ori/'+str(args.frame)+'/'
folder_mc = 'JDM_mc/'+str(args.frame)+'/'
path_train_ori = root + train + folder_ori
path_train_mc = root + train + folder_mc

num_classes = 51

save_model_root = 'D:/graduation_project/JDM_training/InceptionV3/shared/'
if not os.path.exists(save_model_root):
    os.makedirs(save_model_root)

def load_data(ori_file_list, batch_size, index, path_ori, path_mc):
    begin_time = time.time()
    begin = index * batch_size
    end = min([(index + 1) * batch_size, len(ori_file_list)])
    file_list = ori_file_list[begin:end]
    x_ori = []
    x_mc = []
    y = []
    for file in file_list:
        tmpy = int(file.split('.')[0].split('_')[2]) - 1
        tmpy = keras.utils.to_categorical(tmpy, num_classes)
        x_ori.append(cv2.imread(path_ori + file))
        x_mc.append(cv2.imread(path_mc + file))
        y.append(tmpy)
    print('\033[1;33;44m', 'Load data', index, 'done:', time.time()-begin_time, '\033[0m')
    return np.array(x_ori), np.array(x_mc), np.array(y)


def generate_batch_traindata_random(x_train_ori, x_train_mc, y_train, batch_size):
    """逐步提取batch数据到显存，降低对显存的占用"""
    tlen = len(x_train_ori)
    total = tlen // batch_size
    read_count = 0
    while (True):
        read_count += 1
        index = read_count % total
        begin, end = index * batch_size, min([(index + 1) * batch_size, tlen])
        yield [x_train_ori[begin:end], x_train_mc[begin:end]], y_train[begin:end]


def get_model(first=True, model_path=''):
    if first:
        model = Two_input_shared_InceptionV3()
        for layer in model.layers:
            layer.trainable = True
    else:
        model = load_model(model_path)
    return model

if __name__ == '__main__':
    load_model_path = save_model_root + args.model_name
    model = get_model(args.first, load_model_path)
    # num = 14
    # x = cv2.imread(path_train_mc + '0_0_1.jpg')
    # x1 = []
    # x2 = []
    # for i in range(num):
    #     x1.append(x)
    #     x2.append(x)
    # out = model.predict([np.array(x1), np.array(x2)])
    # print(out.shape)


    # # model = Base_cnn()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    spe = 10
    epochs_per_round = 700
    round = 3
    batch_size = 10
    ori_file_list = preprocess_file_list(os.listdir(path_train_ori))
    random.shuffle(ori_file_list)

    for e in range(args.echo_begin, args.echo_end):
        for i in range(round):
            x_train_ori, x_train_mc, y_train = load_data(ori_file_list, epochs_per_round * batch_size, i, path_train_ori, path_train_mc)
            begin_time = time.time()
            history = model.fit_generator(generate_batch_traindata_random(x_train_ori, x_train_mc, y_train, batch_size),
                samples_per_epoch=spe, epochs=epochs_per_round,
                # validation_data=generate_batch_testdata_random(batch_size),
                # validation_steps=1,
                verbose=1)
            model_name = 'e' + str(e) + '_spe' + str(spe) + '_round' + str(i) + '.h5'
            model.save(save_model_root + model_name)
            print('\033[1;33;44m', 'echo:', e, 'round:', i, 'time:', time.time() - begin_time, '\033[0m')
            x_train_ori = []
            x_train_mc = []
            y_train = []
