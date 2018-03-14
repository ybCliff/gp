import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Conv1D, LSTM, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop
from keras import regularizers
import argparse
import numpy as np
import os, time, cv2
from scipy.stats import mode
import random

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--layer', type=str, default='block5_pool')
parser.add_argument('--fusion', type=str, default='mean')   # mean  or  max
parser.add_argument('--folder', type=str, default='JTM_mc/15')   # JTM_mc/x, JTM_ori/x, JDM_mc/x, JDM_ori/x, spatial_x/frame
parser.add_argument('--frame', type=int, default=15)
parser.add_argument('--split', type=int, default=1)
args = parser.parse_args()

num_classes = 51
batch_size = 256
epochs = 70
alternate = False

root = "D:/graduation_project/workspace/dataset/"+args.dataset+'/'
folder_name = args.model + '_' + args.layer + '_' + args.fusion
train = "train" + str(args.split)
test = "test" + str(args.split)
fmap_train_path = root + train + '/' + args.folder + '/' + folder_name + '/'
fmap_test_path = root + test + '/' + args.folder + '/' + folder_name + '/'

def load_data(path, read_loop, loop):

    if not os.path.exists(path):
        exit(0)
    file_list = os.listdir(path)
    print(path, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()
    myset = []
    file_num = []
    for file in file_list:
        count += 1
        if count % 500 == 0:
            print(count, time.time()-beginTime)
        file_to_read = open(path+file, 'r')
        content = file_to_read.read()
        file_to_read.close()

        content = [float(i) for i in content.split(',')]
        tmpy = file.split('.')[0].split('_')[2]
        # file_num.append(file)
        if read_loop:
            myset.append(content)
            if count % loop == 0:
                # print(file_num, tmpy)
                x.append(myset)
                y.append(int(tmpy) - 1)
                myset = []
                # file_num = []
        else:
            x.append(content)
            y.append(int(tmpy) - 1)

    return np.array(x), np.array(y)

def load_spatial_data(path, detail_path, loop):
    if not os.path.exists(path) or not os.path.exists(detail_path):
        print("ERROR OPENING FILE!")
        exit(0)
    file_to_read = open(detail_path, 'r')
    line = file_to_read.readline()
    detail = []
    while line and line != "":
        line = line.replace('\n', '')
        detail.append(int(line.split(' ')[1]))
        line = file_to_read.readline()
    file_to_read.close()
    print('Detail:', detail)

    file_list = os.listdir(path)
    print(path, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()

    for index in range(len(detail)):
        round = detail[index] // loop
        remain = detail[index] % loop
        pre_rec = []

        for i in range(round):
            myset = []
            for j in range(loop):
                file_to_read = open(path + file_list[count], 'r')
                content = file_to_read.read()
                file_to_read.close()

                content = [float(i) for i in content.split(',')]
                tmpy = file_list[count].split('.')[0].split('_')[2]
                myset.append(content)
                if remain != 0:
                    pre_rec.append(content)
                count += 1
            x.append(myset)
            y.append(int(tmpy) - 1)

        if remain != 0:
            pre = random.sample(pre_rec, loop-remain) if len(pre_rec) != 0 else []
            for j in range(remain):
                file_to_read = open(path + file_list[count], 'r')
                content = file_to_read.read()
                file_to_read.close()

                content = [float(i) for i in content.split(',')]
                tmpy = file_list[count].split('.')[0].split('_')[2]
                pre.append(content)
                count += 1
            if len(pre_rec) == 0:
                print(np.array(pre).shape)
                left = loop - remain
                for k in range(left):
                    tmp2 = random.sample(pre, 1)
                    if k == 0:
                        tmp = tmp2
                    else:
                        tmp = np.concatenate((tmp, tmp2), axis=0)
                pre = np.concatenate((pre, tmp), axis=0)

            x.append(np.array(pre).reshape((loop, 512)).tolist())
            y.append(int(tmpy) - 1)

        if index % 50 == 0:
            print(index, time.time() - beginTime)
    print('Count:', count)
    return np.array(x), np.array(y), detail

def alternate_load_data(path1, path2):
    if not os.path.exists(path1) or not os.path.exists(path2):
        exit(100)
    file_list = os.listdir(path1)
    print(path1, path2, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()
    myset1 = []
    myset2 = []

    for file in file_list:
        count += 1
        if count % 500 == 0:
            print(count, time.time()-beginTime)
        file_to_read = open(path1+file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content = [float(i) for i in content.split(',')]
        myset1.append(content)

        file_to_read = open(path2 + file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content = [float(i) for i in content.split(',')]
        myset2.append(content)

        if count % loop_num == 0:
            tmpy = file.split('.')[0].split('_')[2]
            x.append(np.append(myset1, myset2, axis=0))
            y.append(int(tmpy) - 1)
            myset1 = []
            myset2 = []

    return np.array(x), np.array(y)


def mean_max_load(path1, path2):
    if not os.path.exists(path1) or not os.path.exists(path2):
        exit(100)
    file_list = os.listdir(path1)
    print(path1, path2, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()
    myset1 = []
    for file in file_list:
        count += 1
        if count % 500 == 0:
            print(count, time.time()-beginTime)
        file_to_read = open(path1+file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content1 = np.array([float(i) for i in content.split(',')]).reshape((1, 512))

        file_to_read = open(path2 + file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content2 = np.array([float(i) for i in content.split(',')]).reshape((1, 512))
        tmp = np.append(content1, content2, axis=0)

        myset1.append(np.max(tmp, axis=0))

        if count % loop_num == 0:
            tmpy = file.split('.')[0].split('_')[2]
            x.append(myset1)
            y.append(int(tmpy) - 1)
            myset1 = []

    return np.array(x), np.array(y)

def mlp(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(512,)))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    # model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=2,
                        validation_data=(x_test, y_test))


def cnn_lstm(x_train, y_train, x_test, y_test):
    # Convolution
    kernel_size = 5
    filters = 64
    pool_size = 4

    # LSTM
    lstm_output_size = 128

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    # print(x_train.ndim)
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(512,1)))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


def lstm(x_train, y_train, x_test, y_test):
    drop = 0.4
    model = Sequential()
    model.add(LSTM(128, input_shape=(loop_num * 2 if alternate else loop_num, 512), return_sequences=True, dropout=drop))
    # model.add(LSTM(512, return_sequences=True, dropout=drop))
    model.add(LSTM(64, dropout=drop))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    print('Train...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              verbose=2)
    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)
    return model


def evaluate1(model, x, y, intervel=5):
    begin = 0
    correct = 0
    all = 0
    print("Begin to evaluate")
    while begin < x.shape[0]:
        ground_true = np.where(y[begin]==1)[0][0]
        tmpx = x[begin:begin+intervel]
        tmpy = model.predict(tmpx)
        res = np.argmax(tmpy, axis=1)
        Mode = mode(res)[0][0]
        begin += intervel
        all += 1
        if Mode == ground_true:
            correct += 1.0
    print("Accuracy:", correct/all)


def evaluate2(model, x, y, spatial_num, loop, level1=0, level2=0, pre=0, write_csv=True):
    intervel = spatial_num//loop
    trg_path = 'D:/graduation_project/JTM_training/statistics/'
    begin = 0
    correct = 0
    all = 0
    type_count = [0] * 51
    correct_count = [0] * 51
    # print("Begin to evaluate")
    while begin < x.shape[0]:
        ground_true = np.where(y[begin]==1)[0][0]
        tmpx = x[begin:begin+intervel]
        tmpy = model.predict(tmpx)

        tmp = np.zeros((1, num_classes))
        tmp[tmp==0] = 1.0
        for k in range(intervel):
            tmp *= tmpy[k]

        res = np.argmax(tmp, axis=1)
        begin += intervel
        all += 1

        type_count[ground_true] += 1
        if res[0] == ground_true:
            correct_count[ground_true] += 1
            correct += 1.0
    print("Accuracy:", correct/all)
    res = []
    res_count = []
    for i in range(51):
        if type_count[i] == 0:
            continue
        res.append(correct_count[i] / type_count[i])
        res_count.append(type_count[i])
        # print(i, type_count[i], correct_count[i] / type_count[i])

    if correct/all > pre and write_csv:
        file_to_write = open(trg_path + str(spatial_num)+ '_b' + str(batch_size) + '_'+str(level1) + '_' + str(level2) + '.csv', 'w')
        file_to_write.write(','.join([str(i) for i in res])+'\n')
        file_to_write.write(','.join([str(i) for i in res_count]) + '\n')
        file_to_write.write(str(correct/all))
        file_to_write.close()
    return correct/all

def evaluate3(model, x, y, intervel=5):
    begin = 0
    correct = 0
    all = 0
    print("Begin to evaluate")
    while begin < x.shape[0]:
        ground_true = np.where(y[begin]==1)[0][0]
        tmpx = x[begin:begin+intervel]
        tmpy = model.predict(tmpx)

        res = np.max(tmpy, axis=0)
        res = np.argmax(res, axis=0)
        begin += intervel
        all += 1
        if res == ground_true:
            correct += 1.0
    print("Accuracy:", correct/all)


test_detail = []
def spatial_evaluate(model, x, y, level1, level2, loop, pre=0, write_csv=False, trg_path=''):
    begin = 0
    correct = 0
    all = 0
    type_count = [0] * 51
    correct_count = [0] * 51
    index = 0
    # print("Begin to evaluate")
    while begin < x.shape[0]:
        intervel = test_detail[index] // loop
        intervel += 1 if test_detail[index] % loop != 0 else 0

        ground_true = np.where(y[begin] == 1)[0][0]
        tmpx = x[begin:begin + intervel]
        tmpy = model.predict(tmpx)

        tmp = np.zeros((1, num_classes))
        tmp[tmp == 0] = 1.0
        # print(index, test_detail[index], intervel)
        for k in range(intervel):
            tmp *= tmpy[k]

        res = np.argmax(tmp, axis=1)
        begin += intervel
        all += 1
        index += 1

        type_count[ground_true] += 1
        if res[0] == ground_true:
            correct_count[ground_true] += 1
            correct += 1.0
    print("Accuracy:", correct / all)
    res = []
    res_count = []
    for i in range(51):
        if type_count[i] == 0:
            continue
        res.append(correct_count[i] / type_count[i])
        res_count.append(type_count[i])
        # print(i, type_count[i], correct_count[i] / type_count[i])

    if correct / all > pre and write_csv:
        file_to_write = open(
            trg_path + str(spatial_num) + '_b' + str(batch_size) + '_' + str(level1) + '_' + str(level2) + '.csv', 'w')
        file_to_write.write(','.join([str(i) for i in res]) + '\n')
        file_to_write.write(','.join([str(i) for i in res_count]) + '\n')
        file_to_write.write(str(correct / all))
        file_to_write.close()
    return correct / all



def specific_level_test(level1, level2, x_train, y_train, x_test, y_test, loop, save_model=True, trg_path='D:/graduation_project/JTM_training/model/'):
    drop = 0.4
    if not os.path.exists(trg_path):
        os.makedirs(trg_path)
    model = Sequential()
    model.add(LSTM(level1, input_shape=(loop * 2 if alternate else loop, 512), return_sequences=True, dropout=drop))
    model.add(LSTM(level2, dropout=drop))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    pre_best = 0
    best_epochs = 0
    best_model = None
    for i in range(epochs):
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_test, y_test),
                  verbose=2)
        if history.history['val_acc'][0] > 0.62:
            # tmp = evaluate2(model, x_test, y_test, level1=level1, level2=level2, pre=pre_best, loop=loop)
            tmp = spatial_evaluate(model, x_test, y_test, level1, level2, pre=pre_best, loop=loop)
            if tmp > pre_best:
                pre_best = tmp
                if save_model:
                    model.save(trg_path + 'loop'+str(loop) + '_frame10_' + str(level1) + '_' + str(level2) + '_acc'+str(int(pre_best*100000))+'.h5')
                pre_best = 0
                best_epochs = i
    # my_model = load_model(trg_path + 'b' + str(batch_size) + '_' + str(level1) + '_' + str(level2) + '.h5')
    return pre_best#, my_model

def loop_test(x_train, y_train, x_test, y_test):
    rec = {}

    # for level1 in range(320, 1025, 64):
    #     for level2 in range(64, level1+1, 64):
    for level1 in range(128, 1025, 128):
        for level2 in range(64, level1+1, 64):
            print(level1,'_',level2)
            pre_best= specific_level_test(level1, level2, x_train, y_train, x_test, y_test)
            rec[str(level1)+'_'+str(level2)]=pre_best
    print(rec)
    for item in rec:
        print(item)
    # evaluate2(model, x_test, y_test, write_csv=False)

if __name__ == '__main__':
    # train_mc_path = root + train + '/JTM_mc/25/' + folder_name + '/'
    # train_ori_path = root + train + '/JTM_ori/25/' + folder_name + '/'
    # test_mc_path = root + test + '/JTM_mc/25/' + folder_name + '/'
    # test_ori_path = root + test + '/JTM_ori/25/' + folder_name + '/'
    # x_train, y_train = alternate_load_data(train_mc_path, train_ori_path)
    # x_test, y_test = alternate_load_data(test_mc_path, test_ori_path)


    # x_train, y_train = load_data(fmap_train_path)
    # x_test, y_test = load_data(fmap_test_path)

    # train_mc_path = root + train + '/JTM_mc/' + str(args.frame) + '/' + folder_name + '/'
    # train_ori_path = root + train + '/JTM_ori/' + str(args.frame) + '/' + folder_name + '/'
    # test_mc_path = root + test + '/JTM_mc/' + str(args.frame) + '/' + folder_name + '/'
    # test_ori_path = root + test + '/JTM_ori/' + str(args.frame) + '/' + folder_name + '/'
    # x_train, y_train = mean_max_load(train_mc_path, train_ori_path)
    # x_test, y_test = mean_max_load(test_mc_path, test_ori_path)
    #
    # print("train samples shape:", x_train.shape)
    # print("test samples shape:", x_test.shape)
    #
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    # loop_test(x_train, y_train, x_test, y_test)

######################################
    # train_mc_path = root + train + '/JTM_mc/' + str(args.frame) + '/' + folder_name + '/'
    # test_mc_path = root + test + '/JTM_mc/' + str(args.frame) + '/' + folder_name + '/'
    # x_train, y_train = load_data(train_mc_path)
    # x_test, y_test = load_data(test_mc_path)
    #
    # print("train samples shape:", x_train.shape)
    # print("test samples shape:", x_test.shape)
    #
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    #
    # best = specific_level_test(384, 64, x_train, y_train, x_test, y_test, save_model=False)
    # print('Acc', best)



######################################
    # test_mc_path = root + test + '/JTM_mc/' + str(args.frame) + '/' + folder_name + '/'
    # test_ori_path = root + test + '/JTM_ori/' + str(args.frame) + '/' + folder_name + '/'
    # x_test, y_test = mean_max_load(test_mc_path, test_ori_path)
    #
    # print("test samples shape:", x_test.shape)
    #
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    # model = load_model('D:/graduation_project/JTM_training/model/b256_256_128_3505_e35.h5')
    # evaluate2(model, x_test, y_test, write_csv=False)

######################################
    train_path = root + train + '/spatial_10/frame/' + folder_name + '/'
    test_path = root + test + '/spatial_10/frame/' + folder_name + '/'
    train_detail_path = root + train + '/spatial_10/detail/_frame_num.txt'
    test_detail_path = root + test + '/spatial_10/detail/_frame_num.txt'

    x_train, y_train, _ = load_spatial_data(train_path, train_detail_path, loop=5)
    x_test, y_test, detail = load_spatial_data(test_path, test_detail_path, loop=5)
    test_detail = detail

    print("train samples shape:", x_train.shape)
    print("test samples shape:", x_test.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #
    trg_path = 'D:/graduation_project/SPATIAL_training/LSTM/'
    # mlp(x_train, y_train, x_test, y_test)
    specific_level_test(256, 64, x_train, y_train, x_test, y_test, save_model=True, trg_path=trg_path, loop=5)


