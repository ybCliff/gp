from sklearn.svm import SVC
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
parser.add_argument('--layer', type=str, default='block4_pool')
parser.add_argument('--fusion', type=str, default='mean')   # mean  or  max
parser.add_argument('--folder', type=str, default='JTM_mc/10')   # JTM_mc/x, JTM_ori/x, JDM_mc/x, JDM_ori/x, spatial_x/frame
parser.add_argument('--frame', type=int, default=10)
parser.add_argument('--folder2', type=str, default=None)   # JTM_mc/x, JTM_ori/x, JDM_mc/x, JDM_ori/x, spatial_x/frame
parser.add_argument('--read_type', type=str, default='mean')
parser.add_argument('--split', type=int, default=1)

parser.add_argument('--loop', type=int, default=1)
parser.add_argument('--acc_limit', type=float, default=0.285)
parser.add_argument('--feature', type=int, default=512)

parser.add_argument('--write_csv', type=int, default=1)
parser.add_argument('--csv_path', type=str, default='./')
parser.add_argument('--version', type=int, default=0)

parser.add_argument('--save_model', type=int, default=0)
parser.add_argument('--model_path', type=str, default='')

parser.add_argument('--level1', type=int, default=128)
parser.add_argument('--level2', type=int, default=64)
parser.add_argument('--level3', type=int, default=-1)
parser.add_argument('--drop', type=float, default=0.4)

parser.add_argument('--evaluate', type=str, default='mean')

args = parser.parse_args()
args.drop = 0.2
num_classes = 51
batch_size = 256
epochs = 150
alternate = False

root = "D:/graduation_project/workspace/dataset/"+args.dataset+'/'
folder_name = args.model + '_' + args.layer + '_' + args.fusion
train = "train" + str(args.split)
test = "test" + str(args.split)

def tmp_load_data(path):
    x = []
    y = []
    file_list = os.listdir(path)
    count = 0
    for file in file_list:
        if '.txt' not in file:
            continue
        file_to_read = open(path + file, 'r')
        tmp = file_to_read.read()
        tmp = tmp.replace('[', '').replace(']','')
        tmpx = [float(i) for i in tmp.split(',')]
        file_to_read.close()

        x.append(tmpx)
        y.append(int(file.split('.')[0].split('_')[1]) - 1)
        count+=1
        if count % 500 == 0:
            print(count)
    return np.array(x), np.array(y)


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

def alternate_load_data(path1, path2, loop):
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

        if count % loop == 0:
            tmpy = file.split('.')[0].split('_')[2]
            x.append(np.append(myset1, myset2, axis=0))
            y.append(int(tmpy) - 1)
            myset1 = []
            myset2 = []

    return np.array(x), np.array(y)


def mean_max_load(path1, path2, loop, type):
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

        if type == 'max':
            myset1.append(np.max(tmp, axis=0))
        else:
            myset1.append(np.mean(tmp, axis=0))

        if count % loop == 0:
            tmpy = file.split('.')[0].split('_')[2]
            x.append(myset1)
            y.append(int(tmpy) - 1)
            myset1 = []

    return np.array(x), np.array(y)

def mlp(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(2048,)))
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


def evaluate2(model, x, y, spatial_num, loop):
    intervel = spatial_num//loop
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

        if args.evaluate == 'mult':
            tmp = np.zeros((1, num_classes))
            tmp[tmp==0] = 1.0
            for k in range(intervel):
                tmp *= tmpy[k]
        elif args.evaluate == 'mean':
            tmp = np.mean(tmpy, axis=0)
        elif args.evaluate == 'max':
            tmp = np.max(tmpy, axis=0)

        res = np.argmax(tmp)
        begin += intervel
        all += 1

        type_count[ground_true] += 1
        if res == ground_true:
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
def spatial_evaluate(model, x, y, loop):
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

        if args.evaluate == 'mult':
            tmp = np.zeros((1, num_classes))
            tmp[tmp==0] = 1.0
            for k in range(intervel):
                tmp *= tmpy[k]
        elif args.evaluate == 'mean':
            tmp = np.mean(tmpy, axis=0)
        elif args.evaluate == 'max':
            tmp = np.max(tmpy, axis=0)

        res = np.argmax(tmp)
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

    return correct / all



def specific_level_test(x_train, y_train, x_test, y_test, loop):
    drop = args.drop
    if args.save_model and not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    model = Sequential()
    model.add(LSTM(args.level1, input_shape=(loop * 2 if alternate else loop, args.feature), return_sequences=True, dropout=drop))
    if args.level3 != -1:
        model.add(LSTM(args.level2, dropout=drop, return_sequences=True))
    else:
        model.add(LSTM(args.level2, dropout=drop))
    if args.level3 != -1:
        model.add(LSTM(args.level3, dropout=drop))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    pre_best = 0

    if args.write_csv == 1:
        args.csv_path += (folder_name+'/')
        if not os.path.exists(args.csv_path):
            os.makedirs(args.csv_path)
        key = (args.folder).split('/')[1]
        fname = (args.folder).split('/')[0]
        if args.folder2 != None:
            fname += ('_mc_' + args.read_type)
        fname += ('_' + key + '_v' + str(args.version) +'.txt')
        file_to_write = open(args.csv_path + fname, 'w')

    for i in range(epochs):
        history = model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=1,
                  validation_data=(x_test, y_test),
                  verbose=2)
        if history.history['val_acc'][0] > args.acc_limit:
            tmp = evaluate2(model, x_test, y_test, spatial_num=args.frame, loop=loop)
            if tmp > pre_best:
                pre_best = tmp
                if args.save_model == 1:
                    model.save(args.model_path + 'loop'+str(loop) + '_frame10_' + str(args.level1) + '_' + str(args.level2) + '_acc'+str(int(pre_best*100000))+'.h5')
                if args.write_csv:
                    print(i, pre_best, file=file_to_write)

    if args.write_csv:
        file_to_write.close()

    return pre_best

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

def svc(traindata,trainlabel,testdata,testlabel):
    # beginTime = time.time()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # traindata = min_max_scaler.fit_transform(traindata)
    # testdata = min_max_scaler.transform(testdata)
    # print("transform data:", time.time() - beginTime)

    print("Start training SVM...")
    beginTime = time.time()
    svcClf = SVC(C=1,kernel="linear",cache_size=3000, max_iter=5000, tol=1)
    svcClf.fit(traindata,trainlabel)
    print("SVM training time:", time.time() - beginTime)

    beginTime = time.time()
    pred_trainlabel = svcClf.predict(traindata)
    print("predict training time:", time.time() - beginTime)
    num = len(pred_trainlabel)
    print(pred_trainlabel)
    accuracy = len([1 for i in range(num) if trainlabel[i] == pred_trainlabel[i]]) / float(num)
    print("train Accuracy:", accuracy)

    beginTime = time.time()
    pred_testlabel = svcClf.predict(testdata)
    print("predict testing time:", time.time() - beginTime)
    num = len(pred_testlabel)
    print(pred_testlabel)
    accuracy = len([1 for i in range(num) if testlabel[i]==pred_testlabel[i]])/float(num)
    print("test Accuracy:",accuracy)

    return svcClf

if __name__ == '__main__':
    print(args)
    train_path = root + train + '/' + args.folder + '/' + folder_name + '/'
    test_path = root + test + '/' + args.folder + '/' + folder_name + '/'

    if args.folder2 != None:
        train_path2 = root + train + '/' + args.folder2 + '/' + folder_name + '/'
        test_path2 = root + test + '/' + args.folder2 + '/' + folder_name + '/'
        x_train, y_train = mean_max_load(train_path, train_path2, loop=args.loop, type=args.read_type)
        x_test, y_test = mean_max_load(test_path, test_path2, loop=args.loop, type=args.read_type)
    else:
        x_train, y_train = load_data(train_path, read_loop=True, loop=args.loop)
        x_test, y_test = load_data(test_path, read_loop=True, loop=args.loop)

    print("train samples shape:", x_train.shape)
    print("test samples shape:", x_test.shape)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    best = specific_level_test(x_train, y_train, x_test, y_test, loop=args.loop)
    print("Best:", best)

