from __future__ import print_function
from sklearn.svm import SVC
from sklearn import preprocessing
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np
import argparse, os
from keras import regularizers
from scipy.stats import mode
import logging, time

def svc(traindata,trainlabel,testdata,testlabel):
    # beginTime = time.time()
    # min_max_scaler = preprocessing.MinMaxScaler()
    # traindata = min_max_scaler.fit_transform(traindata)
    # testdata = min_max_scaler.transform(testdata)
    # print("transform data:", time.time() - beginTime)

    print("Start training SVM...")
    beginTime = time.time()
    svcClf = SVC(C=0.1,kernel="rbf",cache_size=3000, max_iter=1000, tol=1)
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

read_loop = False
loop_num = 1

def load_data(path):

    if not os.path.exists(path):
        print(path)
        exit(100)
    file_list = os.listdir(path)
    print(path, len(file_list))
    x = []
    y = []
    count = 0
    beginTime = time.time()
    myset = []
    for file in file_list:
        count += 1
        if count % 500 == 0:
            print(count, time.time()-beginTime)
        file_to_read = open(path+file, 'r')
        content = file_to_read.read()
        file_to_read.close()

        content = [float(i) for i in content.split(',')]
        tmpy = file.split('.')[0].split('_')[2]

        if read_loop:
            myset.append(content)
            if count % loop_num == 0:
                x.append(myset)
                y.append(int(tmpy) - 1)
                myset = []
        else:
            x.append(content)
            y.append(int(tmpy) - 1)

    return np.array(x), np.array(y)

if __name__ == '__main__':
    root = "D:/graduation_project/workspace/dataset/HMDB51/"
    train = "train1"
    test = "test1"
    frame = 10
    folder_name = "vgg19_block5_pool_mean"
    # train_mc_path = root + train + '/JTM_mc/' + str(frame) + '/' + folder_name + '/'
    # test_mc_path = root + test + '/JTM_mc/' + str(frame) + '/' + folder_name + '/'

    train_path = root + train + '/JDM_InceptionV3_shared/10/fc1/'
    test_path = root + test + '/JDM_InceptionV3_shared/10/fc1/'
    x_train, y_train = load_data(train_path)
    x_test, y_test = load_data(test_path)
    svc(x_train, y_train, x_test, y_test)