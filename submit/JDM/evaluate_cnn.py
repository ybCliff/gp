from keras.applications.inception_v3 import InceptionV3
from Model_and_funcs import preprocess_file_list
import argparse
import numpy as np
import os, time, cv2, random
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model
from scipy.stats import mode


parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--folder', type=str, default='JDM_ori/10/')
args = parser.parse_args()

root = 'D:/graduation_project/workspace/dataset/HMDB51/'
train = 'train'+str(args.split)+'/'
test = 'test'+str(args.split)+'/'
path_train = root + train + args.folder
path_test = root + test + args.folder
num_classes = 51
loop_num = 10


def evaluate(model, path):
    file_list = preprocess_file_list(os.listdir(path))
    count = 0
    correct1 = 0
    correct2 = 0
    sample_count = 0
    x = []
    for file in file_list:
        count += 1
        tmpx = cv2.imread(path + file)
        x.append(tmpx)
        if count % loop_num == 0:
            ground_true = int(file.split('.')[0].split('_')[2]) - 1
            out = model.predict(np.array(x))

            mul = np.zeros((1, num_classes))
            mul[mul == 0] = 1.0
            for i in range(loop_num):
                mul *= out[i]
            res = np.argmax(mul, axis=1)
            if res == ground_true:
                correct1 += 1

            tmp = np.argmax(out, axis=1)
            Mode = mode(tmp)[0][0]
            if Mode == ground_true:
                correct2 += 1

            sample_count += 1
            print(sample_count, ground_true, res[0], Mode)
            x = []
    print('Mulitiple-wise Accuracy:', correct1/sample_count)
    print('Mode Accuracy:', correct2 / sample_count)

if __name__ == '__main__':
    load_model_root = 'D:/graduation_project/JDM_cnn/'
    model_name = 'e1_spe1_round2.h5'
    model = load_model(load_model_root + model_name)
    evaluate(model, path_train)