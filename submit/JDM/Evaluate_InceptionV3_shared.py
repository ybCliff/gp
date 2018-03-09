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
parser.add_argument('--frame', type=int, default=10)
parser.add_argument('--model_name', type=str, default='')
args = parser.parse_args()

root = 'D:/graduation_project/workspace/dataset/HMDB51/'
train = 'train'+str(args.split)+'/'
test = 'test'+str(args.split)+'/'

folder_ori = 'JDM_ori/'+str(args.frame)+'/'
folder_mc = 'JDM_mc/'+str(args.frame)+'/'

path_train_ori = root + train + folder_ori
path_train_mc = root + train + folder_mc

path_test_ori = root + test + folder_ori
path_test_mc = root + test + folder_mc

num_classes = 51
loop_num = 10

load_model_root = 'D:/graduation_project/JDM_training/InceptionV3/shared/'
save_csv_path = load_model_root + 'statistics/'
if not os.path.exists(save_csv_path):
    os.makedirs(save_csv_path)

def evaluate(model, path_ori, path_mc, csv_name='', write_csv=False):
    file_list = preprocess_file_list(os.listdir(path_ori))
    count = 0
    correct1 = 0
    correct2 = 0
    type_count = [0] * num_classes
    type_correct1 = [0] * num_classes
    type_correct2 = [0] * num_classes
    sample_count = 0
    x_ori = []
    x_mc = []
    for file in file_list:
        count += 1
        x_ori.append(cv2.imread(path_ori + file))
        x_mc.append(cv2.imread(path_mc + file))
        if count % loop_num == 0:
            ground_true = int(file.split('.')[0].split('_')[2]) - 1
            type_count[ground_true] += 1
            out = model.predict([np.array(x_ori), np.array(x_mc)])

            mul = np.zeros((1, num_classes))
            mul[mul == 0] = 1.0
            for i in range(loop_num):
                mul *= out[i]
            res = np.argmax(mul, axis=1)
            if res == ground_true:
                correct1 += 1
                type_correct1[ground_true] += 1

            tmp = np.argmax(out, axis=1)
            Mode = mode(tmp)[0][0]
            if Mode == ground_true:
                correct2 += 1
                type_correct2[ground_true] += 1

            sample_count += 1
            # print(sample_count, ground_true, res[0], Mode)
            x_ori = []
            x_mc = []

    print('Mulitiple-wise Accuracy:', correct1/sample_count)
    print('Mode Accuracy:', correct2 / sample_count)

    if write_csv:
        acc1 = []
        acc2 = []
        index = []
        for i in range(num_classes):
            if type_count[i] == 0:
                continue
            index.append(i)
            acc1.append(type_correct1[i] / type_count[i])
            acc2.append(type_correct2[i] / type_count[i])

        file_to_write = open(save_csv_path+csv_name, 'w')
        print(','.join(['Mult Acc', str(correct1/sample_count), 'Mode Acc', str(correct2 / sample_count)]), file=file_to_write)

        print('index,', end="", file=file_to_write)
        print(','.join([str(i) for i in index]), file=file_to_write)

        print('count,', end="", file=file_to_write)
        print(','.join([str(int(i)) for i in type_count[index]]), file=file_to_write)

        print('mult,', end="", file=file_to_write)
        print(','.join([str(i) for i in acc1]), file=file_to_write)

        print('mode,', end="", file=file_to_write)
        print(','.join([str(i) for i in acc2]), file=file_to_write)
        file_to_write.close()


if __name__ == '__main__':
    model = load_model(load_model_root + args.model_name)
    pre = args.model_name.split('.')[0] + '.csv'
    evaluate(model, path_train_ori, path_train_mc, csv_name='train_'+pre, write_csv=True)
    evaluate(model, path_test_ori, path_test_mc, csv_name='test_' + pre, write_csv=True)