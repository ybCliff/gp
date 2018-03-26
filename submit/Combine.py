from keras.applications.inception_v3 import InceptionV3
import argparse
import numpy as np
import os, time, cv2, random
import keras
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model
from scipy.stats import mode

parser = argparse.ArgumentParser(description='Generate feature map from joints')
parser.add_argument('--dataset', type=str, default='HMDB51')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--model', type=str, default='vgg19')
parser.add_argument('--layer', type=str, default='block5_pool')
parser.add_argument('--fusion', type=str, default='mean')   # mean  or  max
parser.add_argument('--JTM_frame', type=int, default=10)
parser.add_argument('--JDM_frame', type=int, default=10)
parser.add_argument('--SPATIAL_frame', type=int, default=10)
args = parser.parse_args()

root = "D:/graduation_project/workspace/dataset/"+args.dataset+'/'
train = "train" + str(args.split)
test = "test" + str(args.split)
folder_name = args.model + '_' + args.layer + '_' + args.fusion
num_classes = 51

JTM_model_path = "D:/graduation_project/JTM_training/split1/normal/model/loop2_frame10_128_64_acc34287.h5"
JDM_model_path = "D:/graduation_project/JDM_training/split1/InceptionV3/shared/loop2_frame10_128_64_acc41480.h5"
SPATIAL_model_path = "D:/graduation_project/SPATIAL_training/LSTM/loop5_frame10_6571/loop5_frame10_128_64_acc6571.h5"

detail_path = root + test + '/spatial_'+ str(args.SPATIAL_frame)+ '/detail/_frame_num.txt'
JTM_ori_path = root + test + '/JTM_ori/' + str(args.JTM_frame) + '/' + folder_name + '/'
JTM_mc_path = root + test + '/JTM_mc/' + str(args.JTM_frame) + '/' + folder_name + '/'
# JDM_ori_path = root + test + '/JDM_ori/' + str(args.JDM_frame) + '/' + folder_name + '/'
# JDM_mc_path = root + test + '/JDM_mc/' + str(args.JDM_frame) + '/' + folder_name + '/'
JDM_shared_path = root + test + '/JDM_InceptionV3_shared/' + str(args.JDM_frame) + '/fc1/'

SPATIAL_path = root + test + '/spatial_'+ str(args.SPATIAL_frame) + '/frame/' + folder_name + '/'


def get_jpg_list(lst):
    while('.jpg' not in lst[len(lst)-1]):
        lst.pop()
    return lst

def load_detail(detail_path):
    if not os.path.exists(detail_path):
        print("ERROR OPENING DETAIL FILE!")
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
    return detail


def load_spatial_data(path, detail, loop):
    if not os.path.exists(path):
        print("ERROR OPENING FILE!")
        exit(0)

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
    return np.array(x), np.array(y)

def mean_max_load(path1, path2, loop):
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

        if count % loop == 0:
            tmpy = file.split('.')[0].split('_')[2]
            x.append(myset1)
            y.append(int(tmpy) - 1)
            myset1 = []

    return np.array(x), np.array(y)

def evaluate(model, x, y, spatial_num, loop, trg_path="", level1=0, level2=0, pre=0, write_csv=False):
    intervel = spatial_num//loop
    trg_path += 'statistics/'
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

def spatial_evaluate(model, x, y, level1, level2, loop, test_detail, pre=0, write_csv=False, trg_path=''):
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


model_JTM = load_model(JTM_model_path)
model_JDM = load_model(JDM_model_path)
model_SPATIAL = load_model(SPATIAL_model_path)

def get_vec(x_JTM, x_SPATIAL):
    y_JTM = model_JTM.predict(x_JTM)
    y_SPATIAL = model_SPATIAL.predict(x_SPATIAL)
    return y_JTM, y_SPATIAL

def load_data(path, begin, end, num, loop, path2=None, type=None, feature=512):
    if not os.path.exists(path):
        print("ERROR OPENING DATA FILE")
        exit(999)

    file_list = os.listdir(path)
    file_list = file_list[begin:end]

    myset = []
    for file in file_list:
        file_to_read = open(path + file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content = np.array([float(i) for i in content.split(',')]).reshape((1, feature))

        if path2 is not None:
            file_to_read = open(path2 + file, 'r')
            content2 = file_to_read.read()
            file_to_read.close()
            content2 = np.array([float(i) for i in content2.split(',')]).reshape((1, feature))
            tmp = np.append(content, content2, axis=0)
            if type == "max":
                content = np.max(tmp, axis=0)
            elif type == "mean":
                content = np.mean(tmp, axis=0)

        myset.append(content)

    myset = np.array(myset)
    y = int(file_list[0].split('.')[0].split('_')[2])-1

    if myset.shape[0] == num * loop:
        x = np.reshape(myset, (num, loop, feature))
        return x, y
    else:
        remain = num * loop - myset.shape[0]
        lst = [i for i in range(myset.shape[0])]
        rec = [i for i in range(myset.shape[0])]
        for i in range(remain):
            tmp = random.sample(lst, 1)
            rec.append(tmp[0])
        rec = sorted(rec)

        newset = []
        for i in range(num * loop):
            newset.append(myset[rec[i]])

        x = np.reshape(np.array(newset), (num, loop, feature))
        return x, y

def load_data_specific(path, begin, end, sample_num, time_step, path2=None, type=None, feature=512):
    if not os.path.exists(path):
        print("ERROR OPENING DATA FILE")
        exit(999)

    file_list = os.listdir(path)
    file_list = file_list[begin:end]

    myset = []
    for file in file_list:
        file_to_read = open(path + file, 'r')
        content = file_to_read.read()
        file_to_read.close()
        content = np.array([float(i) for i in content.split(',')]).reshape((1, feature))

        if path2 is not None:
            file_to_read = open(path2 + file, 'r')
            content2 = file_to_read.read()
            file_to_read.close()
            content2 = np.array([float(i) for i in content2.split(',')]).reshape((1, feature))
            tmp = np.append(content, content2, axis=0)
            if type == "max":
                content = np.max(tmp, axis=0)
            elif type == "mean":
                content = np.mean(tmp, axis=0)

        myset.append(content)

    myset = np.array(myset)
    ground_true = int(file_list[0].split('.')[0].split('_')[2])-1

    if myset.shape[0] < time_step:
        remain = time_step - myset.shape[0]
        lst = [i for i in range(myset.shape[0])]
        rec = [i for i in range(myset.shape[0])]
        for i in range(remain):
            tmp = random.sample(lst, 1)
            rec.append(tmp[0])
        rec = sorted(rec)

        newset = []
        for i in range(time_step):
            newset.append(myset[rec[i]])
        myset = newset

    x = []
    y = []
    for i in range(sample_num):
        tmp = random.sample(myset.tolist(), time_step)
        print(np.array(tmp).shape)
        x.append(np.reshape(tmp, (time_step, feature)))
        y.append(ground_true)
    return np.array(x), np.array(y)


def get_acc(lst1, lst2):
    count = 0
    for i in range(len(lst1)):
        if lst1[i] == lst2[i]:
            count += 1
    return count * 1.0 / len(lst1)


def normalize(lst):
    # min_value = np.min(lst)
    # max_value = np.max(lst)
    # intervel = max_value - min_value
    for i in range(len(lst)):
        lst[i] = lst[i]**0.4
    return lst

def get_predict(begin, end, begin_SPATIAL, end_SPATIAL):
    loop_JTM = 2
    loop_JDM = 2
    loop_SPATIAL = 5
    x_JTM, y_JTM = load_data(path=JTM_ori_path, path2=JTM_mc_path, begin=begin, end=end, num=5, loop=loop_JTM, type="max")
    vec_JTM = np.mean(model_JTM.predict(x_JTM), axis=0)

    x_JDM, y_JDM = load_data(path=JDM_shared_path, begin=begin, end=end, num=5, loop=loop_JDM, feature=1024)
    vec_JDM = normalize(np.mean(model_JDM.predict(x_JDM), axis=0))

    x_SPATIAL, y_SPATIAL = load_data(path=SPATIAL_path, begin=begin_SPATIAL, end=end_SPATIAL, num=2, loop=loop_SPATIAL)
    vec_SPATIAL = np.mean(model_SPATIAL.predict(x_SPATIAL), axis=0)

    assert y_JTM == y_SPATIAL and y_SPATIAL == y_JDM

    return np.argmax(vec_JTM), np.argmax(vec_JDM), np.argmax(vec_SPATIAL), np.argmax(vec_JTM * vec_JDM), np.argmax(vec_JTM * vec_SPATIAL), np.argmax(vec_JDM * vec_SPATIAL), np.argmax(vec_JTM * vec_JDM * vec_SPATIAL), y_JTM

if __name__ == '__main__':
    # x, y = load_data_specific(path=JTM_ori_path, path2=JTM_mc_path, begin=0, end=10, sample_num=10, time_step=5, type="max")
    # print(x.shape, y.shape)

    detail = load_detail(detail_path)
    max_index = len(detail)
    begin = 0
    begin_SPATIAL = 0
    rec_JTM = []
    rec_JDM = []
    rec_SPATIAL = []

    rec_TD = []
    rec_TS = []
    rec_DS = []
    rec_TDS = []

    rec_ori = []
    begin_time = time.time()
    for index in range(max_index):
        if index % 50 == 0:
            print(index, time.time()-begin_time)
        end = begin + 10
        end_SPATIAL = begin_SPATIAL + detail[index]

        T, D, S, TD, TS, DS, TDS, G = get_predict(begin, end, begin_SPATIAL, end_SPATIAL)

        rec_JTM.append(T)
        rec_JDM.append(D)
        rec_SPATIAL.append(S)
        rec_ori.append(G)
        rec_TD.append(TD)
        rec_TS.append(TS)
        rec_DS.append(DS)
        rec_TDS.append(TDS)

        begin += args.JTM_frame
        begin_SPATIAL += detail[index]

    print("JTM ACC:", get_acc(rec_JTM, rec_ori))
    print("JDM ACC:", get_acc(rec_JDM, rec_ori))
    print("SPA ACC:", get_acc(rec_SPATIAL, rec_ori))

    print(" TD ACC:", get_acc(rec_TD, rec_ori))
    print(" TS ACC:", get_acc(rec_TS, rec_ori))
    print(" DS ACC:", get_acc(rec_DS, rec_ori))
    print("TDS ACC:", get_acc(rec_TDS, rec_ori))


