from keras.applications.inception_v3 import InceptionV3
import argparse
import numpy as np
import os, time, cv2, random
import keras
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model
from Model_and_funcs import preprocess_file_list

parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--frame', type=int, default=10)
parser.add_argument('--model_name', type=str, default='e1_spe4_round2.h5')
parser.add_argument('--layer', type=str, default='fc1')
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

trg_train_fmap = root + train + 'JDM_InceptionV3_shared/' + str(args.frame) + '/' + args.layer + '/'
trg_test_fmap = root + test + 'JDM_InceptionV3_shared/' + str(args.frame) + '/' + args.layer + '/'
if not os.path.exists(trg_train_fmap):
    os.makedirs(trg_train_fmap)
if not os.path.exists(trg_test_fmap):
    os.makedirs(trg_test_fmap)

def run(scope, model):
    path_ori = path_train_ori if scope == 'train' else path_test_ori
    path_mc = path_train_mc if scope == 'train' else path_test_mc
    trg_path = trg_train_fmap if scope == 'train' else trg_test_fmap
    file_list = preprocess_file_list(os.listdir(path_ori))
    for file in file_list:
        x_ori = cv2.imread(path_ori + file)
        x_mc = cv2.imread(path_mc + file)

        x_ori = np.expand_dims(x_ori, axis=0)
        x_mc = np.expand_dims(x_mc, axis=0)

        out = model.predict([x_ori, x_mc])

        txt_name = file.split('.')[0] + '.txt'
        file_to_write = open(trg_path + txt_name, 'w')
        file_to_write.write(','.join([str(i) for i in out[0]]))
        file_to_write.close()





if __name__ == '__main__':
    print(args)
    save_model_root = 'D:/graduation_project/JDM_training/InceptionV3/shared/'
    base_model = load_model(save_model_root + args.model_name)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(args.layer).output)
    run('train', model)
    run('test', model)



