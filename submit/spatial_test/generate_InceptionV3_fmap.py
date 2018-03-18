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

train_spatial = root + train + 'spatial_10/frame/'
test_spatial = root + test + 'spatial_10/frame/'

train_detail = root + train + 'spatial_10/detail/_frame_num.txt'
test_detail = root + test + 'spatial_10/detail/_frame_num.txt'

num_classes = 51

trg_train_fmap = root + train + 'SPATIAL_InceptionV3/' + str(args.frame) + '/' + args.layer + '/'
trg_test_fmap = root + test + 'SPATIAL_InceptionV3/' + str(args.frame) + '/' + args.layer + '/'

if not os.path.exists(trg_train_fmap):
    os.makedirs(trg_train_fmap)
if not os.path.exists(trg_test_fmap):
    os.makedirs(trg_test_fmap)


def run(scope, model):
    spatial_path = train_spatial if scope == 'train' else test_spatial
    trg_path = trg_train_fmap if scope == 'train' else trg_test_fmap
    print(spatial_path)
    print(trg_path)
    file_list = preprocess_file_list(os.listdir(spatial_path))
    for file in file_list:
        x = (cv2.resize(cv2.imread(spatial_path + file), (224, 224)))

        x = np.expand_dims(x, axis=0)

        out = model.predict(x)

        txt_name = file.split('.')[0] + '.txt'
        file_to_write = open(trg_path + txt_name, 'w')
        file_to_write.write(','.join([str(i) for i in out[0]]))
        file_to_write.close()


if __name__ == '__main__':
    print(args)
    load_model_root = 'D:/graduation_project/SPATIAL_training/InceptionV3/'
    base_model = load_model(load_model_root + args.model_name)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(args.layer).output)
    run('train', model)
    run('test', model)



