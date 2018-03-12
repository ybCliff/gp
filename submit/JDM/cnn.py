from keras.applications.inception_v3 import InceptionV3
from Model_and_funcs import preprocess_file_list
import argparse
import numpy as np
import os, time, cv2, random
import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import load_model

parser = argparse.ArgumentParser(description='')
parser.add_argument('--split', type=int, default=1)
parser.add_argument('--folder', type=str, default='JDM_mc/10/')
args = parser.parse_args()

root = 'D:/graduation_project/workspace/dataset/HMDB51/'
train = 'train'+str(args.split)+'/'
test = 'test'+str(args.split)+'/'
path_train = root + train + args.folder
path_test = root + test + args.folder
num_classes = 51

save_model_root = 'D:/graduation_project/JDM_training/InceptionV3/'
if not os.path.exists(save_model_root):
    os.makedirs(save_model_root)

def load_data(ori_file_list, batch_size, index, path):
    begin_time = time.time()
    begin = index * batch_size
    end = min([(index + 1) * batch_size, len(ori_file_list)])
    file_list = ori_file_list[begin:end]
    x = []
    y = []
    for file in file_list:
        tmpx = cv2.imread(path + file)
        tmpy = int(file.split('.')[0].split('_')[2]) - 1
        tmpy = keras.utils.to_categorical(tmpy, num_classes)
        x.append(tmpx)
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
    if first:
        base_model = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)

        for layer in model.layers:
            layer.trainable = True
    else:
        model = load_model(model_path)
    return model

if __name__ == '__main__':
    # data = cv2.imread(path_train+'0_0_1.jpg')
    # data = np.expand_dims(data, 0)
    # out = model.predict(data)
    # print(out, out.shape)
    load_model_path = save_model_root + ''
    model = get_model(True, 'D:/graduation_project/JDM_cnn/e1_spe1_round2.h5')
    for layer in model.layers:
        if layer.trainable:
            print(layer.name)
    # model = Base_cnn()
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    spe = 5
    epochs_per_round = 350
    round = 3
    batch_size = 20
    ori_file_list = preprocess_file_list(os.listdir(path_train))
    random.shuffle(ori_file_list)

    for e in range(0, 2):
        for i in range(round):
            x_train, y_train = load_data(ori_file_list, epochs_per_round * batch_size, i, path_train)
            begin_time = time.time()
            history = model.fit_generator(generate_batch_traindata_random(x_train, y_train, batch_size),
                samples_per_epoch=spe, epochs=epochs_per_round,
                # validation_data=generate_batch_testdata_random(batch_size),
                # validation_steps=1,
                verbose=1)
            model_name = 'e' + str(e) + '_spe' + str(spe) + '_round' + str(i) + '.h5'
            model.save(save_model_root + model_name)
            print('\033[1;33;44m', 'echo:', e, 'round:', i, 'time:', time.time() - begin_time, '\033[0m')
