import numpy as np
import cv2
import random
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg19 import VGG19
from keras.models import Model


def cal_simi_hist_gray(image1, image2, size=(256, 256)):
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    print(np.array(hist1).shape)
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

img = cv2.imread('D:/graduation_project/workspace/dataset/HMDB51/train1/JTM_mc/10/0_0_1.jpg')
img = cv2.resize(img, (224, 224))

print(cal_simi_hist_gray(img, img))

tmp = [1, 1, 1, 2, 2]
print(tmp-0.5)
print(tmp.count(False))

# print(img.shape)
# print(cv2.resize(img, (240, 180)).shape)
# base_model = VGG19(weights='imagenet', include_top=False)
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)
# # for layer in base_model.layers:
# #     print(layer.name)
#
# img = np.expand_dims(img, 0)
# out = model.predict(img)
# print(out.shape)


#

# tmp = np.zeros((3, 2)).tolist()
# for i in range(2):
#     tmp2 = random.sample(tmp, 1)
#     tmp.append(tmp2)
#     print(tmp)
# # tmp2.append(tmp[0])
# # print(np.array(tmp2).shape)