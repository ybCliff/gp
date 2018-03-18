from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, GlobalAveragePooling2D, GlobalAveragePooling1D, Conv3D
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras import backend as K
import keras
from keras.applications.inception_v3 import InceptionV3

def Base_cnn(include_top=True, weights=None,
          input_tensor=None, input_shape=None,
          classes=51):
    # Determine proper input shape
    input_shape = (224, 224, 3)
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    # Block 1
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x) #(None, 112, 112, 8)
    x = Dropout(0.5)(x)

    # Block 2
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x) #(None, 56, 56, 32)
    x = Dropout(0.5)(x)

    # Block 3
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x) #(None, 28, 28, 64)
    x = Dropout(0.5)(x)

    # Block 4
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x) #(None, 14, 14, 128)
    x = Dropout(0.5)(x)

    # Block 5
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)  # (None, 7, 7, 256)
    x = Dropout(0.5)(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    if include_top:
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='JDM_cnn')

    if weights is not None:
        model.load_weights(weights)

    return model

def Two_input_shared_cnn(include_top=True, weights=None,
          input_shape=(224, 224, 3),
          classes=51):
    base_model = Base_cnn(include_top=False)
    img_a = Input(shape=input_shape)
    img_b = Input(shape=input_shape)

    out_a = base_model(img_a)
    out_b = base_model(img_b)

    x = keras.layers.concatenate([out_a, out_b], name='concatenate')
    if include_top:
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model([img_a, img_b], x)
    if weights is not None:
        model.load_weights(weights)

    return model

def My_InceptionV3(include_top=True, weights=None,
          input_shape=(224, 224, 3),
          classes=51):
    base_model = InceptionV3(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)


    if include_top:
        x = Dense(1024, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(512, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    model = Model(base_model.input, x)
    if weights is not None:
        model.load_weights(weights)

    return model


def preprocess_file_list(lst):
    while('.jpg' not in lst[len(lst)-1]):
        lst.pop()
    return lst