import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train = X_train_orig/255
X_test = X_test_orig/255

Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

image_dimensions = X_train.shape[1:]


def model(input_shape):
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2,2),name='max_pool')(X)
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)
    model = Model(inputs=X_input, outputs=X, name='HappyModel')
    return model


def HappyModel(input_shape):
    model_output = model(input_shape)
    return model_output


happyModel = HappyModel(image_dimensions)

happyModel.compile(optimizer='adam', loss='binary_crossentropy')
happyModel.fit(X_train, Y_train, validation_split=0.2, epochs=30)

preds = happyModel.predict(X_test)

print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))