import cv2
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.utils import plot_model
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Input, Flatten, Dropout
from keras.layers import Activation, Concatenate, Conv2D, Multiply
from keras.applications.inception_v3 import InceptionV3
import keras
from keras.applications.inception_v3 import preprocess_input
import tensorflow as tf
from keras import layers
from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np

class RoiExtractor:
    def __init__(self):
        # Model initialization
        self.model = build_image_model()
        self.model.load_weights("https://drive.google.com/file/d/1JqIQHYvW8q5DMjOpcVsfkwxaOZP2CmnL/view?usp=sharing")
        self.heatmap_model = Model(inputs=self.model.inputs, outputs=self.model.layers[-3].output)

        # Outputs
        self.heatmap_1 = None
        self.carpal_img = None
        
        self.heatmap_2 = None
        self.metacarpal_img = None
        
        self.img = None
        self.masked_img = None

    def generate_heatmap(self, img):
        
        # Preprocessing
        i = preprocess_input(img)
        preprocessed_img = np.expand_dims(i, axis=0)

        cbam_output = self.heatmap_model.predict(preprocessed_img)[0]

        #Heatmap generation
        heatmap = np.sum(cbam_output, axis=-1)
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        upsampled_heatmap = cv2.resize(heatmap, (299, 299))
        return upsampled_heatmap


    def get_bounding_box(self, heatmap, threshold=0.7):
        binary_mask = (heatmap > threshold).astype(np.uint8)
        non_zero_indices = np.nonzero(binary_mask)
        if len(non_zero_indices[0]) == 0 or len(non_zero_indices[1]) == 0:
            return None
        min_row, min_col = np.min(non_zero_indices[0]), np.min(non_zero_indices[1])
        max_row, max_col = np.max(non_zero_indices[0]), np.max(non_zero_indices[1])

        bounding_box ={
            'min_row': min_row,
            'min_col': min_col,
            'max_row': max_row,
            'max_col': max_col
        }
        return bounding_box


    def crop_image_by_bounding_box(self, image, bounding_box):
        min_row, min_col = bounding_box['min_row'], bounding_box['min_col']
        max_row, max_col = bounding_box['max_row'], bounding_box['max_col']
        cropped_image = image[min_row:max_row + 1, min_col:max_col + 1, :]
        cropped_resized = cv2.resize(cropped_image, (224, 224))
        return cropped_resized


    def apply_black_rectangle_by_bounding_box(self, image, bounding_box):
        min_row, min_col = bounding_box['min_row'], bounding_box['min_col']
        max_row, max_col = 299, bounding_box['max_col']
        test_img = image.copy()
        test_img = cv2.rectangle(test_img, (min_col, min_row), (max_col, max_row), (0, 0, 0), thickness=cv2.FILLED)
        return test_img

    
    def process_img(self, img): 
        self.img = img
        self.heatmap_1 = self.generate_heatmap(img)
        bounding_box_1 = self.get_bounding_box(self.heatmap_1)
        self.carpal_img = self.crop_image_by_bounding_box(img, bounding_box_1)
        self.masked_img = self.apply_black_rectangle_by_bounding_box(img, bounding_box_1)
        self.heatmap_2 = self.generate_heatmap(self.masked_img)
        bounding_box_2 = self.get_bounding_box(self.heatmap_2)
        self.metacarpal_img = self.crop_image_by_bounding_box(img, bounding_box_2)
    

# Channel Attention Module
def channel_attention_module(x, ratio=8):

    batch,_,_,channel=x.shape
    # shared layers
    l1 = Dense(channel//ratio, activation="relu", use_bias=False)
    l2 = Dense(channel, use_bias= False)

    x1 = GlobalAveragePooling2D()(x)
    x1 = Flatten()(x1)
    x1 = l1(x1)
    x1 = l2(x1)

    x2 = GlobalMaxPooling2D()(x)
    x2 = Flatten()(x2)
    x2 = l1(x2)
    x2 = l2(x2)

    feats = x1 + x2
    feats = Activation("sigmoid")(feats)
    feats = Multiply()([x,feats])

    return feats

# spatical attention module

def spatial_attention_module(x):
    # Average Pooling
    x1 = tf.reduce_mean(x,axis = -1)
    x1 = tf.expand_dims(x1,axis = -1)

    # max pooling
    x2 = tf.reduce_max(x, axis = -1)
    x2 = tf.expand_dims(x2,axis=-1)

    feats = Concatenate()([x1,x2])

    feats = Conv2D(1,kernel_size=7, padding="same",activation="softmax")(feats)
    feats = Multiply()([x,feats])

    return feats

# Cbam module
def cbam(x):
    x = channel_attention_module(x)
    x = spatial_attention_module(x)

    return x


def build_image_model():
    base_model = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(299,299,3)
    )
    for layer in base_model.layers:
        layer.trainable = False
    x = cbam(base_model.output)

    x= GlobalAveragePooling2D()(x)
    predictions = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model
