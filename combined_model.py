from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.utils import plot_model
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Input, Flatten, Dropout
from keras.layers import Activation, Concatenate, Conv2D, Multiply
from keras.applications.resnet import ResNet50
import keras
from keras.applications.inception_v3 import preprocess_input
from updated_resnet_1 import ResNet50

def build_regression_model():
    base_model_carpal = ResNet50(input_shape = (224, 224, 3))
    base_model_carpal.load_weights(r'F:\python\web_streamlit\ALL_IN_ONE\model_weights\model_weights\resnet\carpal_weights_30_epoch_wd_2.h5')
    
    base_model_metacarpal = ResNet50(input_shape = (224, 224, 3))
    base_model_metacarpal.load_weights(r'F:\python\web_streamlit\ALL_IN_ONE\model_weights\model_weights\resnet\metacarpal_weights_30_epoch_wd_2.h5')
    
    gender_input = Input(shape=(1,), name='gender_input')
    for layer in base_model_carpal.layers:
        layer._name = 'carpal_' + layer.name
    for layer in base_model_metacarpal.layers:
        layer._name = 'metacarpal_' + layer.name

    for layer in base_model_carpal.layers[:-9]:
        layer.trainable = False
    for layer in base_model_metacarpal.layers[:-9]:
        layer.trainable = False

    x1 = base_model_carpal.layers[-4].output
    x2 = base_model_metacarpal.layers[-4].output

    # gender_weight = tf.Variable(initial_value=1.0, trainable=True, name='gender_weight')
    # gender_input_weighted = tf.multiply(gender_input, gender_weight)
    x = Concatenate()([x1, x2, gender_input])

    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    predictions = Dense(1)(x) 

    model = Model(inputs=(base_model_carpal.input, base_model_metacarpal.input, gender_input), outputs=predictions)

    return model