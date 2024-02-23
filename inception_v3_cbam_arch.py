#importing libraries
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.utils import plot_model
from keras.layers import BatchNormalization
from keras.layers import AveragePooling2D
import tensorflow as tf
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, Input
from keras.layers import Activation, Concatenate, Conv2D, Multiply


Img_input = Input(shape= (299,299,3))
channel_axis = 3

# In inception v3 architecture, Every convolutin layer had batch normalization and relu activation function
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1,1)):
    x=Conv2D(filters, (num_row, num_col), strides=strides, padding=padding)(x)
    x=BatchNormalization(axis=3, scale=False)(x)
    x=Activation('relu')(x)
    return x

#Inception Block-A
def inc_block_a(x):
    branch1x1 = conv2d_bn(x,64,1,1)
    
    branch3x3 = conv2d_bn(x,48,1,1)
    branch3x3 = conv2d_bn(branch3x3,64,3,3)
    
    branch3x3db1 = conv2d_bn(x,64,1,1)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)
    
    branch_pool = AveragePooling2D((3,3), strides=(1,1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,32,1,1)
    
    x = Concatenate(axis= channel_axis)([branch1x1,branch3x3,branch3x3db1,branch_pool])
    return x

#Reduction Block-A
def reduction_block_a(x):
    branch3x3 = conv2d_bn(x,384,3,3,strides=(2,2),padding='valid')
    
    branch3x3db1 = conv2d_bn(x,64,1,1)
    branch3x3db1 = conv2d_bn(branch3x3db1,96,3,3)
    branch3x3db1 = conv2d_bn(branch3x3db1, 96,3,3,strides=(2,2),padding='valid')
    
    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)
    
    x=Concatenate(axis=channel_axis)([branch3x3,branch3x3db1,branch_pool])
    
    return x

#Inception Block-B
def inc_block_b(x):
    branch1x1 = conv2d_bn(x,192,1,1)
    
    branch7x7 = conv2d_bn(x,128,1,1)
    branch7x7 = conv2d_bn(branch7x7,128,1,7)
    branch7x7 = conv2d_bn(branch7x7,192,7,1)
    
    branch7x7db1 = conv2d_bn(x,128,1,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,128,7,1)
    branch7x7db1 = conv2d_bn(branch7x7db1,128,1,7)  
    branch7x7db1 = conv2d_bn(branch7x7db1,128,7,1) 
    branch7x7db1 = conv2d_bn(branch7x7db1,192,1,7) 
    
    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,192,1,1)
    
    x = Concatenate(axis = channel_axis)([branch1x1,branch7x7,branch7x7db1,branch_pool])
    
    return x

#Reduction Block-B
def reduction_block_b(x):
    branch3x3 = conv2d_bn(x,192,1,1)
    branch3x3 = conv2d_bn(branch3x3, 320,3,3,strides=(2,2),padding='valid')
    
    branch7x7x3 = conv2d_bn(x,192,1,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,1,7)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,7,1)
    branch7x7x3 = conv2d_bn(branch7x7x3,192,3,3,strides=(2,2),padding='valid')
    
    branch_pool = MaxPooling2D((3,3),strides=(2,2))(x)
    
    x = Concatenate( axis=channel_axis)([branch3x3,branch7x7x3,branch_pool])
    
    return x

#Inception Block-C
def inc_block_c(x):
    branch1x1 = conv2d_bn(x,320,1,1)
    
    branch3x3 = conv2d_bn(x,384,1,1)
    branch3x3_1 = conv2d_bn(branch3x3,384,1,3)
    branch3x3_2 = conv2d_bn(branch3x3,384,3,1)
    branch3x3 = Concatenate(axis=channel_axis)([branch3x3_1,branch3x3_2])
    
    branch3x3db1 = conv2d_bn(x,448,1,1)
    branch3x3db1 = conv2d_bn(branch3x3db1,384,3,3)
    branch3x3db1_1 = conv2d_bn(branch3x3db1,384,1,3)
    branch3x3db1_2 = conv2d_bn(branch3x3db1,384,3,1)
    branch3x3db1 = Concatenate(axis= channel_axis)([branch3x3db1_1,branch3x3db1_2])
    
    branch_pool = AveragePooling2D((3,3),strides=(1,1),padding='same')(x)
    branch_pool = conv2d_bn(branch_pool,192,1,1)
    
    x= Concatenate(axis=channel_axis)([branch1x1, branch3x3, branch3x3db1,branch_pool])
    
    return x


# Channel Attention Module
# Channel Attention Module
def channel_attention_module(x, ratio=8):
    batch,_,_,channel=x.shape
    # shared layers
    l1 = Dense(channel//ratio, activation="relu", use_bias=False)
    l2 = Dense(channel, use_bias= False)
    
    x1 = GlobalAveragePooling2D()(x)
    x1 = l1(x1)
    x1 = l2(x1)
    
    x2 = GlobalMaxPooling2D()(x)
    x2 = l1(x2)
    x2 = l2(x2)
    
    feats = x1 + x2
    feats = Activation("sigmoid")(feats)
    feats = Multiply()([x,feats])
    
    return feats

# Spatial Attention Module
# spatical attention module

def spatial_attention_module(x):
    # Average Pooling
    x1 = tf.reduce_mean(x,axis = -1)
    x1 = tf.expand_dims(x1,axis = -1)
    
    # max pooling
    x2 = tf.reduce_max(x, axis = -1)
    x2 = tf.expand_dims(x2,axis=-1)
    
    feats = Concatenate()([x1,x2])
    
    feats = Conv2D(1,kernel_size=7, padding="same",activation="sigmoid")(feats)
    feats = Multiply()([x,feats])
    
    return feats

# CBAM (Convolutional Block Attention Mechanism) Modules
def cbam(x):
    x = channel_attention_module(x)
    x = spatial_attention_module(x)
    
    return x

# Building model layer by layer integrating CBAM
def inception_cbam_model(Img_input):
    x = conv2d_bn(Img_input, 32,3,3,strides=(2,2),padding='valid')
    x = conv2d_bn(x,32,3,3,padding='valid')
    x = conv2d_bn(x,64,3,3)
    x = MaxPooling2D((3,3), strides=(2,2))(x)

    x = conv2d_bn(x,80,1,1,padding='valid')
    x = conv2d_bn(x,192,3,3,padding='valid')
    x=MaxPooling2D((3,3),strides=(2,2))(x)

    x = inc_block_a(x)
    x = inc_block_a(x)
    x = inc_block_a(x)

    x = reduction_block_a(x)

    x = inc_block_b(x)
    x = inc_block_b(x)
    x = inc_block_b(x)
    x = inc_block_b(x)

    x = reduction_block_b(x)

    x = inc_block_c(x)
    x = inc_block_c(x)

    x = AveragePooling2D()(x)
    
    base_model = x

    model = cbam(base_model)
    
    # Add additional layers 
#     model.add(layers.Flatten())
#     model.add(layers.Dense(256, activation='relu'))
#     model.add(layers.Dropout(0.5))
#     model.add(layers.Dense(1, activation='linear'))  # regression task for bone age
    # create model
    inputs = Img_input
    model = Model(inputs,model,name='inception_v3')
    
    return model

image_model = inception_cbam_model(Img_input)
print(image_model.summary())

# Compile the model
image_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])



