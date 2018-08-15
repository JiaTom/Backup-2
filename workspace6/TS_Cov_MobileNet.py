import os
import numpy as np

# Keras Core
from keras import models

# Backend
from keras import backend as K
from keras import backend

# layers
from keras.layers import CuDNNGRU,CuDNNLSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers import Input, Dropout, Dense, Flatten, Activation, Concatenate
from keras.layers import ZeroPadding2D, Conv2D, BatchNormalization, DepthwiseConv2D
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, MaxPool2D, Reshape
from keras.activations import relu


def relu6(x):
    return backend.relu(x, max_value=6)


def T_conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = TimeDistributed(ZeroPadding2D(padding=((0, 1), (0, 1))), name='conv1_pad')(inputs)
    x = TimeDistributed(Conv2D(filters, kernel,
                               padding='valid',
                               use_bias=False,
                               strides=strides),
                        name='conv1')(x)
    x = TimeDistributed(BatchNormalization(axis=channel_axis),
                        name='conv1_bn')(x)
    return TimeDistributed(Activation(relu6), name='conv1_relu')(x)
    # return TimeDistributed(Activation(relu(x,max_value=6)), name='conv1_relu')(x)


def T_depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                           depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = TimeDistributed(ZeroPadding2D(((0, 1), (0, 1))),
                            name='conv_pad_%d' % block_id)(inputs)
    x = TimeDistributed(DepthwiseConv2D((3, 3),
                                        padding='same' if strides == (1, 1) else 'valid',
                                        depth_multiplier=depth_multiplier,
                                        strides=strides,
                                        use_bias=False),
                        name='conv_dw_%d' % block_id)(x)
    x = TimeDistributed(BatchNormalization(axis=channel_axis),
                        name='conv_dw_%d_bn' % block_id)(x)
    x = TimeDistributed(Activation(relu6), name='conv_dw_%d_relu' % block_id)(x)
    # x = TimeDistributed(Activation(relu(max_value=6)), name='conv_dw_%d_relu' % block_id)(x)

    x = TimeDistributed(Conv2D(pointwise_conv_filters, (1, 1),
                               padding='same',
                               use_bias=False,
                               strides=(1, 1)),
                        name='conv_pw_%d' % block_id)(x)
    x = TimeDistributed(BatchNormalization(axis=channel_axis),
                        name='conv_pw_%d_bn' % block_id)(x)

    return TimeDistributed(Activation(relu6), name='conv_pw_%d_relu' % block_id)(x)
    # return TimeDistributed(Activation(relu(max_value=6)), name='conv_pw_%d_relu' % block_id)(x)


def TS_ConvMobileNet(input_shape=None,
                     alpha=1.0,
                     depth_multiplier=1,
                     dropout=1e-3,
                     pooling=None,
                     classes=1000):
    rows = None
    #     filter_num = [32,64,128,256,512,1024]
    filter_num = [16, 32, 64, 128, 256, 512]
    if input_shape:
        if len(input_shape) == 4 and input_shape[0] == 128 and input_shape[3] in [1,3]:
            img_input = Input(shape=input_shape)
        else:
            print('Please check the entered input shape.\n',
                  'The first input (timestep) must be 128\n',
                  'The last input must in 1 or 3')
            raise ValueError
    else:
        img_input = Input(shape=(128, 64, 64, 1))

    x = T_conv_block(img_input, filter_num[0], alpha, strides=(2, 2))
    x = T_depthwise_conv_block(x, filter_num[1], alpha, depth_multiplier, block_id=1)

    x = T_depthwise_conv_block(x, filter_num[2], alpha, depth_multiplier,
                               strides=(2, 2), block_id=2)
    x = T_depthwise_conv_block(x, filter_num[2], alpha, depth_multiplier, block_id=3)

    x = T_depthwise_conv_block(x, filter_num[3], alpha, depth_multiplier,
                               strides=(2, 2), block_id=4)
    x = T_depthwise_conv_block(x, filter_num[3], alpha, depth_multiplier, block_id=5)

    x = T_depthwise_conv_block(x, filter_num[4], alpha, depth_multiplier,
                               strides=(2, 2), block_id=6)
    x = T_depthwise_conv_block(x, filter_num[4], alpha, depth_multiplier, block_id=7)
    x = T_depthwise_conv_block(x, filter_num[4], alpha, depth_multiplier, block_id=8)
    x = T_depthwise_conv_block(x, filter_num[4], alpha, depth_multiplier, block_id=9)
    x = T_depthwise_conv_block(x, filter_num[4], alpha, depth_multiplier, block_id=10)
    x = T_depthwise_conv_block(x, filter_num[4], alpha, depth_multiplier, block_id=11)

    x = T_depthwise_conv_block(x, filter_num[5], alpha, depth_multiplier,
                               strides=(2, 2), block_id=12)
    x = T_depthwise_conv_block(x, filter_num[5], alpha, depth_multiplier, block_id=13)

    if backend.image_data_format() == 'channels_first':
        shape = (1, int(filter_num[5] * alpha))
        channelAxis = 1

    else:
        shape = (int(filter_num[5] * alpha), 1)
        channelAxis = -1

    if pooling == 'max':
        x = TimeDistributed(GlobalMaxPooling2D(), name="MAX_pool")(x)
    else:
        x = TimeDistributed(GlobalAveragePooling2D(), name="AVG_pool")(x)

    x = TimeDistributed(Reshape(shape, name='reshape_1'), name="reshape1")(x)

    # TODO: Conv
    # print(channelAxis)
    TimeFilter_num = [256, 256, 256]
    filter_step = [4, 8, 16] # [2, 16, 32]
    # filter_step = [2, 16, 32]

    conv1 = Conv2D(TimeFilter_num[0], kernel_size=(filter_step[0], int(filter_num[5] * alpha)),
                   padding='valid', name='time_conv_%d_step' % filter_step[0])(x)
    conv1 = BatchNormalization(axis=channelAxis, name='time_bn_%d_step' % filter_step[0])(conv1)
    conv1 = Activation(backend.relu, name='time_relu_%d_step' % filter_step[0])(conv1)
    conv1 = MaxPool2D(pool_size=(128 - filter_step[0] + 1, 1), strides=(1, 1),
                      padding='valid', name='time_Max_%d_step' % filter_step[0])(conv1)

    conv2 = Conv2D(TimeFilter_num[1], kernel_size=(filter_step[1], int(filter_num[5] * alpha)),
                   padding='valid', name='time_conv_%d_step' % filter_step[1])(x)
    conv2 = BatchNormalization(axis=channelAxis, name='time_bn_%d_step' % filter_step[1])(conv2)
    conv2 = Activation(backend.relu, name='time_relu_%d_step' % filter_step[1])(conv2)
    conv2 = MaxPool2D(pool_size=(128 - filter_step[1] + 1, 1), strides=(1, 1),
                      padding='valid', name='time_Max_%d_step' % filter_step[1])(conv2)

    conv3 = Conv2D(TimeFilter_num[2], kernel_size=(filter_step[2], int(filter_num[5] * alpha)),
                   padding='valid', name='time_conv_%d_step' % filter_step[2])(x)
    conv3 = BatchNormalization(axis=channelAxis, name='time_bn_%d_step' % filter_step[2])(conv3)
    conv3 = Activation(backend.relu, name='time_relu_%d_step' % filter_step[2])(conv3)
    conv3 = MaxPool2D(pool_size=(128 - filter_step[2] + 1, 1), strides=(1, 1),
                      padding='valid', name='time_Max_%d_step' % filter_step[2])(conv3)

    concatenated_tensor = Concatenate(axis=1)([conv1, conv2, conv3])
    x = Flatten(name='flatten_concat')(concatenated_tensor)
    x = Dropout(dropout)(x)
    x = Dense(units=256,activation='relu')(x)
    x = Dense(units=classes, activation='softmax')(x)

    # get inputs shape
    inputs = img_input
    print(inputs)

    # Create model.
    model = models.Model(inputs, x, name='mobilenet_%0.2f_%s' % (alpha, rows))

    return model

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = TS_ConvMobileNet(input_shape=(128,64,64,1),
                             alpha=1.0,
                             depth_multiplier=1,
                             dropout=0.1,
                             pooling='avg',
                             classes=3)
    print(model.summary())
