"""
Author: Yifan Wang
Date created: 11/09/2020
This file is partially refered to https://github.com/prediction2020/unet-vessel-segmentation and https://github.com/ellisdg/3DUnetCNN
"""

from keras.models import Model
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Input, UpSampling2D, concatenate, BatchNormalization, Reshape, Multiply, Add, Maximum, Average
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, PReLU, Deconvolution3D, Add
from keras import backend as K
from keras.layers import Layer
from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
from model.metric import dice_coefficient_loss, dice_coefficient
import glob
import os
from keras.utils import multi_gpu_model


K.set_image_data_format("channels_first")
##############################################3D UNET################################################################
def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, deconvolution=False,
                  depth=4, n_base_filters=32, batch_normalization=False, activation_name="sigmoid",ax=1):

    K.set_image_data_format("channels_first")
    inputs_3d = Input(input_shape,name='patch_3d')
    current_layer = inputs_3d
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[ax])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=ax)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[ax],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[ax],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    return inputs_3d, current_layer

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False,ax=1):

    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)

    if batch_normalization:
        layer = BatchNormalization(axis=ax)(layer)
    elif instance_normalization:
        try:
            #from keras_contrib.layers.normalization import InstanceNormalization
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=ax)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)

def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

##################################   Unprojection Layer ##############################################################################

class Mip_arg(Layer):
    def __init__(self):
        super(Mip_arg, self).__init__()
    def build(self, input_shape):
        super(Mip_arg, self).build(input_shape) 
    def call(self,inputs):
        seg=tf.reshape(inputs,(-1,32,3,128,2,128))
        seg=K.permute_dimensions(seg, (0,1,2,4,3,5))
        seg=tf.reshape(seg,(-1,32,6,128,128))
        seg_tile=tf.stack([seg,seg,seg,seg,seg],axis=-1) #5-sliced mip
        return seg_tile

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 32,6,128,128,5)


class Unproject(Layer):
    def __init__(self):      
        super(Unproject, self).__init__()
    def build(self, input_shape):
        super(Unproject, self).build(input_shape) 
    def call(self,inputs):
        a=tf.unstack(inputs,axis=-4)

        a0=tf.pad(a[0],tf.constant([[0,0],[0,0],[0,0],[0,0],[0,11]]))
        a1=tf.pad(a[1],tf.constant([[0,0],[0,0],[0,0],[0,0],[2,9]]))
        a2=tf.pad(a[2],tf.constant([[0,0],[0,0],[0,0],[0,0],[4,7]]))
        a3=tf.pad(a[3],tf.constant([[0,0],[0,0],[0,0],[0,0],[6,5]]))
        a4=tf.pad(a[4],tf.constant([[0,0],[0,0],[0,0],[0,0],[8,3]]))
        a5=tf.pad(a[5],tf.constant([[0,0],[0,0],[0,0],[0,0],[11,0]]))

        a_cancat=tf.stack([a0,a1,a2,a3,a4,a5],axis=-1)
        a_max=tf.reduce_max(a_cancat,axis=-1)

        return a_max
    def compute_output_shape(self, input_shape):
        return (input_shape[0],32,128,128,16)

##################################2D UNET ##############################################################################        

def conv_block(m, num_kernels, kernel_size, strides, padding, activation, dropout, data_format, bn):

    n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
                      data_format=data_format)(m)
    n = BatchNormalization()(n) if bn else n
    n = Dropout(dropout)(n)
    n = Convolution2D(num_kernels, kernel_size, strides=strides, activation=activation, padding=padding,
                      data_format=data_format)(n)
    n = BatchNormalization()(n) if bn else n
    return n


def up_concat_block(m, concat_channels, pool_size, concat_axis, data_format):
    n = UpSampling2D(size=pool_size, data_format=data_format)(m)
    n = concatenate([n, concat_channels], axis=concat_axis)
    return n


def unet_model_2d(patch_size_x,patch_size_y, num_channels, activation, final_activation, dropout,
             kernel_size=(3, 3), pool_size=(2, 2), strides=(1, 1), num_kernels=None, concat_axis=1,
             data_format='channels_first', padding='same', bn=False):

    if num_kernels is None:
        # num_kernels = [64, 128, 256, 512, 1024]
        num_kernels = [32, 64, 128, 256, 512]
    # specify the input shape
    inputs_2d = Input((num_channels, patch_size_x, patch_size_y),name='mip')           
    arg=Input((num_channels,6,128,128,5),name='arg')

    # level 0
    conv_0_down = conv_block(inputs_2d, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_0 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_0_down)

    # level 1
    conv_1_down = conv_block(pool_0, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_1 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_1_down)

    # level 2
    conv_2_down = conv_block(pool_1, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_2 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_2_down)

    # level 3
    conv_3_down = conv_block(pool_2, num_kernels[3], kernel_size, strides, padding, activation, dropout, data_format,
                             bn)
    pool_3 = MaxPooling2D(pool_size=pool_size, data_format=data_format)(conv_3_down)

    # level 4
    conv_4 = conv_block(pool_3, num_kernels[4], kernel_size, strides, padding, activation, dropout, data_format, bn)


    # level 3
    concat_3 = up_concat_block(conv_4, conv_3_down, pool_size, concat_axis, data_format)
    conv_3_up = conv_block(concat_3, num_kernels[3], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)

    # level 2
    concat_2 = up_concat_block(conv_3_up, conv_2_down, pool_size, concat_axis, data_format)
    conv_2_up = conv_block(concat_2, num_kernels[2], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)

    # level 1
    concat_1 = up_concat_block(conv_2_up, conv_1_down, pool_size, concat_axis, data_format)
    conv_1_up = conv_block(concat_1, num_kernels[1], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)

    # level 0
    concat_0 = up_concat_block(conv_1_up, conv_0_down, pool_size, concat_axis, data_format)
    conv_0_up = conv_block(concat_0, num_kernels[0], kernel_size, strides, padding, activation, dropout, data_format,
                           bn)
    #final segmentation from 2d unet
    final_conv = Convolution2D(1, 1, strides=strides, activation=final_activation, padding=padding,
                               data_format=data_format, name='seg_2d')(conv_0_up)                        

    final_reshape = Mip_arg()(conv_0_up)

    back3D=Multiply()([final_reshape,arg])

    final_fea=Unproject()(back3D)
    return inputs_2d,arg, final_conv, final_fea
####################################################################################################################################

def get_vc_net(cube_size,
    patch_size_x=384, patch_size_y=256,num_channels_2d=1, activation_2d='relu', final_activation_2d='sigmoid', dropout_2d=0.0,
    initial_learning_rate=0.0001):

    input_3d,fea_3d=unet_model_3d(input_shape=cube_size)
    input_2d,arg_2d,final_conv,fea_2d=unet_model_2d(patch_size_x=patch_size_x, patch_size_y=patch_size_y,num_channels=num_channels_2d, activation=activation_2d, final_activation=final_activation_2d, dropout=dropout_2d) 
    
    ####################   Final Fusion Stage     ##############################################################
    fea_fuse=concatenate([fea_3d, fea_2d], axis=1,name='final_concat')
    fea_fuse=Conv3D(32, (1, 1, 1))(fea_fuse)
    fea_fuse=Activation('relu')(fea_fuse)
    res_fuse=Conv3D(1, (1, 1, 1))(fea_fuse)
    res_fuse=Activation('sigmoid',name='seg_3d')(res_fuse)
    ############################################################################################################

    model = Model(inputs=[input_3d,input_2d,arg_2d], outputs=[res_fuse,final_conv])
    loss_funcs={
    'seg_3d': dice_coefficient_loss,
    'seg_2d': dice_coefficient_loss
    }
    loss_weights={
    'seg_3d':5.0,     
    'seg_2d':1.0
    }
    metrics={
    'seg_3d':dice_coefficient,
    'seg_2d':dice_coefficient
    }

    initial_learning_rate=initial_learning_rate
    model=multi_gpu_model(model, gpus=2)
    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=loss_funcs, metrics=metrics,loss_weights=loss_weights)

    model.summary()
    return model

############################################################END OF THE FUNCTIONS & CLASSES##########################################################################

def main():
    model=get_vc_net(cube_size=(1,128,128,16),
    patch_size_x=384, patch_size_y=256,num_channels_2d=1, activation_2d='relu', final_activation_2d='sigmoid', dropout_2d=0.0,
    initial_learning_rate=0.00001)

if __name__ == "__main__":
    main()


