"""
Author: Yifan Wang
Date created: 11/09/2020

"""

import tensorflow as tf
import numpy as np
import glob
from model.training_func import fit_vcnet
import os
from model import get_vc_net


config=dict()
config['model_file'] = os.path.abspath('wgt_test.h5')
config['batch_size'] =2
config['learning_rate']=0.0001
config['epoch']=50

config['data_path']='/research/work/2d_unet_wsu/Unet/TubeTK_skel/original_data/github_Patch/'

############################################################################################################
#load the data
###############################     training dataset    ####################################################
train_data_path=config['data_path']+'train/'
train_id_ls=glob.glob(train_data_path+'*_img.npy')
print(train_id_ls)
train_img_ls=[]
train_arg_ls=[]
train_label_ls=[]
train_mip_ls=[]
train_mipgt_ls=[]

for i in range(len(train_id_ls)):
    train_img=np.load(train_id_ls[i])
    train_lbl=np.load(train_id_ls[i][:-7]+'label.npy')
    train_lbl_aft=train_lbl.astype('int8')

    train_arg=np.load(train_id_ls[i][:-7]+'arg.npy')

    train_mip=np.load(train_id_ls[i][:-7]+'mip.npy')

    train_mipgt=np.load(train_id_ls[i][:-7]+'mipgt.npy')


    train_img_ls.append(train_img)
    train_label_ls.append(train_lbl_aft)
    train_mip_ls.append(train_mip)
    train_arg_ls.append(train_arg)
    train_mipgt_ls.append(train_mipgt)

train_img_arr=np.vstack(train_img_ls)
train_label_arr=np.vstack(train_label_ls)
train_arg_arr=np.vstack(train_arg_ls)
train_mip_arr=np.vstack(train_mip_ls)
train_mipgt_arr=np.vstack(train_mipgt_ls)

train_img_arr=train_img_arr[:,np.newaxis,:,:,:]
train_label_arr=train_label_arr[:,np.newaxis,:,:,:]
train_mip_arr=train_mip_arr[:,np.newaxis,:,:]
train_mipgt_arr=train_mipgt_arr[:,np.newaxis,:,:]
train_arg_arr=train_arg_arr[:,np.newaxis,:,:,:,:]
print('Training Data loaded :)')

############validation dataset##############################################
val_data_path=config['data_path']+'val/'
val_id_ls=glob.glob(val_data_path+'*_img.npy')
print(val_id_ls)
val_img_ls=[]
val_arg_ls=[]
val_label_ls=[]
val_mip_ls=[]
val_mipgt_ls=[]

for i in range(len(val_id_ls)):
    val_img=np.load(val_id_ls[i])

    val_lbl=np.load(val_id_ls[i][:-7]+'label.npy')
    val_lbl_aft=val_lbl.astype('int8')

    val_arg=np.load(val_id_ls[i][:-7]+'arg.npy')

    val_mip=np.load(val_id_ls[i][:-7]+'mip.npy')

    val_mipgt=np.load(val_id_ls[i][:-7]+'mipgt.npy')


    val_img_ls.append(val_img)
    val_label_ls.append(val_lbl_aft)
    val_mip_ls.append(val_mip)
    val_arg_ls.append(val_arg)
    val_mipgt_ls.append(val_mipgt)


val_img_arr=np.vstack(val_img_ls)
val_label_arr=np.vstack(val_label_ls)
val_arg_arr=np.vstack(val_arg_ls)
val_mip_arr=np.vstack(val_mip_ls)
val_mipgt_arr=np.vstack(val_mipgt_ls)


val_img_arr=val_img_arr[:,np.newaxis,:,:,:]
val_label_arr=val_label_arr[:,np.newaxis,:,:,:]
val_mip_arr=val_mip_arr[:,np.newaxis,:,:]
val_mipgt_arr=val_mipgt_arr[:,np.newaxis,:,:]
val_arg_arr=val_arg_arr[:,np.newaxis,:,:,:,:]

print('Validation Data loaded :)')




##################             Load model and train!     ####################################################################
model=get_vc_net(cube_size=(1,128,128,16),                 
    patch_size_x=384, patch_size_y=256,num_channels_2d=1, activation_2d='relu', final_activation_2d='sigmoid', dropout_2d=0.0,
    initial_learning_rate=config['learning_rate'])

fit_vcnet(model,config["model_file"],
    {'patch_3d':train_img_arr,
    'mip':train_mip_arr,
    'arg':train_arg_arr},

    {'seg_3d':train_label_arr,
    'seg_2d':train_mipgt_arr},

    config["batch_size"],


    epochs=config['epoch'],


    val_data=[
    {'patch_3d':val_img_arr,
    'mip':val_mip_arr,
    'arg':val_arg_arr},

    {'seg_3d':val_label_arr,
    'seg_2d':val_mipgt_arr},
    ],
    initial_learning_rate=config['learning_rate'])