"""
Author: Yifan Wang
Date created: 11/09/2020

"""

import sys
import os
import numpy as np
import glob
import nibabel as nib
from model import get_vc_net

model=get_vc_net(cube_size=(1,128,128,16),
    patch_size_x=384, patch_size_y=256,num_channels_2d=1, activation_2d='relu', final_activation_2d='sigmoid', dropout_2d=0.0,
    initial_learning_rate=0.0001)

model.load_weights('PATH_TO_WEIGHT/WEIGHT_TO_TEST.h5')    ##PATH TO THE WEIGHT

def whole_image_dice_coef(y_true,y_pred,smooth=0):
	intersection = np.sum(y_true*y_pred)
	union = np.sum(y_pred)+ np.sum(y_true)
	dice = np.mean((2. * intersection + smooth)/(union + smooth))
	return dice

with open('/research/work/2d_unet_wsu/Unet/test_ls.txt') as fp:
	line=fp.readlines()
test_ls_ori=['Normal-' + x.strip() for x in line]
print(len(test_ls_ori))
print(test_ls_ori)
             

patch_size=128
patch_size_z=16

# # # ################################################################################################################
dice_total_fuse3d=0
ptch_valid=0

for case in test_ls_ori:
	print(case)
	nifti_ori=nib.load('PATH_TO_DATASET/'+case+'/brain.nii')      
	img=nifti_ori.get_fdata()
	aff=nifti_ori.affine

	mask_ori=nib.load('PATH_TO_DATASET/'+case+'/mask.nii')
	mask=mask_ori.get_fdata()
	gt_ori=nib.load('PATH_TO_DATASET/'+case+'/vessel.nii')
	gt=gt_ori.get_fdata()	
	x_dim, y_dim, z_dim = img.shape
	x,y,z=np.where(mask)
	x_max=max(x)
	x_min=min(x)
	y_max=max(y)
	y_min=min(y)

	num_x=np.int(np.ceil((x_max-x_min) / patch_size))

	num_y=np.int(np.ceil((y_max-y_min) / patch_size))

	prob_mat=np.zeros((1,)+img.shape, dtype=np.float32)
	prob_mat1=np.zeros((1,)+img.shape, dtype=np.float32)

	z=[*range(0,z_dim,patch_size_z)]
	z[-1]=z_dim - patch_size_z

	count=0
	row_blk=3
	cln_blk=2
	lvl=5


	for i in z:
		for j in range(num_x):
			for k in range(num_y):
				patch_start_x=x_min+patch_size*j
				patch_end_x=x_min+patch_size*(j+1)
				patch_start_y=y_min+patch_size*k
				patch_end_y=y_min+patch_size*(k+1)

				if patch_end_x > x_dim:
					patch_end_x = x_max
					patch_start_x = x_max - patch_size
				if patch_end_y > y_dim:
					patch_end_y = y_max
					patch_start_y = y_max - patch_size		

				cnt=0
				arg_cnt=0

				img_patch = img[patch_start_x: patch_end_x, patch_start_y: patch_end_y, i:i+patch_size_z]
				mip_patch = np.zeros((np.size(img_patch,0)*row_blk,np.size(img_patch,1)*cln_blk),dtype=np.int16)
				arg_patch = np.zeros((6, np.size(img_patch,0),np.size(img_patch,1), lvl))

				for p in range(row_blk):
					for q in range(cln_blk):
						if (p==(row_blk-1) and q==(cln_blk-2)):
							mip_patch[np.size(img_patch,0)*p:(np.size(img_patch,0)*(p+1)),np.size(img_patch,1)*q:(np.size(img_patch,1)*(q+1))]=np.amax(img_patch[:,:,cnt:cnt+lvl],axis=2)
							arg_temp=np.argmax(img_patch[:,:,cnt:cnt+lvl],axis=2)
							arg_patch[arg_cnt]=(np.arange(lvl) == arg_temp[...,None])
							arg_cnt=arg_cnt+1
							cnt=cnt+3
						else:
							mip_patch[np.size(img_patch,0)*p:(np.size(img_patch,0)*(p+1)),np.size(img_patch,1)*q:(np.size(img_patch,1)*(q+1))]=np.amax(img_patch[:,:,cnt:cnt+lvl],axis=2)
							arg_temp=np.argmax(img_patch[:,:,cnt:cnt+lvl],axis=2)
							arg_patch[arg_cnt]=(np.arange(lvl) == arg_temp[...,None])
							arg_cnt=arg_cnt+1
							cnt=cnt+2


				net_patch=np.reshape(img_patch,(1, 1,patch_size, patch_size, patch_size_z))
				mip_net_patch=np.reshape(mip_patch,(1,1,384,256))
				arg_net_patch=np.reshape(arg_patch,(1,1,6,128,128,5))

				# ## double smaple for 2-gpu testing, a workaround way if predict patch one by one using model trained by keras.utils.multi_gpu_model 
				### since it only accept batch_size =k*N (int k>=1, N is GPU # used when training), comment the following 3 lines if model is trained on single GPU or multi-GPU via tf.distribute.MirroredStrategy()
				net_patch=np.concatenate((net_patch,net_patch),axis=0)
				mip_net_patch=np.concatenate((mip_net_patch,mip_net_patch),axis=0)
				arg_net_patch=np.concatenate((arg_net_patch,arg_net_patch),axis=0)

				out_list = model.predict([net_patch,mip_net_patch,arg_net_patch])
				prob_mat[:,patch_start_x: patch_end_x, patch_start_y: patch_end_y, i:i+patch_size_z]=out_list[0][0]

				count=count+1
				ptch_valid=ptch_valid+1



	y_pred_bin=prob_mat[0]>0.5
	y_pred_bin=y_pred_bin.astype(float)


	print('saving prediction nifti for...:', case)
	path_target='PATH_TO_SAVE/'+case+'_pred.nii'
	new_nifti = nib.Nifti1Image(y_pred_bin, aff) 
	nib.save(new_nifti, path_target)	


	dice=whole_image_dice_coef(gt,y_pred_bin,smooth=0)
	dice_total_fuse3d=dice_total_fuse3d+dice
	print('fuse3d: \n',dice)
	print('valid patch # is: \n', count)
			
print('average fuse3d dice: \n',dice_total_fuse3d/len(test_ls_ori))
