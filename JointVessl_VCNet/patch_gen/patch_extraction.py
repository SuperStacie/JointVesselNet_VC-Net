"""
File name: patch_extraction.py
Author: Yifan Wang
Date created: 11/09/2020
This file is partially refered to https://github.com/prediction2020/unet-vessel-segmentation
"""

import os
import sys
import numpy as np 
import nibabel as nib
import argparse

# Specify the what patches to be extracted (you may sample more in a training case image)
parser = argparse.ArgumentParser()
parser.add_argument('--flag', default='train', type=str, help='train or val')

opt=parser.parse_args()
flag=opt.flag

config = dict()
config['patch_size'] = (128,128,16)
config['train_patch'] = 100
config['val_patch'] = 60
config['vessel_focus']=0.6
config['data_path']='/PATH_TO_DATASET/'                     #SPECIFY THE PATH HERE
config['save_path']='../Data/' 

if flag=='train':
    with open('./'+'train'+'_ls.txt') as fp:                #SPECIFY THE PATH HERE
        line=fp.readlines()
    ls_ori=['Normal-' + x.strip() for x in line]
    nr_patches=config['train_patch']
    nr_vessel_patches = int(nr_patches * config['vessel_focus'])
    nr_empty_patches = nr_patches - nr_vessel_patches


if flag=='val':
    with open('./'+'val'+'_ls.txt') as fp:                  #SPECIFY THE PATH HERE
        line=fp.readlines()
    ls_ori=['Normal-' + x.strip() for x in line]    
    nr_patches=config['val_patch']
    nr_vessel_patches = int(nr_patches * config['vessel_focus'])
    nr_empty_patches = nr_patches - nr_vessel_patches


for case in ls_ori:

    # load image, mask and label from nifti file
    print('> Loading image...')
    img_mat = nib.load(config['data_path'] + case + '/brain.nii').get_fdata()
    print('> Loading mask...')
    mask_mat = nib.load(config['data_path'] + case + '/mask.nii').get_fdata()  
    print('> Loading label...')
    label_mat = nib.load(config['data_path'] + case + '/vessel.nii').get_fdata() 

    current_nr_extracted_patches = 0  # counts already extracted patches
    img_patches = []  # list to save image patches
    label_patches = []  # list to save label patches

    mip_patches=[]   # list to save mip patches
    arg_patches=[]   # list to save index info
    mipgt_patches=[]   # list to save mip label patches


    # variables with sizes and ranges for searchable areas
    max_patch_size =  config['patch_size']                               
    half_max_size_x = max_patch_size[0] // 2
    half_max_size_y = max_patch_size[1] // 2
    half_max_size_z = max_patch_size[2] // 2

    max_row = label_mat.shape[0] - half_max_size_x
    max_col = label_mat.shape[1] - half_max_size_y
    max_chn = label_mat.shape[2] - half_max_size_z


    # -----------------------------------------------------------
    # EXTRACT RANDOM PATCHES WITH VESSELS IN THE CENTER OF EACH PATCH
    # -----------------------------------------------------------
    searchable_label_area = label_mat[half_max_size_x: max_row, half_max_size_y: max_col, half_max_size_z:max_chn]
    # find all vessel voxel indices in searchable area
    vessel_inds = np.asarray(np.where(searchable_label_area == 1))

    while current_nr_extracted_patches < nr_vessel_patches:
        # find given number of random vessel indices
        random_vessel_inds = vessel_inds[:,
                             np.random.choice(vessel_inds.shape[1], nr_vessel_patches, replace=False)]
        for i in range(nr_vessel_patches):
            # stop extracting if the desired number of patches has been reached
            if current_nr_extracted_patches == nr_vessel_patches:
                break

            # get the coordinates of the random vessel around which the patch will be extracted
            x = random_vessel_inds[0][i] + half_max_size_x
            y = random_vessel_inds[1][i] + half_max_size_y
            z = random_vessel_inds[2][i] + half_max_size_z

 
            cnt=0
            lvl=5         #compute 5-sliced MIPs
            arg_cnt=0     
            row_blk=3
            cln_blk=2

            half_size_x=half_max_size_x
            half_size_y=half_max_size_y
            half_size_z=half_max_size_z

            ###3D img patch 128 x 128 x 16
            random_img_patch = img_mat[x - half_size_x:x + half_size_x, y - half_size_y:y + half_size_y, z - half_size_z:z + half_size_z] 
            ###3D GT label  128 x 128 x 16
            random_label_patch = label_mat[x - half_size_x:x + half_size_x, y - half_size_y:y + half_size_y, z - half_size_z:z + half_size_z]
            ###2D composited MIP patch  128*3 x 128*2
            random_mip_patch = np.zeros((np.size(random_img_patch,0)*row_blk,np.size(random_img_patch,1)*cln_blk),dtype=np.int16)  
            ### Index info about where the Max-value voxel is picked 6 x 128 x 128 x 5  
            random_arg_patch = np.zeros((6, np.size(random_img_patch,0),np.size(random_img_patch,1), lvl))   
            ### 2D composited MIP GT label 128*3 x 128*2          
            random_mipgt_patch = np.zeros((np.size(random_img_patch,0)*row_blk,np.size(random_img_patch,1)*cln_blk))

            for p in range(row_blk):
                for q in range(cln_blk):
                    if (p==(row_blk-1) and q==(cln_blk-2)):
                        random_mip_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_mipgt_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_label_patch[:,:,cnt:cnt+lvl],axis=2)
                        arg_temp=np.argmax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_arg_patch[arg_cnt]=(np.arange(lvl) == arg_temp[...,None])
                        arg_cnt=arg_cnt+1
                        cnt=cnt+3

                    else:
                        random_mip_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_mipgt_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_label_patch[:,:,cnt:cnt+lvl],axis=2)
                        arg_temp=np.argmax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_arg_patch[arg_cnt]=(np.arange(lvl) == arg_temp[...,None])
                        arg_cnt=arg_cnt+1
                        cnt=cnt+2

            # just sanity check if the patch is already in the list
            if any((random_img_patch == x).all() for x in img_patches):
                print('Skip patch because already extracted. size:', size)
                break
            else:
                # append the extracted patches to the dictionaries
                img_patches.append(random_img_patch)
                label_patches.append(random_label_patch)
                mip_patches.append(random_mip_patch)
                arg_patches.append(random_arg_patch)
                mipgt_patches.append(random_mipgt_patch)
                current_nr_extracted_patches += 1
                if current_nr_extracted_patches % 50 == 0:
                    print(current_nr_extracted_patches, 'PATCHES CREATED')



    # -----------------------------------------------------------
    # EXTRACT RANDOM PATCHES NOT FOCUSED ON VESSEL
    # -----------------------------------------------------------
    searchable_mask_area = mask_mat[half_max_size_x: max_row, half_max_size_y: max_col, half_max_size_z:max_chn]
    # find all brain voxel indices
    brain_inds = np.asarray(np.where(searchable_mask_area == 1))

    # keep extracting patches while the desired number of patches has not been reached yet, this just in case some
    # patches would be skipped, because they were already extracted (we do not want to have same patches in the set
    # more than once)
    while current_nr_extracted_patches < nr_patches:
        # find given number of random indices in the brain area
        random_brain_inds = brain_inds[:, np.random.choice(brain_inds.shape[1], nr_empty_patches, replace=False)]
        for i in range(nr_empty_patches):
            # stop extracting if the desired number of patches has been reached
            if current_nr_extracted_patches == nr_patches:
                break

            # get the coordinates of the random brain voxel around which the patch will be extracted
            x = random_brain_inds[0][i] + half_max_size_x
            y = random_brain_inds[1][i] + half_max_size_y
            z = random_brain_inds[2][i] + half_max_size_z

            cnt=0
            lvl=5
            arg_cnt=0
            row_blk=3
            cln_blk=2

            half_size_x=half_max_size_x
            half_size_y=half_max_size_y
            half_size_z=half_max_size_z

            random_img_patch = img_mat[x - half_size_x:x + half_size_x, y - half_size_y:y + half_size_y, z - half_size_z:z + half_size_z]
            random_label_patch = label_mat[x - half_size_x:x + half_size_x, y - half_size_y:y + half_size_y, z - half_size_z:z + half_size_z]
            random_mip_patch = np.zeros((np.size(random_img_patch,0)*row_blk,np.size(random_img_patch,1)*cln_blk),dtype=np.int16)
            random_arg_patch = np.zeros((6, np.size(random_img_patch,0),np.size(random_img_patch,1), lvl)) #6x5 mips
            random_mipgt_patch = np.zeros((np.size(random_img_patch,0)*row_blk,np.size(random_img_patch,1)*cln_blk))

            for p in range(row_blk):
                for q in range(cln_blk):
                    if (p==(row_blk-1) and q==(cln_blk-2)):
                        random_mip_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_mipgt_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_label_patch[:,:,cnt:cnt+lvl],axis=2)
                        arg_temp=np.argmax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_arg_patch[arg_cnt]=(np.arange(lvl) == arg_temp[...,None])
                        arg_cnt=arg_cnt+1
                        cnt=cnt+3
                    else:
                        random_mip_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_mipgt_patch[np.size(random_img_patch,0)*p:(np.size(random_img_patch,0)*(p+1)),np.size(random_img_patch,1)*q:(np.size(random_img_patch,1)*(q+1))]=np.amax(random_label_patch[:,:,cnt:cnt+lvl],axis=2)
                        arg_temp=np.argmax(random_img_patch[:,:,cnt:cnt+lvl],axis=2)
                        random_arg_patch[arg_cnt]=(np.arange(lvl) == arg_temp[...,None])
                        arg_cnt=arg_cnt+1
                        cnt=cnt+2
            # just sanity check if the patch is already in the list
            if any((random_img_patch == x).all() for x in img_patches):
                print('Skip patch because already extracted. size:', size)
                break
            else:
                # append the extracted patches to the dictionaries
                img_patches.append(random_img_patch)
                label_patches.append(random_label_patch)
                mip_patches.append(random_mip_patch)
                arg_patches.append(random_arg_patch)
                mipgt_patches.append(random_mipgt_patch)                        
                current_nr_extracted_patches += 1
                if current_nr_extracted_patches % 50 == 0:
                    print(current_nr_extracted_patches, 'PATCHES CREATED')



    assert current_nr_extracted_patches == nr_patches, "The number of extracted patches is  " + str(
        current_nr_extracted_patches) + " but should be " + str(
        nr_patches)

    # save extracted patches as numpy arrays
    print('number of extracted image patches:', len(img_patches))
    print('number of extracted label patches:', len(label_patches))
    if flag=='train':
        directory = config['save_path'] + 'train'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + '/' + case +  '_img', np.asarray(img_patches))
        np.save(directory + '/' + case +  '_label', np.asarray(label_patches))
        np.save(directory + '/' + case + '_mip', np.asarray(mip_patches))
        np.save(directory + '/' + case + '_mipgt', np.asarray(mipgt_patches))
        np.save(directory + '/' + case + '_arg', np.asarray(arg_patches))
    if flag=='val':
        directory = config['save_path'] + 'val'
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(directory + '/' + case +  '_img', np.asarray(img_patches))
        np.save(directory + '/' + case +  '_label', np.asarray(label_patches))
        np.save(directory + '/' + case + '_mip', np.asarray(mip_patches))
        np.save(directory + '/' + case + '_mipgt', np.asarray(mipgt_patches))
        np.save(directory + '/' + case + '_arg', np.asarray(arg_patches))        
                                    

    print('Saved....:',case)
print('DONE')
