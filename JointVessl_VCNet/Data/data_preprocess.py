"""
File name: data_preprocess.py
Author: Yifan Wang
Date created: 11/09/2020
"""

import os
import glob
import nibabel as nib 
import SimpleITK as sitk
from nipype.interfaces import fsl
import shutil


ls=['{0:0=3d}'.format(a) for a in range(1,110)]
print(ls)

btr = fsl.BET()
fsl.FSLCommand.set_default_output_type('NIFTI')
btr.inputs.frac = 0.05

for i in ls:
	if not os.path.exists('./TubeTK_MRA/Normal-'+i):
		os.makedirs('./TubeTK_MRA/Normal-'+i)
	print(i)
	mha=sitk.ReadImage('./TubeTK_wholeSet/Normal-'+i+'/MRA/Normal'+i+'-MRA.mha')
	sitk.WriteImage(mha,'./TubeTK_MRA/Normal-'+i+'/mra.nii')

	btr.inputs.in_file = './TubeTK_MRA/Normal-'+i+'/mra.nii'
	btr.inputs.out_file = './TubeTK_MRA/Normal-'+i+'/brain.nii'
	res = btr.run()	
	img=sitk.ReadImage('./TubeTK_MRA/Normal-'+i+'/brain.nii.gz')
	sitk.WriteImage(img,'./TubeTK_MRA/Normal-'+i+'/brain.nii')
	os.remove('./TubeTK_MRA/Normal-'+i+'/brain.nii.gz')

	##################compute mask#############################################
	image=sitk.ReadImage('./TubeTK_MRA/Normal-'+i+'/brain.nii')
	background_image=image!=0
	sitk.WriteImage(background_image, './TubeTK_MRA/Normal-'+i+'/mask.nii')