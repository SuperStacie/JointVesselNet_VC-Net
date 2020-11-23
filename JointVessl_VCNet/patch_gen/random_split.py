"""
File name: patch_extraction.py
Author: Yifan Wang
Date created: 11/09/2020
"""

import os
import glob
import numpy as np
import random

with open('id_list.txt') as fp:
	line=fp.readlines()
full_ls=[x.strip() for x in line]
# print(full_ls)

shuffle_ls=random.sample(full_ls,len(full_ls))
print(shuffle_ls)

train_ls=shuffle_ls[:33]
print('training case #: \n',len(train_ls))
val_ls=shuffle_ls[33:36]
print('val case #: \n',len(val_ls))
test_ls=shuffle_ls[36:]
print('test case #: \n',len(test_ls))


with open('train_ls.txt', 'w') as f:
    for item in train_ls:
        f.write("%s\n" % item)

with open('val_ls.txt', 'w') as f:
    for item in val_ls:
        f.write("%s\n" % item)


with open('test_ls.txt', 'w') as f:
    for item in test_ls:
        f.write("%s\n" % item)        