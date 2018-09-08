# -*- coding:utf-8 -*-
"""
@Function: Transform .mat to .npy

@Author: Ling Bao

@Date: July 12, 2017
@Version: 0.0.1
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def mat2npy(mat_path, npy_path):
    """
    Transform .mat to .npy

    :param mat_path: Folder for .mat
    :param npy_path: Folder for .npy
    :return: None
    """
    mat_files = os.listdir(mat_path)
    for mat_file in mat_files:
        # 载入.mat
        mat_full_path = mat_path + '/' + mat_file
        train_gt = sio.loadmat(mat_full_path)
        train_gt = train_gt['outputD_map']

        # 显示
        print(mat_file)
        plt.imshow(train_gt)
        plt.show()
        
        # 输出转换信息
        print (mat_file + " is transform successfully.")

        # 保存.npy
        mat_name = mat_file.split('.')[0]
        npy_full_path = npy_path + '/' + mat_name + '.npy'
        np.save(npy_full_path, train_gt)

if __name__ == '__main__':
    cur_path = sys.path[0]
    mat_path_ = '/home/bl/ACSCP/Data_mat/train'
    npy_path_ = '/home/bl/ACSCP/Data/Data_gt/train_gt'
    mat2npy(mat_path_, npy_path_)
