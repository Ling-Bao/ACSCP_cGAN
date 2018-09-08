# -*- coding=utf-8 -*-
"""
@Brief: 用于从视频中采集训练/测试数据集合

@Author: Ling Bao
@Date: 星期二, 11. 七月 2017 04:09下午

@Version: 0.0.1
"""

import cv2
import numpy as np
import sys


def data_collect(types='train', nums=200):
    """
    @Brief: 用于从视频中采集训练/测试数据集合

    :param types: train or test
    :param nums: 采集训练/测试数据集样本个数
    :return: None
    """
    if types == 'train':
        # Train Data Collection
        file_read = sys.path[0] + '/DJI_0005.MOV'
        file_write = sys.path[0] + '/UAVData/train_data/train_img/'
        prefix = 'IMG_TR_'
        time_interval = 90  # 视频帧计数间隔频率
    elif types == 'test':
        # Test Data Collection
        file_read = sys.path[0] + '/DJI_0005.MOV'
        file_write = sys.path[0] + '/UAVData/test_data/test_img/'
        prefix = 'IMG_TE_'
        time_interval = 100  # 视频帧计数间隔频率
    else:
        print('Please give a parameter between train and test!')
        exit(0)

    vc = cv2.VideoCapture(file_read)
    if vc.isOpened():  # 判断是否正常打开
        r_val, frame = vc.read()
    else:
        r_val = False
    print(r_val)

    i_timer = 1
    c = 1
    while r_val:  # 循环读取视频帧
        _, frame = vc.read()
        if i_timer % time_interval == 0:  # 每隔timeF帧进行存储操作
            height = frame.shape[0]
            width = frame.shape[1]

            # Random crop image four times
            for i in range(0, 2):
                # 获取Crop图片高/宽
                rand_height = int(np.ceil(np.random.uniform(600, 900)))
                rand_width = int(np.ceil(np.random.uniform(900, 1300)))
                rand_height -= (rand_height % 50)
                rand_width -= (rand_width % 50)

                # 获取左上角坐标
                rand_y1 = int(np.ceil(np.random.uniform(200, height - rand_height)))
                rand_x1 = int(np.ceil(np.random.uniform(300, width - rand_width)))
                rand_y1 -= (rand_y1 % 50)
                rand_x1 -= (rand_x1 % 50)

                # 获取右下角坐标
                rand_y2 = rand_y1 + rand_height
                rand_x2 = rand_x1 + rand_width

                # Crop图片
                temp_img = frame[rand_y1:rand_y2, rand_x1:rand_x2]

                # 打印调试信息
                print(temp_img.shape)
                # cv2.imshow('abc', temp_img)
                # cv2.waitKey(300000)

                # 写图片数据
                cv2.imwrite(file_write + prefix + str(c) + '.jpg', temp_img)  # 存储为图像
                c += 1

        if c >= nums:
            break

        i_timer += 1
        cv2.waitKey(1)

    vc.release()


# 采集训练数据集样本个数
train_num = 200
data_collect('train', train_num)

# 采集测试数据集样本个数
test_num = 100
data_collect('test', test_num)
