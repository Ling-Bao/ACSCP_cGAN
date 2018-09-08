# -*- coding:utf-8 -*-

"""
@Brief
ACSCP model：model building, model training and testing
source：Crowd Counting via Adversarial Cross-Scale Consistency Pursuit
        https://pan.baidu.com/s/1mjPpKqG

@Description
image data load and augmentation

@Reference

@Author: Ling Bao

@Data: April 12, 2018
@Version: 0.1.0
"""

# 系统库
from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
import cv2


def data_augmentation(img, mp, load_size=720, fine_size=240, flip=True, is_test=False):
    """
    对原始图片与标签图片进行处理，注意：测试集与训练集有所不同，训练集可能会考虑数据增广

    :param img: 图片数据
    :param mp: 密度图数据
    :param load_size: 图片初始大小
    :param fine_size: 图片增广或的大小
    :param flip: 是否进行图片翻转操作
    :param is_test: 是否是测试数据
    :return: 处理后的图片img/mp数据
    """
    if is_test:
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
        img_tmp = img[h1:h1 + fine_size, w1:w1 + fine_size]
        mp_tmp = mp[h1:h1 + fine_size, w1:w1 + fine_size]

    else:
        iter_num = 0.0
        sum_mp = 0.0
        img_tmp = None
        mp_tmp = None

        while (sum_mp < 10.0) and (iter_num < 10.0):
            h1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            w1 = int(np.ceil(np.random.uniform(1e-2, load_size - fine_size)))
            img_tmp = img[h1:h1 + fine_size, w1:w1 + fine_size]
            mp_tmp = mp[h1:h1 + fine_size, w1:w1 + fine_size]

            sum_mp = sum(sum(sum(mp_tmp))) / 3
            iter_num += 1

        # 翻转
        if flip and np.random.random() > 0.5:
            img_tmp = np.fliplr(img_tmp)
            mp_tmp = np.fliplr(mp_tmp)

    return img_tmp, mp_tmp


def load_data(image_path, args, flip=True, is_test=False, is_sample=False):
    """
    载入数据操作

    :param image_path: 图片路径
    :param args: 全局配置参数变量
    :param flip: 是否进行图片翻转操作
    :param is_test: 是否是测试数据
    :param is_sample: 是否为验证数据
    :return: 进行拼接后的数据
    """
    # 根据训练与测试条件获取对应路径
    if is_test:
        im_path = args.test_im_dir
        gt_path = args.test_gt_dir
    else:
        im_path = args.train_im_dir
        gt_path = args.train_gt_dir

    # 获取待读取数据完整路径
    name = image_path.split('/')[-1].split('.')[0]
    im_name = name + '.jpg'
    gt_name = name + '.npy'
    im_path += im_name
    gt_path += gt_name

    # 读取图片与密度图
    img = cv2.imread(im_path)
    mp = np.array(np.load(gt_path))

    if is_test:
        # 可视化保存原始测试图与密度图
        cv2.imwrite('./{}/om_{}.jpg'.format(args.test_dir, name), img)
        plt.imsave('./{}/og_{}.jpg'.format(args.test_dir, name), mp, cmap=plt.get_cmap('jet'))
        print("counting:%.2f" % (sum(sum(mp))))

    if is_sample:
        # 可视化保存原始测试图与密度图
        cv2.imwrite('./{}/om_{}.jpg'.format(args.sample_dir, name), img)
        plt.imsave('./{}/og_{}.jpg'.format(args.sample_dir, name), mp, cmap=plt.get_cmap('jet'))
        print("counting:%.2f" % (sum(sum(mp))))

    # # 显示并保存图片
    # show_img(img)
    # show_img(mp)

    # 单通道密度图变为3通道
    mp = np.transpose(np.array([mp, mp, mp]), [1, 2, 0])

    # crop操作进行数据增广(训练)
    img_tmp, mp_tmp = data_augmentation(img, mp, load_size=args.load_size,
                                        fine_size=args.fine_size, flip=flip, is_test=is_test)

    # # 归一化处理
    # img_tmp = img_tmp / 127.5 - 1.

    # 图片拼接
    # img_mp shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    img_mp = np.concatenate((img_tmp, mp_tmp), axis=2)

    return img_mp


def get_real_count(gt_path, img_name):
    """
    获取真实的人数
    :param gt_path: 人群密度真图路径
    :param img_name: 人群密度真图名称
    :return:
    """
    mp = np.array(np.load(gt_path))

    real_count = sum(sum(mp))
    map_name = "real_" + img_name
    print("Real count is %4d" % round(real_count))
    plt.imsave("../" + map_name + ".png", mp, cmap=plt.get_cmap('jet'))


if __name__ == "__main__":
    img_path = "../data/data_gt/test_gt/"
    img_name = "IMG_1_B"
    get_real_count(img_path + img_name + ".npy", img_name)
