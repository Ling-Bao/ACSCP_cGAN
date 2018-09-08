# -*- coding:utf-8 -*-

"""
@Brief
ACSCP model：model building, model training and testing
source：Crowd Counting via Adversarial Cross-Scale Consistency Pursuit
        https://pan.baidu.com/s/1mjPpKqG

@Description
using trained ACSCP model to estimate crowd map, this faces with production environment

@Reference

@Author: Ling Bao

@Data: April 12, 2018
@Version: 0.1.0
"""
# 系统库
from math import ceil
import numpy as np
import time
from matplotlib import pyplot as plt
import cv2

# 项目库
from lib.ops import *

# 机器学习库
import tensorflow as tf

slim = tf.contrib.slim


class ProductMap(object):
    def __init__(self, use_dropout=True):
        # 模型相关
        self.sess = tf.Session()
        self.checkpoints = "./gan_mp_bn_1_240/"
        self.g_large_name = "g_large.ckpt"
        # self.g_large_path = "./product_model/g_large_model_412/"  # have dropout
        self.g_large_path = "./product_model/g_large_model_414/"  # no dropout

        # 输入图像
        self.x = tf.placeholder(tf.float32, [1, 720, 720, 3])

        # 批量归一化——large生成器
        self.g_L_bn_e1 = batch_norm(name='g_L_bn_e1')
        self.g_L_bn_e2 = batch_norm(name='g_L_bn_e2')
        self.g_L_bn_e3 = batch_norm(name='g_L_bn_e3')
        self.g_L_bn_e4 = batch_norm(name='g_L_bn_e4')
        self.g_L_bn_e5 = batch_norm(name='g_L_bn_e5')
        self.g_L_bn_e6 = batch_norm(name='g_L_bn_e6')
        self.g_L_bn_e7 = batch_norm(name='g_L_bn_e7')
        self.g_L_bn_e8 = batch_norm(name='g_L_bn_e8')
        self.g_L_bn_d1 = batch_norm(name='g_L_bn_d1')
        self.g_L_bn_d2 = batch_norm(name='g_L_bn_d2')
        self.g_L_bn_d3 = batch_norm(name='g_L_bn_d3')
        self.g_L_bn_d4 = batch_norm(name='g_L_bn_d4')
        self.g_L_bn_d5 = batch_norm(name='g_L_bn_d5')
        self.g_L_bn_d6 = batch_norm(name='g_L_bn_d6')
        self.g_L_bn_d7 = batch_norm(name='g_L_bn_d7')

        # 构建模型
        self.crowd_map = self.generator_large(self.x, use_dropout=use_dropout)

    def generator_large(self, image, batch_size=1, use_dropout=True, reuse=False):
        """
        Large生成器网络
        :param image: 输入数据
        :param batch_size 批量数,默认为1
        :param use_dropout: 是否使用dropout
        :param reuse:
        :return: 生成图片
        """
        with tf.variable_scope("generator_large"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            # input image size is 240 x 240 x 3, input_c_dim = output_c_dim = 3
            # (240 x 240 x input_c_dim) --> e1(120 x 120 x 64) --> e2(60 x 60 x 64) --> e3(30 x 30 x 64) -->
            # e4(15 x 15 x 64) --> e5(8 x 8 x 64) --> e6(4 x 4 x 64) --> e7(2 x 2 x 64) --> e8(2 x 2 x 64) <--
            # d1(2 x 2 x 64*2) <-- d2(4 x 4 x 64*2) <-- d3(8 x 8 x 64*2) <-- d4(15 x 15 x 64*2) <--
            # d5(30 x 30 x 64*2) <-- d6(60 x 60 x 64*2) <-- d7(120 x 120 x 64*2) <-- (240 x 240 x output_c_dim)

            # general method, input image size is w x h x c, limit to w, h more greater 120, c is equal to 3
            # (w x h x c) --> e1(c[w/2] x c[h/2] x 64) --> e2(c[w/4] x c[h/4] x 64) --> e3(c[w/8] x c[h/8] x 64) -->
            # e4(c[w/16] x c[h/16] x 64) --> e5(c[w/32] x c[h/32] x 64) --> e6(c[w/64] x c[h/64] x 64) -->
            # e7(c[w/128] x c[h/128] x 64) --> e8(c[w/128] x c[h/128] x 64) <-- d1(c[w/128] x c[h/128] x 64*2) <--
            # d2(c[w/64] x c[h/64] x 64*2) <-- d3(c[w/32] x c[h/32] x 64*2) <-- d4(c[w/16] x c[h/16] x 64*2) <--
            # d5(c[w/8] x c[h/8] x 64*2) <-- d6(c[w/4] x c[h/4] x 64*2) <-- d7(c[w/2] x c[h/2] x 64*2) <-- (w x h x c)

            w = int(np.array(self.x.shape[1]))
            h = int(np.array(self.x.shape[2]))

            if use_dropout:
                e1 = self.g_L_bn_e2(conv2d(image, output_dim=64, k_h=6, k_w=6, d_h=2, d_w=2, name='g_L_e1_con'))
            else:
                e1 = self.g_L_bn_e1(conv2d(image, output_dim=64, k_h=6, k_w=6, d_h=2, d_w=2, name='g_L_e1_con'))
            e2 = self.g_L_bn_e2(conv2d(lrelu(e1), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e2_con'))
            e3 = self.g_L_bn_e3(conv2d(lrelu(e2), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e3_con'))
            e4 = self.g_L_bn_e4(conv2d(lrelu(e3), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e4_con'))
            e5 = self.g_L_bn_e5(conv2d(lrelu(e4), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e5_con'))
            e6 = self.g_L_bn_e6(conv2d(lrelu(e5), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e6_con'))
            e7 = self.g_L_bn_e7(conv2d(lrelu(e6), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e7_con'))
            e8 = self.g_L_bn_e8(conv2d(lrelu(e7), output_dim=64, k_h=4, k_w=4, d_h=1, d_w=1, name='g_L_e8_con'))

            d1, _, _ = deconv2d(lrelu(e8), [batch_size, int(ceil(w / 128.)), int(ceil(h / 128.)), 64], k_h=4, k_w=4,
                                d_h=1, d_w=1, name='g_L_d1', with_w=True)
            if use_dropout:
                d1 = tf.nn.dropout(self.g_L_bn_d1(d1), 0.5)
            d1 = tf.concat([d1, e7], 3)

            d2, _, _ = deconv2d(tf.nn.relu(d1), [batch_size, int(ceil(w / 64.)), int(ceil(h / 64.)), 64], k_h=4, k_w=4,
                                d_h=2, d_w=2, name='g_L_d2', with_w=True)
            if use_dropout:
                d2 = tf.nn.dropout(self.g_L_bn_d2(d2), 0.5)
            d2 = tf.concat([d2, e6], 3)

            d3, _, _ = deconv2d(tf.nn.relu(d2), [batch_size, int(ceil(w / 32.)), int(ceil(h / 32.)), 64], k_h=4, k_w=4,
                                d_h=2, d_w=2, name='g_L_d3', with_w=True)
            if use_dropout:
                d3 = tf.nn.dropout(self.g_L_bn_d3(d3), 0.5)
            d3 = tf.concat([d3, e5], 3)

            d4, _, _ = deconv2d(tf.nn.relu(d3), [batch_size, int(ceil(w / 16.)), int(ceil(h / 16.)), 64], k_h=4, k_w=4,
                                d_h=2, d_w=2, name='g_L_d4', with_w=True)
            d4 = self.g_L_bn_d4(d4)
            d4 = tf.concat([d4, e4], 3)

            d5, _, _ = deconv2d(tf.nn.relu(d4), [batch_size, int(ceil(w / 8.)), int(ceil(h / 8.)), 64], k_h=4, k_w=4,
                                d_h=2, d_w=2, name='g_L_d5', with_w=True)
            d5 = self.g_L_bn_d5(d5)
            d5 = tf.concat([d5, e3], 3)

            d6, _, _ = deconv2d(tf.nn.relu(d5), [batch_size, int(ceil(w / 4.)), int(ceil(h / 4.)), 64], k_h=4, k_w=4,
                                d_h=2, d_w=2, name='g_L_d6', with_w=True)
            d6 = self.g_L_bn_d6(d6)
            d6 = tf.concat([d6, e2], 3)

            d7, _, _ = deconv2d(tf.nn.relu(d6), [batch_size, int(ceil(w / 2.)), int(ceil(h / 2.)), 64], k_h=4, k_w=4,
                                d_h=2, d_w=2, name='g_L_d7', with_w=True)
            d7 = self.g_L_bn_d7(d7)
            d7 = tf.concat([d7, e1], 3)

            d8, _, _ = deconv2d(tf.nn.relu(d7), [batch_size, int(w), int(h), 3], k_h=6, k_w=6,
                                d_h=2, d_w=2, name='g_L_d8', with_w=True)

            return tf.nn.relu(tf.nn.sigmoid(d8))

    def generator_large_save(self):
        """
        保存ACSCP模型中的generator_large模型
        """
        with tf.Session() as sess:
            # 载入ACSCP模型参数并对generator_large模型参数进行初始化
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.checkpoints)
            saver.restore(sess, ckpt.model_checkpoint_path)

            # 保存vgg2模型
            saver.save(sess, self.g_large_path + self.g_large_name)

            # 关闭session
            sess.close()

    def generator_large_load(self):
        """
        载入generator_large模型
        """
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(self.g_large_path)
        saver.restore(self.sess, ckpt.model_checkpoint_path)

    def run(self, image):
        """
        利用vgg2模型对images进行特征提取
        :param image 待估计图像
        :return: 图像特征
        """
        start_time = time.time()
        data = np.array([image]).astype(np.float32)

        tmp_mp = self.sess.run(self.crowd_map, feed_dict={self.x: data})

        run_time = time.time() - start_time
        mp_crowd = np.mean(tmp_mp[0], axis=2)

        return mp_crowd, run_time


class VGGFTest(tf.test.TestCase):
    """
    对vgg_2模型进行单元测试
    """
    def test_build(self):
        batch_size = 1
        height, width = 720, 720
        with self.test_session():
            inputs = tf.random_uniform((batch_size, height, width, 3))
            g_large = ProductMap()
            g_large.x = inputs

            expected_names = [
                'generator_large/g_L_bn_d2/moving_variance',
                'generator_large/g_L_bn_d4/beta',
                'generator_large/g_L_bn_d3/moving_variance',
                'generator_large/g_L_bn_d7/gamma',
                'generator_large/g_L_bn_e8/moving_variance',
                'generator_large/g_L_bn_d5/gamma',
                'generator_large/g_L_bn_e7/moving_mean',
                'generator_large/g_L_bn_d3/gamma',
                'generator_large/g_L_bn_e6/moving_variance',
                'generator_large/g_L_bn_e6/beta',
                'generator_large/g_L_bn_e4/beta',
                'generator_large/g_L_bn_e5/moving_mean',
                'generator_large/g_L_bn_d1/moving_variance',
                'generator_large/g_L_bn_e7/gamma',
                'generator_large/g_L_bn_d2/moving_mean',
                'generator_large/g_L_bn_d4/moving_variance',
                'generator_large/g_L_bn_d1/moving_mean',
                'generator_large/g_L_bn_e5/gamma',
                'generator_large/g_L_bn_e8/moving_mean',
                'generator_large/g_L_bn_d5/moving_mean',
                'generator_large/g_L_bn_e5/beta',
                'generator_large/g_L_bn_e6/gamma',
                'generator_large/g_L_bn_d1/beta',
                'generator_large/g_L_bn_e4/moving_mean',
                'generator_large/g_L_bn_e3/beta',
                'generator_large/g_L_bn_e2/gamma',
                'generator_large/g_L_bn_e8/gamma',
                'generator_large/g_L_bn_d2/gamma',
                'generator_large/g_L_bn_e4/gamma',
                'generator_large/g_L_bn_d3/moving_mean',
                'generator_large/g_L_bn_e6/moving_mean',
                'generator_large/g_L_bn_e8/beta',
                'generator_large/g_L_bn_d4/moving_mean',
                'generator_large/g_L_bn_d4/gamma',
                'generator_large/g_L_bn_d5/moving_variance',
                'generator_large/g_L_bn_d7/beta',
                'generator_large/g_L_bn_d6/moving_mean',
                'generator_large/g_L_bn_d6/moving_variance',
                'generator_large/g_L_bn_e4/moving_variance',
                'generator_large/g_L_bn_e2/beta',
                'generator_large/g_L_bn_d7/moving_variance',
                'generator_large/g_L_bn_d5/beta',
                'generator_large/g_L_bn_e7/beta',
                'generator_large/g_L_bn_d3/beta',
                'generator_large/g_L_bn_e3/moving_mean',
                'generator_large/g_L_bn_e5/moving_variance',
                'generator_large/g_L_bn_d2/beta',
                'generator_large/g_L_bn_e7/moving_variance',
                'generator_large/g_L_bn_e2/moving_mean',
                'generator_large/g_L_bn_d1/gamma',
                'generator_large/g_L_bn_e3/gamma',
                'generator_large/g_L_bn_d7/moving_mean',
                'generator_large/g_L_bn_d6/beta',
                'generator_large/g_L_bn_e2/moving_variance',
                'generator_large/g_L_bn_e3/moving_variance',
                'generator_large/g_L_bn_d6/gamma']

            model_variables = [v.op.name for v in slim.get_model_variables()]
            self.assertSetEqual(set(model_variables), set(expected_names))


if __name__ == "__main__":
    # ****************************************************模型测试***************************************************** #
    # # TF模型结构单元测试
    # tf.test.main()

    # ****************************************************接口示例***************************************************** #
    # 载入图像
    img_path = "data/data_im/test_im/"
    img_name = "IMG_2_A"
    image = cv2.imread(img_path + img_name + ".jpg")
    if image is None:
        print("Please check image path!!")
        exit(0)

    # 人群密度估计
    # product = ProductMap(True)  # have dropout
    product = ProductMap(False)  # no dropout
    # product.generator_large_save()  # 仅用于提取generator_larger模型参数并保存
    product.generator_large_load()
    mp, time = product.run(image)

    # 保存估计的人群密度图
    mp_name = img_name
    print("Time: %4.4f, Estimation numbers: %4d" % (time, round(sum(sum(mp)))))
    plt.imsave(mp_name + ".png", mp, cmap=plt.get_cmap('jet'))