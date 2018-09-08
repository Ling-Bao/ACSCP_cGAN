# -*- coding:utf-8 -*-

"""
@Brief
ACSCP model：model building, model training and testing
source：Crowd Counting via Adversarial Cross-Scale Consistency Pursuit
        https://pan.baidu.com/s/1mjPpKqG

@Description
1 using Adam to optimization;
2 using padding not resize to get input image;
3 first 100 epoch set lambda_c=0,last 200 epoch set lambda_c=10

@Reference

@Author: Ling Bao

@Data: April 12, 2018
@Version: 0.1.0
"""

# 系统库
from __future__ import division
import os
import time
from glob import glob
from six.moves import xrange
from matplotlib import pyplot as plt_model

# 项目库
from lib_ops.ops import *
from lib_ops.utils import *
from vgg_feature import VGG2

# 机器学习库
import tensorflow as tf

slim = tf.contrib.slim


class ACSCP(object):
    def __init__(self, sess, image_size=240, batch_size=16, sample_size=1, output_size=240, df_dim=48,
                 input_c_dim=3, output_c_dim=3, data_set_name='facades', checkpoint_dir=None, lambda_e=150,
                 lambda_p=150, lambda_c=10):
        # 通用变量
        self.sess = sess
        self.is_gray = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.lambda_E = lambda_e
        self.lambda_P = lambda_p
        self.lambda_C = lambda_c
        self.sample_size = sample_size
        self.output_size = output_size
        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        # 批量归一化——large判别器
        self.d_L_bn0 = batch_norm(name='d_L_bn0')
        self.d_L_bn1 = batch_norm(name='d_L_bn1')
        self.d_L_bn2 = batch_norm(name='d_L_bn2')
        self.d_L_bn3 = batch_norm(name='d_L_bn3')
        self.d_L_bn4 = batch_norm(name='d_L_bn4')

        # 批量归一化——small判别器
        self.d_S_bn0 = batch_norm(name='d_S_bn0')
        self.d_S_bn1 = batch_norm(name='d_S_bn1')
        self.d_S_bn2 = batch_norm(name='d_S_bn2')
        self.d_S_bn3 = batch_norm(name='d_S_bn3')
        self.d_S_bn4 = batch_norm(name='d_S_bn4')

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

        # 批量归一化——small判别器
        self.g_S_bn_e1 = batch_norm(name='g_S_bn_e1')
        self.g_S_bn_e2 = batch_norm(name='g_S_bn_e2')
        self.g_S_bn_e3 = batch_norm(name='g_S_bn_e3')
        self.g_S_bn_e4 = batch_norm(name='g_S_bn_e4')
        self.g_S_bn_e5 = batch_norm(name='g_S_bn_e5')
        self.g_S_bn_e6 = batch_norm(name='g_S_bn_e6')
        self.g_S_bn_e7 = batch_norm(name='g_S_bn_e7')
        self.g_S_bn_d1 = batch_norm(name='g_S_bn_d1')
        self.g_S_bn_d2 = batch_norm(name='g_S_bn_d2')
        self.g_S_bn_d3 = batch_norm(name='g_S_bn_d3')
        self.g_S_bn_d4 = batch_norm(name='g_S_bn_d4')
        self.g_S_bn_d5 = batch_norm(name='g_S_bn_d5')
        self.g_S_bn_d6 = batch_norm(name='g_S_bn_d6')

        # 0 损失相关通用变量
        self.real_data, self.real_im, self.real_mp = None, None, None

        # 1.1 small判别器损失相关
        self.real_im_small, self.real_mp_small, self.fake_mp_small = None, None, None
        self.real_concat_small, self.fake_concat_small = None, None
        self.d_s_x, self.d_s_y = None, None
        self.d_s_x_, self.d_s_y_ = None, None
        self.d_s_loss_real, self.d_s_loss_fake = None, None
        self.d_small_loss_a = None

        # 1.2 small生成器损失相关
        self.g_small_loss_a, self.g_small_loss_e, self.g_small_loss_p, self.g_small_loss_one = None, None, None, None

        # 1.3 small训练概要设置相关
        self.d_s_real_sum, self.d_s_fake_sum, self.g_s_fake_sum, self.d_s_loss_sum = None, None, None, None
        self.g_s_loss_a_sum, self.g_s_loss_e_sum, self.g_s_loss_p_sum, self.g_s_loss_one_sum = None, None, None, None

        # 2.1 large判别器损失相关
        self.real_im_large, self.real_mp_large, self.fake_mp_large = None, None, None
        self.real_concat_large, self.fake_concat_large = None, None
        self.d_l_x, self.d_l_y = None, None
        self.d_l_x_, self.d_l_y_ = None, None
        self.d_l_loss_real, self.d_l_loss_fake = None, None
        self.d_large_loss_a = None

        # 2.2 large生成器损失相关
        self.g_large_loss_a, self.g_large_loss_e, self.g_large_loss_p, self.g_large_loss_one = None, None, None, None

        # 2.3 large训练概要设置相关
        self.d_l_real_sum, self.d_l_fake_sum, self.g_l_fake_sum, self.d_l_loss_sum = None, None, None, None
        self.g_l_loss_a_sum, self.g_l_loss_e_sum, self.g_l_loss_p_sum, self.g_l_loss_one_sum = None, None, None, None

        # 3 交叉尺度损失相关
        self.fake_mp_small_, self.fake_mp_large_, self.cross_scale_loss_two = None, None, None
        self.cc_loss_sum = None

        # 4 生成器总损失相关
        self.g_s_loss, self.g_l_loss = None, None
        self.g_s_loss_sum, self.g_l_loss_sum = None, None
        self.merged_summary_op = None

        # 5 模型参数训练与保存相关
        self.d_l_vars, self.d_s_vars, self.g_l_vars, self.g_s_vars = None, None, None, None
        self.saver = None
        self.g_l_sum, self.d_l_sum, self.g_s_sum, self.d_s_sum = None, None, None, None
        self.writer = None

        self.data_set_name = data_set_name
        self.checkpoint_dir = checkpoint_dir

        # 6 构建模型
        self.build_model()

    def build_model(self):
        # ××××××××××××××××××××××××××××××××××××××××××××××××前期准备××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 0 前期准备
        # 0.1 small判别器与生成器输入尺寸
        w_small = int(self.image_size / 2)
        h_small = int(self.image_size / 2)

        # 0.2 large模型输入数据
        c_ = self.input_c_dim + self.output_c_dim
        self.real_data = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, c_],
                                        name="image_and_mp")
        self.real_im = self.real_data[:, :, :, :self.input_c_dim]
        self.real_mp = self.real_data[:, :, :, self.input_c_dim:c_]

        # 0.3 small模型输入数据
        small_im_1 = self.real_im[:, :w_small, :h_small, :]
        small_im_2 = self.real_im[:, w_small:w_small + h_small, :h_small, :]
        small_im_3 = self.real_im[:, :w_small, h_small:h_small + w_small, :]
        small_im_4 = self.real_im[:, w_small:w_small + h_small, h_small:h_small + w_small, :]
        small_concat_im = tf.concat([small_im_1, small_im_2, small_im_3, small_im_4], 0)

        small_mp_1 = self.real_mp[:, :w_small, :h_small, :]
        small_mp_2 = self.real_mp[:, w_small:w_small + h_small, :h_small, :]
        small_mp_3 = self.real_mp[:, :w_small, h_small:h_small + w_small, :]
        small_mp_4 = self.real_mp[:, w_small:w_small + h_small, h_small:h_small + w_small, :]
        small_concat_mp = tf.concat([small_mp_1, small_mp_2, small_mp_3, small_mp_4], 0)

        # 0.4 VGG2网络初始化，用于感知损失计算
        vgg2 = VGG2()
        vgg2.vgg_2_load()

        # ××××××××××××××××××××××××××××××××××××××××××××××small部分××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 1 small部分
        # 1.1 D_small损失
        # 1.1.1 获取输入数据
        self.real_im_small = small_concat_im
        self.real_mp_small = small_concat_mp
        self.fake_mp_small = self.generator_small(self.real_im_small, 4 * self.batch_size)

        # 1.1.2 真假判别
        self.real_concat_small = tf.concat([self.real_im_small, self.real_mp_small], 3)
        self.fake_concat_small = tf.concat([self.real_im_small, self.fake_mp_small], 3)
        self.d_s_x, self.d_s_y = self.discriminator_small(self.real_concat_small, 4 * self.batch_size, reuse=False)
        self.d_s_x_, self.d_s_y_ = self.discriminator_small(self.fake_concat_small, 4 * self.batch_size, reuse=True)

        # 1.1.3 small判别器对抗损失
        self.d_s_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_s_x, labels=tf.ones_like(self.d_s_y)))
        self.d_s_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_s_x_, labels=tf.zeros_like(self.d_s_y_)))
        self.d_small_loss_a = self.d_s_loss_real + self.d_s_loss_fake

        # 1.1.4 small判别器训练概要设置(**后续再考虑，用于追踪损失与生成器图像质量**)
        self.d_s_real_sum = tf.summary.histogram("d_small_real", self.d_s_y)
        self.d_s_fake_sum = tf.summary.histogram("d_small_fake", self.d_s_y_)
        self.g_s_fake_sum = tf.summary.image("g_small", self.fake_mp_small)
        self.d_s_loss_sum = tf.summary.scalar("d_s_loss", self.d_small_loss_a)

        # 1.2 G_small损失
        # 1.2.1 small生成器对抗损失
        self.g_small_loss_a = self.d_s_loss_fake

        # 1.2.2 L2损失--Euclidean loss
        self.g_small_loss_e = tf.reduce_mean(
            tf.abs(self.real_mp_small - self.fake_mp_small) * tf.abs(self.real_mp_small - self.fake_mp_small))

        # 1.2.3 small生成器感知损失
        vgg2.x = self.real_mp_small
        vgg2.vgg_2()
        f_real_mp_small = vgg2.net
        vgg2.x = self.fake_mp_small
        vgg2.vgg_2()
        f_fake_mp_small = vgg2.net
        self.g_small_loss_p = tf.reduce_mean(tf.abs(f_real_mp_small - f_fake_mp_small)
                                             * tf.abs(f_real_mp_small - f_fake_mp_small))

        # 1.2.4 small生成器第一部分损失
        self.g_small_loss_one = \
            self.g_small_loss_a + self.lambda_E * self.g_small_loss_e + self.lambda_P * self.g_small_loss_p

        # 1.2.5 small生成器训练概要设置
        self.g_s_loss_a_sum = tf.summary.scalar("g_s_loss_a", self.g_small_loss_a)
        self.g_s_loss_e_sum = tf.summary.scalar("g_s_loss_e", self.g_small_loss_e)
        self.g_s_loss_p_sum = tf.summary.scalar("g_s_loss_p", self.g_small_loss_p)
        self.g_s_loss_one_sum = tf.summary.scalar("g_s_loss_one", self.g_small_loss_one)

        # ××××××××××××××××××××××××××××××××××××××××××××××large部分××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 2 large部分
        # 2.1 D_large损失
        # 2.1.1 获取输入数据
        self.real_im_large = self.real_im
        self.real_mp_large = self.real_mp
        self.fake_mp_large = self.generator_large(self.real_im_large, self.batch_size)

        # 2.1.2 真假判别
        self.real_concat_large = tf.concat([self.real_im_large, self.real_mp_large], 3)
        self.fake_concat_large = tf.concat([self.real_im_large, self.fake_mp_large], 3)
        self.d_l_x, self.d_l_y = self.discriminator_large(self.real_concat_large, self.batch_size, reuse=False)
        self.d_l_x_, self.d_l_y_ = self.discriminator_large(self.fake_concat_large, self.batch_size, reuse=True)

        # 2.1.3 large判别器对抗损失
        self.d_l_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_l_x, labels=tf.ones_like(self.d_l_y)))
        self.d_l_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.d_l_x_, labels=tf.zeros_like(self.d_l_y_)))
        self.d_large_loss_a = self.d_l_loss_real + self.d_l_loss_fake

        # 2.1.4 large判别器训练概要设置(**后续再考虑，用于追踪损失与生成器图像质量**)
        self.d_l_real_sum = tf.summary.histogram("d_large_real", self.d_l_y)
        self.d_l_fake_sum = tf.summary.histogram("d_large_fake", self.d_l_y_)
        self.g_l_fake_sum = tf.summary.image("g_large", self.fake_mp_large)
        self.d_l_loss_sum = tf.summary.scalar("d_l_loss", self.d_large_loss_a)

        # 2.2 G_large损失
        # 2.2.1 large生成器对抗损失
        self.g_large_loss_a = self.d_l_loss_fake

        # 2.2.2 L2损失--Euclidean loss
        self.g_large_loss_e = tf.reduce_mean(
            tf.abs(self.real_mp_large - self.fake_mp_large) * tf.abs(self.real_mp_large - self.fake_mp_large))

        # 2.2.3 large生成器感知损失
        vgg2.x = self.real_mp_large
        vgg2.vgg_2()
        f_real_mp_large = vgg2.net
        vgg2.x = self.fake_mp_large
        vgg2.vgg_2()
        f_fake_mp_large = vgg2.net
        self.g_large_loss_p = tf.reduce_mean(tf.abs(f_real_mp_large - f_fake_mp_large)
                                             * tf.abs(f_real_mp_large - f_fake_mp_large))

        # 2.2.4 large生成器第一部分损失
        self.g_large_loss_one = \
            self.g_large_loss_a + self.lambda_E * self.g_large_loss_e + self.lambda_P * self.g_large_loss_p

        # 2.2.5 large生成器训练概要设置
        self.g_l_loss_a_sum = tf.summary.scalar("g_l_loss_a", self.g_large_loss_a)
        self.g_l_loss_e_sum = tf.summary.scalar("g_l_loss_e", self.g_large_loss_e)
        self.g_l_loss_p_sum = tf.summary.scalar("g_l_loss_p", self.g_large_loss_p)
        self.g_l_loss_one_sum = tf.summary.scalar("g_sl_loss_one", self.g_large_loss_one)

        # ×××××××××××××××××××××××××××××××××××××××××××××交叉尺度损失××××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 3 交叉尺度损失
        # 3.1 获取large与small判别器生成图片
        self.fake_mp_small_ = self.fake_mp_small
        fml = self.fake_mp_large
        fml_1 = fml[:, :w_small, :h_small, :]
        fml_2 = fml[:, w_small:w_small + h_small, :h_small, :]
        fml_3 = fml[:, :w_small, h_small:h_small + w_small, :]
        fml_4 = fml[:, w_small:w_small + h_small, h_small:h_small + w_small, :]
        self.fake_mp_large_ = tf.concat([fml_1, fml_2, fml_3, fml_4], 0)

        # 3.2 计算交叉尺度损失
        cc_loss = tf.reduce_mean(
            tf.abs(self.fake_mp_small_ - self.fake_mp_large_) * tf.abs(self.fake_mp_small_ - self.fake_mp_large_))
        self.cross_scale_loss_two = self.lambda_C * cc_loss

        # 3.3 交叉尺度损失训练概要设置
        self.cc_loss_sum = tf.summary.scalar("cross_scale_loss", self.cross_scale_loss_two)

        # ××××××××××××××××××××××××××××××××××××××××××××××生成器总损失×××××××××××××××××××××××××××××××××××××××××××××××××××× #
        # 4 生成器总损失
        # 4.1 small生成器总损失
        self.g_s_loss = self.g_small_loss_one + self.cross_scale_loss_two

        # 4.2 large生成器总损失
        self.g_l_loss = self.g_large_loss_one + self.cross_scale_loss_two

        # 4.3 生成器总损失训练概要设置
        self.g_s_loss_sum = tf.summary.scalar("g_s_loss", self.g_s_loss)
        self.g_l_loss_sum = tf.summary.scalar("g_l_loss", self.g_l_loss)

        # 5 模型参数训练与保存
        t_vars = tf.trainable_variables()
        self.d_l_vars = [var for var in t_vars if 'd_L_' in var.name]
        self.g_l_vars = [var for var in t_vars if 'g_L_' in var.name]
        self.d_s_vars = [var for var in t_vars if 'd_S_' in var.name]
        self.g_s_vars = [var for var in t_vars if 'g_S_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, args):
        # 设置优化器
        d_s_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_small_loss_a, var_list=self.d_s_vars)
        d_l_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.d_large_loss_a, var_list=self.d_l_vars)
        g_s_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_s_loss, var_list=self.g_s_vars)
        g_l_op = tf.train.AdamOptimizer(args.lr, beta1=args.beta1).minimize(self.g_l_loss, var_list=self.g_l_vars)

        # 初始化变量并创建会话
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # 合并概要并写图结构到日志文件
        self.merged_summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        # 如果存在已保存模型断点，则进行模型载入
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        counter = 1
        for epoch in xrange(args.epoch):
            # 获取训练数据路径列表
            data = glob('{}/*.jpg'.format(args.train_im_dir))
            np.random.shuffle(data)

            # 配置最大训练样本数目
            batch_idx_set = min(len(data), args.train_size)
            batch_idx_set /= self.batch_size
            batch_idx_set = int(np.floor(batch_idx_set))

            # 开始进行本批次样本训练
            for idx in xrange(0, batch_idx_set):
                # 获取本轮训练的数据
                batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch = [load_data(batch_file, args) for batch_file in batch_files]

                # 转换为numpy数组
                batch_images = np.array(batch).astype(np.float32)

                # 更新large判别器网络/large生成器网络/small判别器网络/small生成器网络
                _ = self.sess.run([d_l_op], feed_dict={self.real_data: batch_images})
                _ = self.sess.run([g_l_op], feed_dict={self.real_data: batch_images})
                _ = self.sess.run([d_s_op], feed_dict={self.real_data: batch_images})
                _ = self.sess.run([g_s_op], feed_dict={self.real_data: batch_images})

                # 记录全局迭代步数
                counter += 1

                # 保存概述数据
                if np.mod(counter, 100) == 0:
                    summary_str = self.sess.run(self.merged_summary_op, feed_dict={self.real_data: batch_images})
                    self.writer.add_summary(summary_str, counter)

                    f_l = self.fake_mp_large.eval({self.real_data: batch_images})
                    f_s = self.fake_mp_small.eval({self.real_data: batch_images})

                    r_sum = sum(sum(batch[0][:, :, 3]))
                    f_l_sum = sum(sum(sum(f_l[0]))) / 3
                    f_s_sum = sum(sum(sum(f_s[0]))) / 3
                    print("\n******************************************************************")
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, real: %.4f, l_fake: %.4f, s_fake: %.4f"
                          % (epoch, idx, batch_idx_set, time.time() - start_time, r_sum, f_l_sum, f_s_sum))
                    print("******************************************************************\n")

                    im_path = "./sample/"
                    im_name = "fake_large_" + str(epoch) + ".jpg"
                    cv2.imwrite(im_path + im_name, f_l[0])

                # 打印每一步训练过程信息
                if np.mod(counter, 10) == 0:
                    # 获取损失模型损失
                    err_d_s_a = self.d_small_loss_a.eval({self.real_data: batch_images})
                    err_d_l_a = self.d_large_loss_a.eval({self.real_data: batch_images})
                    err_g_s = self.g_s_loss.eval({self.real_data: batch_images})
                    err_g_l = self.g_l_loss.eval({self.real_data: batch_images})

                    # 打印训练信息
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_s_a_loss: %.8f, d_l_a_loss: %.8f, g_s_loss: %.8f,"
                          " g_l_loss: %.8f" % (epoch, idx, batch_idx_set, time.time() - start_time, err_d_s_a,
                                               err_d_l_a, err_g_s, err_g_l))

                # # 每训练固定批次便进行一次验证（此次为200批次）
                # if np.mod(counter, 400) == 0:
                #     self.sample_model(args)

                # 每训练固定批次便进行一次模型保存（此次为500批次）
                if np.mod(counter, 5000) == 0:
                    self.save(args.checkpoint_dir, counter)

    def test(self, args):
        # 如果存在已保存模型断点，则进行模型载入
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 获取训练数据路径列表
        data = glob('{}/*.jpg'.format(args.test_im_dir))

        # 配置最大训练样本数目
        batch_idx_set = len(data)
        batch_idx_set /= self.batch_size
        batch_idx_set = int(np.floor(batch_idx_set))

        # 计算平均绝对误差与平均均方误差
        sum_mae = 0.0
        sum_mse = 0.0

        # 开始进行本批次样本训练
        for idx in xrange(0, batch_idx_set):
            # 获取本轮训练的数据
            batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
            batch = [load_data(batch_file, args) for batch_file in batch_files]

            # 转换为numpy数组
            batch_images = np.array(batch).astype(np.float32)

            f_l = self.fake_mp_large.eval({self.real_data: batch_images})

            r_sum = sum(sum(batch[0][:, :, 3]))
            f_l_sum = sum(sum(sum(f_l[0]))) / 3
            abs_tmp = abs(r_sum - f_l_sum)
            sqr_tmp = pow(r_sum - f_l_sum, 2)

            print("Image: [%4d/%4d] time: %4.4f, real: %.4f, l_fake: %.4f, abs_diff: %.4f, sqr_diff: %.4f"
                  % (idx, batch_idx_set, time.time() - start_time, r_sum, f_l_sum, abs_tmp, sqr_tmp))

            sum_mae += abs_tmp
            sum_mse += sqr_tmp

            mp = np.mean(f_l[0], axis=2)
            mp_name = batch_files[0].split("/")[-1].split('.')[0]
            plt_model.imsave(args.test_dir + mp_name + ".png", mp, cmap=plt_model.get_cmap('jet'))
            cv2.imwrite(args.test_dir + mp_name + ".jpg", batch[0][:, :, :3])

        mae = sum_mae / batch_idx_set
        mse = np.sqrt(sum_mse / batch_idx_set)
        print("\n******************************************************************")
        print("MAE: %.8f, MSE: %.8f" % (mae, mse))
        print("******************************************************************\n")

    def inference(self, img, mp_name):
        """
        用于人群密度估计推理
        :param img: 待估计图片
        :param mp_name: 密度图名称
        :return: None
        """
        # 如果存在已保存模型断点，则进行模型载入
        start_time = time.time()
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 分割为9等分
        im_size = img.shape
        w = int(im_size[0] / 3)
        h = int(im_size[0] / 3)

        # 分别对9个等分图像进行群密度估计
        concat_mp = np.array(img).astype(np.float32)
        for i in range(3):
            for j in range(3):
                img_tmp = img[i * w:(i + 1) * w, j * h:(j + 1) * h, :]
                img_concat_tmp = np.concatenate((img_tmp, img_tmp), axis=2)
                img_tmp_np = np.array([img_concat_tmp]).astype(np.float32)
                mp_tmp = self.sess.run([self.fake_mp_large], feed_dict={self.real_data: img_tmp_np})

                concat_mp[i * w:(i + 1) * w, j * h:(j + 1) * h, :] = mp_tmp[0]

        run_time = time.time() - start_time
        mp = np.mean(concat_mp, axis=2)

        return mp, run_time

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.data_set_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def generator_large(self, image, batch_size, reuse=False):
        """
        Large生成器网络
        :param image: 输入数据
        :param batch_size
        :param reuse:
        :return: 生成图片
        """
        with tf.variable_scope("generator_large"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            # (240 x 240 x input_c_dim) --> e1(120 x 120 x 64) --> e2(60 x 60 x 64) --> e3(30 x 30 x 64) -->
            # e4(15 x 15 x 64) --> e5(8 x 8 x 64) --> e6(4 x 4 x 64) --> e7(2 x 2 x 64) --> e8(2 x 2 x 64) <--
            # d1(2 x 2 x 64*2) <-- d2(4 x 4 x 64*2) <-- d3(8 x 8 x 64*2) <-- d4(15 x 15 x 64*2) <--
            # d5(30 x 30 x 64*2) <-- d6(60 x 60 x 64*2) <-- d7(120 x 120 x 64*2) <-- (240 x 240 x output_c_dim)
            e1 = self.g_L_bn_e2(conv2d(image, output_dim=64, k_h=6, k_w=6, d_h=2, d_w=2, name='g_L_e1_con'))
            e2 = self.g_L_bn_e2(conv2d(lrelu(e1), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e2_con'))
            e3 = self.g_L_bn_e3(conv2d(lrelu(e2), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e3_con'))
            e4 = self.g_L_bn_e4(conv2d(lrelu(e3), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e4_con'))
            e5 = self.g_L_bn_e5(conv2d(lrelu(e4), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e5_con'))
            e6 = self.g_L_bn_e6(conv2d(lrelu(e5), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e6_con'))
            e7 = self.g_L_bn_e7(conv2d(lrelu(e6), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_L_e7_con'))
            e8 = self.g_L_bn_e8(conv2d(lrelu(e7), output_dim=64, k_h=4, k_w=4, d_h=1, d_w=1, name='g_L_e8_con'))

            d1, _, _ = deconv2d(lrelu(e8), [batch_size, 2, 2, 64], k_h=4, k_w=4, d_h=1, d_w=1,
                                name='g_L_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_L_bn_d1(d1), 0.5)
            d1 = tf.concat([d1, e7], 3)

            d2, _, _ = deconv2d(tf.nn.relu(d1), [batch_size, 4, 4, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_L_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_L_bn_d2(d2), 0.5)
            d2 = tf.concat([d2, e6], 3)

            d3, _, _ = deconv2d(tf.nn.relu(d2), [batch_size, 8, 8, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_L_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_L_bn_d3(d3), 0.5)
            d3 = tf.concat([d3, e5], 3)

            d4, _, _ = deconv2d(tf.nn.relu(d3), [batch_size, 15, 15, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_L_d4', with_w=True)
            d4 = self.g_L_bn_d4(d4)
            d4 = tf.concat([d4, e4], 3)

            d5, _, _ = deconv2d(tf.nn.relu(d4), [batch_size, 30, 30, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_L_d5', with_w=True)
            d5 = self.g_L_bn_d5(d5)
            d5 = tf.concat([d5, e3], 3)

            d6, _, _ = deconv2d(tf.nn.relu(d5), [batch_size, 60, 60, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_L_d6', with_w=True)
            d6 = self.g_L_bn_d6(d6)
            d6 = tf.concat([d6, e2], 3)

            d7, _, _ = deconv2d(tf.nn.relu(d6), [batch_size, 120, 120, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_L_d7', with_w=True)
            d7 = self.g_L_bn_d7(d7)
            d7 = tf.concat([d7, e1], 3)

            d8, _, _ = deconv2d(tf.nn.relu(d7), [batch_size, 240, 240, self.output_c_dim], k_h=6, k_w=6, d_h=2,
                                d_w=2,
                                name='g_L_d8', with_w=True)

            return tf.nn.relu(tf.nn.sigmoid(d8))

    def generator_small(self, image, batch_size, reuse=False):
        """
        Small生成器网络

        :param image: 输入数据
        :param batch_size:
        :param reuse:
        :return: 生成图片
        """
        with tf.variable_scope("generator_small"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False
            # (120 x 120 x input_c_dim) --> e1(60 x 60 x 64) --> e2(30 x 30 x 64) --> e3(15 x 15 x 64) -->
            # e4(8 x 8 x 64) --> e5(4 x 4 x 64) --> e6(2 x 2 x 64) --> e7(2 x 2 x 64) <-- d1(2 x 2 x 64*2) <--
            # d2(4 x 4 x 64*2) <-- d3(8 x 8 x 64*2) <-- d4(15 x 15 x 64*2) <-- d5(30 x 30 x 64*2) <--
            # d6(60 x 60 x 64*2) <-- d7(240 x 240 x output_c_dim)
            e1 = self.g_S_bn_e1(conv2d(image, output_dim=64, k_h=6, k_w=6, d_h=2, d_w=2, name='g_S_e1_con'))
            e2 = self.g_S_bn_e2(conv2d(lrelu(e1), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_S_e2_con'))
            e3 = self.g_S_bn_e3(conv2d(lrelu(e2), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_S_e3_con'))
            e4 = self.g_S_bn_e4(conv2d(lrelu(e3), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_S_e4_con'))
            e5 = self.g_S_bn_e5(conv2d(lrelu(e4), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_S_e5_con'))
            e6 = self.g_S_bn_e6(conv2d(lrelu(e5), output_dim=64, k_h=4, k_w=4, d_h=2, d_w=2, name='g_S_e6_con'))
            e7 = self.g_S_bn_e7(conv2d(lrelu(e6), output_dim=64, k_h=4, k_w=4, d_h=1, d_w=1, name='g_S_e7_con'))

            d1, _, _ = deconv2d(lrelu(e7), [batch_size, 2, 2, 64], k_h=4, k_w=4, d_h=1, d_w=1,
                                name='g_S_d1', with_w=True)
            d1 = tf.nn.dropout(self.g_S_bn_d1(d1), 0.5)
            d1 = tf.concat([d1, e6], 3)

            d2, _, _ = deconv2d(tf.nn.relu(d1), [batch_size, 4, 4, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_S_d2', with_w=True)
            d2 = tf.nn.dropout(self.g_S_bn_d2(d2), 0.5)
            d2 = tf.concat([d2, e5], 3)

            d3, _, _ = deconv2d(tf.nn.relu(d2), [batch_size, 8, 8, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_S_d3', with_w=True)
            d3 = tf.nn.dropout(self.g_S_bn_d3(d3), 0.5)
            d3 = tf.concat([d3, e4], 3)

            d4, _, _ = deconv2d(tf.nn.relu(d3), [batch_size, 15, 15, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_S_d4', with_w=True)
            d4 = self.g_S_bn_d4(d4)
            d4 = tf.concat([d4, e3], 3)

            d5, _, _ = deconv2d(tf.nn.relu(d4), [batch_size, 30, 30, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_S_d5', with_w=True)
            d5 = self.g_S_bn_d5(d5)
            d5 = tf.concat([d5, e2], 3)

            d6, _, _ = deconv2d(tf.nn.relu(d5), [batch_size, 60, 60, 64], k_h=4, k_w=4, d_h=2, d_w=2,
                                name='g_S_d6', with_w=True)
            d6 = self.g_S_bn_d6(d6)
            d6 = tf.concat([d6, e1], 3)

            d7, _, _ = deconv2d(tf.nn.relu(d6), [batch_size, 120, 120, self.output_c_dim], k_h=4, k_w=4, d_h=2,
                                d_w=2,
                                name='g_S_d7', with_w=True)

            return tf.nn.relu(tf.nn.sigmoid(d7))

    def discriminator_large(self, image, batch_size, reuse=False):
        """
        Large判别器
        :param image:
        :param batch_size
        :param reuse:
        :return:
        """
        with tf.variable_scope("discriminator_large"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            # (240 x 240 x (input_c_dim + output_c_dim)) --> (120 x 120 x 48) --> (60 x 60 x 96) -->
            # (30x 30 x 192) --> (30 x 30 x 384) --> (30 x 30 x 1)
            h0 = lrelu(self.d_L_bn0(conv2d(image, 48, k_h=4, k_w=4, d_h=2, d_w=2, name='d_L_h0_con')))
            h1 = lrelu(self.d_L_bn1(conv2d(h0, 96, k_h=4, k_w=4, d_h=2, d_w=2, name='d_L_h1_con')))
            h2 = lrelu(self.d_L_bn2(conv2d(h1, 192, k_h=4, k_w=4, d_h=2, d_w=2, name='d_L_h2_con')))
            h3 = lrelu(self.d_L_bn3(conv2d(h2, 384, k_h=4, k_w=4, d_h=1, d_w=1, name='d_L_h3_con')))
            h4 = lrelu(self.d_L_bn4(conv2d(h3, 1, k_h=4, k_w=4, d_h=1, d_w=1, name='d_L_h4_con')))
            l_h4 = linear(tf.reshape(h4, [batch_size, -1]), 1, 'd_L_h4_lin')

            return l_h4, tf.nn.tanh(l_h4)

    def discriminator_small(self, image, batch_size, reuse=False):
        """
        Small判别器
        :param image:
        :param batch_size
        :param reuse:
        :return:
        """
        with tf.variable_scope("discriminator_small"):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse is False

            # (120 x 120 x (input_c_dim + output_c_dim)) --> (60 x 60 x 48) --> (30 x 30 x 96) -->
            # (15 x 15 x 192) --> (15 x 15 x 384) --> (15 x 15 x 1)
            h0 = lrelu(self.d_S_bn0(conv2d(image, 48, k_h=4, k_w=4, d_h=2, d_w=2, name='d_S_h0_con')))
            h1 = lrelu(self.d_S_bn1(conv2d(h0, 96, k_h=4, k_w=4, d_h=2, d_w=2, name='d_S_h1_con')))
            h2 = lrelu(self.d_S_bn2(conv2d(h1, 192, k_h=4, k_w=4, d_h=2, d_w=2, name='d_S_h2_con')))
            h3 = lrelu(self.d_S_bn3(conv2d(h2, 384, k_h=4, k_w=4, d_h=1, d_w=1, name='d_S_h3_con')))
            h4 = lrelu(self.d_S_bn4(conv2d(h3, 1, k_h=4, k_w=4, d_h=1, d_w=1, name='d_S_h4_con')))
            l_h4 = linear(tf.reshape(h4, [batch_size, -1]), 1, 'd_S_h4_lin')

            return l_h4, tf.nn.tanh(l_h4)

    def save(self, checkpoint_dir, step):
        model_name = "mp_gan.model"
        model_dir = "%s_%s_%s" % (self.data_set_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
