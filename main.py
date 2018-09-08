# -*- coding:utf-8 -*-
"""
@Brief
main function：parameter setting
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
import argparse
import os
import cv2
from matplotlib import pyplot as plt_model

# 机器学习库
import tensorflow as tf

# 项目库
from model import ACSCP
from product import ProductMap

# 参数设置
parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_set_name', dest='data_set_name', default='gan_mp', help='数据集名')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='迭代步数')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='批量大小')
parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='被用于训练的图片最大数量')
parser.add_argument('--load_size', dest='load_size', type=int, default=720, help='输入图像尺寸')
parser.add_argument('--fine_size', dest='fine_size', type=int, default=240, help='裁剪尺寸')
parser.add_argument('--input_nc', dest='input_nc', type=int, default=3, help='输入图片的通道数')
parser.add_argument('--output_nc', dest='output_nc', type=int, default=3, help='输出图片的通道数')
parser.add_argument('--lr', dest='lr', type=float, default=0.00005, help='初始学习率')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='动量因子')
parser.add_argument('--beta2', dest='beta2', type=float, default=0.999, help='RMSProp因子')
parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
parser.add_argument('--phase', dest='phase', default='train', help='train, test, inference, product')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoints/', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample/', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test/', help='test sample are saved here')
parser.add_argument('--lambda_e', dest='lambda_e', type=float, default=150, help='weight on L2 term')
parser.add_argument('--lambda_p', dest='lambda_p', type=float, default=150, help='weight on perceptual loss term')
parser.add_argument('--lambda_c', dest='lambda_c', type=float, default=10, help='weight on Cross-Scale term')
parser.add_argument('--train_im_dir', dest='train_im_dir', default='./data/data_im/train_im/', help='训练图片路径')
parser.add_argument('--train_gt_dir', dest='train_gt_dir', default='./data/data_gt/train_gt/', help='训练密度图路径')
parser.add_argument('--test_im_dir', dest='test_im_dir', default='./data/data_im/test_im/', help='测试图片路径')
parser.add_argument('--test_gt_dir', dest='test_gt_dir', default='./data/data_gt/test_gt/', help='测试密度图路径')
args = parser.parse_args()


def main(_):
    """
    CGAN主函数入口

    :param _:
    :return:
    """
    # 创建训练/测试过程中所需的文件目录
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    # 创建会话/构建CGAN网络/训练网络/测试网络
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.80)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)) as sess:
        if args.phase == 'product':
            # 载入图像
            img_path = "data/data_im/test_im/"
            img_name = "IMG_2_A"
            image = cv2.imread(img_path + img_name + ".jpg")
            if image is None:
                print("Please check image path!!")
                return -1

            # 人群密度估计
            # product = ProductMap(True)
            product = ProductMap(False)
            # product.generator_large_save()  # 仅用于提取generator_larger模型参数并保存
            product.generator_large_load()
            mp, run_time = product.run(image)

            # 保存估计的人群密度图
            mp_name = img_name
            print("Time: %4.4f, Estimation numbers: %4d" % (run_time, round(sum(sum(mp)))))
            plt_model.imsave(mp_name + ".png", mp, cmap=plt_model.get_cmap('jet'))
        else:
            model = ACSCP(sess, image_size=args.fine_size, batch_size=args.batch_size, sample_size=1, output_size=240,
                          df_dim=48, input_c_dim=3, output_c_dim=3, data_set_name='gan_mp',
                          checkpoint_dir=args.checkpoint_dir, lambda_e=args.lambda_e, lambda_p=args.lambda_p,
                          lambda_c=args.lambda_c)

            if args.phase == 'train':
                model.train(args)
            elif args.phase == 'test':
                model.test(args)
            elif args.phase == 'inference':
                # 载入图像
                img_path = "data/data_im/test_im/"
                img_name = "IMG_1_A"
                image = cv2.imread(img_path + img_name + ".jpg")
                if image is None:
                    print("Please check image path!!")
                    return -1

                # 人群密度估计
                mp, run_time = model.inference(img=image, mp_name=img_name + "_mp")

                # 保存估计的人群密度图
                mp_name = img_name
                print("Time: %4.4f, Estimation numbers: %4d" % (run_time, round(sum(sum(mp)))))
                plt_model.imsave(mp_name + ".png", mp, cmap=plt_model.get_cmap('jet'))
            else:
                print("args.phase is train, test or inference!!")


if __name__ == '__main__':
    tf.app.run()
