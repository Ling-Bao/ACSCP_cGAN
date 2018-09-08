# 人数密度估计标签制作文档
```
@Author: Ling Bao
@Date: 星期二, 11. 七月 2017 04:09下午 
@Version: 0.0.1
```

## 1 Step1
:	**（1）采集训练/测试数据集合**
	> **功能描述：**从给定视频中获取训练/测试数据集合，图片尺寸大小限制条件：width{600-900} || height{900-1300}；随机Crop。
:	**（2）数据采集函数**
	> **data_collect.py**
	data_collect(type='train', nums=200)
   	>> ***@Brief:*** *用于从视频中采集训练/测试数据集合*
   	>> *** :param type:*** *train or test*
	>> ***:param nums:*** *采集训练/测试数据集样本个数*
	>> ***:return:*** *None*
:	**（3）使用方法**
	> python data_collect.py

---
## 2 Step2 利用标签制作工具
:	**（1）密度图预标签制作**
	> 将待制作数据拷贝到文件夹 **Image_data**；
	**运行程序pro_mark0/data_marker**打开标签制作工具，按照流程即可完成预标签数据集合的制作。
:	**（2）密度图预标签验证**
	> 利用**pro_check/data_marker**对制作好的预标签进行验证。

---
## 3 Step3 生成标签密度图
:	**（1）预标签转密度图**
	> **map_mcnn.m**
	> **acscp.m**
:	**（2）.mat转换.npy**
	> **mat2npy.py**