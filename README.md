>>>>>>> HEAD

# ACSCP crowd counting model
=======
[![License](http://gplv3.fsf.org/gplv3-127x51.png)](LICENSE)
## Introduction
This is open source project for crowd counting. Implement with paper "Crowd Counting via Adversarial Cross-Scale Consistency Pursuit" from Shanghai Jiao Tong University.  For more details, please refer to our [Baidu Yun](https://pan.baidu.com/s/1mjPpKqG)

<p align="center">
<img src="doc/motivations.png" alt="multimotivations-scale block" width="400px">
</p>
<p align="center">
<img src="doc/loss.png" alt="loss" width="380px">
</p>
<p align="center">
<img src="doc/generator.png" alt="generator" width="380px">
</p>
<p align="center">
<img src="doc/architecture.png" alt="architecture" width="560px">
</p>
<p align="center">
<img src="doc/comparision.png" alt="comparision" width="600px">
</p>
<p align="center">
<img src="doc/loss_result.png" alt="loss_result" width="320px">
</p>
<p align="center">
<img src="doc/pathch_errors.png" alt="pathch_errors" width="320px">
</p>
<p align="center">
<img src="doc/result_ShanghaiTech.png" alt="result_ShanghaiTech" width="240px">
</p>
<p align="center">
<img src="doc/lambda_c.png" alt="lambda_c" width="320px">
</p>
<p align="center">
<img src="doc/tensorboard.png" alt="tensorboard" width="800px">
</p>

### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval/Release](#trainevalrelease)
4. [Additional](#addtional)
5. [Details](#details)

### Installation
1. Configuration requirements

```
python3.x

Please using GPU, suggestion more than GTX960

python-opencv
#tensorflow-gpu==1.0.0
#tensorflow==1.0.0
scipy==1.0.1
matplotlib==2.2.2
numpy==1.14.2

conda install -c https://conda.binstar.org/menpo opencv3
pip install -r requirements.txt
```

2. Get the code

```
git clone git@github.com:Ling-Bao/ACSCP_cGAN.git
cd ACSCP_cGAN
```

### Preparation
1. ShanghaiTech Dataset. 
ShanghaiTech Dataset makes by Zhang Y, Zhou D, Chen S, et al. For more detail, please refer to paper "Single-Image Crowd Counting via Multi-Column Convolutional Neural Network" and click on [here](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Zhang_Single-Image_Crowd_Counting_CVPR_2016_paper.pdf).

2. Get dataset and its corresponding map label
[Baidu Yun](https://pan.baidu.com/s/1gccvnvIeLgQZCVuA6iZEjA) 
Password: yvs1

3. Unzip dataset to ACSCP_cGAN root directory

```
unzip Data.zip
```

### Train/Eval/Release
Train is easy, just using following step.

1. Train. Using [main.py](main.py) to evalute crowd counting model

```
python main.py --phase train
```

2. Eval. Using [main.py](main.py) to evalute crowd counting model

```
python main.py --phase test

OR

python main.py --phase inference
```

### Addtional
1. Crowd map generation tools
Source code store in "data_maker", detail please check [here](data_maker/README.md).
**Note: **This tools write by matlab, please install matlab.

2. Model release
 Model release. Using [product.py](product.py) to release crowd counting model. Download release version 0.1.0, please click on [here](release/version1.0.0.tar.gz)
 
 3. Results
<ul align="center">
<img src="data/data_im/test_im/IMG_2_A.jpg" alt="formulation" width="240px">
<p align="center">Original image</p>
<img src="IMG_2_A_real.png" alt="formulation" width="240px">
<p align="center">Real crowd map, counting is 707</p>
<img src="IMG_2_A.png" alt="formulation" width="240px">
<p align="center">Predict crowd map, counting is 698</p>
</ul>

### Details
1. Tring to delete dropout layers.

2. Improving activation funtion for last layer to adapt crowd counting map estimation.
<p align="center">
<img src="doc/formulation.png" alt="formulation" width="240px">
</p>

=======
[![License](http://gplv3.fsf.org/gplv3-127x51.png)](LICENSE)
>>>>>>> TAIL
