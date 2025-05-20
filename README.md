## 任务描述

微调在ImageNet上预训练的卷积神经网络实现Caltech-101分类

基本要求：

(1) 训练集测试集按照 [Caltech-101]( [https://data.caltech.edu/records/mzrjq-6wc02Links to an external site.](https://data.caltech.edu/records/mzrjq-6wc02)) 标准；
(2) 修改现有的 CNN 架构（如AlexNet，ResNet-18）用于 Caltech-101 识别，通过将其输出层大小设置为 **101** 以适应数据集中的类别数量，其余层使用在ImageNet上预训练得到的网络参数进行初始化；
(3) 在 Caltech-101 数据集上从零开始训练新的输出层，并对其余参数使用较小的学习率进行微调；
(4) 观察不同的超参数，如训练步数、学习率，及其不同组合带来的影响，并尽可能提升模型性能；
(5) 与仅使用 Caltech-101 数据集从随机初始化的网络参数开始训练得到的结果 **进行对比**，观察预训练带来的提升。



## 环境配置

建议使用虚拟环境

```
conda create -n cv_project python=3.8 -y
conda activate cv_project
```

安装pytorch

```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

项目依赖的工具包如下

```
conda install -c conda-forge matplotlib pandas scikit-learn tqdm pyyaml jupyter pillow
```

```
conda install -c conda-forge tensorboard
```



## 模型训练及测试

### 参数设置

您可以通过进入 `generate_configs,py` 设置参数列表和模型标号，点击运行可以在 `model_data` 下生成指定的标号文件夹和模型参数文件。

```python
data_dir = ["./data/caltech101"]
batch_size = [16, 32, 64]
epochs = [30]
lr = [0.00005, 0.0001, 0.00015, 0.0002, 0.0003]
weight_decay = [0.0005]
pretrained = [True, False]
fine_tune = [True]

# 设置模型初始下标，批量生成模型参数文件
start_index = 1
```

### 模型训练

在开始训练前，您需要确保数据集存储在 `data/caltech101` 文件夹下。具体的存储格式可见项目文档说明。

进入 `train.py` 可以通过指定需要训练的模型标号，进行批量模型训练。训练得到的权重将保存在模型目录下的 `best_alexnet.pth`，训练日志将保存在模型目录下的 `log` 文件夹。

### 模型测试

进入 `evaluate.py` 可以通过指定需要测试的模型标号，进行批量模型测试。测试记录将保存在模型目录下的 `log` 文件夹。

### 使用tensorflow查看模型结果

进入项目根目录后在终端中运行以下命令：

```python
# 启动TensorBoard
tensorboard --logdir=model_data
```

打开默认的链接 ` http://localhost:6006/` 即可查看训练结果。



## 项目文件说明

#### data/caltech101

保存了 caltech101 数据集中的图片。目录下图片按照类别保存在各自的文件夹中。

#### datasets/caltech101.py

用于数据集加载和训练集，验证集，测试集划分。

#### images

以 png 形式保存部分训练结果图像。这部分图像同样可以通过 tensorflow 进行查看。

#### model_data

存储了所有训练过程中涉及到的模型，模型参数设置，权重以及日志等。每个模型具有唯一的文件夹。下面以模型1为例介绍包含的信息。

##### alexnet.yaml

包含模型参数设置，共七个参数：

```
batch_size: 16
data_dir: ./data/caltech101
epochs: 30
fine_tune: true
lr: 5.0e-05
pretrained: true
weight_decay: 0.0005
```

##### best_alexnet.pth

包含训练得到的模型权重。

##### logs

包含 `evaluate`, `plots`, `tensorboard`, `training_logs` 四个部分。

`evaluate`: 包含模型对101个类别的图像的分类预测结果，混淆矩阵，以及详细的true-pred记录。

`plots`: 存储该模型训练得到的 accuracy, loss 曲线。

`tensorboard`: 模型训练记录，可以通过tensorboard可视化工具查看。注意，如果一个模型设置被重复训练，将在该目录下生成多次记录。

`training_logs`: 训练epoch, train_loss, val_loss, val_acc 的表格记录。

