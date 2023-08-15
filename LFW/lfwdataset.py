import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch as t

class MyDataSet(Dataset):
    '''
    定义数据集, 用于将读取到的图片数据转换并处理成CNN神经网络需要的格式
    '''
    def __init__(self, DataArray, LabelArray):
        super(MyDataSet, self).__init__()
        self.data = DataArray
        self.label = LabelArray

    def __getitem__(self, index):
        # 对图片的预处理步骤
        # 1. 中心缩放至224(ResNet的输入大小)
        # 2. 随机旋转0-30°
        # 3. 对图片进行归一化，参数来源为pytorch官方文档
        im_trans = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(size=64),
            # transforms.RandomCrop(64, padding=4),
            transforms.RandomRotation((0, 30)), # 随机旋转0-30°
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        return im_trans(self.data[index]), t.tensor(self.label[index], dtype=t.long)

    def __len__(self):
        return self.label.shape[0]

# 读取LFW数据集，将图片数据读入数组并将名字转换为标签
path = r'lfw_funneled'
pathlist = map(lambda x: '/'.join([path, x]), os.listdir(path))
namedict = {}
data, label = [], []
idx = 0
for item in pathlist:
    if not os.path.isdir(item):
        continue
    dirlist = os.listdir(item)
    # 选取拥有30-100张照片的人作为数据来源
    # 太少网络不容易学习到其人脸特征，太多的话则容易过拟合
    if not (30<= len(dirlist) <= 50):
        continue
    # data:     存储人像照片的三通道数据
    # label:    存储人像的对应标签(整数)
    # namedict: 记录label中整数与人名的对应关系
    for picpath in dirlist:
        data.append(image.imread(item + '/' + picpath))
        label.append(idx)
    namedict[str(idx)] = item.split('/')[-1]
    idx += 1

# 随机打乱数据，重新排序并按照8:2的比例分割训练集和测试集
data, label = np.stack(data), np.array(label)

idx = np.random.permutation(data.shape[0])
data, label = data[idx], label[idx]
train_X, test_X, train_Y, test_Y = train_test_split(data, label, test_size=0.2)

# 将分割好的训练集和测试集处理为pytorch所需的格式
TrainSet = MyDataSet(train_X, train_Y)
TestSet = MyDataSet(test_X, test_Y)
# train_loader = DataLoader(TrainSet, batch_size=1, shuffle=True, drop_last=True)
# test_loader = DataLoader(TestSet, batch_size=1, shuffle=True, drop_last=True)
# print(len(namedict))
# for i, (input, target) in enumerate(train_loader):
#     print(input.shape)
#     print(target.shape)
#     break