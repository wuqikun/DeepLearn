"""
案例:
    演示池化层相关操作

池化层解释(Pooling);
    目的:
        降维
    思路:
        最大池化
        平均池化
    特点:
        池化不会改变数据的 通道数
"""
#导包
import torch
import torch.nn as nn


#1.定义函数,演示单通道池化
def dm01():
    #创建一个1通道 3x3 的二维矩阵
    inputs = torch.tensor([[
        [0,1,2],
        [3,4,5],
        [6,7,8]
    ]])
    # print(inputs.shape)
    #创建最大池化层
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
    outputs = pool1(inputs)
    print(outputs,outputs.shape)
    #创建平均池化
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1)
    outputs = pool2(inputs)
    print(outputs,outputs.shape)

#2.定义函数,演示多通道池化
def dm02():
    #创建一个1通道 3x3 的二维矩阵
    inputs = torch.tensor([
        [
        [0,1,2],
        [3,4,5],
        [6,7,8]
    ],
    [
        [10,20,30],
        [30,40,50],
        [60,70,80]
    ],
    [
        [11,22,33],
        [44,55,66],
        [77,88,99]
    ]
    ])
    # print(inputs.shape)
    #创建最大池化层
    pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
    outputs = pool1(inputs)
    print(outputs,outputs.shape)
    #创建平均池化
    pool2 = nn.AvgPool2d(kernel_size=2, stride=1)
    outputs = pool2(inputs)
    print(outputs,outputs.shape)

#3.测试
if __name__ == '__main__':
    # dm01()
    dm02()