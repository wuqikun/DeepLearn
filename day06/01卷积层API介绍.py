"""
案例:
    演示卷积层的API,用于提取图像的局部特征,获取特征图(Feature Map)

卷积神经网络介绍:
    概述:
        全称叫(Convolution nevral network),即包含卷积的神经网络
    组成:
        卷积层(convolution):
            用于提取图像局部特征,结合 卷积核(每个卷积核 = 1个神经元) 实现,处理后的结果叫特征图
        池化层:(pooling):
            用于降维 降采样
        全连接层(Full connected,fc,linear):
        用于 预测结果,并输出结果
特征图计算方式:
    N = (W - F + 2*p) / s  +  1
    W:输入图像的大小,例如 5x5的图像,w=5
    F:卷积核大小
    P:填充的层数
    S:步长
    N:输出图像的大小(特征图)
"""
#导包
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#1定义函数,用于完成图像的加载,卷积,特征图可视化
def dm01():
    #1.加载RGB真彩图
    img = plt.imread('./image/dili.jpg')
    #2.打印图片信息
    # print(f'img:{img}, shape:{img.shape}')  #shape: HWC (681, 500, 3)
    #3.把图形形状HWC -> CHW,思路: img -> 张量 -> 转换维度
    img2 = torch.tensor(img, dtype=torch.float32)
    img2 = img2.permute(2, 0, 1)
    # print(f'img2:{img2}, shape:{img2.shape}')
    #4.因为这里只有一张图,所以给他增加一个维度,从 CHW -> (1,C,H,W),一张3通道的 681x500像素的图
    img3 = img2.unsqueeze(dim=0)
    # print(img3.shape)
    #5.创建卷积层对象,提取特征图
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=1)
    #具体卷积计算
    conv_img = conv(img3)
    #7.打印卷积后的结果
    # print(conv_img.shape)
    #8.查看提取到的4个特征图
    img4 = conv_img[0]
    #9.将上述图像从CHW -> HWC
    img5 = img4.permute(1, 2, 0)
    #10.可视化第一个通道的特征图
    feature1 = img5[:,:,1].detach().numpy()
    plt.imshow(feature1)
    plt.show()



#2.测试
if __name__ == '__main__':
    dm01()
