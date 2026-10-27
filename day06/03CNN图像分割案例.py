"""
案例:
    演示CNN综合案例,图像分类
回顾:深度学习项目步骤
    1.准备数据集
        这里使用 计算机视觉模块 torchvision自带的CIFAR10数据集,包含6w张(32,32,3)的图片,5w张训练集,1w张测试集,每个分类6k张图片
    2.搭建卷积神经网络
    3.模型训练
    4.模型测试
卷积层:
    提取图片局部特征
池化层:
    降维
"""
#导包
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

#每批次样本数
BATCH_SIZE = 8

# 1.准备数据集
def create_dataset():
    #1.获取训练集
    #参1:数据集路径;参2:是否是训练集;参3:数据预处理->张量数据;参4:是否联网下载
    train_dataset = CIFAR10(root='./data', train=True,transform=ToTensor(),download=True)
    test_dataset = CIFAR10(root='./data', train=False,transform=ToTensor())
    #2.返回数据集
    return train_dataset,test_dataset


# 2.搭建卷积神经网络
class ImageModel(nn.Module):
    #1.初始化父类成员,并搭建神经网络
    def __init__(self):
        #1.1初始化父类成员
        super().__init__()
        #1.2搭建神经网络
        #第一个卷积层,池化层
        self.conv1 = nn.Conv2d(3, 6, 3,1)
        self.pool1 = nn.MaxPool2d(2, 2,0)
        #第二个卷积层 池化层
        self.conv2 = nn.Conv2d(6, 16, 3,1)
        self.pool2 = nn.MaxPool2d(2, 2,0)
        #全连接层
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output = nn.Linear(84, 10)

        #定义前向传播
    def forward(self, x):
        #第一层
        x = self.pool1(torch.relu(self.conv1(x)))
        #第二层
        x = self.pool2(torch.relu(self.conv2(x)))
        #第三层
        #全连接层只能处理二维数据,所以要先将数据进行展平
        x = x.reshape(x.size(0), -1)    #8行576列
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.output(x)


# 3.模型训练
def train(train_dataset):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #转GPU速度快些
        # model = ImageModel().to(device)
    #1.创建数据加载器
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    #2.创建模型对象
    model = ImageModel()
    #3.创建损失函数对象
    criterion = nn.CrossEntropyLoss()
    #4.创建优化器对象
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #5.循环遍历epoch,开始每轮的训练
    epochs = 10
    for epoch_idx in range(epochs):
        total_loss,total_samples,total_correct,start = 0.0,0,0,time.time()
        for x,y in dataloader:
            # x, y = x.to(device), y.to(device)
            model.train()               #切换训练模式
            y_pred = model(x)           #模型预测
            loss = criterion(y_pred,y)  #计算损失
            optimizer.zero_grad()       #梯度清零
            loss.backward()             #反向传播
            optimizer.step()            #参数更新

            #统计预测正确的样本个数
            total_correct += (torch.argmax(y_pred,dim=1) == y).sum()
            #统计当前批次的总损失
            total_loss += loss.item() * len(y)
            #统计当前批次的总样本数
            total_samples += len(y)
        #结束一轮训练,打印该轮信息
        print(f'epoch:{epoch_idx+1},loss:{total_loss/total_samples},acc:{total_correct/total_samples:.2f},time:{time.time()-start:.2f}s')
        #保存模型
        torch.save(model.state_dict(),'./model/image_model.pth')


# 4.模型测试
def evaluate(test_dataset):
    #1.创建测试集 数据加载器
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    #2.创建模型对象
    model = ImageModel()
    #3.加载模型参数
    model.load_state_dict(torch.load('./model/image_model.pth'))
    #4.定义变量
    total_correct, total_samples = 0,0
    for x,y in dataloader:
        model.eval()
        y_pred = model(x)
        total_correct += (torch.argmax(y_pred,dim=1) == y).sum()
        total_samples += len(y)

    print(f'预测结果:{total_correct/total_samples:.2f}')


# 5.测试
if __name__ == '__main__':
    #读取数据
    train_dataset,test_dataset = create_dataset()
    # print(f'训练集:{train_dataset.data.shape}')    # (50000, 32, 32, 3)
    # print(f'测试集:{test_dataset.data.shape}')     # (10000, 32, 32, 3)
    # # {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    # print(f'数据集的类别:{train_dataset.class_to_idx}')

    #搭建神经网络
    # model = ImageModel()
    #查看模型参数
    # summary(model,(3,32,32),batch_size=BATCH_SIZE)

    #模型训练
    # train(train_dataset)

    #模型测试
    evaluate(test_dataset)