"""
案例:
    ANN(人工神经网络案例):手机价格分类

背景:
    基于手机的20例特征 -> 预测手机的价格区间(4个区间),

ANN案例的实现步骤:
    1.构建数据集
    2.搭建神经网络
    3.模型训练
    4.模型测试

调优思路:
    1.特征标准化（最可能带来大幅提升）
    2.确认标签类型是 long
    3.用测试集评估 + 记录 loss 曲线
"""

#导包
import torch                                    #pytorch框架,封装了张量的各种操作
from sklearn.preprocessing import StandardScaler
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset      #数据集对象.数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader         #数据加载器
import torch.nn as nn                           #neural network,封装了神经网络的各种操作
import torch.optim as optim                     #优化器
from sklearn.model_selection import train_test_split    #训练集和测试集的划分
import matplotlib.pyplot as plt                 #绘图
import numpy as np                              #数组矩阵操作
import pandas as pd                             #数据处理
import time                                     #时间模块

from torchsummary import summary


# todo 1.定义函数,构建数据集
def creat_dataset():
    #1.加载csv文件数据
    data = pd.read_csv(r'D:\桌面\python\DeepLearn\day05\data\手机价格预测.csv')
    # print(data.head())
    # print(data.shape)

    #2.获取x特征列 和 y标签列
    x,y = data.iloc[:,:-1], data.iloc[:,-1]
    # print(f'x:{x.head()},{x.shape}')    # (2000,20)
    # print(f'y:{y.head()},{y.shape}')    # (2000,1)

    #3.把特征列转成浮点型
    x = x.astype('float32')
    # print(f'x:{x.head},{x.shape}')

    #4.切分训练集和测试集
    #参1:特征;参2:标签;参3:测试集所占比例;参4:随机种子;参5:样本分布(即参考y的类别抽取数据)
    x_train,x_valid,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

    #优化1:数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_valid  = transfer.transform(x_valid)

    #5.把数据集封装成 张量数据集.思路:数据 -> 张量 -> 数据集TensorDataset -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train,dtype=torch.float32),torch.tensor(y_train.values,dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(x_valid,dtype=torch.float32),torch.tensor(y_test.values,dtype=torch.long))
    # print(train_dataset)

    #返回结果                           20(充当 输入特征数)        4(充当输出标签)
    return train_dataset,test_dataset,x_train.shape[1],len(np.unique(y))


# todo 2.搭建神经网络
class PhonePriceModel(nn.Module):
    #1.在init魔法方法中,初始化父类成员,搭建神经网络
    def __init__(self,input_dim,output_dim):
        #1.1初始化父类成员
        super().__init__()
        #1.2搭建神经网络
        #层数多了,容易过拟合
        #优化2:修改深度以及每层神经元个数
        self.linear1 = nn.Linear(input_dim,32)
        self.linear2 = nn.Linear(32,64)
        self.output = nn.Linear(64,output_dim)

    #2.定义前向传播方法 forward()
    def forward(self,x):
        #2.1隐藏层1:加权求和 + 激活函数(relu)
        x = torch.relu(self.linear1(x))
        #2.2隐藏层2:加权求和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))
        #2.3输出层:加权求和 + 激活函数(softmax) -> 这里只需要做 加权求和
        #正常写法,但不需要,后续用多分类交叉熵损失函数CrossEntropyLoss()代替
        # x = torch.softmax(self.output(x),dim=1)
        x = self.output(x)
        return x



# todo 3.模型训练
def train(train_dataset,input_dim,oupput_dim):
    #1.创建数据加载器
    #参1:数据集对象(1600条),参2:每批次的数据条数,参3:是否打乱数据(训练集打乱,测试集不打乱)
    train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
    #2.创建神经网络模型
    model = PhonePriceModel(input_dim,oupput_dim)
    #3.定义损失函数
    criterion = nn.CrossEntropyLoss()
    #4.创建优化器对象
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    # optimizer = optim.Adam(model.parameters(),lr=0.001)
    #5.模型训练
    epochs = 50     #训练轮数要选择合适,多了后面跑空,浪费资源
    train_losses = []       #优化:增加拟合曲线图
    for epoch in range(epochs):
        total_loss = 0.0
        batch_num = 0
        start = time.time()
        for x,y in train_loader:
            model.train()               #训练模式
            y_pred = model(x)           #模型预测
            loss = criterion(y_pred,y)  #计算损失
            optimizer.zero_grad()   #梯度清零
            loss.backward()         #反向传播
            optimizer.step()        #参数更新
            total_loss += loss.item()   #本轮每批次16条的平均损失累计
            batch_num += 1
        train_losses.append(total_loss / batch_num)
        print(f'epoch {epoch+1}, loss: {total_loss/batch_num:.4f},time: {time.time()-start:.2f}s')
    #绘图
    plt.plot(train_losses)
    plt.xlabel('epoch');
    plt.ylabel('loss');
    plt.title('Training Loss')
    plt.savefig('./loss_curve.png')     #保存图片

    # print(f'\n\n模型参数的信息:{model.state_dict()}\n\n')
    torch.save(model.state_dict(),'./model/phone.model.pth')    #保存模型参数


# todo 4.模型测试
def evaluate(test_dataset,input_dim,oupput_dim):
    model = PhonePriceModel(input_dim,oupput_dim)           #创建神经网络分类对象
    model.load_state_dict(torch.load('./model/phone.model.pth'))    #加载模型参数
    test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)  #创建测试集 数据加载对象
    correct = 0     #定义变量,获取预测正确的样本个数
    model.eval()  # 切换模式 -> 预测模式
    for x,y in test_loader:
        y_pred = model(x)
        y_pred = torch.argmax(y_pred,dim=1)
        # print(f'y_pred:{y_pred}')       #打印预测分类
        # print(f'y:{y}')                 #打印真实分类
        # print(y_pred == y)              #预测值与真实值是否一致
        # print((y_pred == y).sum())      #一致的个数
        correct += (y_pred == y).sum().item()

    print(f'准确率:{correct/len(test_dataset):.4f}')


#todo 5.测试
if __name__ == '__main__':
    #1.准备数据集
    train_dataset,test_dataset,input_dim,oupput_dim = creat_dataset()
    # print(f'训练集 数据集对象:{train_dataset}')
    # print(f'测试集 数据集对象:{test_dataset}')
    # print(f'输入特征量:{input_dim}')
    # print(f'输出标签:{oupput_dim}')

    #2.搭建神经网络模型
    # model = PhonePriceModel(input_dim,output_dim)
    # 计算模型参数
    # summary(model,input_size=(16,input_dim))

    # 模型训练
    train(train_dataset,input_dim,oupput_dim)

    #模型测试
    evaluate(test_dataset,input_dim,oupput_dim)
