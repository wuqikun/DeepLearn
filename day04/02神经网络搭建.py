"""
案例:
    演示神经网络搭建流程

神经网络搭建流程:
    1.定义一个类,继承: nn.Module
    2.在 __init__()方法中,搭建神经网络
    3.在 forward()方法中,完成前向传播
"""
#导包
import torch
import torch.nn as nn
from torchsummary import summary


#todo:1搭建神经网络, 即自定义继承 nn.Madule
class ModelDemo(nn.Module):
    def __init__(self):
        #初始化父类成员
        super().__init__()
        #搭建神经网络 -> 隐藏层 + 输出层
        self.linear1 = nn.Linear(3, 3)
        self.linear2 = nn.Linear(3, 2)
        self.output = nn.Linear(2, 2)

        #初始化隐藏层参数
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    #前向传播
    def forward(self,x):
        # x = self.linear1(x)     #加权求和
        # x = torch.sigmiod(x)    #激活函数

        #合并写法
        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.output(x),dim=-1)#dim=-1,表示按行计算
        #返回预测值
        return x


#todo:2模型训练
def train():
    #创建模型对象
    my_model = ModelDemo()

    #创建数据集 => 数入数据
    data = torch.randn(5,3)
    print(f'data: {data}')
    print(f'data.shape: {data.shape}')
    print(f'data.requires_grad: {data.requires_grad}')

    #调用模型
    output = my_model(data)     #底层自动调用 forward()方法,进行前向传播
    print(f'output: {output}')
    print(f'output.shape: {output.shape}')                  #最终会输出[5,2]的数据
    print(f'output.requires_grad: {output.requires_grad}')  #会自动调用自动微分

    #计算 和 查看模型参数
    print('===============计算模型参数======================')
    #参1:神经网络模型对象; 参2:输入数据
    summary(my_model, input_size=(5,3))

    print('===============查看模型参数==========================')
    for name, param in my_model.named_parameters():
        print(f'name: {name}')
        print(f'param: {param}\n')



#todo:3测试
if __name__ == '__main__':
    train()