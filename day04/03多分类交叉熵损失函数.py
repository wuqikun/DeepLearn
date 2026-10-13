"""
案例:
    演示多分类交叉熵损失函数

损失函数介绍:
    损失函数也叫成本函数, 目标函数, 代价函数, 误差函数, 就是用来衡量 模型拟合情况
    分类:
        分类问题:
            多分类交叉熵损失:CrossEntropyLoss
            二分类交叉熵损失:BCELoss
        回归问题:
            MAE:Mean Absolute Error, 平均绝对误差
            MSE:Mean Squared Error,  均方误差
            Smooth L1:结合上述两者的特点做的优化

多分类交叉熵损失:
    设计思路:
        Loss = -

    细节:
        CrossEntropyLoss = Softmax() + 损失计算,如果用这个损失函数,输出层不需额外调softmax()
"""

#导包
import torch
import torch.nn as nn

#1.定义函数,演示: 对分类交叉熵损失
def dm01():
    #手动创建样本的真实值 --> 就是上述公式的y
    # y_true =torch.tensor([[0,1,0],[1,0,0]],dtype=torch.float)
    y_true = torch.tensor([1,2])

    #手动创建样本预测值 -> 就是公式中的f(x)
    y_pred = torch.tensor([[0.1,0.8,0.1],[0.2,0.3,0.2]],requires_grad=True,dtype=torch.float)

   #创建多类分支熵损失函数
    criterion = nn.CrossEntropyLoss()

    #计算损失值
    loss = criterion(y_pred, y_true)
    print(f'损失值:{loss}')

#2.测试
if __name__ == '__main__':
    dm01()