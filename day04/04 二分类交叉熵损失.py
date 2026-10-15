"""
案例:
    演示二分类交叉熵损失函数

公式:
    Loss = -ylog(预测值) - (1-y)log(1-预测值)
    细节:
        公式中没有包含sigmoid激活函数,所以使用BCELoss 需要手动指定sigmoid
"""

#导包
import torch
import torch.nn as nn

#1.定义函数,演示二分类任务的损失函数

def dm01():
    #1.设置真实值
    y_true = torch.tensor([0,1,0], dtype=torch.float)

    #2.设置预测值
    y_pred = torch.tensor([0.691,0.231,0.189], dtype=torch.float)

    #3.创建二分类交叉熵损失函数
    criterion = nn.BCELoss(reduction='mean')

    #4.计算损失
    loss = criterion(y_true, y_pred)
    print(f'损失值:{loss}')


#2.测试
if __name__ == '__main__':
    dm01()