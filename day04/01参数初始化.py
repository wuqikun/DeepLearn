"""
案例:
    参数初始化
"""
#导包
import torch.nn as nn

#1.均匀分布随机初始化
def dm01():
    #创建一个线性层
    lines = nn.Linear(5, 5)
    #权重随机初始化,从0到1均匀分布
    nn.init.uniform_(lines.weight)
    #偏置初始化
    nn.init.uniform_(lines.bias)
    print(lines.weight.data)
    print(lines.bias.data)


#2.固定值初始化
def dm02():
    #创建一个线性层
    lines = nn.Linear(5, 5)
    #权重初始化,固定值为3
    nn.init.constant_(lines.weight,3)
    #偏置初始化,固定值为3
    nn.init.constant_(lines.bias, 3)
    print(lines.weight.data)
    print(lines.bias.data)


#3.全0初始化
def dm03():
    #创建一个线性层
    lines = nn.Linear(5, 5)
    #权重初始化
    nn.init.zeros_(lines.weight)
    #偏置初始化
    nn.init.zeros_(lines.bias)
    print(lines.weight.data)
    print(lines.bias.data)


#4.全1初始化
def dm04():
    #创建一个线性层
    lines = nn.Linear(5, 5)
    #权重初始化
    nn.init.ones_(lines.weight)
    #偏置初始化
    nn.init.ones_(lines.bias)
    print(lines.weight.data)
    print(lines.bias.data)


#5.正态分布随机初始化
def dm05():
    #创建一个线性层
    lines = nn.Linear(5, 5)
    #参数 权重初始化,均匀分布,默认(均值为0,方差为1)
    nn.init.normal_(lines.weight, 0, 1)
    print(lines.weight.data)


#6.kaiming初始化
def dm06():
    #创建一个线性层
    lines = nn.Linear(5, 5)
    #kaiming权重初始化,均匀分布,默认(均值为0,方差为1)
    # nn.init.kaiming_normal_(lines.weight)
    #kaiming权重初始化,正态分布
    nn.init.kaiming_uniform_(lines.weight)
    print(lines.weight.data)


#7.xavier初始化
def dm07():
    #创建一个线性层
    lines = nn.Linear(5, 5)
    #xavier权重初始化,均匀分布,默认(均值为0,方差为1)
    nn.init.xavier_normal_(lines.weight)
    # xavier权重初始化,正态分布
    nn.init.xavier_uniform_(lines.weight)
    print(lines.weight.data)

#8.测试
if __name__ == '__main__':
    # dm01()
    # dm02()
    # dm03()
    # dm04()
    # dm05()
    # dm06()
    dm07()
