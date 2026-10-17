"""
案例:
    演示 随机失活
正则化的作用:
    缓解模型过拟合情况

正则化的方式:
    L1正则化:  权重可以变为0,相当于降维
    L2正则化:  权重可以无限接近0
    DropOut:  随机失活,每批次样本训练时,随机让一部分神经元死亡,防止一些特征对结果的影响较大(防止过拟合)
    BN(批量归一化):
"""

#导包
import torch
import torch.nn as nn

#定义函数,演示随机失活(dropout)
def dm01():
    #1.创建隐藏层输出结果
    t1 = torch.randint(0,10,size=(1,4)).float()
    print(f't1:{t1}')

    #2.进行下一层 加权求和 激活函数 计算
    #2.1创建全连接层(充当线性层)
    #参1:输入特征维度; 参2:输出特征维度
    linear1 = nn.Linear(4,5)

    #2.2.进行下一层 加权求和
    l1 = linear1(t1)
    print(f'l1:{l1}')
    output = torch.relu(l1)
    print(f'output:{output}')

    #对激活值进行随机失活,dropout处理 -> 只有训练阶段有,测试阶段没有
    dropout = nn.Dropout(p=0.2)     #每个神经元都有0.2的概率失活
    d1 = dropout(output)
    print(f'随机失活后的数据:{d1}')




#3.测试
if __name__ == '__main__':
    dm01()