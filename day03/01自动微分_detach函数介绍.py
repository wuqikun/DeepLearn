"""
案例:
    演示detach函数的功能,解决自动微分的一个弊端

回顾:
    自动微分 = 求导,即:基于损失函数,计算梯度
    结合权重更新公式:w新 = w旧 - 学习率 * 梯度 ,来更新权重

问题:
    一个张量一旦设置了自动微分,这个张量旧不能直接转成numpy的ndarray对象了,需要通过detach函数解决
"""

# 导包
import torch
import numpy as np

# 1.定义张量
#参1:数据 ; 参2:是否自动微分 ; #数据类型
t1 = torch.tensor([1,2,3],requires_grad=True, dtype=torch.float32)
print(f't1:{t1},type(t1):{type(t1)}')

#2.尝试把上述张量 -> numpy对象
n1 = t1.detach().numpy()
print(f'n1:{n1},type(n1):{type(n1)}')
