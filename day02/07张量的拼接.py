"""
案例:
    演示张量的拼接操作

涉及到的API:
    cat()       不改变维度数,除了拼接的那个维度外,其他维度要保持一致
    stack()     会改变维度数,所有维度都必须保持一致
"""

#导包
import torch
from numpy.ma.core import shape

#1.创建2个张量
t1 = torch.randint(1,10,(2,3))
print(t1)
t2 = torch.randint(1,10,(2,3))
print(t2)

#2.演示张量拼接
#思路1:cat()
t3 = torch.cat([t1,t2],dim=0)   # 0维拼接 (2,3) + (2,3) = (4,3)
print(t3)

t4 = torch.cat([t1,t2],dim=1)   # 1维拼接 (2,3) + (2,3) = (2,6)
print(t4)

#思路2:stack()
t6 = torch.stack([t1,t2],dim=0)     # (2,3) + (2,3) = (2,2,3)
print(t6)