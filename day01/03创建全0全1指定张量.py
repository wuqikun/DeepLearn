"""
案例:
    演示如何创建全0 全1, 指定值的张量

涉及到的函数:
    torch.ones 和 torch.ones_like 创建全1张量
    torch.zeros 和 torch.zeros_like 创建全0张量
    torch.full 和 torch.full_like 创建全为指定值张量

需要掌握的函数:
    zeros(),full()

"""

#导包
import torch

#场景1:torch.ones 和 torch.ones_like 创建全1张量
t1 = torch.ones(3,4)      #创建3行4列全1张量
print(f"t1 : {t1},type:{type(t1)}")
print('-'*30)

t2 = torch.tensor([[1,2,3],[4,5,6]])
print(f"t2 : {t2},type:{type(t2)}")

t3 = torch.ones_like(t2)
print(f"t3 : {t3},type:{type(t3)}")
print('*'*30)

#场景2:torch.zeros 和 torch.zeros_like 创建全0张量
t1 = torch.zeros(3,4)      #创建3行4列全0张量
print(f"t1 : {t1},type:{type(t1)}")
print('-'*30)

t2 = torch.tensor([[1,2,3],[4,5,6]])
print(f"t2 : {t2},type:{type(t2)}")

t3 = torch.zeros_like(t2)
print(f"t3 : {t3},type:{type(t3)}")
print('*'*30)

#场景3:torch.full 和 torch.full_like 创建全为指定值张量
t1 = torch.full((3,4),255)      #创建3行4列指定值张量
print(f"t1 : {t1},type:{type(t1)}")
print('-'*30)

t2 = torch.tensor([[1,2,3],[4,5,6]])
print(f"t2 : {t2},type:{type(t2)}")

t3 = torch.full_like(t2,255)
print(f"t3 : {t3},type:{type(t3)}")