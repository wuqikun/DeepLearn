"""
案例:
    演示张量常用的运算函数

涉及到的API:
    sum(), min(), max(), mean()                 -> 都有dim 参数,0表示列 ,1表示行
    pow(),sqrt(),exp(),log(),log2(),log10()     ->没有dim参数
"""

#导包
import torch

#1.定义张量,记录初值
t1 = torch.tensor([
    [1,2,3],
    [4,5,6]
],dtype=torch.float)
print(t1)

#2.演示有dim的函数
#sum() 求和
print(t1.sum(dim=0))        #按 列 求和
print(t1.sum(dim=1))        #按 行 求和
print(t1.sum())             #全 局 求和
print('-'*30)

#max() 最大值 ,min() 最小值
print(t1.max(dim=0))        #按列 求 最大值
print(t1.max(dim=1))        #按行 求 最大值
print(t1.max())             #全局 求 最大值
print('-'*30)

#mean() 求平均值
print(t1.mean(dim=0))        #按列 求 平均值
print(t1.mean(dim=1))        #按行 求 平均值
print(t1.mean())             #全局 求 平均值
print('*'*30)

#3.没有dim参数的 函数
#pow() n次幂
print(t1.pow(2))            #每个数的平方
print(t1.pow(3))            #每个数的立方
print(t1**3)                #每个数的立方
print('-'*30)

#sqrt() 平方根
print(t1.sqrt())            #每个数的平方根
print('-'*30)

#exp(),e的n次幂  ,n是矩阵中的每个元素
print(t1.exp())
print('-'*30)

#log() 对数
print(t1.log())
print(t1.log2())
print(t1.log10())
