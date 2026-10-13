"""
案例:
    演示张量 和 numpy之间如何相互转换, 以及 如何从标量张量中 提取其内容

涉及到的API:
    场景1: 张量 -> numpy nd的数组对象
        张量对象.numpy()          共享内存
        张量对象.numpy().copy()   不共享内存,链式编程
    场景2: numpy nd数组 -> 张量
        from_numpy()            共享内存
        torch.tensor(nd数组)      不共享内存
    场景3: 从标量张量中 提取其内容
    标量张量.item()

掌握:
    张量 -> numpy:   张量对象.numpy()
    numpy -> 张量:   torch.tensor(nd数组)
    从标量张量中 提取其内容:  标量张量.item()
"""
import numpy as np
import torch


# 1.定义函数,演示:张量 -> numpy
def dm01():
    #1.创建张量
    t1 = torch.tensor([1,2,3,4,5])
    print(f't1:{t1},type:{type(t1)}')

    #2.张量 -> numpy
    # n1 = t1.numpy()     #共享
    n1 = t1.numpy().copy()     #不共享

    print(f'n1: {n1},type: {type(n1)}')

    #3.演示上述方式 是否共享内存
    n1[0]=100
    print(f'n1:{n1}')   # [100,2,3,4,5]
    print(f't1:{t1}')   # 若[1,2,3,4,5]则不共享内存

#2.定义函数,numpy -> 张量
def dm02():
    #1.创建numpy数组
    n1 = np.array([1,2,3])
    print(f'n1:{n1},type:{type(n1)}')

    #2.转换为张量
    # t1 = torch.from_numpy(n1).type(torch.float)   #转换 + 转类型
    t1 = torch.from_numpy(n1)
    print(f't1:{t1},type:{type(t1)}')       #共享

    t2 = torch.tensor(n1)
    print(f't2:{t2},type:{type(t2)}')       #不共享

    #3.演示上述方式 是否共享内存
    n1[0] = 100
    print(f'n1:{n1}')
    print(f't1:{t1}')
    print(f't2:{t2}')

#2.定义函数,从标量张量中 提取其内容
def dm03():
    #1.创建张量
    t1 = torch.tensor([1])
    print(f't1: {t1},type:{type(t1)}')

    #2.提取
    a = t1.item()
    print(f'a: {a},type:{type(a)}')

#4.测试
if __name__ == '__main__':
    #dm01()
    # dm02()
    dm03()