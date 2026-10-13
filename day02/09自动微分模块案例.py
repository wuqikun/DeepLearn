"""
案例:
    演示自动微分模块,循环实现 计算梯度 更新参数

需求:
    求 y = x**2 + 20 的极小值,并打印y是最小值时 w的值(梯度)
    
解题步骤:
    1.定义点 x=10 requires_grad=True dtype=torch.float
    2.定义函数 y = x**2 + 20
    3.利用梯度下降法 循环迭代100 求最优解
    3.1 正向计算
    3.2 梯度清零
    3.3 反向传播
    3.4 梯度更新 x.data = x.data - 0.01 * x.grad 
"""

#导包
import torch

# 1.定义点 w=10 requires_grad=True dtlosspe=torch.float
#参1:初始值 ;参2:是否自动微分(求导) ;参3:参数类型
w = torch.tensor(10, requires_grad=True, dtype=torch.float)

# 2.定义函数 loss = w**2 + 20
loss = w**2 + 20        #求导:loss' = 2w

# 3.利用梯度下降法 循环迭代100 求最优解
print(f'开始 权重初始值:{w},(0.01 * w.grad):无, loss:{loss}')

#迭代
for i in range(1,101):
    # 3.1 正向计算
    loss = w**2 + 20


    # 3.2 梯度清零,默认梯度会累加,所以需要清零
    #第一次,梯度还没计算,为空,w.grad = None 所以要做非空判断
    if w.grad is not None:
        w.grad.zero_()

    # 3.3 反向传播
    loss.sum().backward()

    # 3.4 梯度更新 w.data = w.data - 0.01 * w.grad
    # print(f'梯度值为:{w.grad}')
    w.data = w.data - 0.01 * w.grad

    #3.5打印本次 梯度更新后的 权重参数
    print(f'第{i}次,权重初始值:{w},(0.01 * w.grad):{0.01 * w.grad:.5f}, loss:{loss:.5f}')

#4.打印最终结果
print(f'最终结果 权重:{w},梯度:{w.grad},loss:{loss}')