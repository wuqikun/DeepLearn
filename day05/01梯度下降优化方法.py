"""
案例:
    演示梯度下降优化方法

梯度下降相关介绍:
    概述:
        梯度下降是结合本次损失函数的导数(梯度) 基于学习率 来更新权重的
    公式:
        W新 = W旧 - 学习率 * 梯度
    存在的问题:
        1.遇到平缓区域,梯度下降(权重更新)可能会比较慢
        2.可能会遇到鞍点(梯度为0),
        3.可能会遇到局部最小值
    解决思路:
        从上数学习率 或 梯度入手, 进行优化,于是有了:动量法Momentum, 自适应学习率AdaGrad,RMSProp, 综合衡量:Adam

    动量法Momentum:
        公式:
            St = β * St-1 + (1 - β) * Gt
        解释:
            St:本次的指数移动加权平均结果
            β: 调节权重系数,越大,数据越平缓,历史指数移动加权平均 比重越大,本次梯度权重越小
            St-1:历史的指数移动加权平均结果
            Gt:本次计算的梯度(不考虑历史梯度)
        加入动量法后的 梯度更新公式:
            W新 = W旧 - 学习率 * St

    自适应学习率:AdaGrad
        公式:
            累计平方梯度:
                St = St-1 + Gt * Gt
                解释:
                    St:累计平方梯度
                    St-1:历史累计平方梯度
                    Gt:本次平方梯度
            学习率:
                学习率 / (sqrt(St) * 小常数)
                解释:小常数:1e-10,目的:防止分母变为0
            梯度下降公式:
                 W新 = W旧 - (调整后的)学习率 * Gt
        缺点:
            可能会导致学习率过早,过量的降低,导致模型后期学习率太小,较难找到最优解

自适应学习率:RMSProp -> 对AdaGrad加入调和权重系数
        公式:
            指数加权平均 累计历史平方梯度:
                St = β * St-1 + (1-β) * Gt * Gt
                解释:
                    St:累计平方梯度
                    St-1:历史累计平方梯度
                    Gt:本次平方梯度
                    β:调和权重系数
            学习率:
                学习率 / (sqrt(St) * 小常数)
                解释:小常数:1e-10,目的:防止分母变为0
            梯度下降公式:
                 W新 = W旧 - (调整后的)学习率 * Gt
        优点:
            通过引入 衰减系数β,控制历史梯度 对 历史梯度信息获取的多少

    自适应矩估计:Adam(Adaptive Moment Estimation
        思路:
            既优化学习了,又优化梯度
        公式:
            一阶矩:算均值
                Mt = β1 * Mt-1 + (1-β1) * Gt        充当:梯度
                St = β2 * St-1 + (1-β2) * Gt *Gt    充当:学习率
            二阶矩:算梯度的方差
                Mt^ = Mt / (1-β1^t)
                St^ = St / (1-β2^t)
            权重更新公式:
                w新 = w旧 - 学习率 / (sqrt(St^) + 小常数) * Mt^

总结:如何选择梯度下降优化方法
    简单任务和较小模型:
        SGD,动量法
    复杂任务或者有大量数据:
        Adam
    需要处理稀疏数据或者文本数据:
    AdaGrad, RMSProp
"""

#导包
import torch
import torch.optim as optim

#1.定义函数,演示:梯度下降优化方法 -> 动量法(Momentum)
def dm01_momentum():
    #1.初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype = torch.float32)
    #2.定义损失函数
    criterion = ((w**2)/2.0).sum()
    #3.创建优化器(函数对象) -> 基于SGD(随机函数下降),加入参数Momentum,就是动量法
    #参1:(待优化的参数列表),参2:学习率,参3:动量参数
    optimizer = optim.SGD([w], lr=0.01, momentum=0.9)   #momentum=0(默认),只考虑:本次梯度
    #计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    #重复上述步骤,第2次权重更新
    #5.1定义损失函数
    criterion = ((w ** 2) / 2.0).sum()
    #5.2计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    #5.3打印结果
    print(f'w:{w},w.grad:{w.grad}')


#2.定义函数,演示:梯度下降优化方法 -> 自适应学习率(AdaGrad)
def dm02_momentum():
    #1.初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype = torch.float32)
    #2.定义损失函数
    criterion = ((w**2)/2.0).sum()
    #3.创建优化器(函数对象)
    # 思路1: 基于SGD(随机函数下降),加入参数Momentum,就是动量法
    #参1:(待优化的参数列表),参2:学习率,参3:动量参数
    # optimizer = optim.SGD([w], lr=0.01, momentum=0.9)   #momentum=0(默认),只考虑:本次梯度

    #思路2:基于自适应学习率(AdaGrad)
    optimizer = optim.Adagrad([w], lr=0.01)

    #计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    #重复上述步骤,第2次权重更新
    #5.1定义损失函数
    criterion = ((w ** 2) / 2.0).sum()
    #5.2计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    #5.3打印结果
    print(f'w:{w},w.grad:{w.grad}')


#3.定义函数,演示:梯度下降优化方法 -> 自适应学习率(RMSProp)
def dm03_momentum():
    #1.初始化权重参数
    w = torch.tensor([1.0],requires_grad=True,dtype = torch.float32)
    #2.定义损失函数
    criterion = ((w**2)/2.0).sum()
    #3.创建优化器(函数对象)
    # 思路1: 基于SGD(随机函数下降),加入参数Momentum,就是动量法
    #参1:(待优化的参数列表),参2:学习率,参3:动量参数
    # optimizer = optim.SGD([w], lr=0.01, momentum=0.9)   #momentum=0(默认),只考虑:本次梯度

    #思路2:基于自适应学习率(AdaGrad)
    # optimizer = optim.Adagrad([w], lr=0.01)

    #思路3:基于RMSProp
    optimizer = optim.RMSprop([w], lr=0.01,alpha=0.99)

    #计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    #重复上述步骤,第2次权重更新
    #5.1定义损失函数
    criterion = ((w ** 2) / 2.0).sum()
    #5.2计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    #5.3打印结果
    print(f'w:{w},w.grad:{w.grad}')


#4.定义函数,演示:梯度下降优化方法 -> 自适应矩估计(Adam)
def dm04_momentum():
    # 1.初始化权重参数
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2.定义损失函数
    criterion = ((w ** 2) / 2.0).sum()
    # 3.创建优化器(函数对象)
    # 思路1: 基于SGD(随机函数下降),加入参数Momentum,就是动量法
    # 参1:(待优化的参数列表),参2:学习率,参3:动量参数
    # optimizer = optim.SGD([w], lr=0.01, momentum=0.9)   #momentum=0(默认),只考虑:本次梯度

    # 思路2:基于自适应学习率(AdaGrad)
    # optimizer = optim.Adagrad([w], lr=0.01)

    # 思路3:基于RMSProp
    optimizer = optim.RMSprop([w], lr=0.01, alpha=0.99)

    #思路4:基于Adam(自适应矩估计)
    optimizer = optim.Adam([w], lr=0.01, betas=(0.9,0.999))     #betas(梯度用的 衰减系数,学习率用的 衰减系数)

    # 计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    print(f'w:{w},w.grad:{w.grad}')

    # 重复上述步骤,第2次权重更新
    # 5.1定义损失函数
    criterion = ((w ** 2) / 2.0).sum()
    # 5.2计算梯度值:梯度清零+反向传播+参数更新
    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()
    # 5.3打印结果
    print(f'w:{w},w.grad:{w.grad}')


#5.测试
if __name__=='__main__':
   # dm01_momentum()
   #  dm02_momentum()
   #  dm03_momentum()
    dm04_momentum()