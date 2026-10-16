"""
案例:
    演示学习率衰减策略
学习率衰减策略介绍:
    目的:
        较之于AdaGrad,RMSProp,Adam方式,可以通过 等间隔 指定间隔,指数间隔,来手动控制学习率的调整

    分类:
        等间隔学习率衰减
        指定间隔学习率衰减
        指数学习率衰减
等间隔学习率衰减:
    step_size:间隔轮次,即多少轮调整一次学习率
    gamma:    学习率衰减系数,即 lr新 = lr旧 * gamma
指定间隔学习率衰减:
    milestones = [50, 125, 160]     里边定义的要调整学习率的轮数
     gamma:    学习率衰减系数,即 lr新 = lr旧 * gamma
指数学习率衰减:
    前期衰减快,更符合梯度下降规律
    公式: lr新 = lr旧 * gamma ** epoch

"""

#导包
import torch
from torch import optim
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler


#1.定义函数,演示指定间隔学习率衰减
def dm01():
    #1.定义变量,记录初始学习率,训练的轮数,每轮训练的批次数
    lr,epochs,iteration = 0.01,200,10

    #2.创建数据集,y_ture,x,w
    #真实值
    y_ture = torch.tensor([0])
    #输入特征
    x = torch.tensor([1.0],dtype=torch.float)
    #权重参数,需要自动微分
    # 参1:(待优化的参数列表),参2:学习率,参3:动量参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    #3.创建优化器对象,动量法 -> 加快模型收敛,减小振荡
    optimizer = optim.SGD([w],lr=lr,momentum=0.9)

    #4.创建学习率衰减策略对象
    #思路1:创建等间隔学习率衰减对象
    #参1:优化器对象, 参2:间隔轮数(多少轮调整一次学习率), 参3:学习率衰减系数
    scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)


    #5.创建两个列表,分别表示:训练轮数,每轮训练用的学习率
    lr_list, epoch_list = [],[]

    #循环遍历训练轮数,进行具体的训练
    for epoch in range(epochs):
        #7.获取当前轮数 学习率,并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()[0])       #获取最后的lr(learning rate,学习率)

        #8.循环遍历,每轮每批次进行训练
        for batch in range(iteration):
            #9.先计算预测值,基于损失函数计算损失
            y_pred = w * x
            #10.计算损失,最小二乘法
            loss = (y_pred - y_ture) ** 2
            #11.梯度清零+反向传播+优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #12.更新学习率
        scheduler.step()
    #13.打印结果
    print(f'lr_list:{lr_list}')

    #14.可视化
    plt.plot(epoch_list,lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.legend("Step LR Scheduler")
    plt.show()


#2.定义函数,演示等间隔学习率衰减
def dm02():
    #1.定义变量,记录初始学习率,训练的轮数,每轮训练的批次数
    lr,epochs,iteration = 0.01,200,10

    #2.创建数据集,y_ture,x,w
    #真实值
    y_ture = torch.tensor([0])
    #输入特征
    x = torch.tensor([1.0],dtype=torch.float)
    #权重参数,需要自动微分
    # 参1:(待优化的参数列表),参2:学习率,参3:动量参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    #3.创建优化器对象,动量法 -> 加快模型收敛,减小振荡
    optimizer = optim.SGD([w],lr=lr,momentum=0.9)

    #4.创建学习率衰减策略对象
    #思路1:创建等间隔学习率衰减对象
    #参1:优化器对象, 参2:间隔轮数(多少轮调整一次学习率), 参3:学习率衰减系数
    scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    # 思路2:创建指定间隔学习率衰减对象
    # 定义变量,记录要修改学习率的轮数
    milestones = [50, 125, 160]
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    #5.创建两个列表,分别表示:训练轮数,每轮训练用的学习率
    lr_list, epoch_list = [],[]

    #循环遍历训练轮数,进行具体的训练
    for epoch in range(epochs):
        #7.获取当前轮数 学习率,并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()[0])       #获取最后的lr(learning rate,学习率)

        #8.循环遍历,每轮每批次进行训练
        for batch in range(iteration):
            #9.先计算预测值,基于损失函数计算损失
            y_pred = w * x
            #10.计算损失,最小二乘法
            loss = (y_pred - y_ture) ** 2
            #11.梯度清零+反向传播+优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #12.更新学习率
        scheduler.step()
    #13.打印结果
    print(f'lr_list:{lr_list}')

    #14.可视化
    plt.plot(epoch_list,lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.legend("Step LR Scheduler")
    plt.show()


#1.定义函数,演示等间隔学习率衰减
def dm03():
    #1.定义变量,记录初始学习率,训练的轮数,每轮训练的批次数
    lr,epochs,iteration = 0.01,200,10

    #2.创建数据集,y_ture,x,w
    #真实值
    y_ture = torch.tensor([0])
    #输入特征
    x = torch.tensor([1.0],dtype=torch.float)
    #权重参数,需要自动微分
    # 参1:(待优化的参数列表),参2:学习率,参3:动量参数
    w = torch.tensor([1.0],requires_grad=True,dtype=torch.float)

    #3.创建优化器对象,动量法 -> 加快模型收敛,减小振荡
    optimizer = optim.SGD([w],lr=lr,momentum=0.9)

    #4.创建学习率衰减策略对象
    #思路1:创建等间隔学习率衰减对象
    #参1:优化器对象, 参2:间隔轮数(多少轮调整一次学习率), 参3:学习率衰减系数
    # scheduler = lr_scheduler.StepLR(optimizer,step_size=50,gamma=0.5)

    # 思路2:创建指定间隔学习率衰减对象
    # 定义变量,记录要修改学习率的轮数
    # milestones = [50, 125, 160]
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    #思路3:创建指数学习率衰减
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.95)

    #5.创建两个列表,分别表示:训练轮数,每轮训练用的学习率
    lr_list, epoch_list = [],[]

    #循环遍历训练轮数,进行具体的训练
    for epoch in range(epochs):
        #7.获取当前轮数 学习率,并保存到列表中
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr()[0])       #获取最后的lr(learning rate,学习率)

        #8.循环遍历,每轮每批次进行训练
        for batch in range(iteration):
            #9.先计算预测值,基于损失函数计算损失
            y_pred = w * x
            #10.计算损失,最小二乘法
            loss = (y_pred - y_ture) ** 2
            #11.梯度清零+反向传播+优化器更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #12.更新学习率
        scheduler.step()
    #13.打印结果
    print(f'lr_list:{lr_list}')

    #14.可视化
    plt.plot(epoch_list,lr_list)
    plt.xlabel('Epoch')
    plt.ylabel('Learning rate')
    plt.legend("Step LR Scheduler")
    plt.show()



#4.测试
if __name__=='__main__':
    # dm01()
    # dm02()
    dm03()
