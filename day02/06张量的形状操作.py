"""
案例:
    演示张量的形状操作

涉及到的API:
    reshape():          在不改变张量的内容的前提下,对其形状进行改变
    unsqueeze():        在指定的轴上增加一个(1)维度
    squeeze():          删除所有为(1)的维度,等价于:降维
    transpose():        一次只能交换2个维度
    permute():          一次可以同时交换多个维度
    view()              只能修改连续的张量形状,连续张量 = 内存中存储顺序 和 张量中显示顺序一致
    contiguous():       把不连续的张量 -> 连续的张量,即:基于张量中显示的顺序,修改内存的存储顺序
    is_contiguous():    判断张量是否连续
"""
#导包
import torch
#指定随机种子
torch.manual_seed(2)

#1.定义函数,演示reshape()函数
def dm01():
    #1.定义一个2行3列的张量
    t1 = torch.randint(1,10,(2,3))
    print(f't1:{t1}, shape:{t1.shape}, row:{t1.shape[0]}, col:{t1.shape[1],t1.shape[-1]}')

    #2.通过reshape(),将t1 -> 3行2列,1行6列,6行1列
    # t1 = t1.reshape(3,2)          #3行2列
    # t1 = t1.reshape(1,6)          #1行6列
    t1 = t1.reshape(6,1)            #6行1列
    print(f't1:{t1}, shape:{t1.shape}, row:{t1.shape[0]}, col:{t1.shape[1]}')

    #尝试通过reshape(),将t1 -> 2行5列
    # t3 = t1.reshape(2,5)          #报错,转之前有2x3个元素,转之后有2x5个元素,不一致
    # print(t3)


#2.定义函数,演示unsqueeze(), squeeze()函数
def dm02():
    #1.定义2行3列的张量
    t1 = torch.randint(1,10,(2,3))
    print(f't1:{t1},shape:{t1.shape}')      # (2,3)
   #2.在0维上,添加一个维度
    t2 = t1.unsqueeze(0)
    print(f't2:{t2},shape:{t2.shape}')      # (1,2,3)
    #3.在1维上添加一个维度
    t3 = t1.unsqueeze(1)
    print(f't3:{t3},shape:{t3.shape}')      # (2,1,3)
    #4.在2维上添加一个维度
    t4 = t1.unsqueeze(2)
    print(f't4:{t4},shape:{t4.shape}')      # (2,3,1)
    #5.不能超出原有维度
    # t5 = t1.unsqueeze(3)                  # 报错,越界
    # print(f't5:{t5},shape:{t5.shape}')      # (2,3,???,1)

    #6.删除所有为1的维度
    t6 = torch.randint(1,10,(2,1,3,1,1))
    print(f't6:{t6},shape:{t6.shape}')
    t7 = t6.squeeze()
    print(f't7:{t7},shape:{t7.shape}')

#3.定义函数,演示transpose(), permute()函数
def dm03():
    #1.定义张量
    t1 = torch.randint(1,10,(2,3,4))
    print(f't1:{t1},shape:{t1.shape}')
    #2.改变维度(2,3,4) -> (3,2,4)
    t2 = t1.transpose(0, 1)     #一次只能2个相互转换,列如(2,3,4)不能转(3,4,2)
    print(f't2:{t2},shape:{t2.shape}')
    #3.(2,3,4) -> (3,4,2)就需要permute()
    t3 = t1.permute(1,2,0)
    print(f't3:{t3},shape:{t3.shape}')

#4.定义函数,演示view(), contiguous(), is_contiguous()
def dm04():
     #1.定义张量
     t1 = torch.randint(1,10,(2,3))
     print(f't1:{t1},shape:{t1.shape}')
     #2.判断张量是否连续
     print(t1.is_contiguous())
     #3.通过view()函数修改张量的形状,(2,3) -> (3,2)
     t2 = t1.view(3,2)
     print(f't2:{t2},shape:{t2.shape}')
     #4.通过transpose()交换 -> 不连续了
     t3 = t1.transpose(0,1)
     print(f't3:{t3},shape:{t3.shape}')
     print(t3.is_contiguous())
     #5.通过contiguous() -> 连续
     t4 = t3.contiguous()
     print(f't4:{t4},shape:{t4.shape}')
     print(t4.is_contiguous())      #Ture,可以用view()转换形状了

#5.测试
if __name__ == "__main__":
    # dm01()
    # dm02()
    # dm03()
     dm04()