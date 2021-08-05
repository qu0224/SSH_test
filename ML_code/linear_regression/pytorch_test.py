from torch.nn import Linear, ReLU, Sequential
net = Sequential(
    Linear(3, 8), # 第1层有8个神经元
    ReLU(), # 第1层神经元的非线性函数是max(*,0)
    Linear(8, 8), # 第2层有8个神经元
    ReLU(), # 第2层的神经元的非线性函数是max(*,0)
    Linear(8, 1), # 第3层有1个神经元
    )

def g(x, y):
    x0, x1, x2 = x[:, 0] ** 0, x[:, 1] ** 1, x[:, 2] ** 2
    y0 = y[:, 0]
    return (x0 + x1 + x2) * y0 - y0 * y0 - x0 * x1 * x2

import torch
from torch.optim import Adam
if __name__ == '__main__':
    optimizer = Adam(net.parameters())
    for step in range(1000):
        optimizer.zero_grad()
        x = torch.randn(1000, 3)
        y = net(x)
        outputs = g(x, y)
        loss = -torch.sum(outputs)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print ('iteration #{}: loss = {}'.format(step, loss.item()))
    # 生成测试数据
    x_test = torch.randn(2, 3)
    print('测试输入: {}'.format(x_test))

    # 查看神经网络的计算结果
    y_test = net(x_test)
    print('人工神经网络计算结果: {}'.format(y_test))
    print('g的值: {}'.format(g(x_test, y_test)))


    # 根据理论计算参考答案
    def argmax_g(x):
        x0, x1, x2 = x[:, 0] ** 0, x[:, 1] ** 1, x[:, 2] ** 2
        return 0.5 * (x0 + x1 + x2)[:, None]


    yref_test = argmax_g(x_test)
    print('理论最优值: {}'.format(yref_test))
    print('g的值: {}'.format(g(x_test, yref_test)))