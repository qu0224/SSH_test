import numpy as np
#coding=UTF-8
class GDLinearRegression:

    def __init__(self, n_iter=200, eta=1e-3, tol=None):
        # 训练迭代次数
        self.n_iter = n_iter
        # 学习率
        self.eta = eta
        # 误差变化阈值
        self.tol = tol
        # 模型参数w(训练时初始化)
        self.w = None

    def _loss(self, y, y_pred):
        '''计算损失'''
        return np.sum((y_pred - y) ** 2) / y.size

    def _gradient(self, X, y, y_pred):
        '''计算梯度'''
        return np.matmul(y_pred - y, X) / y.size

    def _gradient_descent(self, w, X, y):
        '''梯度下降算法'''

        # 若用户指定tol, 则启用早期停止法.
        if self.tol is not None:
            loss_old = np.inf

        # 使用梯度下降至多迭代n_iter次, 更新w.
        for step_i in range(self.n_iter):
            # 预测
            y_pred = self._predict(X, w)
            # 计算损失
            loss = self._loss(y, y_pred)
            print('%4i Loss: %s' % (step_i, loss))

            # 早期停止法
            if self.tol is not None:
                # 如果损失下降不足阈值, 则终止迭代.
                if loss_old - loss < self.tol:
                    break
                loss_old = loss

            # 计算梯度
            grad = self._gradient(X, y, y_pred)
            # 更新参数w
            w -= self.eta * grad

    def _preprocess_data_X(self, X):
        '''数据预处理'''

        # 扩展X, 添加x0列并置1.
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        '''训练'''

        # 预处理X_train(添加x0=1)
        X_train = self._preprocess_data_X(X_train)  

        # 初始化参数向量w
        _, n = X_train.shape
        self.w = np.random.random(n) * 0.05

        # 执行梯度下降训练w
        self._gradient_descent(self.w, X_train, y_train)

    def _predict(self, X, w):
        '''预测内部接口, 实现函数h(x).'''
        return np.matmul(X, w)

    def predict(self, X):
        '''预测'''
        X = self._preprocess_data_X(X)  
        return self._predict(X, self.w)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


if __name__ == '__main__':
    data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=True)
    gd_lr = GDLinearRegression(n_iter=3000, eta=0.05, tol=0.00001)
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ss = StandardScaler()
    ss.fit(X_train)
    StandardScaler(copy=True, with_mean=True, with_std=True)
    X_train_std = ss.transform(X_train)
    X_test_std = ss.transform(X_test)
    gd_lr.train(X_train_std, y_train)
    print(gd_lr.train(X_train_std, y_train))
    y_pred = gd_lr.predict(X_test_std)
    mse = mean_squared_error(y_test, y_pred)
    print("数据归一化、标准化后的均方误差： ", mse)
    mae = mean_absolute_error(y_test, y_pred)
    print("数据归一化、标准化后的平均绝对误差： ", mae)

