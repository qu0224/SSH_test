import numpy as np

class OLSLinearRegression:

    def _ols(self, X, y):
        '''最小二乘法估算w'''
        tmp = np.linalg.inv(np.matmul(X.T, X))
        tmp = np.matmul(tmp, X.T)
        return np.matmul(tmp, y)

        # 若使用较新的python和numpy版本, 可使用如下实现.
        # return np.linalg.inv(X.T @ X) @ X.T @ y

    def _preprocess_data_X(self, X):
        '''数据预处理'''

        # 扩展X, 添加x0列并置1.
        m, n = X.shape
        X_ = np.empty((m, n + 1))
        X_[:, 0] = 1
        X_[:, 1:] = X

        return X_

    def train(self, X_train, y_train):
        '''训练模型'''

        # 预处理X_train(添加x0=1)
        X_train = self._preprocess_data_X(X_train)  

        # 使用最小二乘法估算w
        self.w = self._ols(X_train, y_train)

    def predict(self, X):
        '''预测'''
        # 预处理X_train(添加x0=1)
        X = self._preprocess_data_X(X)  
        return np.matmul(X, self.w)

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    data = np.genfromtxt('winequality-red.csv', delimiter=';', skip_header=True)
    ols_lr = OLSLinearRegression()
    X = data[:, :-1]
    y = data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    ols_lr.train(X_train, y_train)
    y_pred = ols_lr.predict(X_test)
    #print(y_pred)
    mse = mean_squared_error(y_test, y_pred)
    print("模型在测试集上的均方误差： ", mse)
    y_train_pred = ols_lr.predict(X_train)
    mse_train = mean_squared_error(y_train, y_train_pred)
    print("模型在训练集上的均方误差： ",mse_train)
    mae = mean_absolute_error(y_test, y_pred)
    print("模型在测试集上的平均绝对误差： ", mae)

