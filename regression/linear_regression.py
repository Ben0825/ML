# coding:utf-8
"""
desc: 线性回归预测房价
time：2020-02-09
author：ben
"""

from sklearn import datasets
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt


boston = datasets.load_boston()
X = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1/5., random_state=8)

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

plt.title("linear_regression ")
plt.plot(y_test, color='green', marker='o', label='test')
plt.plot(y_pred, color='red', marker='+', label='predict')
plt.legend()
plt.show()


# 用均方误差评估预测结果
mse = mean_squared_error(y_test, y_pred)
print("MSE：" + repr(mse))
