
# 线性回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 生成随机数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 训练线性回归模型
model = LinearRegression()
model.fit(X, y)

# 获取模型参数
w = model.coef_[0][0]
b = model.intercept_[0]
print(f"模型参数: w = {w:.4f}, b = {b:.4f}")

# 预测
X_new = np.array([[0], [2]])
y_pred = model.predict(X_new)

# 计算MSE
y_pred_all = model.predict(X)
mse = mean_squared_error(y, y_pred_all)
print(f"均方误差(MSE): {mse:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label='数据点')
plt.plot(X_new, y_pred, 'r-', linewidth=2, label=f'线性回归: y = {w:.2f}x + {b:.2f}')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归模型')
plt.legend()
plt.grid(True)
plt.savefig('linear_regression.png')
plt.show()