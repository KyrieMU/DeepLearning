import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 生成随机数据
np.random.seed(1)
X = np.random.randn(300, 3)
y = (X[:, 0] + X[:, 1] + 0.5*X[:, 2] > 0).astype(int)  # 使用所有三个特征的分类规则

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X, y)

# 获取模型参数
w = model.coef_[0]
b = model.intercept_[0]
print(f"模型参数: w = [{w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f}], b = {b:.4f}")

# 预测
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"准确率: {accuracy:.4f}")

# 创建网格以可视化决策边界（使用前两个特征进行可视化）
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 为第三个特征创建一个固定值（使用训练数据的平均值）
fixed_z = np.mean(X[:, 2]) * np.ones(xx.ravel().shape)

# 组合三个特征用于预测
Z = model.predict(np.c_[xx.ravel(), yy.ravel(), fixed_z])
Z = Z.reshape(xx.shape)

# 可视化
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='o', alpha=0.6, label='类别 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='x', alpha=0.6, label='类别 1')

# 绘制决策边界
plt.contour(xx, yy, Z, [0.5], linewidths=2, colors='black')

# 添加标签和图例
plt.xlabel('特征 1')
plt.ylabel('特征 2')
plt.title('三特征Logistic回归分类 (第三特征固定为平均值)')
plt.legend()
plt.grid(True)
plt.savefig('logistic_regression.png')
plt.show()