import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 生成随机数据
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 梯度下降参数
# 学习率
learning_rate = 0.01
# 迭代次数
n_iterations = 100
# 数据数量
m = len(X)

# 初始化参数
w = np.random.randn()
b = np.random.randn()

# 存储每次迭代的参数和损失
params_history = []
loss_history = []

# 梯度下降
for i in range(n_iterations):
    # 计算预测值
    y_pred = w * X + b
    
    # 计算损失
    loss = np.mean((y_pred - y) ** 2)
    # 存储损失
    loss_history.append(loss)
    
    # 计算梯度
    # 计算w的梯度（斜率项的梯度）
    dw = (2/m) * np.sum(X * (y_pred - y))
    
    # 计算b的梯度（截距项的梯度）
    db = (2/m) * np.sum(y_pred - y)
    
    # 更新参数
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # 存储参数
    params_history.append((w, b))
    
    if i % 10 == 0:
        print(f"迭代 {i}: w = {w:.4f}, b = {b:.4f}, 损失 = {loss:.4f}")

# 最终参数
print(f"最终参数: w = {w:.4f}, b = {b:.4f}")
print(f"最终损失: {loss_history[-1]:.4f}")

# 可视化损失函数下降过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('梯度下降: 损失函数')
plt.grid(True)

# 可视化拟合过程
plt.subplot(1, 2, 2)
plt.scatter(X, y, alpha=0.6, label='数据点')

# 绘制最终拟合线
X_new = np.array([[0], [2]])
y_pred = w * X_new + b
plt.plot(X_new, y_pred, 'r-', linewidth=2, label=f'最终拟合: y = {w:.2f}x + {b:.2f}')

plt.xlabel('X')
plt.ylabel('y')
plt.title('梯度下降: 线性回归拟合')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('gradient_descent.png')
plt.show()

# 创建动画展示梯度下降过程
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(X, y, alpha=0.6, label='数据点')
line, = ax.plot([], [], 'r-', linewidth=2)
title = ax.set_title('迭代: 0')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.grid(True)
ax.legend()

def init():
    line.set_data([], [])
    return line,

def update(frame):
    w, b = params_history[frame]
    y_line = w * X_new + b
    line.set_data(X_new, y_line)
    title.set_text(f'迭代: {frame}, w = {w:.2f}, b = {b:.2f}')
    return line, title

ani = FuncAnimation(fig, update, frames=range(0, n_iterations, 2),
                    init_func=init, blit=True, interval=100)
ani.save('gradient_descent_animation.gif', writer='pillow', fps=10)
plt.close()