import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha * x, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 减去最大值以提高数值稳定性
    return exp_x / exp_x.sum()

# 定义导数函数
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t**2

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)

# 创建输入数据
x = np.linspace(-10, 10, 1000)
x_softmax = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])  # Softmax示例输入

# 创建图形
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

# 绘制激活函数
axs[0, 0].plot(x, sigmoid(x), 'r-', linewidth=2)
axs[0, 0].set_title('Sigmoid 函数')
axs[0, 0].grid(True)
axs[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axs[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

axs[0, 1].plot(x, tanh(x), 'g-', linewidth=2)
axs[0, 1].set_title('Tanh 函数')
axs[0, 1].grid(True)
axs[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axs[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

axs[1, 0].plot(x, relu(x), 'b-', linewidth=2)
axs[1, 0].set_title('ReLU 函数')
axs[1, 0].grid(True)
axs[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axs[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)

axs[1, 1].plot(x, leaky_relu(x), 'm-', linewidth=2)
axs[1, 1].set_title('Leaky ReLU 函数 (α=0.01)')
axs[1, 1].grid(True)
axs[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axs[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)

# 绘制Softmax函数
softmax_output = softmax(x_softmax)
axs[2, 0].bar(range(len(x_softmax)), softmax_output, color='purple')
axs[2, 0].set_title('Softmax 函数示例')
axs[2, 0].set_xticks(range(len(x_softmax)))
axs[2, 0].set_xticklabels([f'{x:.1f}' for x in x_softmax])
axs[2, 0].set_ylabel('概率')
axs[2, 0].grid(True, axis='y')

# 绘制导数
axs[2, 1].plot(x, sigmoid_derivative(x), 'r-', label='Sigmoid导数', linewidth=2)
axs[2, 1].plot(x, tanh_derivative(x), 'g-', label='Tanh导数', linewidth=2)
axs[2, 1].plot(x, relu_derivative(x), 'b-', label='ReLU导数', linewidth=2)
axs[2, 1].plot(x, leaky_relu_derivative(x), 'm-', label='Leaky ReLU导数', linewidth=2)
axs[2, 1].set_title('激活函数的导数')
axs[2, 1].legend()
axs[2, 1].grid(True)
axs[2, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
axs[2, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
axs[2, 1].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig('e:/Desktop/深度学习/第5章激活函数/activation_functions.png')
plt.show()