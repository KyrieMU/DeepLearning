import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


# 设置随机种子
np.random.seed(42)

# 定义简单的两层神经网络
class SimpleNetwork:
    def __init__(self):
        # 网络结构：2个输入，2个隐藏神经元，1个输出
        self.W1 = np.random.randn(2, 2) * 0.1
        self.b1 = np.zeros((1, 2))
        self.W2 = np.random.randn(2, 1) * 0.1
        self.b2 = np.zeros((1, 1))
        
        # 存储中间值用于反向传播
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None
        
        # 存储梯度
        self.dW1 = None
        self.db1 = None
        self.dW2 = None
        self.db2 = None
        
        # 存储训练历史
        self.loss_history = []
        self.weight_history = []
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward(self, x):
        # 存储输入
        self.x = x
        
        # 第一层
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # 第二层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def compute_loss(self, y):
        # 均方误差损失
        return np.mean((self.a2 - y) ** 2)
    
    def backward(self, y):
        m = y.shape[0]
        
        # 输出层梯度
        dz2 = (self.a2 - y) * self.sigmoid_derivative(self.z2)
        self.dW2 = np.dot(self.a1.T, dz2) / m
        self.db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 隐藏层梯度
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.z1)
        self.dW1 = np.dot(self.x.T, dz1) / m
        self.db1 = np.sum(dz1, axis=0, keepdims=True) / m
    
    def update_parameters(self, learning_rate):
        # 更新权重和偏置
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2
    
    def train_step(self, x, y, learning_rate):
        # 前向传播
        self.forward(x)
        
        # 计算损失
        loss = self.compute_loss(y)
        self.loss_history.append(loss)
        
        # 保存当前权重
        self.weight_history.append((
            self.W1.copy(), self.b1.copy(),
            self.W2.copy(), self.b2.copy()
        ))
        
        # 反向传播
        self.backward(y)
        
        # 更新参数
        self.update_parameters(learning_rate)
        
        return loss

# 生成简单的XOR数据
def generate_xor_data(n_samples=100):
    X = np.random.rand(n_samples, 2) * 2 - 1  # 在[-1, 1]范围内生成随机点
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y

# 可视化反向传播过程
def visualize_backpropagation(network, X, y):
    # 创建图形
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 左上角：绘制原始数据点
    axs[0, 0].scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='blue', marker='o', label='类别 0')
    axs[0, 0].scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='red', marker='x', label='类别 1')
    axs[0, 0].set_title('XOR数据分布')
    axs[0, 0].set_xlabel('特征1')
    axs[0, 0].set_ylabel('特征2')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # 2. 右上角：绘制损失曲线
    loss_line, = axs[0, 1].plot([], [], 'b-', linewidth=2)
    axs[0, 1].set_title('训练损失曲线')
    axs[0, 1].set_xlabel('迭代次数')
    axs[0, 1].set_ylabel('损失值')
    axs[0, 1].set_xlim(0, len(network.loss_history))
    axs[0, 1].set_ylim(0, max(network.loss_history) * 1.1)
    axs[0, 1].grid(True)
    
    # 3. 左下角：准备决策边界可视化
    h = 0.01  # 网格步长
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 绘制初始数据点
    axs[1, 0].scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='blue', marker='o', label='类别 0')
    axs[1, 0].scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='red', marker='x', label='类别 1')
    axs[1, 0].set_title('决策边界演变')
    axs[1, 0].set_xlabel('特征1')
    axs[1, 0].set_ylabel('特征2')
    
    # 4. 右下角：绘制权重变化曲线
    weight_lines = []
    colors = ['r', 'g', 'b', 'y', 'm', 'c']
    labels = ['W1[0,0]', 'W1[0,1]', 'W1[1,0]', 'W1[1,1]', 'W2[0,0]', 'W2[1,0]']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        line, = axs[1, 1].plot([], [], color=color, linewidth=1.5, label=label)
        weight_lines.append(line)
    
    axs[1, 1].set_title('网络权重变化')
    axs[1, 1].set_xlabel('迭代次数')
    axs[1, 1].set_ylabel('权重值')
    axs[1, 1].set_xlim(0, len(network.weight_history))
    axs[1, 1].set_ylim(-2, 2)
    axs[1, 1].legend(loc='upper right', fontsize=8)
    axs[1, 1].grid(True)
    
    # 设置图表布局
    plt.tight_layout()
    
    # 更新函数 - 每一帧动画的更新逻辑
    def update(frame):
        # 1. 更新损失曲线
        loss_line.set_data(range(frame+1), network.loss_history[:frame+1])
        
        # 2. 更新权重曲线
        weight_indices = [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (2, 0, 0), (2, 1, 0)]
        
        for i, (w_idx, line) in enumerate(zip(weight_indices, weight_lines)):
            layer_idx, row_idx, col_idx = w_idx
            data = [network.weight_history[j][layer_idx][row_idx, col_idx] for j in range(frame+1)]
            line.set_data(range(frame+1), data)
        
        # 3. 更新决策边界
        # 获取当前帧的权重
        W1 = network.weight_history[frame][0]
        b1 = network.weight_history[frame][1]
        W2 = network.weight_history[frame][2]
        b2 = network.weight_history[frame][3]
        
        # 计算网格点的预测值
        grid = np.c_[xx.ravel(), yy.ravel()]
        
        # 前向传播计算预测值
        z1 = np.dot(grid, W1) + b1
        a1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(a1, W2) + b2
        a2 = 1 / (1 + np.exp(-z2))
        
        Z = a2.reshape(xx.shape)
        
        # 清除之前的轮廓图
        for c in axs[1, 0].collections:
            c.remove()
        
        # 绘制新的决策边界
        decision_boundary = axs[1, 0].contourf(xx, yy, Z, alpha=0.6, cmap=plt.cm.RdBu, levels=np.linspace(0, 1, 11))
        
        # 返回所有更新的对象
        return [loss_line] + weight_lines + [decision_boundary]
    
    # 创建动画
    ani = FuncAnimation(
        fig, 
        update, 
        frames=len(network.loss_history),
        blit=False,  # 设置为False以避免某些平台上的问题
        interval=50,  # 帧之间的间隔时间(毫秒)
        repeat=False  # 不重复播放
    )
    
    # 保存动画
    try:
        ani.save('backprop_animation.gif', writer='pillow', fps=15)
        print("动画已保存为GIF文件")
    except Exception as e:
        print(f"保存动画时出错: {e}")
        print("尝试显示动画而不保存")
    
    plt.show()

# 主函数
if __name__ == "__main__":
    # 生成XOR数据
    X, y = generate_xor_data(200)
    
    # 创建网络
    network = SimpleNetwork()
    
    # 训练网络
    epochs = 1000
    learning_rate = 0.1
    
    for i in range(epochs):
        loss = network.train_step(X, y, learning_rate)
        if (i+1) % 100 == 0:
            print(f"Epoch {i+1}/{epochs}, Loss: {loss:.6f}")
    
    '''
    # 可视化训练过程
    visualize_backpropagation(network, X, y)
    '''
    # 绘制最终决策边界
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 计算网格点的预测值
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = network.forward(grid).reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[y.ravel() == 0, 0], X[y.ravel() == 0, 1], c='blue', marker='o', label='类别 0')
    plt.scatter(X[y.ravel() == 1, 0], X[y.ravel() == 1, 1], c='red', marker='x', label='类别 1')
    plt.title('XOR问题的最终决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()
    plt.grid(True)
    plt.savefig('xor_decision_boundary.png')
    plt.show()
    
    # 打印最终准确率
    predictions = (network.forward(X) > 0.5).astype(float)
    accuracy = np.mean(predictions == y)
    print(f"最终准确率: {accuracy:.4f}")