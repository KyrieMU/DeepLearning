import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
'''
def createDataSet_1(n_samples=100, noise=0.1, random_state=42):
    """
    创建一个月牙形数据集用于二分类问题
    
    参数:
    n_samples: 样本数量
    noise: 噪声水平
    random_state: 随机种子
    
    返回:
    X: 特征矩阵，形状为(n_samples, 2)
    y: 标签向量，形状为(n_samples,)
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

'''

def createDataSet_2():
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([1, 0, 0, 1])
    return x, y


# 调用函数创建数据集
x, y = createDataSet_2()

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 可视化数据集
plt.figure(figsize=(8, 6))
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('XOR数据集')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.colorbar()
plt.savefig('e:\\GitHub\\DeepLearning\\深度学习\\第4章深度神经网络\\xor_dataset.png')
plt.show()

# 定义神经网络类
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def compute_loss(self, y_pred, y_true):
        # 计算二元交叉熵损失
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8) + (1 - y_true) * np.log(1 - y_pred + 1e-8)) / m
        return loss
    
    def backward(self, X, y, learning_rate):
        # 反向传播
        m = X.shape[0]
        
        # 输出层误差
        dz2 = self.a2 - y.reshape(-1, 1)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 隐藏层误差
        dz1 = np.dot(dz2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        losses = []
        
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = self.compute_loss(y_pred, y)
            losses.append(loss)
            
            # 反向传播
            self.backward(X, y, learning_rate)
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def predict(self, X):
        # 预测
        y_pred = self.forward(X)
        return (y_pred > 0.5).astype(int)
    
    def plot_decision_boundary(self, X, y, title):
        # 创建网格点
        h = 0.01
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        
        # 预测网格点的类别
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界和数据点
        plt.figure(figsize=(8, 6))
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
        plt.title(title)
        plt.xlabel('特征1')
        plt.ylabel('特征2')
        
        # 计算准确率
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred.flatten() == y)
        plt.text(0.05, 0.95, f'准确率: {accuracy:.2f}', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', alpha=0.5))
        
        return plt

# 尝试不同的学习率
learning_rates = [0.1, 0.5, 1.0, 2.0]
hidden_size = 4
epochs = 10000

plt.figure(figsize=(15, 10))
for i, lr in enumerate(learning_rates):
    # 创建并训练模型
    model = SimpleNeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
    losses = model.train(x, y, epochs=epochs, learning_rate=lr)
    
    # 绘制损失曲线
    plt.subplot(2, 2, i+1)
    plt.plot(losses)
    plt.title(f'学习率 = {lr}的损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.grid(True)

plt.tight_layout()
plt.savefig('e:\\GitHub\\DeepLearning\\深度学习\\第4章深度神经网络\\learning_rate_comparison.png')
plt.show()

# 绘制不同学习率的决策边界
plt.figure(figsize=(15, 10))
for i, lr in enumerate(learning_rates):
    # 创建并训练模型
    model = SimpleNeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
    model.train(x, y, epochs=epochs, learning_rate=lr)
    
    # 绘制决策边界
    plt.subplot(2, 2, i+1)
    model.plot_decision_boundary(x, y, f'学习率 = {lr}的决策边界')

plt.tight_layout()
plt.savefig('e:\\GitHub\\DeepLearning\\深度学习\\第4章深度神经网络\\learning_rate_decision_boundaries.png')
plt.show()

# 尝试不同的隐藏层节点数
hidden_sizes = [2, 4, 8, 16]
learning_rate = 0.5  # 选择一个较好的学习率
epochs = 10000

plt.figure(figsize=(15, 10))
for i, hidden_size in enumerate(hidden_sizes):
    # 创建并训练模型
    model = SimpleNeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
    losses = model.train(x, y, epochs=epochs, learning_rate=learning_rate)
    
    # 绘制损失曲线
    plt.subplot(2, 2, i+1)
    plt.plot(losses)
    plt.title(f'隐藏层节点数 = {hidden_size}的损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.grid(True)

plt.tight_layout()
plt.savefig('e:\\GitHub\\DeepLearning\\深度学习\\第4章深度神经网络\\hidden_size_comparison.png')
plt.show()

# 绘制不同隐藏层节点数的决策边界
plt.figure(figsize=(15, 10))
for i, hidden_size in enumerate(hidden_sizes):
    # 创建并训练模型
    model = SimpleNeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
    model.train(x, y, epochs=epochs, learning_rate=learning_rate)
    
    # 绘制决策边界
    plt.subplot(2, 2, i+1)
    model.plot_decision_boundary(x, y, f'隐藏层节点数 = {hidden_size}的决策边界')

plt.tight_layout()
plt.savefig('e:\\GitHub\\DeepLearning\\深度学习\\第4章深度神经网络\\hidden_size_decision_boundaries.png')
plt.show()