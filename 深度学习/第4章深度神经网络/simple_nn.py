import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置随机种子，确保结果可重现
np.random.seed(42)

# 激活函数及其导数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# 简单的神经网络，神经网络类定义
class SimpleNeuralNetwork:
    
    def __init__(self, layers, activation='sigmoid'):
        """
        初始化神经网络
        
        参数:
            layers: 一个列表，包含每层的神经元数量
            activation: 激活函数类型，'sigmoid'或'relu'
        def __init__的作用
        - 接收网络结构参数和激活函数类型
        - 根据激活函数类型选择对应的函数和导数
        - 初始化权重和偏置，使用He初始化(ReLU)或Xavier初始化(Sigmoid)
        """
        self.layers = layers
        self.activation = activation
        
        # 选择激活函数
        if activation == 'sigmoid':
            self.activation_function = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'relu':
            self.activation_function = relu
            self.activation_derivative = relu_derivative
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        # 使用He初始化权重
        for i in range(len(layers) - 1):
            if activation == 'relu':
                scale = np.sqrt(2.0 / layers[i])  # He初始化
            else:
                scale = np.sqrt(1.0 / layers[i])  # Xavier初始化
                
            self.weights.append(np.random.randn(layers[i], layers[i+1]) * scale)
            self.biases.append(np.zeros((1, layers[i+1])))
    
    def forward(self, X):
        """
        前向传播
        - 实现神经网络的前向计算过程
        - 存储每一层的激活值和中间输入，用于反向传播
        - 逐层计算线性变换和激活函数
        参数:
            X: 输入数据，形状为(样本数, 特征数)
            
        返回:
            activations: 每层的激活值
            layer_inputs: 每层的输入值（激活前）
        """
        activations = [X]  # 存储每层的激活值
        layer_inputs = []  # 存储每层的输入值（激活前）
        
        # 逐层计算
        A = X
        for i in range(len(self.weights)):
            # 线性变换: Z = A * W + b
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            layer_inputs.append(Z)
            
            # 应用激活函数: A = activation(Z)
            A = self.activation_function(Z)
            activations.append(A)
        
        return activations, layer_inputs
    
    def backward(self, X, y, activations, layer_inputs):
        """
        反向传播计算梯度
        - 实现神经网络的前向计算过程
        - 存储每一层的激活值和中间输入，用于反向传播
        - 逐层计算线性变换和激活函数
        参数:
            X: 输入数据
            y: 目标值
            activations: 前向传播中每层的激活值
            layer_inputs: 前向传播中每层的输入值
            
        返回:
            weight_gradients: 权重的梯度
            bias_gradients: 偏置的梯度
        """
        m = X.shape[0]  # 样本数量
        
        # 初始化梯度
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # 计算输出层误差
        delta = activations[-1] - y  # 均方误差的导数
        
        # 反向传播误差
        for l in range(len(self.weights) - 1, -1, -1):
            # 计算当前层的权重和偏置梯度
            weight_gradients[l] = np.dot(activations[l].T, delta) / m
            bias_gradients[l] = np.sum(delta, axis=0, keepdims=True) / m
            
            if l > 0:
                # 计算前一层的误差
                delta = np.dot(delta, self.weights[l].T) * self.activation_derivative(layer_inputs[l-1])
        
        return weight_gradients, bias_gradients
    
    def train(self, X, y, learning_rate=0.1, epochs=1000, batch_size=32, verbose=True):
        """
        训练神经网络
        - 实现小批量梯度下降训练算法
        - 每个epoch随机打乱数据
        - 分批次进行前向传播、计算损失、反向传播和参数更新
        - 记录训练损失并可选择性打印
        参数:
            X: 输入数据
            y: 目标值
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批量大小
            verbose: 是否打印训练过程
        """
        m = X.shape[0]  # 样本数量
        losses = []  # 记录损失
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # 小批量梯度下降
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                activations, layer_inputs = self.forward(X_batch)
                
                # 计算损失
                loss = np.mean((activations[-1] - y_batch) ** 2)
                
                # 反向传播
                weight_gradients, bias_gradients = self.backward(X_batch, y_batch, activations, layer_inputs)
                
                # 更新参数
                for l in range(len(self.weights)):
                    self.weights[l] -= learning_rate * weight_gradients[l]
                    self.biases[l] -= learning_rate * bias_gradients[l]
            
            # 计算整个数据集的损失
            activations, _ = self.forward(X)
            loss = np.mean((activations[-1] - y) ** 2)
            losses.append(loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """
        使用训练好的网络进行预测
        - 使用训练好的网络进行预测
        - 只执行前向传播计算
        参数:
            X: 输入数据
            
        返回:
            预测结果
        """
        activations, _ = self.forward(X)
        return activations[-1]

# 生成示例数据：非线性分类问题
def generate_data(n_samples=1000):
    X = np.random.randn(n_samples, 2) * 2
    y = np.zeros((n_samples, 1))
    
    # 创建一个非线性决策边界：圆形
    for i in range(n_samples):
        if X[i, 0]**2 + X[i, 1]**2 < 4:
            y[i] = 1
    
    return X, y

# 可视化决策边界
def plot_decision_boundary(model, X, y):
    # 创建网格点
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdBu, edgecolors='k')
    plt.title('神经网络决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.savefig('decision_boundary.png')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 生成数据
    X, y = generate_data(10000)
    
    # 创建神经网络：2个输入，10个隐藏神经元，1个输出
    nn = SimpleNeuralNetwork([2, 10, 10, 1], activation='sigmoid')#2种激活函数'sigmoid''relu'
    
    # 训练网络
    losses = nn.train(X, y, learning_rate=0.01, epochs=200, batch_size=32)
    
    # 可视化损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()
    
    # 可视化决策边界
    plot_decision_boundary(nn, X, y)
    
    # 计算准确率
    predictions = nn.predict(X)
    predictions = (predictions > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    print(f"准确率: {accuracy:.4f}")