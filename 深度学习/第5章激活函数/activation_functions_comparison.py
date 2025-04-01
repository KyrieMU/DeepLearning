import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义要比较的激活函数
activation_functions = ['sigmoid', 'tanh', 'relu', 'leaky_relu']

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义不同激活函数的神经网络
class NeuralNetwork(nn.Module):
    def __init__(self, activation_function):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)
        
        # 选择激活函数
        if activation_function == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation_function == 'tanh':
            self.activation = nn.Tanh()
        elif activation_function == 'relu':
            self.activation = nn.ReLU()
        elif activation_function == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.01)
        else:
            raise ValueError("不支持的激活函数")
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # 输出层不使用激活函数，因为后面会用交叉熵损失
        return x

# 训练函数
def train_model(model, X_train, y_train, epochs=1000, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        # 前向传播
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    return losses

# 评估函数
def evaluate_model(model, X_test, y_test):
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
    return accuracy

# 可视化决策边界
def plot_decision_boundary(model, X, y, title):
    # 创建网格点
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    Z = np.zeros(xx.shape)
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            with torch.no_grad():
                logits = model(torch.FloatTensor([[xx[i, j], yy[i, j]]]))
                Z[i, j] = torch.argmax(logits, dim=1).item()
    
    # 绘制决策边界和数据点
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdBu, edgecolors='k')
    plt.title(title)
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.savefig(f'decision_boundary_{title}.png')
    plt.show()

# 生成多种数据集
datasets = {
    "月牙形": make_moons(n_samples=1000, noise=0.1, random_state=42),
    "同心圆": make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42),
    "线性可分": make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                              n_informative=2, random_state=42, n_clusters_per_class=1),
    "复杂非线性": make_classification(n_samples=1000, n_features=2, n_redundant=0,
                               n_informative=2, n_clusters_per_class=2, random_state=42)
}

# 标准化所有数据集
for dataset_name in datasets:
    X, y = datasets[dataset_name]
    datasets[dataset_name] = (StandardScaler().fit_transform(X), y)

# 训练和评估不同激活函数在不同数据集上的表现
results = {}

for dataset_name, (X, y) in datasets.items():
    print(f"\n===== 数据集: {dataset_name} =====")
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    dataset_results = {'models': {}, 'losses': {}, 'accuracies': {}}
    
    for activation in activation_functions:
        print(f"\n训练使用 {activation} 激活函数的模型")
        model = NeuralNetwork(activation)
        loss_history = train_model(model, X_train_tensor, y_train_tensor, epochs=500)  # 减少迭代次数以加快运行
        accuracy = evaluate_model(model, X_test_tensor, y_test_tensor)
        
        dataset_results['models'][activation] = model
        dataset_results['losses'][activation] = loss_history
        dataset_results['accuracies'][activation] = accuracy
        
        print(f"{activation} 模型在{dataset_name}数据集上的测试准确率: {accuracy:.4f}")
        
        # 可视化决策边界
        plot_decision_boundary(model, X, y, f"{activation}_{dataset_name}")
    
    results[dataset_name] = dataset_results
    
    # 为每个数据集绘制损失曲线比较
    plt.figure(figsize=(12, 8))
    for activation, loss in dataset_results['losses'].items():
        plt.plot(loss, label=activation)
    plt.title(f'{dataset_name}数据集上不同激活函数的损失曲线比较')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'loss_comparison_{dataset_name}.png')
    plt.show()
    
    # 为每个数据集绘制准确率比较
    plt.figure(figsize=(10, 6))
    plt.bar(dataset_results['accuracies'].keys(), dataset_results['accuracies'].values())
    plt.title(f'{dataset_name}数据集上不同激活函数的测试准确率比较')
    plt.xlabel('激活函数')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    for i, (activation, accuracy) in enumerate(dataset_results['accuracies'].items()):
        plt.text(i, accuracy + 0.01, f'{accuracy:.4f}', ha='center')
    plt.savefig(f'accuracy_comparison_{dataset_name}.png')
    plt.show()

# 绘制所有数据集上的激活函数性能比较
plt.figure(figsize=(15, 10))
bar_width = 0.2
index = np.arange(len(activation_functions))

for i, dataset_name in enumerate(results.keys()):
    accuracies = [results[dataset_name]['accuracies'][act] for act in activation_functions]
    plt.bar(index + i*bar_width, accuracies, bar_width, label=dataset_name)

plt.xlabel('激活函数')
plt.ylabel('准确率')
plt.title('不同激活函数在各数据集上的性能比较')
plt.xticks(index + bar_width, activation_functions)
plt.legend()
plt.tight_layout()
plt.savefig('overall_comparison.png')
plt.show()