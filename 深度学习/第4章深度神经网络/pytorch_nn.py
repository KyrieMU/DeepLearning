import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# 定义神经网络模型
class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
        super(DeepNeuralNetwork, self).__init__()
        
        # 创建网络层
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(hidden_sizes[0]))
        layers.append(nn.Dropout(dropout_rate))
        
        # 添加更多隐藏层
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_sizes[i+1]))
            layers.append(nn.Dropout(dropout_rate))
        
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # 如果是二分类问题，添加Sigmoid激活函数
        if output_size == 1:
            layers.append(nn.Sigmoid())
        
        # 创建Sequential模型
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 生成复杂的非线性数据
def generate_complex_data(n_samples=1000, noise=0.1, boundary_type=1):
    X = np.random.randn(n_samples, 2) * 2
    
    # 创建复杂的非线性决策边界
    radius = np.sqrt(X[:, 0]**2 + X[:, 1]**2)
    angle = np.arctan2(X[:, 1], X[:, 0])
    
    y = np.zeros(n_samples)
    
    # 根据选择的边界类型生成标签
    if boundary_type == 1:
        # 螺旋形状的决策边界
        y[(radius < 4) & (np.sin(3 * angle + radius) > 0)] = 1
    elif boundary_type == 2:
        # 圆形决策边界
        y[((radius > 1.5) & (radius < 3)) | (radius < 0.8)] = 1
    elif boundary_type == 3:
        # 棋盘格状的决策边界
        grid_size = 0.8
        x_grid = np.floor(X[:, 0] / grid_size) % 2
        y_grid = np.floor(X[:, 1] / grid_size) % 2
        y[np.logical_xor(x_grid, y_grid)] = 1
    
    # 添加噪声
    mask = np.random.rand(n_samples) < noise
    y[mask] = 1 - y[mask]
    
    return X, y.reshape(-1, 1)

# 训练模型
def train_model(model, X_train, y_train, X_val, y_val, learning_rate=0.01, epochs=50, batch_size=32):
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 定义损失函数和优化器
    criterion = nn.BCELoss()  # 二元交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # 训练历史记录
    train_losses = []
    val_losses = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()  # 设置为训练模式
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()  # 设置为评估模式
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
            val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

# 可视化决策边界
def plot_decision_boundary(model, X, y):
    # 创建网格点
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    model.eval()
    with torch.no_grad():
        grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(grid).numpy()
    
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和数据点
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap=plt.cm.RdBu, edgecolors='k')
    plt.title('PyTorch深度神经网络决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.savefig('pytorch_decision_boundary.png')
    plt.show()

# 主函数
if __name__ == "__main__":
    # 生成数据 - 选择决策边界类型 (1:螺旋形, 2:圆形, 3:棋盘格)
    boundary_type = 2# 可以修改为1、2或3
    X, y = generate_complex_data(2000, noise=0.05, boundary_type=boundary_type)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    input_size = 2
        
# 根据边界类型选择不同的网络结构
    if boundary_type == 1:  # 螺旋形
            hidden_sizes = [32, 16, 8]  # 漏斗形结构
    elif boundary_type == 2:  # 圆形
            hidden_sizes = [16, 16]  # 较简单的结构
    elif boundary_type == 3:  # 棋盘格
            hidden_sizes = [64, 64, 32, 16]  # 更深的网络
        
    output_size = 1  # 二分类问题


    model = DeepNeuralNetwork(input_size, hidden_sizes, output_size)
    
    # 训练模型
    train_losses, val_losses = train_model(
        model, X_train, y_train, X_test, y_test,
        learning_rate=0.001, epochs=200, batch_size=32
    )
    
    # 可视化训练过程
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('pytorch_training_loss.png')
    plt.show()
    
    # 可视化决策边界
    plot_decision_boundary(model, X, y)
    
    # 计算准确率
    model.eval()
    with torch.no_grad():
        test_predictions = model(torch.FloatTensor(X_test))
        test_predictions = (test_predictions > 0.5).float().numpy()
        accuracy = np.mean(test_predictions == y_test)
        print(f"测试集准确率: {accuracy:.4f}")