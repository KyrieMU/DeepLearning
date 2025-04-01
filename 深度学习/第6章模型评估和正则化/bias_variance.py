import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成带噪声的数据
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = np.sin(X.ravel())
y = y_true + 0.2 * np.random.randn(100)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建不同复杂度的模型
models = {
    '欠拟合模型': MLPRegressor(hidden_layer_sizes=(2,), max_iter=1000, random_state=42),
    '适合模型': MLPRegressor(hidden_layer_sizes=(10,), max_iter=1000, random_state=42),
    '过拟合模型': MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=1000, random_state=42)
}

plt.figure(figsize=(15, 10))

# 训练模型并可视化
for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算误差
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # 可视化
    plt.subplot(2, 2, i+1)
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='训练数据')
    plt.scatter(X_test, y_test, color='green', alpha=0.5, label='测试数据')
    
    # 排序X以便绘制平滑曲线
    X_plot = np.linspace(0, 10, 1000).reshape(-1, 1)
    y_plot = model.predict(X_plot)
    
    plt.plot(X_plot, y_plot, color='red', label='模型预测')
    plt.plot(X_plot, np.sin(X_plot.ravel()), color='black', linestyle='--', label='真实函数')
    
    plt.title(f'{name}\n训练MSE: {train_mse:.4f}, 测试MSE: {test_mse:.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

# 绘制训练集和测试集误差随模型复杂度变化的曲线
plt.subplot(2, 2, 4)
complexities = [1, 2, 5, 10, 20, 50, 100, 200]
train_errors = []
test_errors = []

for complexity in complexities:
    model = MLPRegressor(hidden_layer_sizes=(complexity,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_errors.append(mean_squared_error(y_train, train_pred))
    test_errors.append(mean_squared_error(y_test, test_pred))

plt.plot(complexities, train_errors, 'o-', label='训练误差')
plt.plot(complexities, test_errors, 'o-', label='测试误差')
plt.xscale('log')
plt.xlabel('模型复杂度（隐藏层神经元数）')
plt.ylabel('均方误差')
plt.title('模型复杂度与误差关系')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('fitting_comparison.png', dpi=300)
plt.show()