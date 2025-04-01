import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取数据
data_path = r'e:\GitHub\DeepLearning\深度学习\data\Advertising.csv'
data = pd.read_csv(data_path)

# 提取特征和目标变量
X = data[['TV', 'radio', 'newspaper']].values
y = data['sales'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义RSS损失函数
def compute_rss(X, y, w, b):
    y_pred = X.dot(w) + b
    return np.sum((y_pred - y) ** 2) / len(y)

# 定义梯度下降函数
def gradient_descent(X, y, w_init, b_init, alpha, num_iters, method='batch', batch_size=32):
    """
    实现不同的梯度下降方法
    
    参数:
    X: 特征矩阵
    y: 目标变量
    w_init: 初始权重
    b_init: 初始偏置
    alpha: 学习率
    num_iters: 迭代次数
    method: 梯度下降方法 ('batch', 'sgd', 'mini_batch')
    batch_size: mini-batch的大小
    
    返回:
    w: 最终权重
    b: 最终偏置
    w_history: 权重历史
    b_history: 偏置历史
    loss_history: 损失历史
    """
    w = w_init.copy()
    b = b_init
    w_history = [w.copy()]
    b_history = [b]
    loss_history = [compute_rss(X, y, w, b)]
    m = len(y)
    
    for i in range(num_iters):
        if method == 'batch':
            # 批量梯度下降
            y_pred = X.dot(w) + b
            dw = (2/m) * X.T.dot(y_pred - y)
            db = (2/m) * np.sum(y_pred - y)
            
        elif method == 'sgd':
            # 随机梯度下降
            idx = np.random.randint(0, m)
            xi = X[idx:idx+1]
            yi = y[idx:idx+1]
            y_pred = xi.dot(w) + b
            dw = (2) * xi.T.dot(y_pred - yi)
            db = (2) * np.sum(y_pred - yi)
            
        elif method == 'mini_batch':
            # 小批量梯度下降
            indices = np.random.choice(m, batch_size, replace=False)
            xi = X[indices]
            yi = y[indices]
            y_pred = xi.dot(w) + b
            dw = (2/batch_size) * xi.T.dot(y_pred - yi)
            db = (2/batch_size) * np.sum(y_pred - yi)
        
        # 更新参数
        w = w - alpha * dw
        b = b - alpha * db
        
        # 记录历史
        w_history.append(w.copy())
        b_history.append(b)
        loss_history.append(compute_rss(X, y, w, b))
    
    return w, b, w_history, b_history, loss_history

# 创建标准化和非标准化的数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 初始化参数
w_init = np.zeros(X_train.shape[1])
b_init = 0
alpha = 0.01
num_iters = 10

# 使用不同的梯度下降方法
methods = ['batch', 'sgd', 'mini_batch']
results = {}
results_scaled = {}

for method in methods:
    # 非标准化数据
    w, b, w_history, b_history, loss_history = gradient_descent(
        X_train, y_train, w_init, b_init, alpha, num_iters, method=method)
    results[method] = {
        'w': w, 'b': b, 'w_history': w_history, 
        'b_history': b_history, 'loss_history': loss_history
    }
    
    # 标准化数据
    w_scaled, b_scaled, w_history_scaled, b_history_scaled, loss_history_scaled = gradient_descent(
        X_train_scaled, y_train, w_init, b_init, alpha, num_iters, method=method)
    results_scaled[method] = {
        'w': w_scaled, 'b': b_scaled, 'w_history': w_history_scaled, 
        'b_history': b_history_scaled, 'loss_history': loss_history_scaled
    }

# 绘制损失函数收敛曲线
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
for method in methods:
    plt.plot(results[method]['loss_history'], label=f'{method}')
plt.title('非标准化数据的损失函数收敛曲线')
plt.xlabel('迭代次数')
plt.ylabel('RSS损失')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for method in methods:
    plt.plot(results_scaled[method]['loss_history'], label=f'{method}')
plt.title('标准化数据的损失函数收敛曲线')
plt.xlabel('迭代次数')
plt.ylabel('RSS损失')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\loss_convergence.png', dpi=300)
plt.show()

# 绘制参数w的收敛路径（以TV和radio为例）
plt.figure(figsize=(15, 12))

# 非标准化数据
for i, method in enumerate(methods):
    plt.subplot(2, 3, i+1)
    w_history = np.array(results[method]['w_history'])
    plt.plot(w_history[:, 0], w_history[:, 1], 'o-', markersize=2)
    plt.plot(w_history[0, 0], w_history[0, 1], 'go', markersize=5, label='起点')
    plt.plot(w_history[-1, 0], w_history[-1, 1], 'ro', markersize=5, label='终点')
    plt.title(f'非标准化数据 - {method}方法的参数收敛路径')
    plt.xlabel('w[TV]')
    plt.ylabel('w[radio]')
    plt.legend()
    plt.grid(True)

# 标准化数据
for i, method in enumerate(methods):
    plt.subplot(2, 3, i+4)
    w_history = np.array(results_scaled[method]['w_history'])
    plt.plot(w_history[:, 0], w_history[:, 1], 'o-', markersize=2)
    plt.plot(w_history[0, 0], w_history[0, 1], 'go', markersize=5, label='起点')
    plt.plot(w_history[-1, 0], w_history[-1, 1], 'ro', markersize=5, label='终点')
    plt.title(f'标准化数据 - {method}方法的参数收敛路径')
    plt.xlabel('w[TV]')
    plt.ylabel('w[radio]')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\parameter_convergence.png', dpi=300)
plt.show()

# 绘制RSS的等高线及参数收敛路径（以TV和radio为例）
def plot_contour_and_path(X, y, w_history, title, save_path):
    # 创建网格点 - 扩大范围以确保捕捉到更多特征
    w1_min, w1_max = min(w_history[:, 0]), max(w_history[:, 0])
    w2_min, w2_max = min(w_history[:, 1]), max(w_history[:, 1])
    
    # 扩大范围，确保能看到等高线
    w1_range = w1_max - w1_min
    w2_range = w2_max - w2_min
    w1_vals = np.linspace(w1_min - 0.2 * w1_range, w1_max + 0.2 * w1_range, 100)
    w2_vals = np.linspace(w2_min - 0.2 * w2_range, w2_max + 0.2 * w2_range, 100)
    w1_grid, w2_grid = np.meshgrid(w1_vals, w2_vals)
    
    # 计算每个网格点的RSS
    rss_grid = np.zeros(w1_grid.shape)
    for i in range(len(w1_vals)):
        for j in range(len(w2_vals)):
            w_temp = np.array([w1_grid[j, i], w2_grid[j, i], 0])  # 固定newspaper的权重为0
            rss_grid[j, i] = compute_rss(X, y, w_temp, 0)
    
    # 绘制等高线
    plt.figure(figsize=(10, 8))
    
    # 增加等高线的数量和可见度
    contour = plt.contour(w1_grid, w2_grid, rss_grid, 15, colors='black', linewidths=1.0, alpha=0.7)
    plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
    
    # 填充等高线
    filled_contour = plt.contourf(w1_grid, w2_grid, rss_grid, 100, cmap='jet', alpha=0.6)
    
    # 绘制参数收敛路径
    plt.plot(w_history[:, 0], w_history[:, 1], 'o-', color='white', markersize=3, alpha=0.7)
    plt.plot(w_history[0, 0], w_history[0, 1], 'go', markersize=6, label='起点')
    plt.plot(w_history[-1, 0], w_history[-1, 1], 'ro', markersize=6, label='终点')
    
    plt.colorbar(filled_contour, label='RSS损失')
    plt.title(title)
    plt.xlabel('w[TV]')
    plt.ylabel('w[radio]')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=300)
    plt.show()

# 为每种方法绘制等高线图
for method in methods:
    # 非标准化数据
    plot_contour_and_path(
        X_train, y_train, 
        np.array(results[method]['w_history'])[:, :2],  # 只取TV和radio的权重
        f'非标准化数据 - {method}方法的RSS等高线与参数收敛路径',
        f'e:\\GitHub\\DeepLearning\\深度学习\\第3章线性模型\\contour_{method}.png'
    )
    
    # 标准化数据
    plot_contour_and_path(
        X_train_scaled, y_train, 
        np.array(results_scaled[method]['w_history'])[:, :2],  # 只取TV和radio的权重
        f'标准化数据 - {method}方法的RSS等高线与参数收敛路径',
        f'e:\\GitHub\\DeepLearning\\深度学习\\第3章线性模型\\contour_{method}_scaled.png'
    )

# 打印最终的权重和偏置
print("最终模型参数:")
for method in methods:
    print(f"\n{method}方法:")
    print(f"非标准化数据 - 权重: {results[method]['w']}, 偏置: {results[method]['b']}")
    print(f"标准化数据 - 权重: {results_scaled[method]['w']}, 偏置: {results_scaled[method]['b']}")
    
    # 计算测试集上的RSS
    test_rss = compute_rss(X_test, y_test, results[method]['w'], results[method]['b'])
    test_rss_scaled = compute_rss(X_test_scaled, y_test, results_scaled[method]['w'], results_scaled[method]['b'])
    print(f"测试集RSS - 非标准化: {test_rss:.4f}, 标准化: {test_rss_scaled:.4f}")