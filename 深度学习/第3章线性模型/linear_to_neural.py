import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 生成示例数据
def generate_complex_data(seed=42, sample_size=1000, complexity='medium', random_components=False, sort_data=True):
    """生成复杂的非线性数据，支持随机组件选择
    
    参数:
        seed (int): 随机种子，确保可重复性
        sample_size (int): 样本数量
        complexity (str): 复杂度 - 'low', 'medium', 'high'
        random_components (bool): 是否随机选择非线性组件
        sort_data (bool): 是否对X进行排序，设为False可增加随机性
    
    返回:
        X (ndarray): 输入特征
        y (ndarray): 目标值
    """
    np.random.seed(seed)
    
    # 生成输入特征
    X = 15 * np.random.rand(sample_size, 1)
    if sort_data:
        X = np.sort(X, axis=0)
    
    # 定义所有可能的非线性组件
    components = {
        'sine': lambda x: np.sin(0.8 * x) * 4 * np.exp(-0.1 * x),
        'cosine': lambda x: np.cos(3 * x) * 2 * np.sin(0.5 * x),
        'polynomial': lambda x: 0.5 * (x**2) - 0.2 * (x**2.5),
        'arctan': lambda x: 1.5 * np.arctan(0.5*(x-7)),
        'abs': lambda x: 0.8 * np.abs(x-10),
        'exponential': lambda x: np.exp(0.3 * x) * np.sin(x),
        'piecewise': lambda x: np.where(x > 8, 3 * np.sqrt(np.maximum(x-8, 0)), 0),
        'log': lambda x: 2 * np.log(np.maximum(x, 0.1)),
        'sigmoid': lambda x: 4 / (1 + np.exp(-0.5 * (x - 7))),
        'periodic': lambda x: 2 * np.sin(x) * np.cos(x * 0.4) * np.sin(x * 0.1)
    }
    
    # 添加随机参数变化，使每次生成的函数形状略有不同
    if random_components:
        # 为每个组件添加随机参数变化
        rand_params = {}
        for key in components.keys():
            rand_params[key] = {
                'amp': 0.5 + np.random.rand() * 1.5,  # 振幅变化
                'freq': 0.7 + np.random.rand() * 0.6,  # 频率变化
                'phase': np.random.rand() * np.pi     # 相位变化
            }
        
        # 修改组件函数，加入随机参数
        components = {
            'sine': lambda x, p=rand_params['sine']: p['amp'] * np.sin(p['freq'] * 0.8 * x + p['phase']) * 4 * np.exp(-0.1 * x),
            'cosine': lambda x, p=rand_params['cosine']: p['amp'] * np.cos(p['freq'] * 3 * x + p['phase']) * 2 * np.sin(0.5 * x),
            'polynomial': lambda x, p=rand_params['polynomial']: p['amp'] * (0.5 * (x**2) - 0.2 * (x**p['freq']+1.5)),
            'arctan': lambda x, p=rand_params['arctan']: p['amp'] * 1.5 * np.arctan(p['freq'] * 0.5 * (x - 7 + p['phase'])),
            'abs': lambda x, p=rand_params['abs']: p['amp'] * 0.8 * np.abs(x - (10 + p['phase'])),
            'exponential': lambda x, p=rand_params['exponential']: p['amp'] * np.exp(p['freq'] * 0.3 * x) * np.sin(x + p['phase']),
            'piecewise': lambda x, p=rand_params['piecewise']: p['amp'] * np.where(x > 8, 3 * np.sqrt(np.maximum(x - (8 - p['phase']), 0)), 0),
            'log': lambda x, p=rand_params['log']: p['amp'] * 2 * np.log(np.maximum(x * p['freq'], 0.1)),
            'sigmoid': lambda x, p=rand_params['sigmoid']: p['amp'] * 4 / (1 + np.exp(-p['freq'] * 0.5 * (x - 7 + p['phase']))),
            'periodic': lambda x, p=rand_params['periodic']: p['amp'] * 2 * np.sin(x + p['phase']) * np.cos(p['freq'] * x * 0.4) * np.sin(x * 0.1)
        }
    
    # 根据复杂度设置参数
    if complexity == 'low':
        noise_scale = 0.5
        active_ratio = 0.3
        min_components = 2
        max_components = 3
    elif complexity == 'medium':
        noise_scale = 1.0
        active_ratio = 0.5
        min_components = 3
        max_components = 6
    else:  # high
        noise_scale = 1.5
        active_ratio = 0.7
        min_components = 5
        max_components = 8
    
    # 选择要使用的组件
    if random_components:
        # 随机选择组件数量
        n_components = np.random.randint(min_components, min(max_components + 1, len(components) + 1))
        # 随机选择具体组件
        selected_keys = np.random.choice(list(components.keys()), size=n_components, replace=False)
        selected_components = {k: components[k] for k in selected_keys}
        print(f"随机选择了{n_components}个组件: {', '.join(selected_keys)}")
    else:
        # 使用所有组件，但根据复杂度随机禁用一些
        selected_components = {}
        for key, func in components.items():
            if np.random.rand() < active_ratio:
                selected_components[key] = func
        print(f"使用了{len(selected_components)}个组件: {', '.join(selected_components.keys())}")
    
    # 生成目标值
    y = np.zeros(X.shape[0])
    for func in selected_components.values():
        y += func(X).ravel()
    
    # 添加噪声
    noise_types = [
        np.random.normal(0, 1, X.shape[0]),  # 高斯噪声
        np.random.gumbel(0, 1, X.shape[0]),  # 极值分布噪声
        np.random.poisson(0.5 + X.ravel()/20),  # 泊松噪声
        np.random.rand(X.shape[0]) * (0.3 + 0.1*X.ravel()),  # 均匀噪声
        np.random.rand(X.shape[0]) * (np.random.rand(X.shape[0]) > 0.95) * 5  # 脉冲噪声
    ]
    
    # 随机选择噪声类型
    noise_weights = np.random.rand(len(noise_types))
    noise_weights /= noise_weights.sum()  # 归一化权重
    
    noise = np.zeros(X.shape[0])
    for i, n in enumerate(noise_types):
        noise += noise_weights[i] * n
    
    y += noise_scale * noise
    
    # 添加离群点
    outlier_ratio = 0.05 + 0.05 * np.random.rand()  # 5%-10%的离群点
    outlier_intensity = 10 + 20 * np.random.rand()  # 强度在10-30之间
    
    outlier_mask = np.random.rand(X.shape[0]) > (1 - outlier_ratio)
    y[outlier_mask] += outlier_intensity * np.abs(np.random.randn(np.sum(outlier_mask))) * (1 + X[outlier_mask].ravel()/10)
    
    return X, y

# 生成数据示例 - 不排序且使用随机参数
X, y = generate_complex_data(
    seed=np.random.randint(1, 10000),  # 随机种子
    sample_size=1332,
    complexity='high',
    random_components=True,
    sort_data=True  # 不排序X，增加随机性
)


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 训练线性回归模型
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear_pred = linear_model.predict(X)
linear_mse = mean_squared_error(y, y_linear_pred)

# 训练神经网络模型
nn_model = MLPRegressor(hidden_layer_sizes=(10, 10), activation='relu', 
                        solver='adam', max_iter=2000, random_state=42,
                        tol=1e-4)  # 增加迭代次数并调整收敛容差
nn_model.fit(X, y)
y_nn_pred = nn_model.predict(X)
nn_mse = mean_squared_error(y, y_nn_pred)

# 可视化
plt.figure(figsize=(12, 6))
plt.scatter(X, y, color='navy', s=30, marker='o', label='数据点')
plt.plot(X, y_linear_pred, color='red', linewidth=2, label=f'线性模型 (MSE: {linear_mse:.4f})')
plt.plot(X, y_nn_pred, color='green', linewidth=2, label=f'神经网络 (MSE: {nn_mse:.4f})')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性模型 vs 神经网络')
plt.legend()
plt.grid(True)
plt.savefig('linear_vs_neural.png')
plt.show()


'''
# 可视化神经网络结构
def plot_neural_network():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置层数和每层节点数（简化结构）
    n_input = 1
    n_hidden1 = 10
    n_hidden2 = 5
    n_hidden3 = 10
    n_output = 1
    
    # 绘制参数
    layer_sizes = [n_input, n_hidden1, n_hidden2, n_hidden3, n_output]
    layer_names = ['输入层', '隐藏层1 (ReLU)', '隐藏层2 (ReLU)', '隐藏层3 (ReLU)', '输出层']
    layer_colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFAAFF', '#FFFFAA']
    
    # 设置每层的位置（优化布局）
    layer_positions = [1, 3, 5, 7, 9]
    
    # 绘制节点和连接
    for i, (size, pos, name, color) in enumerate(zip(layer_sizes, layer_positions, layer_names, layer_colors)):
        # 绘制节点
        for j in range(size):
            if size > 5 and 1 < j < size-1:
                if j == 2:
                    ax.text(pos, 0.5, '...', ha='center', va='center', fontsize=20)
                continue
                
            y_pos = 0.9 - (j * 0.8 / max(1, size-1))
            circle = plt.Circle((pos, y_pos), 0.15, fill=True, color=color, ec='black', alpha=0.8)
            ax.add_patch(circle)
            
            # 添加节点标注
            if i == 0:
                ax.text(pos, y_pos, 'x', ha='center', va='center', fontsize=10)
            elif i == len(layer_sizes)-1:
                ax.text(pos, y_pos, 'y', ha='center', va='center', fontsize=10)
        
        # 添加层说明
        ax.text(pos, 1.1, name, ha='center', va='center', fontsize=10, 
               bbox=dict(facecolor=color, alpha=0.3, boxstyle="round"))
        
        # 绘制连接线（增强可见性）
        if i < len(layer_sizes) - 1:
            next_size = layer_sizes[i+1]
            next_pos = layer_positions[i+1]
            
            # 使用渐变色增强连接线可视化
            line_color = plt.cm.Blues(0.3 + i*0.1)  # 根据层深改变颜色
            
            for j in range(size):
                if size > 5 and 1 < j < size-1:
                    continue
                    
                y_start = 0.9 - (j * 0.8 / max(1, size-1))
                
                for k in range(next_size):
                    if next_size > 5 and 1 < k < next_size-1:
                        continue
                        
                    y_end = 0.9 - (k * 0.8 / max(1, next_size-1))
                    # 增强连接线参数
                    ax.plot([pos+0.15, next_pos-0.15], [y_start, y_end], 
                           color=line_color,  # 使用渐变色
                           alpha=0.3,          # 提高透明度
                           linewidth=1.2,      # 加粗线宽
                           solid_capstyle='round')  # 圆角端点
    
    # 添加更多说明（优化文字说明）
    ax.text(5, -0.1, '网络参数:', ha='center', fontsize=12)
    ax.text(5, -0.2, '优化器: Adam | 学习率: 0.001', ha='center', fontsize=10)
    ax.text(5, -0.3, '损失函数: 均方误差', ha='center', fontsize=10)
    
    # 设置图表属性
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.4, 1.2)
    ax.set_title('神经网络结构示意图 (3隐藏层)', fontsize=14)
    ax.axis('off')
    
    plt.savefig('e:/Desktop/深度学习/第3章线性模型/neural_network_structure.png')
    plt.show()

plot_neural_network()
'''