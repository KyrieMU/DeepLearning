import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成非线性数据
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(300, 1))
y = 0.5 * X.ravel()**3 - X.ravel()**2 + 2 * X.ravel() + 2 + np.random.normal(0, 3, size=300)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义不同正则化的模型
def create_model(regularization=None, dropout_rate=0.0):
    model = Sequential()
    
    if regularization == 'l1':
        model.add(Dense(100, activation='relu', kernel_regularizer=l1(0.01), input_shape=(1,)))
    elif regularization == 'l2':
        model.add(Dense(100, activation='relu', kernel_regularizer=l2(0.01), input_shape=(1,)))
    else:
        model.add(Dense(100, activation='relu', input_shape=(1,)))
    
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(100, activation='relu'))
    
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse')
    return model

# 创建不同的模型
models = {
    '无正则化': create_model(),
    'L1正则化': create_model(regularization='l1'),
    'L2正则化': create_model(regularization='l2'),
    'Dropout': create_model(dropout_rate=0.3)
}

# 训练模型并可视化
plt.figure(figsize=(15, 12))

for i, (name, model) in enumerate(models.items()):
    # 训练模型
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )
    
    # 绘制损失曲线
    plt.subplot(3, 2, i+1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'{name}模型的损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('均方误差')
    plt.legend()
    
    # 预测并绘制拟合曲线
    X_plot = np.linspace(-3, 3, 1000).reshape(-1, 1)
    X_plot_scaled = scaler.transform(X_plot)
    y_plot = model.predict(X_plot_scaled).ravel()
    
    if i == len(models) - 1:  # 最后一个子图用于比较所有模型
        continue
        
    plt.subplot(3, 2, 5)
    if i == 0:
        plt.scatter(X, y, alpha=0.3, label='原始数据')
    plt.plot(X_plot, y_plot, label=name)
    plt.title('不同正则化方法的拟合效果对比')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()

# 比较不同模型的权重分布
plt.subplot(3, 2, 6)
for name, model in models.items():
    # 获取第一层的权重
    weights = model.layers[0].get_weights()[0].flatten()
    plt.hist(weights, bins=50, alpha=0.5, label=name)

plt.title('不同正则化方法的权重分布')
plt.xlabel('权重值')
plt.ylabel('频数')
plt.legend()

plt.tight_layout()
plt.savefig('e:\\Desktop\\深度学习\\第6章模型评估和正则化\\regularization_comparison.png', dpi=300)
plt.show()