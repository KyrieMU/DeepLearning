import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成复杂数据
np.random.seed(42)
X = np.random.uniform(-3, 3, size=(500, 2))
y = 0.5 * X[:, 0]**2 + 0.2 * X[:, 1]**2 + 0.3 * X[:, 0] * X[:, 1] + 0.1 * np.sin(X[:, 0] * X[:, 1]) + np.random.normal(0, 0.5, size=500)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 创建模型
def create_model():
    model = Sequential([
        Dense(100, activation='relu', input_shape=(2,)),
        Dense(100, activation='relu'),
        Dense(100, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 创建两个相同的模型
model_with_early_stopping = create_model()
model_without_early_stopping = create_model()

# 设置早停
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True,
    verbose=1
)

# 训练模型
history_with_es = model_with_early_stopping.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping],
    verbose=0
)

history_without_es = model_without_early_stopping.fit(
    X_train, y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=0
)

# 可视化
plt.figure(figsize=(12, 10))

# 绘制损失曲线
plt.subplot(2, 1, 1)
plt.plot(history_with_es.history['loss'], label='训练损失(早停)')
plt.plot(history_with_es.history['val_loss'], label='验证损失(早停)')
plt.plot(history_without_es.history['loss'], label='训练损失(无早停)', linestyle='--')
plt.plot(history_without_es.history['val_loss'], label='验证损失(无早停)', linestyle='--')

# 标记早停点
stopped_epoch = len(history_with_es.history['loss'])
min_val_loss = min(history_with_es.history['val_loss'])
min_val_loss_epoch = history_with_es.history['val_loss'].index(min_val_loss)

plt.axvline(x=min_val_loss_epoch, color='r', linestyle='-', label=f'最佳模型(epoch {min_val_loss_epoch})')
plt.axvline(x=stopped_epoch-1, color='g', linestyle='-', label=f'早停点(epoch {stopped_epoch-1})')

plt.title('早停法效果对比')
plt.xlabel('迭代次数')
plt.ylabel('均方误差')
plt.legend()
plt.grid(True)

# 绘制预测对比
plt.subplot(2, 1, 2)

# 生成测试数据
X_test = np.random.uniform(-3, 3, size=(200, 2))
y_test = 0.5 * X_test[:, 0]**2 + 0.2 * X_test[:, 1]**2 + 0.3 * X_test[:, 0] * X_test[:, 1] + 0.1 * np.sin(X_test[:, 0] * X_test[:, 1]) + np.random.normal(0, 0.5, size=200)
X_test_scaled = scaler.transform(X_test)

# 预测
y_pred_with_es = model_with_early_stopping.predict(X_test_scaled).ravel()
y_pred_without_es = model_without_early_stopping.predict(X_test_scaled).ravel()

# 计算MSE
mse_with_es = np.mean((y_test - y_pred_with_es)**2)
mse_without_es = np.mean((y_test - y_pred_without_es)**2)

plt.scatter(y_test, y_pred_with_es, alpha=0.5, label=f'早停 (MSE: {mse_with_es:.4f})')
plt.scatter(y_test, y_pred_without_es, alpha=0.5, label=f'无早停 (MSE: {mse_without_es:.4f})')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='理想预测')
plt.title('预测值与真实值对比')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('e:\\Desktop\\深度学习\\第6章模型评估和正则化\\early_stopping.png', dpi=300)
plt.show()