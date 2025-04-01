import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

# 构建带正则化的模型（增加L1正则化）
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu', 
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
    Dropout(0.3),
    Dense(64, activation='relu', 
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
    Dropout(0.3),
    Dense(32, activation='relu',
          kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
    Dense(10, activation='softmax')
])

# 添加早停回调
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# 训练模型（增加回调）
history = model.fit(X_train, y_train, 
                    epochs=50,  # 增加最大epoch数
                    validation_split=0.2,
                    callbacks=[early_stopping],
                    verbose=1)

# 增强可视化部分
plt.figure(figsize=(18, 6))

# 损失曲线
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.axvline(x=np.argmin(history.history['val_loss']), color='r', linestyle='--', label='最佳模型')
plt.title('损失曲线对比')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.legend()

# 准确率曲线
plt.subplot(1, 3, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率') 
plt.axhline(y=max(history.history['val_accuracy']), color='g', linestyle='--', label='最高验证准确率')
plt.title('准确率曲线对比')
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.legend()

# 权重分布直方图
plt.subplot(1, 3, 3)
for i, layer in enumerate(model.layers):
    if hasattr(layer, 'kernel'):
        plt.hist(layer.kernel.numpy().flatten(), bins=50, alpha=0.5, label=f'层 {i+1}')
plt.title('权重分布')
plt.xlabel('权重值')
plt.ylabel('频数')
plt.legend()

plt.tight_layout()
plt.savefig('e:\\Desktop\\深度学习\\第7章基于TensorFlow 2建立深度学习模型\\mnist_regularization.png', dpi=300)
plt.show()

# 新增激活分布可视化
plot_model = tf.keras.models.Model(inputs=model.input, outputs=[layer.output for layer in model.layers[1:-1]])
activations = plot_model.predict(X_test[:1])

plt.figure(figsize=(12, 6))
for i, act in enumerate(activations):
    plt.subplot(1, len(activations), i+1)
    plt.hist(act.numpy().flatten(), bins=50)
    plt.title(f'隐藏层 {i+1} 激活分布')
plt.tight_layout()
plt.savefig('e:\\Desktop\\深度学习\\第7章基于TensorFlow 2建立深度学习模型\\activation_dist.png', dpi=300)