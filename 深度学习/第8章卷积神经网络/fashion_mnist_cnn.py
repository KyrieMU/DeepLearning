import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据准备
# 修改数据准备部分
try:
    # 指定明确的数据缓存路径
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data(
        cache_dir='e:\\Desktop\\深度学习\\第8章卷积神经网络\\datasets',
        force_reload=True  # 强制重新下载
    )
except EOFError:
    # 清理损坏的缓存文件
    import shutil
    shutil.rmtree('e:\\Desktop\\深度学习\\第8章卷积神经网络\\datasets\\fashion-mnist')
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# 数据预处理保持不变
X_train, X_test = X_train[..., tf.newaxis]/255.0, X_test[..., tf.newaxis]/255.0

# 构建CNN模型
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# 编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练并保存历史
history = model.fit(X_train, y_train, epochs=10, 
                   validation_split=0.2, verbose=1)

# 可视化训练过程
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失曲线')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('准确率曲线')
plt.legend()
plt.savefig('e:\\Desktop\\深度学习\\第8章卷积神经网络\\cnn_training.png', dpi=300)