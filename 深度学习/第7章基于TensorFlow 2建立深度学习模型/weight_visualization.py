import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载MNIST数据集
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0  # 归一化

# 构建无隐藏层的简单模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将28x28图像展平为784维向量
    tf.keras.layers.Dense(10, activation='softmax')  # 直接输出10个类别的概率
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(X_train, y_train, epochs=5, verbose=1)

# 获取权重矩阵 (784, 10)
weights = model.layers[1].get_weights()[0]

# 可视化权重
plt.figure(figsize=(15, 8))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(weights[:, i].reshape(28, 28), cmap='gray')
    plt.title(f'数字 {i} 的权重模式')
    plt.axis('off')

plt.suptitle('MNIST分类器权重可视化 (无隐藏层模型)')
plt.tight_layout()
plt.savefig('e:\\Desktop\\深度学习\\第7章基于TensorFlow 2建立深度学习模型\\mnist_weights.png', dpi=300)
plt.show()