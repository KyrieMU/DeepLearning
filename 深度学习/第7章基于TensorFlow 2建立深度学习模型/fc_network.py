import tensorflow as tf
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt


# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成数据
np.random.seed(42)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# 构建模型
model = tf.keras.Sequential([
    Dense(4, activation='relu', input_shape=(2,), name='hidden_layer'),
    Dense(1, activation='sigmoid', name='output_layer')
])

# 编译模型
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(X, y, epochs=100, verbose=0)

# 可视化训练过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.title('训练损失')
plt.xlabel('迭代次数')
plt.ylabel('损失值')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'])
plt.title('训练准确率')
plt.xlabel('迭代次数')
plt.ylabel('准确率')

plt.tight_layout()
plt.savefig('e:\\Desktop\\深度学习\\第7章基于TensorFlow 2建立深度学习模型\\training_curve.png', dpi=300)
plt.show()