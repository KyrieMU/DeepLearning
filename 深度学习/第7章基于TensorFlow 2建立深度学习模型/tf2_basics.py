import tensorflow as tf
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建张量
a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)

# 矩阵乘法
c = tf.matmul(a, b)

print("矩阵乘法结果:\n", c.numpy())

# 可视化
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.imshow(a, cmap='viridis')
plt.title('矩阵A')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(b, cmap='viridis')
plt.title('矩阵B')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(c, cmap='viridis')
plt.title('乘积结果')
plt.colorbar()

plt.tight_layout()
plt.savefig('e:\\Desktop\\深度学习\\第7章基于TensorFlow 2建立深度学习模型\\matrix_multiplication.png', dpi=300)
plt.show()