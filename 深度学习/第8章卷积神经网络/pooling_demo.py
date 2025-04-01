import tensorflow as tf
import matplotlib.pyplot as plt

#池化层原理

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成测试数据
data = tf.random.uniform((1, 28, 28, 1), minval=0, maxval=1)

# 构建池化层
max_pool = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)

# 执行池化
pooled = max_pool(data)

# 可视化对比
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('原始特征图')
plt.imshow(data[0,:,:,0], cmap='viridis')

plt.subplot(1,2,2)
plt.title('最大池化结果')
plt.imshow(tf.image.resize(pooled, (28,28))[0,:,:,0], cmap='viridis')
plt.savefig('e:\\Desktop\\深度学习\\第8章卷积神经网络\\pooling_effect.png', dpi=300)
plt.show()