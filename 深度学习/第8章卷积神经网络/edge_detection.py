import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加在文件开头

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np  # 添加这行导入语句

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 载入MNIST样本
mnist = tf.keras.datasets.mnist
(_, _), (X_test, _) = mnist.load_data()
sample = X_test[0].reshape(28, 28, 1)/255.0

# 定义卷积核
edge_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)  # 改为NumPy数组
conv_layer = tf.keras.layers.Conv2D(1, kernel_size=3, 
                                   kernel_initializer=tf.constant_initializer(edge_kernel),
                                   padding='same')

# 执行卷积
feature_map = conv_layer(sample[tf.newaxis,...]).numpy()[0,:,:,0]

# 可视化
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(sample[:,:,0], cmap='gray')
plt.title('原始图像')

plt.subplot(1,2,2)
plt.imshow(feature_map, cmap='gray')
plt.title('边缘特征图')
plt.savefig('/edge_detection.png', dpi=300)