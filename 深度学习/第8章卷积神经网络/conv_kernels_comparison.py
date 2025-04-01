import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 添加在文件开头

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 载入MNIST样本
mnist = tf.keras.datasets.mnist
(_, _), (X_test, _) = mnist.load_data()
sample = X_test[12].reshape(28, 28, 1)/255.0  # 选择一个有明显特征的样本

# 定义不同的卷积核
kernels = {
    '水平边缘检测': np.array([[-1,-1,-1], [0,0,0], [1,1,1]]),
    '垂直边缘检测': np.array([[-1,0,1], [-1,0,1], [-1,0,1]]),
    '锐化': np.array([[0,-1,0], [-1,5,-1], [0,-1,0]]),
    '高斯模糊': np.array([[1,2,1], [2,4,2], [1,2,1]]) / 16
}

# 创建图形
plt.figure(figsize=(15, 10))

# 显示原始图像
plt.subplot(2, 3, 1)
plt.imshow(sample[:,:,0], cmap='gray')
plt.title('原始图像')
plt.axis('off')

# 应用不同卷积核并显示结果
for i, (name, kernel) in enumerate(kernels.items(), 2):
    # 创建卷积层
    kernel_tf = tf.constant(kernel.reshape(3, 3, 1, 1), dtype=tf.float32)
    conv_layer = tf.keras.layers.Conv2D(1, kernel_size=3, 
                                       kernel_initializer=tf.constant_initializer(kernel),
                                       padding='same')
    
    # 执行卷积
    feature_map = conv_layer(sample[tf.newaxis,...]).numpy()[0,:,:,0]
    
    # 显示结果
    plt.subplot(2, 3, i)
    plt.imshow(feature_map, cmap='viridis')
    plt.title(f'{name}卷积核效果')
    plt.axis('off')

# 保存图像
# 直接修改为当前脚本所在目录保存
plt.tight_layout()
plt.savefig('e:\\GitHub\\DeepLearning\\深度学习\\第8章卷积神经网络\\conv_kernels_comparison.png', dpi=300)
plt.show()