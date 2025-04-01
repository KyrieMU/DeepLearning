import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 获取第一个卷积层的权重
conv_weights = model.layers[0].get_weights()[0]

# 可视化前16个卷积核
plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    kernel = conv_weights[:,:,0,i]
    plt.imshow(kernel, cmap='gray')
    plt.axis('off')
plt.suptitle('第一卷积层核可视化')
plt.savefig('e:\\Desktop\\深度学习\\第8章卷积神经网络\\conv_kernels.png', dpi=300)