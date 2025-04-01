import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# 抑制TensorFlow警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=全部信息, 1=INFO, 2=WARNING, 3=ERROR

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建一个简单的5x5输入
input_data = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
]).reshape(1, 5, 5, 1).astype(np.float32)

# 定义一个简单的3x3卷积核
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
]).reshape(3, 3, 1, 1).astype(np.float32)

# 创建不同配置的卷积层
conv_configs = {
    '步长=1，无填充': {'strides': 1, 'padding': 'valid'},
    '步长=1，有填充': {'strides': 1, 'padding': 'same'},
    '步长=2，无填充': {'strides': 2, 'padding': 'valid'},
    '步长=2，有填充': {'strides': 2, 'padding': 'same'}
}

# 创建图形
plt.figure(figsize=(15, 10))  # 创建唯一图形对象

# 显示输入数据 (位置1)
plt.subplot(2, 3, 1)
plt.imshow(input_data[0, :, :, 0], cmap='viridis')  # 新增
plt.title('输入数据 (5x5)')  # 新增
plt.colorbar()  # 新增

# 显示卷积核 (位置2)
plt.subplot(2, 3, 2)
plt.imshow(kernel[:, :, 0, 0], cmap='viridis')  # 新增
plt.title('卷积核 (3x3)')  # 新增
plt.colorbar()  # 新增

# 应用不同配置的卷积并显示结果 (位置3-6)
for i, (name, config) in enumerate(conv_configs.items(), 3):
    plt.subplot(2, 3, i)
    # 创建卷积层
    conv_layer = tf.keras.layers.Conv2D(
        filters=1,
        kernel_size=3,
        strides=config['strides'],
        padding=config['padding'],
        use_bias=False,
        kernel_initializer=tf.constant_initializer(kernel)
    )
    
    # 执行卷积
    output = conv_layer(input_data).numpy()
    
    # 显示结果
    plt.imshow(output[0, :, :, 0], cmap='viridis')
    plt.title(f'{name}\n输出形状: {output.shape[1]}x{output.shape[2]}')
    plt.colorbar()

# 添加图片显示和保存功能
plt.tight_layout()  # 自动调整子图间距
plt.savefig(r'e:\Desktop\深度学习\第8章卷积神经网络\conv_demo_result.png')  # 保存结果图片
plt.show()  # 显示图形