import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取数据
data_path = r'e:\GitHub\DeepLearning\深度学习\data\Advertising.csv'
data = pd.read_csv(data_path)

# 创建一个2x2的子图布局
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

# 绘制各个特征与销售额的散点图
features = ['TV', 'radio', 'newspaper']
colors = ['blue', 'green', 'red']

for i, feature in enumerate(features):
    axes[i].scatter(data[feature], data['sales'], color=colors[i], alpha=0.6)
    axes[i].set_title(f'{feature}广告投入与销售额的关系')
    axes[i].set_xlabel(f'{feature}广告投入')
    axes[i].set_ylabel('销售额')
    
    # 添加趋势线
    z = np.polyfit(data[feature], data['sales'], 1)
    p = np.poly1d(z)
    axes[i].plot(data[feature], p(data[feature]), "r--", alpha=0.8)
    
    # 添加相关系数
    corr = data[feature].corr(data['sales'])
    axes[i].annotate(f'相关系数: {corr:.2f}', 
                     xy=(0.05, 0.95), 
                     xycoords='axes fraction',
                     fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))

# 绘制三维散点图
from mpl_toolkits.mplot3d import Axes3D
ax3d = axes[3]
ax3d = fig.add_subplot(2, 2, 4, projection='3d')
ax3d.scatter(data['TV'], data['radio'], data['sales'], c='purple', marker='o')
ax3d.set_xlabel('TV广告投入')
ax3d.set_ylabel('Radio广告投入')
ax3d.set_zlabel('销售额')
ax3d.set_title('TV和Radio广告投入与销售额的关系')

plt.tight_layout()
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\advertising_scatter.png', dpi=300)
plt.show()