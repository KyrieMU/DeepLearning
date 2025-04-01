import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取数据
data_path = r'e:\GitHub\DeepLearning\深度学习\data\Advertising.csv'
data = pd.read_csv(data_path)

# 为简化起见，只使用TV和radio特征
X = data[['TV', 'radio']].values
y = data['sales'].values

# 定义RSS损失函数
def compute_rss(X, y, w):
    y_pred = X.dot(w)
    return np.sum((y_pred - y) ** 2) / len(y)

# 创建网格点以可视化RSS与w的关系
w1_vals = np.linspace(-0.1, 0.3, 100)
w2_vals = np.linspace(-0.1, 1.0, 100)
w1_grid, w2_grid = np.meshgrid(w1_vals, w2_vals)
rss_grid = np.zeros(w1_grid.shape)

# 计算每个网格点的RSS
for i in range(len(w1_vals)):
    for j in range(len(w2_vals)):
        w_temp = np.array([w1_grid[j, i], w2_grid[j, i]])
        rss_grid[j, i] = compute_rss(X, y, w_temp)

# 绘制RSS与w的关系的3D图
fig = plt.figure(figsize=(15, 12))

# 3D表面图
ax1 = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax1.plot_surface(w1_grid, w2_grid, rss_grid, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w[TV]')
ax1.set_ylabel('w[radio]')
ax1.set_zlabel('RSS损失')
ax1.set_title('RSS损失函数的3D表面')
fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)

# 等高线图
ax2 = fig.add_subplot(2, 2, 2)
contour = ax2.contour(w1_grid, w2_grid, rss_grid, 20, colors='k', alpha=0.4)
ax2.clabel(contour, inline=True, fontsize=8)
filled_contour = ax2.contourf(w1_grid, w2_grid, rss_grid, 100, cmap='viridis', alpha=0.6)
ax2.set_xlabel('w[TV]')
ax2.set_ylabel('w[radio]')
ax2.set_title('RSS损失函数的等高线图')
fig.colorbar(filled_contour, ax=ax2)

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算标准化数据的RSS
rss_grid_scaled = np.zeros(w1_grid.shape)
for i in range(len(w1_vals)):
    for j in range(len(w2_vals)):
        w_temp = np.array([w1_grid[j, i], w2_grid[j, i]])
        rss_grid_scaled[j, i] = compute_rss(X_scaled, y, w_temp)

# 标准化数据的3D表面图
ax3 = fig.add_subplot(2, 2, 3, projection='3d')
surf_scaled = ax3.plot_surface(w1_grid, w2_grid, rss_grid_scaled, cmap='plasma', alpha=0.8)
ax3.set_xlabel('w[TV]')
ax3.set_ylabel('w[radio]')
ax3.set_zlabel('RSS损失')
ax3.set_title('标准化数据的RSS损失函数3D表面')
fig.colorbar(surf_scaled, ax=ax3, shrink=0.5, aspect=5)

# 标准化数据的等高线图
ax4 = fig.add_subplot(2, 2, 4)
contour_scaled = ax4.contour(w1_grid, w2_grid, rss_grid_scaled, 20, colors='k', alpha=0.4)
ax4.clabel(contour_scaled, inline=True, fontsize=8)
filled_contour_scaled = ax4.contourf(w1_grid, w2_grid, rss_grid_scaled, 100, cmap='plasma', alpha=0.6)
ax4.set_xlabel('w[TV]')
ax4.set_ylabel('w[radio]')
ax4.set_title('标准化数据的RSS损失函数等高线图')
fig.colorbar(filled_contour_scaled, ax=ax4)

plt.tight_layout()
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\rss_visualization.png', dpi=300)
plt.show()