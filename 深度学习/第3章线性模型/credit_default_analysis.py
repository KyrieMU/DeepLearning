import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 读取数据
data_path = r'e:\GitHub\DeepLearning\深度学习\data\Default.txt'
# 数据没有表头，添加列名
column_names = ['balance', 'income', 'default']
data = pd.read_csv(data_path, sep='\t', names=column_names)

# 将Yes/No转换为1/0
data['default'] = data['default'].map({'Yes': 1, 'No': 0})

# 查看数据基本信息
print("数据集基本信息:")
print(data.info())
print("\n数据集统计摘要:")
print(data.describe())
print("\n违约与非违约样本数量:")
print(data['default'].value_counts())

# 绘制散点图
plt.figure(figsize=(12, 10))

# 余额与收入的散点图，按违约状态着色
plt.subplot(2, 2, 1)
plt.scatter(data[data['default']==1]['balance'], 
            data[data['default']==1]['income'], 
            c='red', marker='o', label='违约', alpha=0.6)
plt.scatter(data[data['default']==0]['balance'], 
            data[data['default']==0]['income'], 
            c='blue', marker='x', label='非违约', alpha=0.6)
plt.xlabel('信用卡余额')
plt.ylabel('收入')
plt.title('信用卡余额与收入的关系（按违约状态）')
plt.legend()
plt.grid(True, alpha=0.3)

# 余额的直方图，按违约状态分组
plt.subplot(2, 2, 2)
plt.hist(data[data['default']==1]['balance'], bins=20, alpha=0.5, color='red', label='违约')
plt.hist(data[data['default']==0]['balance'], bins=20, alpha=0.5, color='blue', label='非违约')
plt.xlabel('信用卡余额')
plt.ylabel('频数')
plt.title('信用卡余额分布（按违约状态）')
plt.legend()
plt.grid(True, alpha=0.3)

# 收入的直方图，按违约状态分组
plt.subplot(2, 2, 3)
plt.hist(data[data['default']==1]['income'], bins=20, alpha=0.5, color='red', label='违约')
plt.hist(data[data['default']==0]['income'], bins=20, alpha=0.5, color='blue', label='非违约')
plt.xlabel('收入')
plt.ylabel('频数')
plt.title('收入分布（按违约状态）')
plt.legend()
plt.grid(True, alpha=0.3)

# 箱线图比较
plt.subplot(2, 2, 4)
data.boxplot(column=['balance', 'income'], by='default', grid=True)
plt.title('余额和收入的箱线图（按违约状态）')
plt.suptitle('')  # 移除自动生成的标题

plt.tight_layout()
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\credit_default_scatter.png', dpi=300)
plt.show()

# 准备数据进行逻辑回归
X = data[['balance', 'income']].values
y = data['default'].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练逻辑回归模型
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# 模型评估
y_pred = model.predict(X_test_scaled)
print("\n分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 计算ROC曲线
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('接收者操作特征曲线')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\credit_default_roc.png', dpi=300)
plt.show()

# 可视化决策边界
def plot_decision_boundary(X, y, model, scaler, title):
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1000, X[:, 0].max() + 1000
    y_min, y_max = X[:, 1].min() - 10000, X[:, 1].max() + 10000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 100),
                         np.arange(y_min, y_max, 1000))
    
    # 对网格点进行预测
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和散点图
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
    plt.contour(xx, yy, Z, colors='k', linewidths=0.5)
    
    # 绘制散点图
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', label='违约', alpha=0.6)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='x', label='非违约', alpha=0.6)
    
    plt.xlabel('信用卡余额')
    plt.ylabel('收入')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 添加模型系数信息
    coef = model.coef_[0]
    intercept = model.intercept_[0]
    scaled_coef = coef / scaler.scale_
    scaled_intercept = intercept - np.sum(coef * scaler.mean_ / scaler.scale_)
    
    equation = f'决策边界方程: {scaled_coef[0]:.4f} × 余额 + {scaled_coef[1]:.4f} × 收入 + {scaled_intercept:.4f} = 0'
    plt.annotate(equation, xy=(0.05, 0.05), xycoords='axes fraction', 
                 bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3))
    
    return plt

# 绘制决策边界
plot_decision_boundary(X, y, model, scaler, '信用卡违约预测的逻辑回归决策边界')
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\credit_default_decision_boundary.png', dpi=300)
plt.show()

# 绘制概率分布
def plot_probability_distribution(X, y, model, scaler):
    # 创建网格点
    x_min, x_max = X[:, 0].min() - 1000, X[:, 0].max() + 1000
    y_min, y_max = X[:, 1].min() - 10000, X[:, 1].max() + 10000
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 100),
                         np.arange(y_min, y_max, 1000))
    
    # 对网格点进行预测概率
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.predict_proba(grid_points_scaled)[:, 1]
    Z = Z.reshape(xx.shape)
    
    # 绘制概率分布和散点图
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(xx, yy, Z, 20, cmap=plt.cm.RdBu_r, alpha=0.8)
    plt.colorbar(contour, label='违约概率')
    
    # 绘制决策边界
    plt.contour(xx, yy, Z, [0.5], colors='k', linewidths=2)
    
    # 绘制散点图
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', marker='o', label='违约', alpha=0.6)
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', marker='x', label='非违约', alpha=0.6)
    
    plt.xlabel('信用卡余额')
    plt.ylabel('收入')
    plt.title('信用卡违约概率分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return plt

# 绘制概率分布
plot_probability_distribution(X, y, model, scaler)
plt.savefig(r'e:\GitHub\DeepLearning\深度学习\第3章线性模型\credit_default_probability.png', dpi=300)
plt.show()

# 输出模型系数和截距
print("\n逻辑回归模型系数:")
print(f"余额系数: {model.coef_[0][0]:.6f}")
print(f"收入系数: {model.coef_[0][1]:.6f}")
print(f"截距: {model.intercept_[0]:.6f}")

# 计算原始尺度的系数
original_coef = model.coef_[0] / scaler.scale_
original_intercept = model.intercept_[0] - np.sum(model.coef_[0] * scaler.mean_ / scaler.scale_)

print("\n原始尺度的逻辑回归方程:")
print(f"log(p/(1-p)) = {original_coef[0]:.6f} × 余额 + {original_coef[1]:.6f} × 收入 + {original_intercept:.6f}")

# 计算违约概率为0.5时的决策边界方程
print("\n违约概率为0.5的决策边界方程:")
print(f"{original_coef[0]:.6f} × 余额 + {original_coef[1]:.6f} × 收入 + {original_intercept:.6f} = 0")