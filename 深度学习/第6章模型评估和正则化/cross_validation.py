import numpy as np
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 生成示例数据
np.random.seed(42)
X = np.random.rand(100, 2)
y = (X[:, 0] + X[:, 1] > 1).astype(int)

# 设置k折交叉验证
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# 存储每折的准确率
fold_accuracies = []

plt.figure(figsize=(15, 10))
for i, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # 训练模型
    model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测并计算准确率
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    fold_accuracies.append(accuracy)
    
    # 可视化每折的训练集和测试集
    plt.subplot(2, 3, i+1)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.6, marker='o', label='训练集')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', alpha=1.0, marker='x', s=100, label='测试集')
    plt.title(f'第{i+1}折 (准确率: {accuracy:.2f})')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.legend()

# 显示平均准确率
plt.subplot(2, 3, 6)
plt.bar(range(1, k+1), fold_accuracies)
plt.axhline(y=np.mean(fold_accuracies), color='r', linestyle='-', label=f'平均: {np.mean(fold_accuracies):.2f}')
plt.xlabel('折数')
plt.ylabel('准确率')
plt.title('各折准确率对比')
plt.legend()

plt.tight_layout()
plt.savefig('cross_validation.png', dpi=300)
plt.show()