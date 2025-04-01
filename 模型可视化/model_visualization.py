import torch
import torch.nn as nn
from torchviz import make_dot

# 定义一个简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(16 * 112 * 112, 10)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 112 * 112)
        x = self.fc(x)
        return x

# 创建模型和输入
model = SimpleCNN()
x = torch.randn(1, 3, 224, 224)
y = model(x)

# 可视化模型
dot = make_dot(y, params=dict(model.named_parameters()))
dot.render("model_architecture", format="png")