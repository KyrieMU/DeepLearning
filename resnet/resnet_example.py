
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义基本的残差块
class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入和输出维度不同，需要使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(identity)  # 残差连接
        out = self.relu(out)
        
        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

# 创建ResNet-18模型
def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# 数据预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车')

# 设置设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 初始化模型、损失函数和优化器
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# 训练函数
def train(epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'训练轮次: {epoch} [{batch_idx}/{len(trainloader)}] | 损失: {train_loss/(batch_idx+1):.3f} | 准确率: {100.*correct/total:.3f}%')
    
    return train_loss/len(trainloader), 100.*correct/total

# 测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f'测试轮次: {epoch} | 损失: {test_loss/len(testloader):.3f} | 准确率: {100.*correct/total:.3f}%')
    
    return test_loss/len(testloader), 100.*correct/total

# 可视化一些图像
def imshow(img):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.figure(figsize=(10, 4))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()

# 可视化ResNet结构
def visualize_resnet():
    plt.figure(figsize=(12, 8))
    plt.title('ResNet残差块结构示意图', fontsize=16)
    
    # 绘制常规网络结构
    plt.subplot(1, 2, 1)
    plt.title('常规网络层', fontsize=14)
    plt.plot([0, 1], [0, 1], 'b-', linewidth=2)
    plt.plot([1, 2], [1, 2], 'b-', linewidth=2)
    plt.plot([2, 3], [2, 3], 'b-', linewidth=2)
    plt.text(0, 0, '输入', fontsize=12)
    plt.text(1, 1, '层1', fontsize=12)
    plt.text(2, 2, '层2', fontsize=12)
    plt.text(3, 3, '输出', fontsize=12)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 3.5)
    plt.axis('off')
    
    # 绘制残差网络结构
    plt.subplot(1, 2, 2)
    plt.title('残差网络层', fontsize=14)
    plt.plot([0, 1], [0, 1], 'b-', linewidth=2)
    plt.plot([1, 2], [1, 2], 'b-', linewidth=2)
    plt.plot([0, 3], [0, 3], 'r-', linewidth=2)  # 残差连接
    plt.plot([2, 3], [2, 3], 'b-', linewidth=2)
    plt.text(0, 0, '输入', fontsize=12)
    plt.text(1, 1, '层1', fontsize=12)
    plt.text(2, 2, '层2', fontsize=12)
    plt.text(3, 3, '输出 = 层2输出 + 输入', fontsize=12)
    plt.text(1.5, 0.5, '残差连接', color='red', fontsize=12)
    plt.xlim(-0.5, 3.5)
    plt.ylim(-0.5, 3.5)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 可视化训练过程
def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(test_losses, label='测试损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.title('训练和测试损失')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='训练准确率')
    plt.plot(test_accs, label='测试准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.title('训练和测试准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 可视化ResNet结构
    visualize_resnet()
    
    # 显示一些训练图像
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    imshow(torchvision.utils.make_grid(images[:5]))
    print(' '.join(f'{classes[labels[j]]}' for j in range(5)))
    
    # 训练模型
    epochs = 5  # 为了演示，只训练5轮
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(epochs):
        train_loss, train_acc = train(epoch)
        test_loss, test_acc = test(epoch)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        scheduler.step()
    
    # 可视化训练历史
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    
    # 在测试集上进行预测并可视化一些结果
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    # 将图像移回CPU以便可视化
    images = images.cpu()
    
    # 显示预测结果
    imshow(torchvision.utils.make_grid(images[:5]))
    print('真实标签: ', ' '.join(f'{classes[labels[j]]}' for j in range(5)))
    print('预测标签: ', ' '.join(f'{classes[predicted[j]]}' for j in range(5)))

if __name__ == '__main__':
    main()