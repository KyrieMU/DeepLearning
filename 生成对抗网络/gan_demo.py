import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

##设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 超参数 - 调整超参数
batch_size = 128  # 增大批次大小
z_dim = 2  # 保持潜在空间维度为2
lr = 0.001  # 降低学习率，提高稳定性
beta1 = 0.5
epochs = 400  # 增加训练轮数
sample_interval = 20

# 创建输出目录
os.makedirs('gan_output', exist_ok=True)

# 创建简单的2D高斯分布数据集 - 保持不变
def get_data_batch(batch_size):
    centers = [(3, 3), (3, -3), (-3, 3), (-3, -3)]  # 增大中心点间距
    center_idx = np.random.randint(len(centers), size=batch_size)
    # 减小噪声尺度
    point = np.random.normal(loc=[x, y], scale=0.1, size=2)

class Generator(nn.Module):
    def forward(self, z):
        return self.model(z) * 4  # 同步增大输出范围

def save_sample_data(epoch):
    plt.xlim(-5, 5)  # 扩大坐标范围
    plt.ylim(-5, 5)
    # 生成数据点
    z = torch.randn(500, z_dim).to(device)
    with torch.no_grad():
        gen_data = generator(z).detach().cpu().numpy()
    
    # 获取真实数据分布
    real_data = np.vstack([get_data_batch(100).numpy() for _ in range(5)])
    
    # 绘制数据分布
    plt.figure(figsize=(10, 5))
    
    # 绘制真实数据
    plt.subplot(1, 2, 1)
    plt.scatter(real_data[:, 0], real_data[:, 1], c='blue', alpha=0.5, label='真实数据')
    plt.title('真实数据分布')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    
    # 绘制生成数据
    plt.subplot(1, 2, 2)
    plt.scatter(gen_data[:, 0], gen_data[:, 1], c='red', alpha=0.5, label='生成数据')
    plt.title(f'生成数据分布 (Epoch {epoch})')
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'gan_output/epoch_{epoch}.png')
    plt.close()

# 确保在调用前正确定义训练函数
def train_gan():
    # 记录损失
    g_losses = []
    d_losses = []
    
    for epoch in range(epochs):
        # 获取真实数据批次
        real_data = get_data_batch(batch_size).to(device)
        
        # 创建标签 - 添加标签平滑化
        valid = torch.ones(batch_size, 1).to(device) * 0.9  # 标签平滑化
        fake = torch.zeros(batch_size, 1).to(device) + 0.1  # 标签平滑化
        
        # -----------------
        # 训练判别器 - 先训练判别器
        # -----------------
        optimizer_D.zero_grad()
        
        # 生成随机噪声
        z = torch.randn(batch_size, z_dim).to(device)
        
        # 生成假数据
        gen_data = generator(z)
        
        # 计算判别器对真实数据和生成数据的损失
        real_loss = adversarial_loss(discriminator(real_data), valid)
        fake_loss = adversarial_loss(discriminator(gen_data.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        
        # 反向传播和优化
        d_loss.backward()
        optimizer_D.step()
        
        # -----------------
        # 训练生成器 - 每训练n次判别器，训练一次生成器
        # -----------------
        if epoch % 2 == 0:  # 每2次迭代训练一次生成器
            optimizer_G.zero_grad()
            
            # 重新生成随机噪声
            z = torch.randn(batch_size, z_dim).to(device)
            
            # 生成假数据
            gen_data = generator(z)
            
            # 计算生成器损失
            g_loss = adversarial_loss(discriminator(gen_data), valid)
            
            # 反向传播和优化
            g_loss.backward()
            optimizer_G.step()
        
        # 打印训练信息
        if epoch % 10 == 0:
            print(
                f"[Epoch {epoch}/{epochs}] "
                f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
            )
        
        # 记录每个epoch的损失
        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        
        # 保存生成的数据样本
        if epoch % sample_interval == 0 or epoch == epochs - 1:
            save_sample_data(epoch)
    
    # 保存损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='生成器损失')
    plt.plot(d_losses, label='判别器损失')
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.legend()
    plt.savefig('gan_output/loss_curve.png')
    plt.close()
    
    # 保存模型
    torch.save(generator.state_dict(), 'gan_output/generator.pth')
    torch.save(discriminator.state_dict(), 'gan_output/discriminator.pth')

# 运行训练
if __name__ == "__main__":
    train_gan()
    print("训练完成！生成的图像保存在 'gan_output' 文件夹中。")