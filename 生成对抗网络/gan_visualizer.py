import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
import glob

##设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def create_gan_animation():
    """创建GAN训练过程的动画"""
    # 获取所有生成的图像
    image_files = sorted(glob.glob('gan_output/epoch_*.png'), 
                         key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    if not image_files:
        print("未找到生成的图像文件，请先运行gan_demo.py")
        return
    
    # 读取图像
    images = [np.array(Image.open(file)) for file in image_files]
    
    # 创建动画
    fig = plt.figure(figsize=(10, 5))
    plt.axis('off')
    
    ims = [[plt.imshow(img, animated=True)] for img in images]
    
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    
    # 保存动画
    ani.save('gan_output/gan_training_process.gif', writer='pillow', fps=2)
    
    print("GAN训练过程动画已保存为 'gan_output/gan_training_process.gif'")

def plot_latent_space_exploration():
    """探索2D潜在空间，适用于简化版GAN模型"""
    import torch
    try:
        # 尝试从gan_demo导入必要的组件
        from gan_demo import Generator, z_dim, device
        
        # 创建生成器实例
        generator = Generator().to(device)
        
        # 如果存在模型文件则加载，否则使用未训练的模型
        try:
            generator.load_state_dict(torch.load('gan_output/generator.pth'))
            generator.eval()
            print("已加载训练好的生成器模型")
        except:
            print("未找到生成器模型文件，使用未训练的模型进行可视化")
        
        # 创建潜在空间网格
        grid_size = 20
        z_range = 3.0
        
        # 创建均匀分布的网格点
        z1 = np.linspace(-z_range, z_range, grid_size)
        z2 = np.linspace(-z_range, z_range, grid_size)
        z1_grid, z2_grid = np.meshgrid(z1, z2)
        
        # 将网格点转换为潜在向量
        z_grid = np.column_stack([z1_grid.flatten(), z2_grid.flatten()])
        z_tensor = torch.FloatTensor(z_grid).to(device)
        
        # 生成数据点
        with torch.no_grad():
            gen_data = generator(z_tensor).detach().cpu().numpy()
        
        # 绘制潜在空间映射
        plt.figure(figsize=(12, 10))
        
        # 绘制生成的数据点
        plt.scatter(gen_data[:, 0], gen_data[:, 1], c='blue', alpha=0.6, s=10)
        
        # 绘制潜在空间网格
        for i in range(grid_size):
            # 绘制垂直线
            start_idx = i * grid_size
            end_idx = (i + 1) * grid_size - 1
            plt.plot(gen_data[start_idx:end_idx+1, 0], gen_data[start_idx:end_idx+1, 1], 'r-', alpha=0.3)
            
            # 绘制水平线
            for j in range(grid_size):
                start_idx = j * grid_size + i
                plt.plot([gen_data[start_idx, 0]], [gen_data[start_idx, 1]], 'k.', alpha=0.5)
        
        plt.title('潜在空间映射到数据空间')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.grid(True, alpha=0.3)
        plt.savefig('gan_output/latent_space_mapping.png')
        plt.close()
        
        print("潜在空间映射图像已保存为 'gan_output/latent_space_mapping.png'")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保gan_demo.py中包含必要的类和变量")

if __name__ == "__main__":
    # 确保输出目录存在
    os.makedirs('gan_output', exist_ok=True)
    
    # 创建GAN训练过程动画
    create_gan_animation()
    
    # 绘制潜在空间映射
    plot_latent_space_exploration()