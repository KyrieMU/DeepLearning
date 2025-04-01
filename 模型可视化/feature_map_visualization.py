import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练模型
model = models.vgg16(pretrained=True)
model.eval()

# 准备输入图像
img_path = 'szu1.png'
img = Image.open(img_path)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

# 提取特征图
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

# 注册钩子
model.features[0].register_forward_hook(get_activation('conv1'))
model.features[5].register_forward_hook(get_activation('conv2'))
model.features[10].register_forward_hook(get_activation('conv3'))

# 前向传播
_ = model(input_tensor)

# 可视化特征图
def visualize_feature_maps(activation, layer_name):
    feature_maps = activation[layer_name].squeeze(0)
    num_features = feature_maps.size(0)
    
    # 选择前16个特征图显示
    num_to_show = min(16, num_features)
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(num_to_show):
        row, col = i // 4, i % 4
        axes[row][col].imshow(feature_maps[i].cpu().numpy(), cmap='viridis')
        axes[row][col].set_title(f'Filter {i}')
        axes[row][col].axis('off')
    
    plt.suptitle(f'Feature Maps of {layer_name}')
    plt.savefig(f'e:/GitHub/DeepLearning/{layer_name}_feature_maps.png')
    plt.close()

# 可视化不同层的特征图
visualize_feature_maps(activation, 'conv1')
visualize_feature_maps(activation, 'conv2')
visualize_feature_maps(activation, 'conv3')
