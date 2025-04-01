import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 加载预训练的ResNet模型
model = models.resnet50(pretrained=True)
model.eval()

# 图像预处理
img_path = 'szu1.png'
img = Image.open(img_path)
original_img = np.array(img)

# 预处理图像
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = transform(img).unsqueeze(0)

# 获取最后一个卷积层的特征图和预测类别
feature_blobs = []
def hook_feature(module, input, output):
    feature_blobs.append(output.data.cpu().numpy())

model._modules.get('layer4').register_forward_hook(hook_feature)
logits = model(input_tensor)
probs = torch.nn.functional.softmax(logits, dim=1)
predicted_class = torch.argmax(probs, dim=1).item()

# 获取预测类别对应的权重
params = list(model.parameters())
weight_softmax = params[-2].data.cpu().numpy()
weight_softmax_class = weight_softmax[predicted_class, :]

# 生成CAM热力图
cam = np.zeros(feature_blobs[0].shape[2:], dtype=np.float32)
for i, w in enumerate(weight_softmax_class):
    cam += w * feature_blobs[0][0, i, :, :]

# 归一化并上采样到原始图像大小
cam = cv2.resize(cam, (224, 224))
cam = np.maximum(cam, 0)
cam = (cam - cam.min()) / (cam.max() - cam.min())
cam = np.uint8(255 * cam)

# 应用热力图到原始图像
heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
original_img_resized = cv2.resize(original_img, (224, 224))
result = heatmap * 0.3 + original_img_resized * 0.7
result = result.astype(np.uint8)

# 保存结果
cv2.imwrite('e:/GitHub/DeepLearning/cam_heatmap.jpg', result)

# 显示结果
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_img_resized)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cam, cmap='jet')
plt.title('Class Activation Map')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('CAM Overlay')
plt.axis('off')

plt.savefig('e:/GitHub/DeepLearning/cam_visualization_result.png')
plt.close()