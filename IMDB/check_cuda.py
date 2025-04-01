import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"CUDA设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"CUDA设备{i}名称: {torch.cuda.get_device_name(i)}")
    print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 创建一个小张量并移动到GPU，测试是否正常工作
    x = torch.tensor([1.0, 2.0, 3.0])
    x = x.to('cuda')
    print(f"张量设备: {x.device}")
    print("CUDA测试成功!")
else:
    print("CUDA不可用，请检查您的PyTorch安装和NVIDIA驱动")

import torch
print(torch.cuda.is_available())  # 输出True表示支持CUDA