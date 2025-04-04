import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from PIL import Image, ImageDraw
import os
import threading
import time

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 在文件开头添加 MNISTNet 类定义
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

# 数字到中文的映射
DIGIT_TO_CHINESE = {
    0: "零", 1: "一", 2: "二", 3: "三", 
    4: "四", 5: "五", 6: "六", 7: "七", 
    8: "八", 9: "九"
}

class MultiDigitRecognizer:
    def __init__(self, root):
        self.root = root
        self.root.title("多位数手写数字识别")
        self.root.geometry("1000x600")
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTNet().to(self.device)
        
        if os.path.exists('nmist/mnist_model.pth'):
            self.model.load_state_dict(torch.load('nmist/mnist_model.pth', map_location=self.device))
            self.model.eval()
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 创建画布
        self.canvas_frame = tk.Frame(root, width=600, height=300, bg="black")
        self.canvas_frame.pack(side=tk.TOP, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=600, height=300, bg="black")
        self.canvas.pack()
        
        # 创建PIL图像用于绘制
        self.image = Image.new("L", (600, 300), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # 控制面板
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.LEFT, padx=10)
        
        # 清除按钮
        self.clear_button = tk.Button(self.control_frame, text="清除画布", command=self.clear_canvas)
        self.clear_button.pack(pady=5)
        
        # 识别按钮
        self.recognize_button = tk.Button(self.control_frame, text="识别数字", command=self.recognize_digits)
        self.recognize_button.pack(pady=5)
        
        # 结果显示区域
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 结果显示标签
        self.result_label = tk.Label(self.result_frame, text="请书写数字后点击识别", font=("SimHei", 24))
        self.result_label.pack(pady=20)
        
        # 数字分割线显示
        self.divider_label = tk.Label(self.result_frame, text="数字分割线将显示在这里", font=("SimHei", 12))
        self.divider_label.pack()
        
        # 绘图变量
        self.last_x = None
        self.last_y = None
        self.brush_size = 15
    
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_draw(self, event):
        self.last_x = None
        self.last_y = None
    
    def draw_line(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                  fill="white", width=self.brush_size, 
                                  capstyle=tk.ROUND, joinstyle=tk.ROUND)
            self.draw.line([self.last_x, self.last_y, x, y], 
                          fill=255, width=self.brush_size, joint="curve")
            self.last_x = x
            self.last_y = y
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (600, 300), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="请书写数字后点击识别")
        self.divider_label.config(text="数字分割线将显示在这里")
    
    def recognize_digits(self):
        # 垂直投影分割数字
        img_array = np.array(self.image)
        vertical_projection = np.sum(img_array > 128, axis=0)
        
        # 找到分割点
        dividers = []
        in_digit = False
        for i in range(1, len(vertical_projection)):
            if vertical_projection[i-1] == 0 and vertical_projection[i] > 0:
                dividers.append(i)
            elif vertical_projection[i-1] > 0 and vertical_projection[i] == 0:
                dividers.append(i)
        
        # 确保有偶数个分割点
        if len(dividers) % 2 != 0:
            dividers = dividers[:-1]
        
        # 显示分割线
        self.canvas.delete("divider")
        for i, pos in enumerate(dividers):
            color = "red" if i % 2 == 0 else "green"
            self.canvas.create_line(pos, 0, pos, 300, fill=color, tags="divider", width=2)
        
        self.divider_label.config(text=f"检测到 {len(dividers)//2} 个数字")
        
        # 识别每个数字
        digits = []
        for i in range(0, len(dividers), 2):
            left, right = dividers[i], dividers[i+1]
            digit_img = self.image.crop((left, 0, right, 300))
            
            # 预处理
            try:
                digit_img = digit_img.resize((28, 28), Image.LANCZOS)
            except AttributeError:
                digit_img = digit_img.resize((28, 28), Image.Resampling.LANCZOS)
            
            img_tensor = self.transform(digit_img).unsqueeze(0).to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                predicted = torch.argmax(probabilities).item()
                digits.append(predicted)
        
        # 显示结果
        arabic_result = "".join(map(str, digits))
        chinese_result = "".join([DIGIT_TO_CHINESE[d] for d in digits])
        self.result_label.config(text=f"识别结果: {arabic_result}\n中文: {chinese_result}")

if __name__ == "__main__":
    root = tk.Tk()
    app = MultiDigitRecognizer(root)
    root.mainloop()

