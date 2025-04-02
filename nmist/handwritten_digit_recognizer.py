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

# 设置环境变量以解决OpenMP错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# 定义与训练时相同的神经网络模型
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

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别")
        self.root.geometry("800x600")
        
        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MNISTNet().to(self.device)
        
        # 检查模型文件是否存在
        if os.path.exists('nmist/mnist_model.pth'):
            self.model.load_state_dict(torch.load('nmist/mnist_model.pth', map_location=self.device))
            self.model.eval()
            print("模型已加载")
        else:
            print("模型文件不存在，请先运行 train_mnist.py 训练模型")
        
        # 数据预处理
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 创建画布
        self.canvas_frame = tk.Frame(root, width=280, height=280, bg="black")
        self.canvas_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=280, height=280, bg="black")
        self.canvas.pack()
        
        # 创建PIL图像用于绘制
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # 创建按钮框架
        self.button_frame = tk.Frame(root)
        self.button_frame.pack(side=tk.TOP, pady=10)
        
        # 创建按钮
        self.clear_button = tk.Button(self.button_frame, text="清除", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        
        # 添加笔刷大小滑块
        self.brush_size_label = tk.Label(self.button_frame, text="笔刷大小:", font=("SimHei", 10))
        self.brush_size_label.pack(side=tk.LEFT, padx=5)
        self.brush_size = tk.IntVar(value=8)  # 默认笔刷大小
        self.brush_slider = tk.Scale(self.button_frame, from_=1, to=20, orient=tk.HORIZONTAL, 
                                    variable=self.brush_size, length=100)
        self.brush_slider.pack(side=tk.LEFT, padx=5)
        
        # 创建结果显示区域
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图形用于显示预测概率
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.result_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化绘图
        self.ax.set_title("预测概率")
        self.ax.set_xlabel("数字")
        self.ax.set_ylabel("概率")
        self.ax.set_xticks(range(10))
        self.ax.set_ylim(0, 1)
        self.bars = self.ax.bar(range(10), [0] * 10)
        self.canvas_plot.draw()
        
        # 显示预测结果的标签
        self.result_label = tk.Label(self.result_frame, text="请在左侧画布上绘制数字", font=("SimHei", 16))
        self.result_label.pack(pady=10)
        
        # 记录上一个点的位置
        self.last_x = None
        self.last_y = None
        
        # 用于控制实时预测的变量
        self.is_drawing = False
        self.prediction_thread = None
        self.stop_prediction = False
    
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
        self.is_drawing = True
        
        # 绘制起始点
        brush_size = self.brush_size.get()
        # 在Canvas上绘制圆形
        self.canvas.create_oval(event.x-brush_size/2, event.y-brush_size/2, 
                               event.x+brush_size/2, event.y+brush_size/2, 
                               fill="white", outline="white")
        # 在PIL图像上绘制圆形
        self.draw.ellipse([event.x-brush_size/2, event.y-brush_size/2, 
                          event.x+brush_size/2, event.y+brush_size/2], fill=255)
        
        # 启动实时预测线程
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.stop_prediction = False
            self.prediction_thread = threading.Thread(target=self.real_time_predict)
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
    
    def stop_draw(self, event):
        self.is_drawing = False
        self.last_x = None
        self.last_y = None
        # 在停止绘制时进行一次最终预测
        self.predict()
    
    def draw_line(self, event):
        if self.last_x and self.last_y:
            x, y = event.x, event.y
            # 获取当前笔刷大小
            brush_size = self.brush_size.get()
            
            # 在Canvas上绘制圆形笔刷
            # 1. 绘制从上一点到当前点的线条
            self.canvas.create_line(self.last_x, self.last_y, x, y, 
                                   fill="white", width=brush_size, 
                                   capstyle=tk.ROUND, joinstyle=tk.ROUND)
            # 2. 在当前点绘制一个圆形，确保连续性
            self.canvas.create_oval(x-brush_size/2, y-brush_size/2, 
                                   x+brush_size/2, y+brush_size/2, 
                                   fill="white", outline="white")
            
            # 在PIL图像上绘制
            # 1. 绘制线条
            self.draw.line([self.last_x, self.last_y, x, y], 
                          fill=255, width=brush_size, joint="curve")
            # 2. 绘制圆形端点
            self.draw.ellipse([x-brush_size/2, y-brush_size/2, 
                              x+brush_size/2, y+brush_size/2], fill=255)
            
            self.last_x = x
            self.last_y = y
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="请在左侧画布上绘制数字")
        # 重置概率条形图
        for bar in self.bars:
            bar.set_height(0)
        self.canvas_plot.draw()
    
    def real_time_predict(self):
        """实时预测线程函数"""
        while not self.stop_prediction:
            try:
                if self.is_drawing:
                    self.predict()
            except Exception as e:
                print(f"预测出错: {e}")
            time.sleep(0.1)  # 每100毫秒预测一次
    
    def predict(self):
        # 调整图像大小并进行预处理
        try:
            img = self.image.resize((28, 28), Image.LANCZOS)
        except AttributeError:
            # 对于较新版本的PIL
            img = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 进行预测
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted = torch.argmax(probabilities).item()
        
        # 更新概率条形图
        for i, bar in enumerate(self.bars):
            bar.set_height(probabilities[i].item())
        self.canvas_plot.draw()
        
        # 更新预测结果标签
        self.result_label.config(text=f"预测结果: {predicted}")

if __name__ == "__main__":
    print("启动手写数字识别程序...")
    root = tk.Tk()
    try:
        app = DrawingApp(root)
        print("程序初始化完成，开始主循环")
        root.mainloop()
    except Exception as e:
        print(f"程序出错: {e}")