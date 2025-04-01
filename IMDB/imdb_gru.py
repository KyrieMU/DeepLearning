import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import re
import os
from tqdm import tqdm
# 添加可视化所需的库
import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib as mpl

# 设置中文显示
def set_chinese_font():
    try:
        from fontTools.ttLib import TTFont
        
        # 检查SimHei字体是否存在
        font_path = None
        for font in font_manager.findSystemFonts():
            if 'simhei' in font.lower():
                font_path = font
                break
                
        if font_path:
            # 验证字体有效性
            try:
                TTFont(font_path)
                plt.rcParams['font.sans-serif'] = ['SimHei']
                print(f"使用中文字体: {font_path}")
            except:
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
                print("SimHei字体损坏，使用微软雅黑替代")
        else:
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
            print("未找到中文字体，使用英文字体")
            
    except ImportError:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        print("fonttools未安装，使用默认字体设置")
    
    plt.rcParams['axes.unicode_minus'] = False
    
    # 检查是否有中文字体，如果没有则尝试其他字体
    fonts = font_manager.findSystemFonts()
    chinese_fonts = [f for f in fonts if 'simhei' in f.lower() or 'msyh' in f.lower() or 'simsun' in f.lower()]
    
    if chinese_fonts:
        # 使用找到的第一个中文字体
        font_path = chinese_fonts[0]
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        print(f"使用中文字体: {font_path}")
    else:
        print("警告: 未找到中文字体，图表中文可能显示为方块")

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 数据预处理
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 移除特殊字符
    text = re.sub(r'[^\w\s]', '', text)
    # 移除数字
    text = re.sub(r'\d+', '', text)
    # 移除多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 加载数据
def load_data(reviews_path, labels_path):
    with open(reviews_path, 'r', encoding='utf-8') as f:
        reviews = f.readlines()
    
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    
    # 预处理评论
    reviews = [preprocess_text(review.strip()) for review in reviews]
    # 处理标签
    labels = [1 if label.strip() == 'positive' else 0 for label in labels]
    
    return reviews, labels

# 构建词汇表
def build_vocab(texts, max_vocab_size=10000):
    all_words = []
    for text in texts:
        all_words.extend(text.split())
    
    # 计算词频
    word_counts = Counter(all_words)
    # 选择最常见的词构建词汇表
    vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(max_vocab_size-2)]
    # 创建词到索引的映射
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    
    return vocab, word_to_idx

# 将文本转换为索引序列
def text_to_indices(texts, word_to_idx, max_length=200):
    indices = []
    for text in texts:
        words = text.split()
        # 截断或填充到固定长度
        if len(words) > max_length:
            words = words[:max_length]
        else:
            words = words + ['<PAD>'] * (max_length - len(words))
        # 将词转换为索引
        indices.append([word_to_idx.get(word, word_to_idx['<UNK>']) for word in words])
    
    return np.array(indices)

# 自定义数据集类
class IMDBDataset(Dataset):
    def __init__(self, reviews_indices, labels):
        self.reviews_indices = torch.LongTensor(reviews_indices)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.reviews_indices[idx], self.labels[idx]

# GRU模型定义
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, bidirectional=True):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, 
                          batch_first=True, dropout=dropout if n_layers > 1 else 0,
                          bidirectional=bidirectional)
        # 如果是双向GRU，输出维度需要乘以2
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.bidirectional = bidirectional
        
    def forward(self, text):
        # text: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(text))
        # embedded: [batch_size, seq_len, embedding_dim]
        
        output, hidden = self.gru(embedded)
        # output: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden: [n_layers * num_directions, batch_size, hidden_dim]
        
        if self.bidirectional:
            # 连接最后一层的前向和后向隐藏状态
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            # 使用最后一个时间步的隐藏状态
            hidden = hidden[-1, :, :]
        # hidden: [batch_size, hidden_dim * num_directions]
        
        return self.fc(hidden)

# 训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        reviews, labels = batch
        reviews, labels = reviews.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(reviews).squeeze(1)
        loss = criterion(predictions, labels)
        
        # 计算准确率
        predicted_labels = torch.round(torch.sigmoid(predictions))
        correct = (predicted_labels == labels).float().sum()
        acc = correct / len(labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# 评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            reviews, labels = batch
            reviews, labels = reviews.to(device), labels.to(device)
            
            predictions = model(reviews).squeeze(1)
            loss = criterion(predictions, labels)
            
            # 计算准确率
            predicted_labels = torch.round(torch.sigmoid(predictions))
            correct = (predicted_labels == labels).float().sum()
            acc = correct / len(labels)
            
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# 绘制训练过程图表
def plot_training_process(train_losses, train_accs, val_losses, val_accs, save_path):
    # 设置中文字体
    set_chinese_font()
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='训练损失')
    ax1.plot(epochs, val_losses, 'r-', label='验证损失')
    ax1.set_title('训练和验证损失')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(epochs, train_accs, 'b-', label='训练准确率')
    ax2.plot(epochs, val_accs, 'r-', label='验证准确率')
    ax2.set_title('训练和验证准确率')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('准确率')
    ax2.legend()
    ax2.grid(True)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"训练过程图表已保存至: {save_path}")

def main():
    # 文件路径
    reviews_path = 'e:\\GitHub\\DeepLearning\\IMDB\\reviews.txt'
    labels_path = 'e:\\GitHub\\DeepLearning\\IMDB\\labels.txt'
    
    # 检查文件是否存在
    if not os.path.exists(reviews_path) or not os.path.exists(labels_path):
        print(f"Error: 文件不存在，请确认路径: {reviews_path} 和 {labels_path}")
        return
    
    # 加载数据
    print("加载数据...")
    reviews, labels = load_data(reviews_path, labels_path)
    
    # 划分训练集和验证集
    train_reviews, val_reviews, train_labels, val_labels = train_test_split(
        reviews, labels, test_size=0.2, random_state=42
    )
    
    # 构建词汇表
    print("构建词汇表...")
    vocab, word_to_idx = build_vocab(train_reviews)
    
    # 尝试加载预训练词向量
    try:
        import gensim.downloader as api
        print("加载预训练词向量...")
        word_vectors = api.load("glove-wiki-gigaword-100")
        
        # 创建嵌入矩阵
        embedding_matrix = np.zeros((len(vocab), 100))
        for i, word in enumerate(vocab):
            if word in word_vectors:
                embedding_matrix[i] = word_vectors[word]
        
        print(f"成功加载预训练词向量，覆盖率: {np.count_nonzero(np.sum(embedding_matrix, axis=1)) / len(vocab):.2f}")
        use_pretrained = True
    except Exception as e:
        print(f"加载预训练词向量失败: {e}")
        use_pretrained = False
    
    # 将文本转换为索引
    print("将文本转换为索引...")
    train_indices = text_to_indices(train_reviews, word_to_idx)
    val_indices = text_to_indices(val_reviews, word_to_idx)
    
    # 创建数据集和数据加载器
    print("创建数据加载器...")
    train_dataset = IMDBDataset(train_indices, train_labels)
    val_dataset = IMDBDataset(val_indices, val_labels)
    
    batch_size = 64
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    
    # 模型参数
    vocab_size = len(vocab)
    embedding_dim = 200  # 从100增加到200
    hidden_dim = 512     # 从256增加到512
    output_dim = 1
    n_layers = 3         # 从2增加到3
    dropout = 0.5
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化模型
    model = GRUModel(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, bidirectional=True)
    
    # 如果有预训练词向量，加载到模型中
    if use_pretrained:
        model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        # 可选：冻结嵌入层
        # model.embedding.weight.requires_grad = False
        print("已加载预训练词向量到模型")
    
    model = model.to(device)
    
    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # 添加权重衰减
    criterion = nn.BCEWithLogitsLoss()
    
    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练模型
    n_epochs = 5  # 从5增加到15
    best_val_loss = float('inf')
    patience = 3  # 早停耐心值
    counter = 0   # 早停计数器
    
    # 用于记录训练过程的列表
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    print(f"开始训练，共{n_epochs}个epochs...")
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        
        train_loss, train_acc = train(model, train_dataloader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_dataloader, criterion, device)
        
        # 记录训练过程
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f"Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}")
        print(f"Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}")
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'e:\\GitHub\\DeepLearning\\IMDB\\best_gru_model.pt')
            print("保存最佳模型")
            counter = 0  # 重置计数器
        else:
            counter += 1  # 增加计数器
            
            # 早停检查
            if counter >= patience:
                print(f"早停: 验证损失在{patience}个epoch内没有改善")
                break
    
    print("训练完成！")
    
    # 绘制并保存训练过程图表
    plot_path = 'e:\\GitHub\\DeepLearning\\IMDB\\training_process.png'
    plot_training_process(train_losses, train_accs, val_losses, val_accs, plot_path)
    
    # 可视化词嵌入（使用t-SNE降维）
    try:
        from sklearn.manifold import TSNE
        import pandas as pd
        
        # 获取嵌入权重
        embeddings = model.embedding.weight.cpu().detach().numpy()
        
        # 选择前1000个词进行可视化（全部可视化会很慢）
        num_visualize = min(1000, len(vocab))
        
        print(f"正在使用t-SNE对词嵌入进行降维，这可能需要一些时间...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings[:num_visualize])
        
        # 创建DataFrame以便绘图
        df = pd.DataFrame(embeddings_2d, columns=['x', 'y'])
        df['word'] = vocab[:num_visualize]
        
        # 设置中文字体
        set_chinese_font()
        
        # 绘制词嵌入散点图
        plt.figure(figsize=(12, 10))
        plt.scatter(df['x'], df['y'], alpha=0.5)
        
        # 为一些常见词添加标签
        common_words = ['good', 'bad', 'excellent', 'terrible', 'love', 'hate', 'movie', 'film']
        for word in common_words:
            if word in df['word'].values:
                row = df[df['word'] == word]
                plt.annotate(word, (row['x'].values[0], row['y'].values[0]), fontsize=12)
        
        plt.title('词嵌入t-SNE可视化')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.tight_layout()
        
        # 保存图表
        embedding_plot_path = 'e:\\GitHub\\DeepLearning\\IMDB\\word_embeddings.png'
        plt.savefig(embedding_plot_path)
        plt.show()
        print(f"词嵌入可视化已保存至: {embedding_plot_path}")
    except Exception as e:
        print(f"词嵌入可视化失败: {e}")

if __name__ == "__main__":
    main()