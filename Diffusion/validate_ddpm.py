import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from validation import extract, Training, Sampling, TimeEmbedding

# 定义一个简单的UNet骨干网络（用于预测噪声）
class SimpleUnet(nn.Module):
    def __init__(self, image_channels=1, hidden_dims=[32, 64, 128], embedding_dim=128):
        super().__init__()
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(embedding_dim)
        
        # 下采样部分 - 使用stride代替池化
        self.downs = nn.ModuleList([])
        in_channels = image_channels
        
        # 第一层不做下采样
        self.downs.append(nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims[0], 3, padding=1),
            nn.GroupNorm(8, hidden_dims[0]),
            nn.GELU()
        ))
        
        # 后续层做下采样
        for i in range(len(hidden_dims) - 1):
            self.downs.append(nn.Sequential(
                nn.Conv2d(hidden_dims[i], hidden_dims[i+1], 4, 2, 1),  # stride=2下采样
                nn.GroupNorm(8, hidden_dims[i+1]),
                nn.GELU()
            ))
        
        # 中间层
        mid_dim = hidden_dims[-1]
        self.mid = nn.Sequential(
            nn.Conv2d(mid_dim, mid_dim, 3, padding=1),
            nn.GroupNorm(8, mid_dim),
            nn.GELU()
        )
        
        # 上采样部分
        self.ups = nn.ModuleList([])
        for i in range(len(hidden_dims) - 1, 0, -1):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i-1], 4, 2, 1),  # stride=2上采样
                nn.GroupNorm(8, hidden_dims[i-1]),
                nn.GELU()
            ))
        
        # 最终输出层
        self.final = nn.Conv2d(hidden_dims[0], image_channels, 3, padding=1)
        
    def forward(self, x, t):
        # 时间嵌入
        t_emb = self.time_embedding(t)  # [B, embedding_dim]
        
        # 下采样并存储中间结果
        h = []
        for down in self.downs:
            x = down(x)
            h.append(x)
        
        # 中间层处理
        x = self.mid(x)
        
        # 上采样和特征融合 - 不使用torch.cat，而是使用加法
        h = h[:-1]  # 排除最后的特征图
        h.reverse()
        
        for i, up in enumerate(self.ups):
            x = up(x)
            x = x + h[i]  # 使用加法而非拼接，避免尺寸不匹配问题
        
        # 最终输出
        return self.final(x)

def validate_ddpm():
    # 参数设置 - 对Mac用户使用MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # 加载MNIST数据集作为测试
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 仅加载少量数据用于验证
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # 创建模型
    model = SimpleUnet(image_channels=1).to(device)
    
    # 创建训练和采样器
    T = 200  # 减少步数以加速验证
    beta_1 = 1e-4
    beta_T = 0.02
    
    trainer = Training(model, T, beta_1, beta_T).to(device)
    sampler = Sampling(model, beta_1, beta_T, T).to(device)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练几个批次来验证代码
    num_epochs = 1
    for epoch in range(num_epochs):
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # 计算损失
            loss = trainer(data).mean()
            
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 只训练几个批次用于验证
            if batch_idx >= 5:
                break
    
    # 生成图像来验证采样过程
    print("Generating images...")
    with torch.no_grad():
        # 从随机噪声开始
        x_T = torch.randn(8, 1, 28, 28).to(device)
        
        # 使用采样器生成图像
        samples = sampler(x_T)
    
    # 显示结果
    plt.figure(figsize=(8, 4))
    for i in range(8):
        plt.subplot(2, 4, i+1)
        plt.imshow(samples[i, 0].cpu().numpy(), cmap='gray')
        plt.axis('off')
    
    plt.savefig('ddpm_validation.png')
    print("Validation completed! Check ddpm_validation.png for generated samples.")

if __name__ == "__main__":
    validate_ddpm()