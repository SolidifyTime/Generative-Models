import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """
    使用正弦位置编码实现时间步嵌入
    将标量时间步映射到高维向量
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: [batch_size] tensor
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class TimeEmbedding(nn.Module):
    """
    时间嵌入模块，包含SinusoidalPositionEmbeddings和MLP
    """
    def __init__(self, time_embedding_dim=256):
        super().__init__()
        self.time_embedding_dim = time_embedding_dim
        self.sinusoidal_embedding = SinusoidalPositionEmbeddings(time_embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, time_embedding_dim * 2),
            nn.SiLU(),
            nn.Linear(time_embedding_dim * 2, time_embedding_dim)
        )
        
    def forward(self, timesteps):
        # timesteps: [batch_size] tensor of integer timesteps
        x = self.sinusoidal_embedding(timesteps)
        x = self.time_mlp(x)
        return x 