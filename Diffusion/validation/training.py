import torch
import torch.nn as nn
import torch.nn.functional as F
from .extract import extract

class Training(nn.Module):
    def __init__(self, model, T, beta_1, beta_T):
        super().__init__()
        self.T = T
        self.model = model

        # 存储超参数到模型中
        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).float()
        )
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar)
        )
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar)
        )

    def forward(self, x_0):
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
              extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + 
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)
        
        # reduction = 'none' 逐元素求mse
        loss = F.mse_loss(
            self.model(x_t, t), noise, reduction='none'
        )
        return loss 