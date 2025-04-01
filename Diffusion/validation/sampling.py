import torch
import torch.nn as nn
import torch.nn.functional as F
from .extract import extract

class Sampling(nn.Module):
   def __init__(self, model, beta_1, beta_T, T):
      super().__init__()

      self.model = model
      self.T = T

      self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
      alphas = 1. - self.betas
      alphas_bar = torch.cumprod(alphas, dim=0)
      # pad 参数是一个长度可变的列表或元组，遵循从最后一个维度向前的逆序填充规则
      # [1, 0] 表示只在最后一个维度左侧填充 1 个元素，右侧不填充
      alphas_bar_prev = F.pad(alphas_bar, pad=[1,0], value=1)[:T]

      self.register_buffer('coeff1', 1./ torch.sqrt(alphas_bar))
      self.register_buffer('coeff2', self.coeff1 * self.betas / torch.sqrt(1-alphas_bar))

      self.register_buffer('posterior_var', self.betas * (1 - alphas_bar_prev) / (1 - alphas_bar))
      
   def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
      assert x_t.shape == eps.shape
      mean = extract(self.coeff1, t, x_t.shape) * x_t - \
             extract(self.coeff2, t, x_t.shape) * eps
      return mean
   
   def p_mean_variance(self, x_t, t):
      var = extract(self.posterior_var, t, x_t.shape)
      std = torch.sqrt(var)

      eps = self.model(x_t, t)
      xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps)
      return xt_prev_mean, std
   
   def forward(self, x_T):
      x_t = x_T
      for time_step in reversed(range(self.T)):
         if time_step > 0:
            noise = torch.randn_like(x_t, device=x_t.device)
         else:
            noise = torch.zeros_like(x_t, device=x_t.device)
         t = x_t.new_ones([x_T.shape[0],], dtype=torch.long) * time_step
         mean, std = self.p_mean_variance(x_t, t)
         x_t = mean + noise * std
         assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
      x_0 = x_t
      return torch.clamp(x_0, -1, 1) 