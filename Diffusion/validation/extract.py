import torch

def extract(alphas, timestep, x_shape):
    device = timestep.device
    out = torch.gather(alphas, index=timestep, dim=0).float().to(device)
    return out.view([timestep.shape[0]] + (len(x_shape)-1) * [1]) 