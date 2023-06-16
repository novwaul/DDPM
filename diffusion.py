import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod

class GaussianDiffusion(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.steps = 1000

        # register constants for cuda transition
        self.register_buffer("betas", torch.linspace(start=1e-4, end=2e-2, steps=self.steps, dtype=torch.float64)) # for precision
        self.register_buffer("alphas", 1. - self.betas)
        self.register_buffer("alphas_bar", torch.cumprod(self.alphas, dim=0))

    def extract(self, values, times, dimension_num):
        B, *_ = times.shape
        selected_values = torch.gather(values, dim=0, index=times)
        return selected_values.reshape((B, *[1 for _ in range(dimension_num-1)])) # to broadcast coefficients
    
    @abstractmethod
    def forward(self):
        pass
    
class GaussianDiffusionLoss(GaussianDiffusion):
    def __init__(self, model):
        super().__init__()
        self.model = model

        # register constants for cuda transition
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(self.alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - self.alphas_bar))
        
    def forward(self, x_0):
        # Algorithm 1
        B, *_ = x_0.shape
        dimension_num = len(x_0.shape)
        
        times = torch.randint(self.steps, size=(B, ), device=x_0.device)
        epsilon = torch.randn_like(x_0, device=x_0.device)
        x_t = self.extract(self.sqrt_alphas_bar, times, dimension_num) * x_0 + self.extract(self.sqrt_one_minus_alphas_bar, times, dimension_num) * epsilon
        
        loss = F.mse_loss(self.model(x_t, times), epsilon, reduction="mean") 
        return loss

class GaussianDiffusionSampler(GaussianDiffusion):
    def __init__(self):
        super().__init__()
        # register constants for cuda transition
        self.register_buffer("one_over_sqrt_alphas", torch.sqrt(1. / self.alphas))
        self.register_buffer("betas_over_sqrt_one_minus_alphas_bar", self.betas / torch.sqrt(1. - self.alphas_bar)) # 1 - alpha = beta
        self.register_buffer("sigmas", torch.sqrt(self.betas))
       
    def forward(self, x_T):
        B, *_ = x_T.shape
        dimension_num = len(x_T.shape)
        x_t = x_T
        for step in reversed(range(self.steps)):
            times = step * x_t.new_ones((B, ))
            epsilon = self.model(x_t, times)
            one_over_sqrt_alpha = self.extract(self.one_over_sqrt_alphas, times, dimension_num)
            beta_over_sqrt_one_minus_alpha_bar = self.extract(self.betas_over_sqrt_one_minus_alphas_bar, times, dimension_num)
            sigma = self.extact(self.sigmas, times, dimension_num)
            z = torch.randn_like(x_t, device=x_t.device) if step > 0 else 0
            # get x_{t-1} from x_{t} and set to new x_{t}
            x_t = one_over_sqrt_alpha * ( x_t - beta_over_sqrt_one_minus_alpha_bar * epsilon ) + sigma * z

        x_0 = x_t
        return torch.clip(x_0, -1, 1)