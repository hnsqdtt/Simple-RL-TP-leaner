import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Normal
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0

class CNNEncoder(nn.Module):
    def __init__(self, in_ch=1, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.fc   = nn.Linear(64*4*4, out_dim)
    def forward(self, x):
        z = self.pool(self.conv(x)).flatten(1)
        return F.relu(self.fc(z))

class VecEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim,128), nn.ReLU(),
                                 nn.Linear(128,out_dim), nn.ReLU())
    def forward(self, v): return self.net(v)

class Actor(nn.Module):
    def __init__(self, vec_dim, action_dim=3, img_ch=1):
        super().__init__()
        self.cnn = CNNEncoder(img_ch, 256)
        self.vec = VecEncoder(vec_dim, 128)
        self.trunk = nn.Sequential(nn.Linear(256+128,256), nn.ReLU())
        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
    def forward(self, img, vec):
        h = torch.cat([self.cnn(img), self.vec(vec)], dim=1)
        h = self.trunk(h)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std
    def sample(self, img, vec, limits):
        mu, log_std = self.forward(img, vec)
        std = log_std.exp()
        eps = torch.randn_like(mu)
        pre_tanh = mu + std*eps
        a = torch.tanh(pre_tanh)
        # tanh 修正后的 logπ
        dist = Normal(mu, std)
        logp = dist.log_prob(pre_tanh) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(-1, keepdim=True)
        # 依据 limits 缩放到物理上限（[v,v,ω]）
        a_scaled = a * limits
        return a_scaled, logp

class Critic(nn.Module):
    def __init__(self, vec_dim, action_dim=3, img_ch=1):
        super().__init__()
        self.cnn = CNNEncoder(img_ch, 256)
        self.vec = VecEncoder(vec_dim + action_dim, 128)
        self.q   = nn.Sequential(nn.Linear(256+128,256), nn.ReLU(), nn.Linear(256,1))
    def forward(self, img, vec, act):
        z = torch.cat([self.cnn(img), self.vec(torch.cat([vec, act], dim=1))], dim=1)
        return self.q(z)
