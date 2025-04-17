import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveSampler(nn.Module):
    def __init__(self, config):
        super(AdaptiveSampler, self).__init__()
        
        self.n_samples = config.n_samples
        self.n_importance = config.n_importance
        self.perturb = config.perturb
        self.gradient_based = config.gradient_based
        
        if self.gradient_based:
            # 基于梯度的采样策略网络
            self.sampling_net = nn.Sequential(
                nn.Linear(config.input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
        
        self.near = config.near
        self.far = config.far
        
    def forward(self, static_repr, dynamic_repr, rays, config):
        """
        生成自适应采样策略
        
        Args:
            static_repr: 静态场景表示 [B, C, H, W]
            dynamic_repr: 动态场景表示 [B, T, C, H, W]
            rays: 光线参数 (origins, directions)
            config: 采样配置
            
        Returns:
            sampling_strategy: 采样策略对象
        """
        B = rays[0].shape[0]
        
        # 提取光线方向对应的特征
        ray_origins, ray_directions = rays
        
        # 创建采样策略
        if self.gradient_based:
            # 基于梯度的采样策略
            sampling_strategy = GradientBasedSamplingStrategy(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                static_repr=static_repr,
                dynamic_repr=dynamic_repr,
                sampling_net=self.sampling_net,
                n_samples=self.n_samples,
                n_importance=self.n_importance,
                near=self.near,
                far=self.far,
                perturb=self.perturb
            )
        else:
            # 基本的分层采样策略
            sampling_strategy = HierarchicalSamplingStrategy(
                ray_origins=ray_origins,
                ray_directions=ray_directions,
                n_samples=self.n_samples,
                n_importance=self.n_importance,
                near=self.near,
                far=self.far,
                perturb=self.perturb
            )
        
        return sampling_strategy

class SamplingStrategy:
    """采样策略基类"""
    def __init__(self, ray_origins, ray_directions, n_samples, n_importance, near, far, perturb):
        self.ray_origins = ray_origins
        self.ray_directions = ray_directions
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.near = near
        self.far = far
        self.perturb = perturb
        
    def generate_samples(self, rays):
        """生成采样点"""
        raise NotImplementedError
        
    def _sample_along_rays(self, ray_origins, ray_directions, z_vals):
        """根据z值采样点"""
        # [N_rays, N_samples, 3]
        pts = ray_origins[..., None, :] + ray_directions[..., None, :] * z_vals[..., :, None]
        return pts

class HierarchicalSamplingStrategy(SamplingStrategy):
    """分层采样策略"""
    def __init__(self, ray_origins, ray_directions, n_samples, n_importance, near, far, perturb):
        super().__init__(ray_origins, ray_directions, n_samples, n_importance, near, far, perturb)
        
    def generate_samples(self, rays=None):
        """生成分层采样点"""
        if rays is not None:
            ray_origins, ray_directions = rays
        else:
            ray_origins, ray_directions = self.ray_origins, self.ray_directions
            
        N_rays = ray_origins.shape[0]
        
        # 粗采样
        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=ray_origins.device)
        z_vals = self.near * (1.-t_vals) + self.far * t_vals
        
        # 扩展到每条光线
        z_vals = z_vals.expand(N_rays, self.n_samples)
        
        # 扰动采样点
        if self.perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # 生成采样点
        pts = self._sample_along_rays(ray_origins, ray_directions, z_vals)
        
        # 返回采样点、方向和z值
        return pts, ray_directions[..., None, :].expand_as(pts), z_vals

class GradientBasedSamplingStrategy(SamplingStrategy):
    """基于梯度的自适应采样策略"""
    def __init__(self, ray_origins, ray_directions, static_repr, dynamic_repr, sampling_net, 
                 n_samples, n_importance, near, far, perturb):
        super().__init__(ray_origins, ray_directions, n_samples, n_importance, near, far, perturb)
        self.static_repr = static_repr
        self.dynamic_repr = dynamic_repr
        self.sampling_net = sampling_net
        
    def generate_samples(self, rays=None):
        """生成基于梯度的自适应采样点"""
        if rays is not None:
            ray_origins, ray_directions = rays
        else:
            ray_origins, ray_directions = self.ray_origins, self.ray_directions
            
        N_rays = ray_origins.shape[0]
        
        # 首先进行粗采样
        t_vals = torch.linspace(0., 1., steps=self.n_samples, device=ray_origins.device)
        z_vals = self.near * (1.-t_vals) + self.far * t_vals
        
        # 扩展到每条光线
        z_vals = z_vals.expand(N_rays, self.n_samples)
        
        # 扰动采样点
        if self.perturb > 0.:
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            t_rand = torch.rand(z_vals.shape, device=z_vals.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # 生成粗采样点
        pts = self._sample_along_rays(ray_origins, ray_directions, z_vals)
        
        # 计算采样点的重要性权重
        with torch.no_grad():
            # 这里应该根据静态和动态表示计算每个点的重要性
            # 简化实现，仅使用采样网络
            pts_flat = pts.reshape(-1, 3)
            dirs_flat = ray_directions[..., None, :].expand_as(pts).reshape(-1, 3)
            
            # 计算重要性权重
            inputs = torch.cat([pts_flat, dirs_flat], -1)
            weights = self.sampling_net(inputs).reshape(N_rays, self.n_samples)
            weights = F.softmax(weights, -1)
            
            # 基于权重进行重要性采样
            z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            z_samples = self._sample_pdf(z_vals_mid, weights[..., 1:-1], self.n_importance)
            z_samples = z_samples.detach()
            
            # 合并采样点
            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            
        # 生成最终采样点
        pts = self._sample_along_rays(ray_origins, ray_directions, z_vals)
        
        return pts, ray_directions[..., None, :].expand_as(pts), z_vals
    
    def _sample_pdf(self, bins, weights, N_samples):
        """基于权重的概率密度函数采样"""
        # 获取PDF
        weights = weights + 1e-5  # 防止除零
        pdf = weights / torch.sum(weights, -1, keepdim=True)
        cdf = torch.cumsum(pdf, -1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
        
        # 取N_samples个均匀样本
        u = torch.linspace(0., 1., steps=N_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
        
        # 反向采样
        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.max(torch.zeros_like(inds-1), inds-1)
        above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
        inds_g = torch.stack([below, above], -1)
        
        matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
        cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
        bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)
        
        denom = (cdf_g[..., 1] - cdf_g[..., 0])
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        
        return samples
