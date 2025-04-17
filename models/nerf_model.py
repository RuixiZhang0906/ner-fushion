import torch
import torch.nn as nn
from models.feature_extraction import FeatureExtractor
from models.optical_flow import OpticalFlowEstimator
from models.octree import OctreeBuilder
from models.scene_decomposition import SceneDecomposer
from models.adaptive_sampling import AdaptiveSampler

class FusionNeRF(nn.Module):
    def __init__(self, config):
        super(FusionNeRF, self).__init__()
        
        # 初始化各个模块
        self.feature_extractor = FeatureExtractor(config.feature_extractor)
        self.optical_flow = OpticalFlowEstimator(config.optical_flow)
        self.octree_builder = OctreeBuilder(config.octree)
        self.scene_decomposer = SceneDecomposer(config.scene_decomposer)
        self.adaptive_sampler = AdaptiveSampler(config.adaptive_sampler)
        
        # NeRF MLP网络
        self.static_mlp = self._build_mlp(config.static_mlp)
        self.dynamic_mlp = self._build_mlp(config.dynamic_mlp)
        
        self.config = config
        
    def _build_mlp(self, mlp_config):
        """构建MLP网络"""
        layers = []
        input_dim = mlp_config.input_dim
        
        for dim in mlp_config.hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
            
        layers.append(nn.Linear(input_dim, mlp_config.output_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, inputs):
        """前向传播"""
        # 1. 特征提取与对齐
        features = self.feature_extractor(inputs['frames'])
        
        # 2. 光流估计与时空一致性分析
        flow, consistency = self.optical_flow(features)
        
        # 3. 八叉树构建
        octree = self.octree_builder(features, flow)
        
        # 4. 动态场景解耦
        static_repr, dynamic_repr = self.scene_decomposer(features, flow, consistency, octree)
        
        # 5. 梯度分析与采样策略生成
        sampling_strategy = self.adaptive_sampler(
            static_repr, dynamic_repr, inputs['rays'], self.config.sampling
        )
        
        # 6. 渲染结果
        results = self._render(
            static_repr, dynamic_repr, 
            inputs['rays'], inputs['time'], 
            sampling_strategy
        )
        
        return {
            'rgb': results['rgb'],
            'depth': results['depth'],
            'weights': results['weights'],
            'static_weights': results['static_weights'],
            'dynamic_weights': results['dynamic_weights'],
            'sampling_points': results['sampling_points'],
            'consistency': consistency
        }
    
    def _render(self, static_repr, dynamic_repr, rays, time, sampling_strategy):
        """渲染过程"""
        # 根据采样策略生成采样点
        points, dirs, z_vals = sampling_strategy.generate_samples(rays)
        
        # 计算静态和动态成分
        static_output = self.static_mlp(torch.cat([points, dirs], dim=-1))
        dynamic_output = self.dynamic_mlp(torch.cat([points, dirs, time.unsqueeze(-1)], dim=-1))
        
        # 解析输出
        static_sigma = static_output[..., 0]
        static_rgb = torch.sigmoid(static_output[..., 1:4])
        
        dynamic_sigma = dynamic_output[..., 0]
        dynamic_rgb = torch.sigmoid(dynamic_output[..., 1:4])
        
        # 根据解耦权重合成最终结果
        blend_weights = torch.sigmoid(dynamic_output[..., 4:5])
        sigma = (1 - blend_weights) * static_sigma + blend_weights * dynamic_sigma
        rgb = (1 - blend_weights) * static_rgb + blend_weights * dynamic_rgb
        
        # 体积渲染
        weights = self._compute_weights(sigma, z_vals)
        rgb_map = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)
        depth_map = torch.sum(weights * z_vals, dim=-1)
        
        return {
            'rgb': rgb_map,
            'depth': depth_map,
            'weights': weights,
            'static_weights': (1 - blend_weights) * weights,
            'dynamic_weights': blend_weights * weights,
            'sampling_points': points
        }
    
    def _compute_weights(self, sigma, z_vals):
        """计算体积渲染权重"""
        delta = z_vals[..., 1:] - z_vals[..., :-1]
        delta = torch.cat([delta, torch.ones_like(delta[..., :1]) * 1e10], dim=-1)
        
        alpha = 1.0 - torch.exp(-sigma * delta)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[..., :-1]
        
        return weights
