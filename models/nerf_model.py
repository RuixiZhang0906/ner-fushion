import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from models.feature_extraction import FeatureExtractor
from models.optical_flow_v0 import OpticalFlowEstimator
from models.octree_v0 import OctreeBuilder
from models.scene_decomposition import SceneDecomposer
from models.adaptive_sampling import AdaptiveSampler

def positional_encoding(x, num_freqs, log_sampling=True):
    """
    对输入应用位置编码
    
    Args:
        x: 输入张量 [..., C]
        num_freqs: 频率数量
        log_sampling: 是否使用对数采样频率
        
    Returns:
        encoded: 编码后的张量 [..., C * (2 * num_freqs + 1)]
    """
    if log_sampling:
        freq_bands = 2.0 ** torch.linspace(0, num_freqs-1, num_freqs, device=x.device)
    else:
        freq_bands = torch.linspace(1.0, 2.0**(num_freqs-1), num_freqs, device=x.device)
    
    encodings = [x]
    for freq in freq_bands:
        encodings.append(torch.sin(x * freq))
        encodings.append(torch.cos(x * freq))
    
    return torch.cat(encodings, dim=-1)

class FusionNeRF(nn.Module):
    def __init__(self, config):
        super(FusionNeRF, self).__init__()
        
        # 位置编码配置
        self.pos_enc_levels = config.pos_enc_levels if hasattr(config, 'pos_enc_levels') else 10  # 位置编码频率数量
        self.dir_enc_levels = config.dir_enc_levels if hasattr(config, 'dir_enc_levels') else 4   # 方向编码频率数量
        self.time_enc_levels = config.time_enc_levels if hasattr(config, 'time_enc_levels') else 4  # 时间编码频率数量
        
        # 计算编码后的维度
        self.pos_enc_dim = 3 * (2 * self.pos_enc_levels + 1)  # 位置编码后的维度
        self.dir_enc_dim = 3 * (2 * self.dir_enc_levels + 1)  # 方向编码后的维度
        self.time_enc_dim = 1 * (2 * self.time_enc_levels + 1) if hasattr(config, 'time_enc_levels') else 0  # 时间编码后的维度
        
        # 初始化各个模块
        self.feature_extractor = FeatureExtractor(config.feature_extractor)
        self.optical_flow = OpticalFlowEstimator(config.optical_flow)
        self.octree_builder = OctreeBuilder(config.octree)
        self.scene_decomposer = SceneDecomposer(config.scene_decomposer)
        self.adaptive_sampler = AdaptiveSampler(config.adaptive_sampler)
        
        # 更新MLP输入维度以适应位置编码
        # 静态MLP输入: 编码后的位置 + 编码后的方向
        static_mlp_config = config.static_mlp
        static_mlp_config.input_dim = self.pos_enc_dim + self.dir_enc_dim
        
        # 动态MLP输入: 编码后的位置 + 编码后的方向 + 编码后的时间
        dynamic_mlp_config = config.dynamic_mlp
        dynamic_mlp_config.input_dim = self.pos_enc_dim + self.dir_enc_dim + self.time_enc_dim
        
        # NeRF MLP网络
        self.static_mlp = self._build_mlp(static_mlp_config)
        self.dynamic_mlp = self._build_mlp(dynamic_mlp_config)
        
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
        # features = self.feature_extractor(inputs['frames'])
        # 特征提取和光流估计可能产生大量中间结果，占用大量内存，使用梯度检查点减少内存使用
        # 1. 特征提取与对齐
        features = checkpoint(self.feature_extractor, inputs['frames'])
        # 2. 光流估计与时空一致性分析
        flow, consistency = checkpoint(self.optical_flow, features)
        # flow, consistency = self.optical_flow(features)
        
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
        
        # 应用位置编码
        encoded_points = positional_encoding(points, self.pos_enc_levels)
        encoded_dirs = positional_encoding(dirs, self.dir_enc_levels)
        
        # 应用时间编码
        encoded_time = positional_encoding(time.unsqueeze(-1), self.time_enc_levels) if time is not None else None
        
        # 计算静态和动态成分
        static_input = torch.cat([encoded_points, encoded_dirs], dim=-1)
        static_output = self.static_mlp(static_input)
        
        if encoded_time is not None:
            dynamic_input = torch.cat([encoded_points, encoded_dirs, encoded_time], dim=-1)
        else:
            # 如果没有时间信息，可以使用零张量替代
            dummy_time = torch.zeros((points.shape[0], self.time_enc_dim), device=points.device)
            dynamic_input = torch.cat([encoded_points, encoded_dirs, dummy_time], dim=-1)
            
        dynamic_output = self.dynamic_mlp(dynamic_input)
        
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
