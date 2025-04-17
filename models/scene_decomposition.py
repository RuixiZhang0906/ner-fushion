# models/scene_decomposition.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SceneDecomposer(nn.Module):
    def __init__(self, config):
        super(SceneDecomposer, self).__init__()
        
        # 静态/动态分割网络
        self.segmentation_net = nn.Sequential(
            nn.Conv2d(config.input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1)
        )
        
        # 静态场景编码器
        self.static_encoder = nn.Sequential(
            nn.Conv2d(config.feature_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, config.output_dim, kernel_size=1)
        )
        
        # 动态场景编码器
        self.dynamic_encoder = nn.Sequential(
            nn.Conv2d(config.feature_dim + 2, 128, kernel_size=3, padding=1),  # +2 for flow
            nn.ReLU(),
            nn.Conv2d(128, config.output_dim, kernel_size=1)
        )
        
    def forward(self, features, flows, consistency, octree):
        """
        Args:
            features: 特征列表
            flows: 光流列表
            consistency: 时空一致性分数
            octree: 八叉树结构
        
        Returns:
            static_repr: 静态场景表示
            dynamic_repr: 动态场景表示
        """
        B, C, H, W = features[0].shape
        T = len(features)
        
        # 基于一致性分数和光流生成静态/动态分割掩码
        static_masks = []
        for t in range(T):
            # 拼接特征和一致性
            if t < T-1:
                input_tensor = torch.cat([features[t], consistency], dim=1)
            else:
                # 最后一帧没有对应的一致性，使用前一帧的
                input_tensor = torch.cat([features[t], consistency], dim=1)
            
            # 生成分割掩码
            mask = torch.sigmoid(self.segmentation_net(input_tensor))
            static_masks.append(mask)
        
        # 编码静态场景表示
        static_features = []
        for t in range(T):
            # 应用掩码
            masked_feature = features[t] * static_masks[t]
            # 编码
            static_feat = self.static_encoder(masked_feature)
            static_features.append(static_feat)
        
        # 取平均作为最终静态表示
        static_repr = torch.stack(static_features).mean(dim=0)
        
        # 编码动态场景表示
        dynamic_features = []
        for t in range(T-1):
            # 应用反掩码
            masked_feature = features[t] * (1 - static_masks[t])
            # 拼接光流
            input_tensor = torch.cat([masked_feature, flows[t]], dim=1)
            # 编码
            dynamic_feat = self.dynamic_encoder(input_tensor)
            dynamic_features.append(dynamic_feat)
        
        # 最后一帧
        if T > 1:
            masked_feature = features[-1] * (1 - static_masks[-1])
            # 使用零光流
            zero_flow = torch.zeros_like(flows[0])
            input_tensor = torch.cat([masked_feature, zero_flow], dim=1)
            dynamic_feat = self.dynamic_encoder(input_tensor)
            dynamic_features.append(dynamic_feat)
        
        # 动态表示保持时序信息
        dynamic_repr = torch.stack(dynamic_features, dim=1)  # [B, T, C, H, W]
        
        return static_repr, dynamic_repr
