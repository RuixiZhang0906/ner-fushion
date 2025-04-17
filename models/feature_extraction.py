import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        
        # CNN特征提取器
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # 特征对齐模块
        self.alignment = FeatureAlignment(config.alignment)
        
    def forward(self, frames):
        """
        Args:
            frames: [B, T, C, H, W] 多帧输入
        Returns:
            aligned_features: 对齐后的特征
        """
        B, T, C, H, W = frames.shape
        
        # 提取每一帧的特征
        features = []
        for t in range(T):
            feat = self.cnn(frames[:, t])
            features.append(feat)
        
        # 特征对齐
        aligned_features = self.alignment(features)
        
        return aligned_features

class FeatureAlignment(nn.Module):
    def __init__(self, config):
        super(FeatureAlignment, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.feature_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
    def forward(self, features):
        """
        Args:
            features: 列表，包含每帧的特征 [B, C, H, W]
        Returns:
            aligned_features: 对齐后的特征
        """
        B, C, H, W = features[0].shape
        T = len(features)
        
        # 重塑特征以适应注意力机制
        reshaped_features = []
        for feat in features:
            # [B, C, H, W] -> [B, H*W, C]
            reshaped = feat.view(B, C, H*W).permute(0, 2, 1)
            reshaped_features.append(reshaped)
        
        # 拼接所有帧的特征
        # [B, T*H*W, C]
        concat_features = torch.cat(reshaped_features, dim=1)
        
        # 应用自注意力进行特征对齐
        aligned_features = []
        for t in range(T):
            query = reshaped_features[t]
            attn_output, _ = self.attention(query, concat_features, concat_features)
            # [B, H*W, C] -> [B, C, H, W]
            aligned = attn_output.permute(0, 2, 1).view(B, C, H, W)
            aligned_features.append(aligned)
        
        return aligned_features
