import torch
import torch.nn as nn
import torch.nn.functional as F

class OpticalFlowEstimator(nn.Module):
    def __init__(self, config):
        super(OpticalFlowEstimator, self).__init__()
        
        # 光流估计网络
        self.flow_net = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
        
        # 时空一致性分析模块
        self.consistency_net = ConsistencyAnalyzer(config.consistency)
        
    def forward(self, features):
        """
        Args:
            features: 列表，包含每帧对齐后的特征 [B, C, H, W]
        Returns:
            flows: 帧间光流
            consistency: 时空一致性分数
        """
        T = len(features)
        flows = []
        
        # 计算相邻帧之间的光流
        for t in range(T-1):
            # 拼接相邻帧的特征
            concat_feat = torch.cat([features[t], features[t+1]], dim=1)
            # 估计光流
            flow = self.flow_net(concat_feat)
            flows.append(flow)
        
        # 分析时空一致性
        consistency = self.consistency_net(features, flows)
        
        return flows, consistency

class ConsistencyAnalyzer(nn.Module):
    def __init__(self, config):
        super(ConsistencyAnalyzer, self).__init__()
        
        self.conv1 = nn.Conv2d(config.input_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
    def forward(self, features, flows):
        """
        Args:
            features: 列表，包含每帧特征
            flows: 列表，包含相邻帧之间的光流
        Returns:
            consistency: 时空一致性分数 [B, 1, H, W]
        """
        T = len(features)
        B, C, H, W = features[0].shape
        
        # 使用光流进行特征对齐
        warped_features = []
        for t in range(T-1):
            # 使用光流对特征进行变形
            grid = self._create_grid(flows[t], H, W)
            warped = F.grid_sample(features[t+1], grid, align_corners=True)
            warped_features.append(warped)
        
        # 计算原始特征与变形特征之间的差异
        diffs = []
        for t in range(T-1):
            diff = torch.abs(features[t] - warped_features[t])
            diffs.append(diff)
        
        # 如果只有一帧差异，直接使用它
        if len(diffs) == 1:
            diff_tensor = diffs[0]
        else:
            # 否则取平均
            diff_tensor = torch.stack(diffs).mean(dim=0)
        
        # 通过网络计算一致性分数
        x = F.relu(self.conv1(diff_tensor))
        x = F.relu(self.conv2(x))
        consistency = torch.sigmoid(self.conv3(x))
        
        return consistency
    
    def _create_grid(self, flow, H, W):
        """创建用于网格采样的坐标网格"""
        B = flow.shape[0]
        
        # 创建基础网格
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, H, W, 1).repeat(B, 1, 1, 1)
        yy = yy.view(1, H, W, 1).repeat(B, 1, 1, 1)
        grid = torch.cat([xx, yy], dim=-1).float().to(flow.device)
        
        # 应用光流偏移
        flow_permuted = flow.permute(0, 2, 3, 1)
        grid = grid + flow_permuted
        
        # 归一化到[-1, 1]
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        
        return grid
