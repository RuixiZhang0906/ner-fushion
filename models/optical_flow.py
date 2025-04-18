import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

class OpticalFlowEstimator(nn.Module):
    def __init__(self, config):
        super(OpticalFlowEstimator, self).__init__()
        
        # 配置参数
        self.use_multiscale = config.get('use_multiscale', True)
        self.num_scales = config.get('num_scales', 3)
        self.use_checkpointing = config.get('use_checkpointing', False)
        
        # 多尺度光流估计
        if self.use_multiscale:
            self.flow_nets = nn.ModuleList()
            for i in range(self.num_scales):
                self.flow_nets.append(self._create_flow_net(512))
        else:
            # 光流估计网络
            self.flow_net = self._create_flow_net(512)
        
        # 时空一致性分析模块
        self.consistency_net = ConsistencyAnalyzer(config.consistency)
    
    def _create_flow_net(self, input_dim):
        """创建光流估计网络"""
        return nn.Sequential(
            nn.Conv2d(input_dim, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 2, kernel_size=3, padding=1)
        )
    
    def forward(self, features):
        """
        Args:
            features: 列表，包含每帧对齐后的特征 [B, C, H, W]
        Returns:
            flows: 帧间光流
            consistency: 时空一致性分数
        """
        T = len(features)
        
        if self.use_multiscale:
            return self._forward_multiscale(features)
        else:
            return self._forward_single_scale(features)
    
    def _forward_single_scale(self, features):
        """单尺度光流估计"""
        T = len(features)
        flows = []
        
        # 计算相邻帧之间的光流
        for t in range(T-1):
            # 拼接相邻帧的特征
            concat_feat = torch.cat([features[t], features[t+1]], dim=1)
            
            # 估计光流
            if self.use_checkpointing and self.training:
                flow = checkpoint.checkpoint(self.flow_net, concat_feat)
            else:
                flow = self.flow_net(concat_feat)
                
            flows.append(flow)
        
        # 分析时空一致性
        consistency = self.consistency_net(features, flows)
        
        return flows, consistency
    
    def _forward_multiscale(self, features):
        """多尺度光流估计"""
        T = len(features)
        B, C, H, W = features[0].shape
        flows = []
        
        for t in range(T-1):
            # 创建图像金字塔
            feat1_pyramid = self._create_pyramid(features[t])
            feat2_pyramid = self._create_pyramid(features[t+1])
            
            # 从最粗糙尺度开始估计
            flow = torch.zeros(B, 2, H//2**(self.num_scales-1), W//2**(self.num_scales-1)).to(features[0].device)
            
            # 从粗到细逐步优化
            for i in range(self.num_scales-1, -1, -1):
                # 上采样流场到当前尺度
                if i < self.num_scales-1:
                    flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
                
                # 当前尺度的特征
                feat1 = feat1_pyramid[i]
                feat2 = feat2_pyramid[i]
                
                # 使用光流扭曲特征
                warped_feat2 = self._warp_features(feat2, flow)
                
                # 拼接特征并估计残差流
                concat_feat = torch.cat([feat1, warped_feat2], dim=1)
                
                if self.use_checkpointing and self.training:
                    flow_delta = checkpoint.checkpoint(self.flow_nets[i], concat_feat)
                else:
                    flow_delta = self.flow_nets[i](concat_feat)
                
                # 更新流场
                flow = flow + flow_delta
            
            flows.append(flow)
        
        # 分析时空一致性
        consistency = self.consistency_net(features, flows)
        
        return flows, consistency
    
    def _create_pyramid(self, feature, num_scales=None):
        """创建特征金字塔"""
        if num_scales is None:
            num_scales = self.num_scales
            
        pyramid = [feature]
        for i in range(1, num_scales):
            pyramid.append(F.avg_pool2d(pyramid[-1], kernel_size=2))
        
        return pyramid
    
    def _warp_features(self, feature, flow):
        """使用光流扭曲特征"""
        B, C, H, W = feature.shape
        
        # 创建网格
        grid = self._create_grid(flow, H, W)
        
        # 扭曲特征
        warped = F.grid_sample(feature, grid, align_corners=True)
        
        return warped
    
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
    
    def compute_smoothness_loss(self, flows):
        """计算光流平滑性损失"""
        loss = 0.0
        for flow in flows:
            # 计算x和y方向的梯度
            dx = torch.abs(flow[:, :, :, :-1] - flow[:, :, :, 1:])
            dy = torch.abs(flow[:, :, :-1, :] - flow[:, :, 1:, :])
            
            # 计算平滑性损失
            loss += torch.mean(dx) + torch.mean(dy)
            
        return loss / len(flows)
    
    def compute_warping_loss(self, features, flows):
        """计算光流扭曲损失"""
        loss = 0.0
        T = len(features)
        
        for t in range(T-1):
            # 使用光流扭曲特征
            warped_feat = self._warp_features(features[t+1], flows[t])
            
            # 计算扭曲损失
            diff = torch.abs(features[t] - warped_feat)
            loss += torch.mean(diff)
            
        return loss / (T-1)

class ConsistencyAnalyzer(nn.Module):
    def __init__(self, config):
        super(ConsistencyAnalyzer, self).__init__()
        
        # 配置参数
        self.use_attention = config.get('use_attention', True)
        self.use_residual = config.get('use_residual', True)
        input_dim = config.input_dim
        
        # 注意力机制
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(input_dim, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1),
                nn.Sigmoid()
            )
        
        # 主干网络
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(128)
        
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        
        # 残差连接
        if self.use_residual:
            self.skip = nn.Conv2d(input_dim, 64, kernel_size=1)
        
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
        
        # 应用注意力机制
        if self.use_attention:
            attention = self.attention(diff_tensor)
            diff_tensor = diff_tensor * attention
        
        # 通过网络计算一致性分数
        x = F.relu(self.norm1(self.conv1(diff_tensor)))
        x = F.relu(self.norm2(self.conv2(x)))
        
        # 应用残差连接
        if self.use_residual:
            skip_x = self.skip(diff_tensor)
            x = x + skip_x
        
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
    
    def compute_consistency_loss(self, consistency, target_mask=None):
        """计算一致性损失
        
        Args:
            consistency: 预测的一致性分数 [B, 1, H, W]
            target_mask: 可选的目标掩码 [B, 1, H, W]
        
        Returns:
            loss: 一致性损失
        """
        if target_mask is not None:
            # 如果有目标掩码，计算二元交叉熵损失
            loss = F.binary_cross_entropy(consistency, target_mask)
        else:
            # 否则鼓励高一致性
            loss = torch.mean(1.0 - consistency)
        
        return loss
