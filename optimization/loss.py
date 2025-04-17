import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_loss(outputs, targets, config):
    """
    计算训练损失
    
    Args:
        outputs: 模型输出
        targets: 真实值
        config: 损失配置
        
    Returns:
        loss_dict: 损失字典
    """
    loss_dict = {}
    
    # RGB重建损失
    rgb_loss = compute_rgb_loss(outputs['rgb'], targets['rgb'], config)
    loss_dict['rgb_loss'] = rgb_loss
    
    # 时空一致性损失
    if 'consistency' in outputs and config.use_consistency_loss:
        consistency_loss = compute_consistency_loss(
            outputs['consistency'], 
            outputs['static_weights'], 
            outputs['dynamic_weights'],
            config
        )
        loss_dict['consistency_loss'] = consistency_loss
    else:
        consistency_loss = 0.0
    
    # 深度平滑损失
    if 'depth' in outputs and config.use_depth_loss:
        depth_loss = compute_depth_loss(outputs['depth'], config)
        loss_dict['depth_loss'] = depth_loss
    else:
        depth_loss = 0.0
    
    # 总损失
    total_loss = (
        config.rgb_weight * rgb_loss +
        config.consistency_weight * consistency_loss +
        config.depth_weight * depth_loss
    )
    
    loss_dict['total_loss'] = total_loss
    
    return loss_dict

def compute_rgb_loss(pred_rgb, gt_rgb, config):
    """计算RGB重建损失"""
    if config.rgb_loss_type == 'l1':
        return F.l1_loss(pred_rgb, gt_rgb)
    elif config.rgb_loss_type == 'l2':
        return F.mse_loss(pred_rgb, gt_rgb)
    elif config.rgb_loss_type == 'huber':
        return F.smooth_l1_loss(pred_rgb, gt_rgb, beta=config.huber_delta)
    else:
        raise ValueError(f"Unsupported RGB loss type: {config.rgb_loss_type}")

def compute_consistency_loss(consistency, static_weights, dynamic_weights, config):
    """计算时空一致性损失"""
    # 鼓励动态区域的一致性低，静态区域的一致性高
    static_consistency_loss = F.binary_cross_entropy(
        consistency, 
        torch.ones_like(consistency),
        reduction='none'
    )
    static_consistency_loss = (static_weights * static_consistency_loss).mean()
    
    dynamic_consistency_loss = F.binary_cross_entropy(
        consistency,
        torch.zeros_like(consistency),
        reduction='none'
    )
    dynamic_consistency_loss = (dynamic_weights * dynamic_consistency_loss).mean()
    
    return static_consistency_loss + dynamic_consistency_loss

def compute_depth_loss(depth, config):
    """计算深度平滑损失"""
    # 计算深度梯度
    dy_depth = depth[:, 1:] - depth[:, :-1]
    dx_depth = depth[:, :, 1:] - depth[:, :, :-1]
    
    # 梯度平滑损失
    dy_loss = torch.mean(torch.abs(dy_depth))
    dx_loss = torch.mean(torch.abs(dx_depth))
    
    return dy_loss + dx_loss
