import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

def compute_metrics(pred, target):
    """
    计算评估指标
    
    Args:
        pred: 预测图像 [H, W, 3]
        target: 目标图像 [H, W, 3]
        
    Returns:
        metrics: 指标字典
    """
    metrics = {}
    
    # 确保输入是CPU上的numpy数组
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 确保值范围在[0, 1]
    pred = np.clip(pred, 0, 1)
    target = np.clip(target, 0, 1)
    
    # 计算PSNR
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        metrics['psnr'] = 100.0
    else:
        metrics['psnr'] = -10 * np.log10(mse)
    
    # 计算SSIM
    metrics['ssim'] = ssim_fn(
        target, pred, 
        multichannel=True, 
        data_range=1.0
    )
    
    # 计算LPIPS（如果可用）
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex')
        pred_tensor = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0) * 2 - 1
        metrics['lpips'] = loss_fn(pred_tensor, target_tensor).item()
    except ImportError:
        # 如果LPIPS不可用，使用感知损失代替
        metrics['lpips'] = compute_perceptual_loss(
            torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0),
            torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
        ).item()
    
    return metrics

def compute_perceptual_loss(pred, target):
    """
    计算简单的感知损失（如果LPIPS不可用）
    
    Args:
        pred: 预测图像 [B, C, H, W]
        target: 目标图像 [B, C, H, W]
        
    Returns:
        loss: 感知损失
    """
    # 使用预训练的VGG特征
    try:
        from torchvision.models import vgg16
        from torchvision.transforms import Normalize
        
        # 加载预训练模型
        model = vgg16(pretrained=True).features[:16].eval()
        
        # 标准化
        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pred_norm = normalize(pred)
        target_norm = normalize(target)
        
        # 提取特征
        pred_features = model(pred_norm)
        target_features = model(target_norm)
        
        # 计算特征距离
        return F.mse_loss(pred_features, target_features)
    except:
        # 如果VGG不可用，返回MSE
        return F.mse_loss(pred, target)
