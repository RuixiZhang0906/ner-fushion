import torch

def volume_render(rgb, sigma, z_vals, white_bkgd=False):
    """
    体积渲染
    
    Args:
        rgb: 颜色值 [N_rays, N_samples, 3]
        sigma: 密度值 [N_rays, N_samples]
        z_vals: 采样深度 [N_rays, N_samples]
        white_bkgd: 是否使用白色背景
        
    Returns:
        rgb_map: 渲染的RGB图像 [N_rays, 3]
        depth_map: 渲染的深度图 [N_rays]
        weights: 渲染权重 [N_rays, N_samples]
        acc_map: 累积不透明度 [N_rays]
    """
    # 计算相邻采样点之间的距离
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)  # [N_rays, N_samples]
    
    # 计算alpha值（不透明度）
    alpha = 1.0 - torch.exp(-sigma * dists)  # [N_rays, N_samples]
    
    # 计算权重
    # T_i = exp(-sum_{j=1}^{i-1} sigma_j * delta_j)
    # weights = alpha_i * T_i
    T = torch.cumprod(torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1)[..., :-1]
    weights = alpha * T  # [N_rays, N_samples]
    
    # 渲染RGB
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [N_rays, 3]
    
    # 渲染深度
    depth_map = torch.sum(weights * z_vals, dim=-1)  # [N_rays]
    
    # 累积不透明度（用于混合背景）
    acc_map = torch.sum(weights, dim=-1)  # [N_rays]
    
    # 处理白色背景
    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])
    
    return {
        'rgb': rgb_map,
        'depth': depth_map,
        'weights': weights,
        'acc': acc_map
    }

def render_rays(model, rays_o, rays_d, near, far, n_samples, n_importance=0, perturb=0, time=None):
    """
    渲染光线
    
    Args:
        model: NeRF模型
        rays_o: 光线起点 [N_rays, 3]
        rays_d: 光线方向 [N_rays, 3]
        near: 近平面距离
        far: 远平面距离
        n_samples: 粗采样点数量
        n_importance: 细采样点数量
        perturb: 扰动强度
        time: 时间值 (用于动态场景)
        
    Returns:
        results: 渲染结果
    """
    # 粗采样
    pts, z_vals = stratified_sampling(rays_o, rays_d, near, far, n_samples, perturb)
    
    # 扩展方向
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, None].expand(pts.shape)
    
    # 准备模型输入
    if time is not None:
        time_expanded = time[:, None].expand(pts.shape[:-1] + (1,))
        pts_input = torch.cat([pts, viewdirs, time_expanded], dim=-1)
    else:
        pts_input = torch.cat([pts, viewdirs], dim=-1)
    
    # 模型推理
    raw = model(pts_input)
    
    # 解析输出
    rgb = torch.sigmoid(raw[..., :3])
    sigma = raw[..., 3]
    
    # 体积渲染
    coarse_render = volume_render(rgb, sigma, z_vals)
    
    results = {
        'coarse': coarse_render
    }
    
    # 如果需要细采样
    if n_importance > 0:
        # 重要性采样
        pts_fine, z_vals_fine = importance_sampling(
            rays_o, rays_d, z_vals, coarse_render['weights'], n_importance, perturb
        )
        
        # 扩展方向
        viewdirs_fine = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        viewdirs_fine = viewdirs_fine[:, None].expand(pts_fine.shape)
        
        # 准备模型输入
        if time is not None:
            time_expanded = time[:, None].expand(pts_fine.shape[:-1] + (1,))
            pts_input_fine = torch.cat([pts_fine, viewdirs_fine, time_expanded], dim=-1)
        else:
            pts_input_fine = torch.cat([pts_fine, viewdirs_fine], dim=-1)
        
        # 模型推理
        raw_fine = model(pts_input_fine)
        
        # 解析输出
        rgb_fine = torch.sigmoid(raw_fine[..., :3])
        sigma_fine = raw_fine[..., 3]
        
        # 体积渲染
        fine_render = volume_render(rgb_fine, sigma_fine, z_vals_fine)
        
        results['fine'] = fine_render
    
    return results
