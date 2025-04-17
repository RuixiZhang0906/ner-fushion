import torch

def generate_rays(camera_params, H, W):
    """
    生成相机光线
    
    Args:
        camera_params: 相机参数 (位置, 方向, 内参)
        H, W: 图像高度和宽度
        
    Returns:
        rays_o: 光线起点 [H*W, 3]
        rays_d: 光线方向 [H*W, 3]
    """
    if isinstance(camera_params, dict):
        # 从字典中提取相机参数
        c2w = camera_params.get('c2w', torch.eye(4))
        K = camera_params.get('K', None)
    else:
        # 假设camera_params是变换矩阵
        c2w = camera_params
        K = None
    
    if isinstance(c2w, torch.Tensor):
        device = c2w.device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        c2w = torch.tensor(c2w, dtype=torch.float32, device=device)
    
    # 如果没有提供内参，使用默认值
    if K is None:
        focal = 0.5 * W / torch.tan(torch.tensor(0.5 * 3.14159 / 3))  # 60度视场角
        K = torch.tensor([
            [focal, 0, W/2],
            [0, focal, H/2],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
    elif not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32, device=device)
    
    # 生成像素坐标
    i, j = torch.meshgrid(
        torch.linspace(0, W-1, W, device=device),
        torch.linspace(0, H-1, H, device=device)
    )
    i = i.t()  # 转置以匹配图像坐标
    j = j.t()
    
    # 像素坐标转相机坐标
    dirs = torch.stack([
        (i - K[0, 2]) / K[0, 0],
        -(j - K[1, 2]) / K[1, 1],
        -torch.ones_like(i)
    ], dim=-1)
    
    # 相机坐标转世界坐标
    rays_d = dirs @ c2w[:3, :3].t()
    
    # 归一化方向
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    
    # 光线起点（相机位置）
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    
    # 展平
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    return rays_o, rays_d

def stratified_sampling(rays_o, rays_d, near, far, n_samples, perturb=0):
    """
    分层采样
    
    Args:
        rays_o: 光线起点 [N_rays, 3]
        rays_d: 光线方向 [N_rays, 3]
        near: 近平面距离
        far: 远平面距离
        n_samples: 采样点数量
        perturb: 扰动强度 (0-1)
        
    Returns:
        pts: 采样点 [N_rays, n_samples, 3]
        z_vals: 采样深度 [N_rays, n_samples]
    """
    N_rays = rays_o.shape[0]
    device = rays_o.device
    
    # 均匀采样深度
    t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
    z_vals = near * (1.-t_vals) + far * t_vals
    
    # 扩展到每条光线
    z_vals = z_vals.expand(N_rays, n_samples)
    
    # 扰动采样点
    if perturb > 0.:
        # 获取相邻采样点的中点
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        
        # 在每个区间内随机采样
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand
    
    # 计算3D采样点
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    
    return pts, z_vals

def importance_sampling(rays_o, rays_d, z_vals, weights, n_importance, perturb=0):
    """
    重要性采样
    
    Args:
        rays_o: 光线起点 [N_rays, 3]
        rays_d: 光线方向 [N_rays, 3]
        z_vals: 初始采样深度 [N_rays, n_samples]
        weights: 初始权重 [N_rays, n_samples]
        n_importance: 重要性采样点数量
        perturb: 扰动强度 (0-1)
        
    Returns:
        pts: 新采样点 [N_rays, n_importance, 3]
        z_vals_combined: 合并后的采样深度 [N_rays, n_samples+n_importance]
    """
    device = rays_o.device
    
    # 获取相邻采样点的中点
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    
    # 归一化权重
    weights = weights[..., 1:-1] + 1e-5  # 防止除零
    weights = weights / torch.sum(weights, dim=-1, keepdim=True)
    
    # 从权重分布中采样
    z_samples = sample_pdf(z_vals_mid, weights, n_importance, perturb=perturb, device=device)
    
    # 合并采样点并排序
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, z_samples], dim=-1), dim=-1)
    
    # 计算新的3D采样点
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
    
    return pts, z_vals_combined

def sample_pdf(bins, weights, N_samples, perturb=0, device=None):
    """
    从PDF中采样
    
    Args:
        bins: 区间边界 [N_rays, M]
        weights: 权重 [N_rays, M]
        N_samples: 采样数量
        perturb: 扰动强度
        device: 设备
        
    Returns:
        samples: 采样结果 [N_rays, N_samples]
    """
    if device is None:
        device = weights.device
    
    # 计算PDF和CDF
    weights = weights + 1e-5  # 防止除零
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [N_rays, M+1]
    
    # 均匀采样
    u = torch.linspace(0., 1., steps=N_samples, device=device)
    u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    
    # 扰动采样
    if perturb > 0:
        u = u + torch.rand_like(u) * (1.0 / N_samples)
        u = torch.clamp(u, 0, 1)
    
    # 反向CDF采样
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds-1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1]-1)
    
    inds_g = torch.stack([below, above], dim=-1)  # [N_rays, N_samples, 2]
    
    # 获取相应的CDF和bin值
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape[:-1] + [bins.shape[-1]]), 
                         dim=-1, index=inds_g)
    
    # 线性插值
    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    
    return samples

