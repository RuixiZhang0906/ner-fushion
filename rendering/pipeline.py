import torch
import torch.nn as nn
from rendering.ray_sampling import generate_rays
from rendering.volume_rendering import volume_render

class RenderingPipeline:
    def __init__(self, model, config):
        """
        初始化渲染管线
        
        Args:
            model: NeRF模型
            config: 渲染配置
        """
        self.model = model
        self.config = config
        self.chunk_size = config.chunk_size
        
    def render(self, camera_params, time=None, resolution=None):
        """
        渲染给定相机参数的视图
        
        Args:
            camera_params: 相机参数 (位置, 方向, 内参)
            time: 时间戳 (用于动态场景)
            resolution: 渲染分辨率 (H, W)
            
        Returns:
            rendered: 渲染结果 (RGB, 深度等)
        """
        if resolution is None:
            H, W = self.config.height, self.config.width
        else:
            H, W = resolution
            
        # 生成光线
        rays_o, rays_d = generate_rays(camera_params, H, W)
        
        # 分块渲染
        all_rgb, all_depth = [], []
        for i in range(0, rays_o.shape[0], self.chunk_size):
            chunk_rays_o = rays_o[i:i+self.chunk_size]
            chunk_rays_d = rays_d[i:i+self.chunk_size]
            
            # 渲染当前块
            if time is not None:
                chunk_time = time.expand(chunk_rays_o.shape[0])
                chunk_rendered = self._render_chunk(chunk_rays_o, chunk_rays_d, chunk_time)
            else:
                chunk_rendered = self._render_chunk(chunk_rays_o, chunk_rays_d)
                
            all_rgb.append(chunk_rendered['rgb'])
            all_depth.append(chunk_rendered['depth'])
            
        # 合并结果
        rgb = torch.cat(all_rgb, 0).reshape(H, W, 3)
        depth = torch.cat(all_depth, 0).reshape(H, W)
        
        return {
            'rgb': rgb,
            'depth': depth
        }
    
    def _render_chunk(self, rays_o, rays_d, time=None):
        """渲染一个光线块"""
        # 准备输入
        inputs = {
            'rays': (rays_o, rays_d)
        }
        
        if time is not None:
            inputs['time'] = time
            
        # 前向传播
        with torch.no_grad():
            outputs = self.model(inputs)
            
        return {
            'rgb': outputs['rgb'],
            'depth': outputs['depth']
        }
