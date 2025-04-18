import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class BlenderDataset(Dataset):
    """Blender 数据集加载器，支持标准的 NeRF 合成数据集格式"""
    
    def __init__(self, data_dir, split='train', img_wh=(800, 800), transform=None):
        """
        初始化 Blender 数据集
        
        参数:
            data_dir: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            img_wh: 输出图像的宽高
            transform: 图像预处理变换
        """
        self.data_dir = data_dir
        self.split = split
        self.img_wh = img_wh
        self.transform = transform
        self.white_bg = True  # Blender 数据集使用白色背景
        
        # 加载数据集
        self._load_data()
        
        # 定义像素坐标
        self.define_pixel_coordinates()
        
    def _load_data(self):
        """加载 Blender 格式数据集"""
        # 加载相应的 transforms.json 文件
        transforms_path = os.path.join(self.data_dir, f'transforms_{self.split}.json')
        if not os.path.exists(transforms_path):
            raise ValueError(f"找不到数据文件: {transforms_path}")
        
        with open(transforms_path, 'r') as f:
            self.meta = json.load(f)
            
        # 提取相机参数
        if 'camera_angle_x' in self.meta:
            self.camera_angle_x = float(self.meta['camera_angle_x'])
        else:
            raise ValueError("transforms.json 中缺少 camera_angle_x 字段")
            
        # 计算相机内参
        W, H = self.img_wh
        self.focal = 0.5 * W / np.tan(0.5 * self.camera_angle_x)  # 焦距
        
        # 相机内参矩阵
        self.K = np.array([
            [self.focal, 0, W/2],
            [0, self.focal, H/2],
            [0, 0, 1]
        ])
        
        # 提取所有帧
        self.frames = []
        self.poses = []
        self.images = []
        
        print(f"Loading {self.split} split from {self.data_dir}")
        
        for frame in self.meta['frames']:
            # 帧文件路径
            file_path = frame['file_path']
            if file_path.startswith('./'):
                file_path = file_path[2:]
            
            # 构建完整的图像路径
            image_path = os.path.join(self.data_dir, file_path + '.png')
            if not os.path.exists(image_path):
                # 尝试其他可能的路径格式
                image_path = os.path.join(self.data_dir, f"{file_path}.png")
                if not os.path.exists(image_path):
                    image_path = os.path.join(self.data_dir, file_path)
                    if not os.path.exists(image_path):
                        print(f"警告: 找不到图像 {image_path}")
                        continue
            
            # 加载并预处理图像
            img = Image.open(image_path)
            
            # 调整图像大小
            if img.size != self.img_wh:
                img = img.resize(self.img_wh, Image.LANCZOS)
            
            # 转换为 RGB 格式
            img = img.convert('RGB')
            
            # 如果没有指定变换，执行默认变换
            if self.transform is None:
                img = np.array(img) / 255.0  # 归一化到 [0, 1]
                
                # 处理 alpha 通道（如果有）
                if img.shape[-1] == 4:  # RGBA
                    # 预乘 alpha
                    alpha = img[..., 3:4]
                    img = img[..., :3] * alpha + (1 - alpha) if self.white_bg else img[..., :3] * alpha
            else:
                img = self.transform(img)
            
            self.images.append(img)
            self.frames.append(frame['file_path'])
            
            # 提取相机姿态
            pose = np.array(frame['transform_matrix'])
            self.poses.append(pose)
        
        # 转换为 NumPy 数组
        self.poses = np.stack(self.poses)
        
        # 计算场景边界
        self.near = 2.0  # Blender 默认近平面
        self.far = 6.0   # Blender 默认远平面
        if 'near' in self.meta:
            self.near = self.meta['near']
        if 'far' in self.meta:
            self.far = self.meta['far']
            
        print(f"Loaded {len(self.frames)} frames from {self.split} split")
        print(f"Image dimensions: {self.img_wh}")
        print(f"Focal length: {self.focal}")
        print(f"Near/far planes: {self.near}/{self.far}")
    
    def define_pixel_coordinates(self):
        """定义像素坐标，用于生成光线"""
        W, H = self.img_wh
        
        # 像素坐标网格
        i, j = np.meshgrid(
            np.arange(W, dtype=np.float32),
            np.arange(H, dtype=np.float32),
            indexing='xy'
        )
        
        self.directions = np.stack(
            [(i - self.K[0, 2]) / self.K[0, 0],
             -(j - self.K[1, 2]) / self.K[1, 1],
             -np.ones_like(i)],
            axis=-1
        )  # (H, W, 3)
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        # 加载图像
        img = self.images[idx]
        
        # 如果图像不是张量，转换为张量
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).float()
            
        # 确保图像是 [3, H, W] 格式
        if img.shape[0] == self.img_wh[1] and img.shape[1] == self.img_wh[0]:
            img = img.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
        
        # 获取相机姿态
        pose = self.poses[idx]
        pose = torch.from_numpy(pose).float()
        
        # 生成光线
        rays_o, rays_d = self.get_rays(pose)
        
        # 返回数据字典
        sample = {
            'img_id': idx,
            'pose': pose,  # [4, 4]
            'img': img,    # [3, H, W]
            'rays_o': rays_o,  # [H*W, 3]
            'rays_d': rays_d,  # [H*W, 3]
            'near': self.near,
            'far': self.far,
        }
        
        # 如果是训练集，添加目标RGB值
        if self.split == 'train':
            rgb = img.view(3, -1).permute(1, 0)  # [3, H, W] -> [H*W, 3]
            sample['rgb'] = rgb
        
        return sample
    
    def get_rays(self, c2w):
        """
        根据相机姿态生成光线
        
        参数:
            c2w: 相机到世界坐标系的变换矩阵 [4, 4]
            
        返回:
            rays_o: 光线起点 [H*W, 3]
            rays_d: 光线方向 [H*W, 3]
        """
        # 将方向向量从相机坐标系转换到世界坐标系
        directions = self.directions.reshape(-1, 3)  # [H*W, 3]
        rays_d = np.sum(directions[:, np.newaxis, :] * c2w[:3, :3], axis=-1)  # [H*W, 3]
        
        # 光线起点就是相机中心
        rays_o = np.broadcast_to(c2w[:3, 3], rays_d.shape)  # [H*W, 3]
        
        # 转换为张量
        rays_o = torch.from_numpy(rays_o).float()
        rays_d = torch.from_numpy(rays_d).float()
        
        return rays_o, rays_d
    
    def get_render_poses(self, n_views=40):
        """
        生成用于渲染的相机姿态
        
        参数:
            n_views: 要生成的视图数量
            
        返回:
            render_poses: 渲染姿态列表 [n_views, 4, 4]
        """
        # 计算所有相机位置的中心
        center = self.poses[:, :3, 3].mean(0)
        
        # 计算相机到中心的平均距离
        radius = np.linalg.norm(self.poses[:, :3, 3] - center, axis=-1).mean()
        
        # 生成圆形路径
        render_poses = []
        for th in np.linspace(0, 2*np.pi, n_views, endpoint=False):
            # 在 xz 平面上的圆形路径
            camorigin = np.array([radius * np.cos(th), 0, radius * np.sin(th)]) + center
            
            # 相机始终朝向中心
            lookat = center
            
            # 计算相机坐标系
            forward = (lookat - camorigin) / np.linalg.norm(lookat - camorigin)
            up = np.array([0, 1, 0])  # 假设 y 轴向上
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, forward)
            
            # 构建相机到世界的变换矩阵
            pose = np.stack([right, up, -forward, camorigin], axis=1)
            pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
            
            render_poses.append(torch.from_numpy(pose).float())
        
        return render_poses


class MultiViewBlenderDataset(BlenderDataset):
    """支持多视角输入的 Blender 数据集"""
    
    def __init__(self, data_dir, split='train', img_wh=(800, 800), transform=None, num_views=3):
        """
        初始化多视角 Blender 数据集
        
        参数:
            data_dir: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test')
            img_wh: 输出图像的宽高
            transform: 图像预处理变换
            num_views: 每个样本包含的视图数量
        """
        super().__init__(data_dir, split, img_wh, transform)
        self.num_views = num_views
        
        # 确保数据集足够大
        if len(self.frames) < num_views:
            raise ValueError(f"数据集中的帧数 ({len(self.frames)}) 少于请求的视图数 ({num_views})")
        
        # 创建样本索引
        self.create_samples()
    
    def create_samples(self):
        """创建多视角样本"""
        n_frames = len(self.frames)
        
        # 对于训练集，使用滑动窗口创建样本
        if self.split == 'train':
            self.samples = []
            for i in range(n_frames):
                # 选择当前帧作为中心，周围的帧作为上下文
                context_indices = []
                for j in range(-(self.num_views//2), self.num_views//2 + 1):
                    if j == 0:
                        continue  # 跳过中心帧自身
                    idx = (i + j) % n_frames  # 循环索引
                    context_indices.append(idx)
                
                # 如果上下文帧不足，从其他位置补充
                while len(context_indices) < self.num_views - 1:
                    rand_idx = np.random.randint(0, n_frames)
                    if rand_idx != i and rand_idx not in context_indices:
                        context_indices.append(rand_idx)
                
                # 中心帧放在第一位
                self.samples.append([i] + context_indices)
        else:
            # 对于验证和测试集，每个样本都是单独的
            self.samples = [[i] + [(i + j) % n_frames for j in range(1, self.num_views)]
                           for i in range(n_frames)]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取多视角数据样本"""
        frame_indices = self.samples[idx]
        
        # 中心帧
        center_idx = frame_indices[0]
        center_sample = super().__getitem__(center_idx)
        
        # 上下文帧
        context_imgs = []
        context_poses = []
        
        for i in frame_indices[1:]:
            context_sample = super().__getitem__(i)
            context_imgs.append(context_sample['img'])
            context_poses.append(context_sample['pose'])
        
        # 堆叠上下文帧
        if context_imgs:
            context_imgs = torch.stack(context_imgs)  # [N-1, 3, H, W]
            context_poses = torch.stack(context_poses)  # [N-1, 4, 4]
        
        # 构建返回字典
        result = {
            'img_id': center_idx,
            'pose': center_sample['pose'],  # [4, 4]
            'img': center_sample['img'],    # [3, H, W]
            'rays_o': center_sample['rays_o'],  # [H*W, 3]
            'rays_d': center_sample['rays_d'],  # [H*W, 3]
            'near': center_sample['near'],
            'far': center_sample['far'],
            'context_imgs': context_imgs,      # [N-1, 3, H, W]
            'context_poses': context_poses,    # [N-1, 4, 4]
        }
        
        # 如果是训练集，添加目标RGB值
        if self.split == 'train' and 'rgb' in center_sample:
            result['rgb'] = center_sample['rgb']
        
        return result
