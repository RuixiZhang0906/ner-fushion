import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class MultiFrameDataset(Dataset):
    def __init__(self, data_dir, frame_window=3, transform=None, split='train'):
        """
        多帧输入数据集
        
        Args:
            data_dir: 数据目录
            frame_window: 每个样本的帧窗口大小
            transform: 数据变换
            split: 数据集划分 ('train', 'val', 'test')
        """
        self.data_dir = data_dir
        self.frame_window = frame_window
        self.transform = transform
        self.split = split
        
        # 加载数据集信息
        self.metadata = self._load_metadata()
        
        # 获取当前划分的帧列表
        self.frames = self.metadata['frames'][split]
        
        # 相机内参
        self.K = np.array(self.metadata['camera']['K']).reshape(3, 3)
        
        # 生成样本索引
        self.samples = self._generate_samples()
        
    def _load_metadata(self):
        """加载数据集元数据"""
        metadata_path = os.path.join(self.data_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        else:
            # 如果没有元数据文件，尝试自动生成
            return self._generate_metadata()
    
    def _generate_metadata(self):
        """生成数据集元数据"""
        metadata = {'frames': {}, 'camera': {}}
        
        # 查找所有图像文件
        image_dir = os.path.join(self.data_dir, 'images')
        if not os.path.exists(image_dir):
            raise ValueError(f"Image directory not found: {image_dir}")
        
        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # 划分训练、验证和测试集
        n_images = len(image_files)
        n_train = int(n_images * 0.8)
        n_val = int(n_images * 0.1)
        
        metadata['frames']['train'] = image_files[:n_train]
        metadata['frames']['val'] = image_files[n_train:n_train+n_val]
        metadata['frames']['test'] = image_files[n_train+n_val:]
        
        # 尝试加载相机参数
        camera_path = os.path.join(self.data_dir, 'camera.json')
        if os.path.exists(camera_path):
            with open(camera_path, 'r') as f:
                metadata['camera'] = json.load(f)
        else:
            # 使用默认相机参数
            # 打开第一张图像获取尺寸
            img_path = os.path.join(image_dir, image_files[0])
            with Image.open(img_path) as img:
                W, H = img.size
            
            # 默认相机内参
            focal = max(H, W)
            metadata['camera']['K'] = [
                focal, 0, W/2,
                0, focal, H/2,
                0, 0, 1
            ]
            metadata['camera']['width'] = W
            metadata['camera']['height'] = H
        
        # 保存元数据
        with open(os.path.join(self.data_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return metadata
    
    def _generate_samples(self):
        """生成数据样本"""
        samples = []
        
        # 对于每个可能的起始帧
        for i in range(len(self.frames) - self.frame_window + 1):
            # 创建一个包含frame_window帧的样本
            frame_indices = list(range(i, i + self.frame_window))
            samples.append(frame_indices)
        
        return samples
    
    def __len__(self):
        """数据集长度"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取数据样本"""
        frame_indices = self.samples[idx]
        
        # 加载帧
        frames = []
        poses = []
        for i in frame_indices:
            # 加载图像
            frame_path = os.path.join(self.data_dir, 'images', self.frames[i])
            frame = Image.open(frame_path).convert('RGB')
            
            # 应用变换
            if self.transform:
                frame = self.transform(frame)
            else:
                frame = np.array(frame) / 255.0
                frame = torch.from_numpy(frame).float().permute(2, 0, 1)
            
            frames.append(frame)
            
            # 加载相机姿态
            pose_path = os.path.join(self.data_dir, 'poses', self.frames[i].replace('.png', '.txt').replace('.jpg', '.txt'))
            if os.path.exists(pose_path):
                pose = np.loadtxt(pose_path).reshape(4, 4)
            else:
                # 如果没有姿态文件，使用单位矩阵
                pose = np.eye(4)
            
            poses.append(torch.from_numpy(pose).float())
        
        # 堆叠帧
        frames = torch.stack(frames)
        poses = torch.stack(poses)
        
        # 计算光线
        H, W = self.metadata['camera']['height'], self.metadata['camera']['width']
        K = torch.from_numpy(self.K).float()
        
        # 中心帧的光线
        center_idx = len(frames) // 2
        rays_o, rays_d = self._get_rays(H, W, K, poses[center_idx])
        
        # 目标RGB值
        target_rgb = frames[center_idx].view(3, -1).permute(1, 0)  # [H*W, 3]
        
        # 时间戳
        timestamps = torch.linspace(0, 1, len(frames))
        
        return {
            'frames': frames,  # [T, 3, H, W]
            'poses': poses,    # [T, 4, 4]
            'rays_o': rays_o,  # [H*W, 3]
            'rays_d': rays_d,  # [H*W, 3]
            'rgb': target_rgb, # [H*W, 3]
            'time': timestamps[center_idx],  # 中心帧时间戳
            'timestamps': timestamps,        # 所有时间戳
            'H': H,
            'W': W
        }
    
    def _get_rays(self, H, W, K, c2w):
        """
        计算光线起点和方向
        
        Args:
            H, W: 图像高度和宽度
            K: 相机内参
            c2w: 相机到世界的变换矩阵
            
        Returns:
            rays_o: 光线起点
            rays_d: 光线方向
        """
        i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))
        i = i.t()
        j = j.t()
        
        # 像素坐标转相机坐标
        dirs = torch.stack([(i - K[0, 2]) / K[0, 0], 
                           -(j - K[1, 2]) / K[1, 1], 
                           -torch.ones_like(i)], -1)
        
        # 相机坐标转世界坐标
        rays_d = dirs @ c2w[:3, :3].t()
        rays_o = c2w[:3, 3].expand(rays_d.shape)
        
        # 展平
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        
        return rays_o, rays_d
    
    def get_eval_dataset(self):
        """获取评估数据集"""
        return MultiFrameDataset(
            data_dir=self.data_dir,
            frame_window=self.frame_window,
            transform=self.transform,
            split='val'
        )
    
    def get_render_poses(self):
        """获取渲染路径"""
        # 如果有预定义的渲染路径，加载它
        render_path = os.path.join(self.data_dir, 'render_path.json')
        if os.path.exists(render_path):
            with open(render_path, 'r') as f:
                render_data = json.load(f)
                return [torch.tensor(pose).float() for pose in render_data['poses']]
        
        # 否则，生成一个圆形路径
        if len(self.frames) > 0:
            # 使用训练姿态的中心作为参考
            poses = []
            for i in range(min(20, len(self.frames))):
                pose_path = os.path.join(self.data_dir, 'poses', self.frames[i].replace('.png', '.txt').replace('.jpg', '.txt'))
                if os.path.exists(pose_path):
                    pose = np.loadtxt(pose_path).reshape(4, 4)
                    poses.append(pose)
            
            if poses:
                # 计算中心位置
                center = np.mean([p[:3, 3] for p in poses], axis=0)
                
                # 计算平均距离
                radius = np.mean([np.linalg.norm(p[:3, 3] - center) for p in poses])
                
                # 生成圆形路径
                render_poses = []
                for th in np.linspace(0, 2*np.pi, 40, endpoint=False):
                    # 圆形路径上的位置
                    pos = center + radius * np.array([np.cos(th), 0, np.sin(th)])
                    
                    # 看向中心的旋转
                    z = (center - pos) / np.linalg.norm(center - pos)
                    y = np.array([0, 1, 0])
                    x = np.cross(y, z)
                    x = x / np.linalg.norm(x)
                    y = np.cross(z, x)
                    
                    pose = np.stack([x, y, z, pos], axis=1)
                    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]])], axis=0)
                    render_poses.append(torch.from_numpy(pose).float())
                
                return render_poses
        
        # 如果没有参考姿态，使用默认路径
        render_poses = []
        for th in np.linspace(0, 2*np.pi, 40, endpoint=False):
            pose = np.eye(4)
            pose[:3, :3] = np.array([
                [np.cos(th), 0, -np.sin(th)],
                [0, 1, 0],
                [np.sin(th), 0, np.cos(th)]
            ])
            pose[:3, 3] = np.array([2 * np.cos(th), 0, 2 * np.sin(th)])
            render_poses.append(torch.from_numpy(pose).float())
        
        return render_poses
