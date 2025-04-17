import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from optimization.loss import compute_loss
from rendering.pipeline import RenderingPipeline
from utils.metrics import compute_metrics
from utils.visualization import visualize_results

class Trainer:
    def __init__(self, model, dataset, config, output_dir, resource_monitor=None):
        """
        初始化训练器
        
        Args:
            model: NeRF模型
            dataset: 数据集
            config: 训练配置
            output_dir: 输出目录
            resource_monitor: 资源监控器
        """
        self.model = model
        self.dataset = dataset
        self.config = config
        self.output_dir = output_dir
        self.resource_monitor = resource_monitor
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
        
        # 初始化优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=config.lr_decay_rate
        )
        
        # 渲染管线
        self.rendering_pipeline = RenderingPipeline(model, config.rendering)
        
        # 训练状态
        self.start_epoch = 0
        self.global_step = 0
        
        # 加载检查点（如果有）
        if config.resume and os.path.exists(config.resume):
            self._load_checkpoint(config.resume)
            
    def train(self):
        """训练模型"""
        print(f"Starting training from epoch {self.start_epoch}...")
        
        # 创建数据加载器
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )
        
        # 训练循环
        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            # 每个批次
            for batch_idx, batch in enumerate(train_loader):
                # 移动数据到设备
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.model.device)
                
                # 前向传播
                outputs = self.model(batch)
                
                # 计算损失
                loss_dict = compute_loss(outputs, batch, self.config.loss)
                loss = loss_dict['total_loss']
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 更新状态
                epoch_loss += loss.item()
                self.global_step += 1
                
                # 监控资源使用
                if self.resource_monitor and batch_idx % self.config.resource_monitor_freq == 0:
                    resource_info = self.resource_monitor.get_stats()
                    self._adjust_resources(resource_info)
                
                # 记录进度
                if batch_idx % self.config.log_freq == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Time: {time.time() - epoch_start_time:.2f}s")
                    
                    # 可视化中间结果
                    if batch_idx % self.config.vis_freq == 0:
                        self._visualize_batch(batch, outputs, epoch, batch_idx)
            
            # 更新学习率
            self.scheduler.step()
            
            # 保存检查点
            if epoch % self.config.save_freq == 0:
                self._save_checkpoint(epoch)
            
            # 评估
            if epoch % self.config.eval_freq == 0:
                self.evaluate()
                
            print(f"Epoch {epoch} completed, Avg Loss: {epoch_loss / len(train_loader):.4f}, "
                  f"Time: {time.time() - epoch_start_time:.2f}s")
    
    def evaluate(self):
        """评估模型"""
        print("Evaluating model...")
        
        # 切换到评估模式
        self.model.eval()
        
        # 创建评估数据加载器
        eval_dataset = self.dataset.get_eval_dataset()
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        # 评估指标
        metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'lpips': 0.0
        }
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(eval_loader):
                # 移动数据到设备
                for k in batch:
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.model.device)
                
                # 渲染
                camera_params = batch['camera_params']
                time_value = batch.get('time', None)
                rendered = self.rendering_pipeline.render(camera_params, time_value)
                
                # 计算指标
                gt_rgb = batch['rgb']
                batch_metrics = compute_metrics(rendered['rgb'], gt_rgb)
                
                # 更新指标
                for k in metrics:
                    metrics[k] += batch_metrics[k]
                
                # 可视化评估结果
                if batch_idx % self.config.eval_vis_freq == 0:
                    self._visualize_eval(batch, rendered, batch_idx)
        
        # 平均指标
        for k in metrics:
            metrics[k] /= len(eval_loader)
            
        print(f"Evaluation metrics: PSNR: {metrics['psnr']:.2f}, "
              f"SSIM: {metrics['ssim']:.4f}, "
              f"LPIPS: {metrics['lpips']:.4f}")
        
        # 切换回训练模式
        self.model.train()
        
        return metrics
    
    def render(self):
        """渲染新视角"""
        print("Rendering novel views...")
        
        # 切换到评估模式
        self.model.eval()
        
        # 加载渲染路径
        render_poses = self.dataset.get_render_poses()
        
        # 渲染每个视角
        rendered_frames = []
        with torch.no_grad():
            for i, pose in enumerate(render_poses):
                print(f"Rendering view {i+1}/{len(render_poses)}")
                
                # 渲染
                time_value = torch.tensor([i / len(render_poses)]).to(self.model.device) if self.config.dynamic else None
                rendered = self.rendering_pipeline.render(pose, time_value)
                
                # 保存渲染结果
                rendered_frames.append(rendered['rgb'].cpu())
                
                # 可视化
                self._save_rendered_image(rendered['rgb'], i)
        
        # 保存渲染视频
        self._save_rendered_video(rendered_frames)
        
        # 切换回训练模式
        self.model.train()
    
    def _adjust_resources(self, resource_info):
        """根据资源使用情况调整参数"""
        # 如果GPU内存使用率过高，减少采样点数量
        if resource_info['gpu_memory_percent'] > self.config.gpu_memory_threshold:
            self.model.adaptive_sampler.n_samples = max(
                self.model.adaptive_sampler.n_samples - self.config.sample_reduction,
                self.config.min_samples
            )
            print(f"Reduced sampling points to {self.model.adaptive_sampler.n_samples} due to high GPU usage")
    
    def _save_checkpoint(self, epoch):
        """保存检查点"""
        checkpoint_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        self.start_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    def _visualize_batch(self, batch, outputs, epoch, batch_idx):
        """可视化批次结果"""
        vis_path = os.path.join(
            self.output_dir, 'visualizations', 
            f'epoch_{epoch}_batch_{batch_idx}.png'
        )
        visualize_results(batch, outputs, vis_path)
    
    def _visualize_eval(self, batch, rendered, batch_idx):
        """可视化评估结果"""
        vis_path = os.path.join(
            self.output_dir, 'visualizations', 
            f'eval_batch_{batch_idx}.png'
        )
        visualize_results(batch, rendered, vis_path, is_eval=True)
    
    def _save_rendered_image(self, image, idx):
        """保存渲染图像"""
        save_path = os.path.join(
            self.output_dir, 'visualizations', 
            f'render_{idx:03d}.png'
        )
        visualize_results(None, {'rgb': image}, save_path, is_render=True)
    
    def _save_rendered_video(self, frames):
        """保存渲染视频"""
        # 实现视频保存逻辑
        pass
