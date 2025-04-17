import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_results(batch, outputs, save_path, is_eval=False, is_render=False):
    """
    可视化结果
    
    Args:
        batch: 输入批次
        outputs: 模型输出
        save_path: 保存路径
        is_eval: 是否为评估模式
        is_render: 是否为渲染模式
    """
    plt.figure(figsize=(15, 10))
    
    if is_render:
        # 仅渲染输出
        rgb = outputs['rgb']
        if isinstance(rgb, torch.Tensor):
            rgb = rgb.detach().cpu().numpy()
        
        plt.imshow(np.clip(rgb, 0, 1))
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        return
    
    # 训练或评估模式
    if is_eval:
        # 评估模式：对比预测和真实值
        plt.subplot(1, 3, 1)
        gt_rgb = batch['rgb']
        if isinstance(gt_rgb, torch.Tensor):
            gt_rgb = gt_rgb.detach().cpu().numpy()
        plt.imshow(np.clip(gt_rgb, 0, 1))
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        pred_rgb = outputs['rgb']
        if isinstance(pred_rgb, torch.Tensor):
            pred_rgb = pred_rgb.detach().cpu().numpy()
        plt.imshow(np.clip(pred_rgb, 0, 1))
        plt.title('Prediction')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        error = np.abs(gt_rgb - pred_rgb)
        plt.imshow(error, cmap='hot', vmin=0, vmax=0.5)
        plt.title('Error')
        plt.axis('off')
        
        plt.colorbar(plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(0, 0.5)), 
                     ax=plt.gca())
    else:
        # 训练模式：显示更多调试信息
        plt.subplot(2, 3, 1)
        gt_rgb = batch['rgb']
        if isinstance(gt_rgb, torch.Tensor):
            gt_rgb = gt_rgb.detach().cpu().numpy()
        plt.imshow(np.clip(gt_rgb, 0, 1))
        plt.title('Ground Truth')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        pred_rgb = outputs['rgb']
        if isinstance(pred_rgb, torch.Tensor):
            pred_rgb = pred_rgb.detach().cpu().numpy()
        plt.imshow(np.clip(pred_rgb, 0, 1))
        plt.title('Prediction')
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        error = np.abs(gt_rgb - pred_rgb)
        plt.imshow(error, cmap='hot', vmin=0, vmax=0.5)
        plt.title('Error')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        depth = outputs['depth']
        if isinstance(depth, torch.Tensor):
            depth = depth.detach().cpu().numpy()
        plt.imshow(depth, cmap='viridis')
        plt.title('Depth')
        plt.axis('off')
        plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=plt.gca())
        
        plt.subplot(2, 3, 5)
        if 'consistency' in outputs:
            consistency = outputs['consistency']
            if isinstance(consistency, torch.Tensor):
                consistency = consistency.detach().cpu().numpy()
                if len(consistency.shape) > 2:
                    consistency = consistency.squeeze()
            plt.imshow(consistency, cmap='coolwarm', vmin=0, vmax=1)
            plt.title('Consistency')
            plt.axis('off')
            plt.colorbar(plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(0, 1)), 
                         ax=plt.gca())
        else:
            plt.text(0.5, 0.5, 'Consistency not available', 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        plt.subplot(2, 3, 6)
        if 'weights' in outputs:
            weights = outputs['weights']
            if isinstance(weights, torch.Tensor):
                weights = weights.detach().cpu().numpy()
                if len(weights.shape) > 2:
                    weights = weights.mean(axis=-1)  # 平均权重
            plt.imshow(weights, cmap='plasma')
            plt.title('Sampling Weights')
            plt.axis('off')
            plt.colorbar(plt.cm.ScalarMappable(cmap='plasma'), ax=plt.gca())
        else:
            plt.text(0.5, 0.5, 'Weights not available', 
                     horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_octree(octree, save_path):
    """
    可视化八叉树结构
    
    Args:
        octree: 八叉树对象
        save_path: 保存路径
    """
    try:
        import open3d as o3d
        
        # 创建点云
        points = []
        colors = []
        
        # 收集所有叶节点
        for node_id, node in octree.nodes.items():
            if node.is_leaf:
                # 添加节点中心点
                points.append(node.center.numpy())
                
                # 使用深度作为颜色
                depth_color = [node.depth / octree.max_depth, 
                              1.0 - node.depth / octree.max_depth, 
                              0.5]
                colors.append(depth_color)
        
        # 创建Open3D点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        # 可视化
        o3d.visualization.draw_geometries([pcd])
        
        # 保存图像
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()
        
    except ImportError:
        print("Open3D not available for octree visualization")
        
        # 使用matplotlib进行简单2D投影
        plt.figure(figsize=(10, 10))
        
        # 收集所有叶节点
        xs, ys, sizes, depths = [], [], [], []
        for node_id, node in octree.nodes.items():
            if node.is_leaf:
                center = node.center.numpy()
                xs.append(center[0])
                ys.append(center[1])
                sizes.append(node.size * 100)  # 缩放以便可视化
                depths.append(node.depth)
        
        plt.scatter(xs, ys, s=sizes, c=depths, cmap='viridis', alpha=0.6)
        plt.colorbar(label='Depth')
        plt.title('Octree Leaf Nodes (2D Projection)')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
