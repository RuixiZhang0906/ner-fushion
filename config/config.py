import os
import yaml
import argparse

def get_config(config_path):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        config: 配置对象
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 转换为Config对象
    return Config(config)

class Config:
    """配置类，支持嵌套访问"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def to_dict(self):
        """将配置转换为字典"""
        result = {}
        for key in dir(self):
            if not key.startswith('__') and not callable(getattr(self, key)):
                value = getattr(self, key)
                if isinstance(value, Config):
                    result[key] = value.to_dict()
                else:
                    result[key] = value
        return result

def create_default_config():
    """创建默认配置"""
    config = {
        'model': {
            'feature_extractor': {
                'input_channels': 3,
                'feature_dim': 64,
                'alignment': {
                    'feature_dim': 64,
                    'num_heads': 4,
                    'dropout': 0.1
                }
            },
            'optical_flow': {
                'input_dim': 128,
                'consistency': {
                    'input_dim': 256
                }
            },
            'octree': {
                'max_depth': 8,
                'min_voxel_size': 0.01,
                'feature_dim': 32
            },
            'scene_decomposer': {
                'input_dim': 65,  # feature_dim + 1 (consistency)
                'feature_dim': 64,
                'output_dim': 32
            },
            'adaptive_sampler': {
                'n_samples': 64,
                'n_importance': 64,
                'perturb': 1.0,
                'gradient_based': True,
                'input_dim': 6,  # position (3) + direction (3)
                'near': 0.1,
                'far': 10.0
            },
            'static_mlp': {
                'input_dim': 6,  # position (3) + direction (3)
                'hidden_dims': [256, 256, 256, 256],
                'output_dim': 4  # RGB (3) + sigma (1)
            },
            'dynamic_mlp': {
                'input_dim': 7,  # position (3) + direction (3) + time (1)
                'hidden_dims': [256, 256, 256, 256],
                'output_dim': 5  # RGB (3) + sigma (1) + blend weight (1)
            }
        },
        'data': {
            'frame_window': 3,
            'transform': None
        },
        'training': {
            'batch_size': 1024,
            'num_epochs': 300,
            'learning_rate': 5e-4,
            'lr_decay_rate': 0.1,
            'lr_decay_steps': 50000,
            'resume': None,
            'num_workers': 4,
            'log_freq': 10,
            'vis_freq': 100,
            'save_freq': 10,
            'eval_freq': 10,
            'resource_monitor_freq': 50,
            'gpu_memory_threshold': 90,
            'sample_reduction': 16,
            'min_samples': 32,
            'loss': {
                'rgb_loss_type': 'l1',
                'rgb_weight': 1.0,
                'consistency_weight': 0.1,
                'depth_weight': 0.05,
                'use_consistency_loss': True,
                'use_depth_loss': True,
                'huber_delta': 0.1
            }
        },
        'rendering': {
            'chunk_size': 4096,
            'height': 400,
            'width': 400
        },
        'resource': {
            'monitor_interval': 2.0,
            'print_stats': False,
            'gpu_memory_high_threshold': 85,
            'gpu_util_low_threshold': 30,
            'cpu_high_threshold': 80
        },
        'eval': {
            'eval_vis_freq': 5
        }
    }
    
    return config

def save_default_config(path='config/default.yaml'):
    """保存默认配置到文件"""
    config = create_default_config()
    
    # 确保目录存在
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Default config saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='配置文件工具')
    parser.add_argument('--create-default', action='store_true', help='创建默认配置文件')
    parser.add_argument('--output', type=str, default='config/default.yaml', help='输出配置文件路径')
    
    args = parser.parse_args()
    
    if args.create_default:
        save_default_config(args.output)
