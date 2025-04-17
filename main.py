import argparse
import torch
from config.config import get_config
from data.dataset import MultiFrameDataset
from models.nerf_model import FusionNeRF
from optimization.training import Trainer
from utils.resource_monitor import ResourceMonitor

def parse_args():
    parser = argparse.ArgumentParser(description='NeRF融合方案')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='配置文件路径')
    parser.add_argument('--data_dir', type=str, required=True, help='数据集目录')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'render'], default='train', help='运行模式')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config(args.config)
    
    # 初始化资源监控器
    resource_monitor = ResourceMonitor(config.resource)
    
    # 加载数据集
    dataset = MultiFrameDataset(
        data_dir=args.data_dir,
        frame_window=config.data.frame_window,
        transform=config.data.transform
    )
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FusionNeRF(config.model).to(device)
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        dataset=dataset,
        config=config.training,
        output_dir=args.output_dir,
        resource_monitor=resource_monitor
    )
    
    # 根据模式执行不同操作
    if args.mode == 'train':
        trainer.train()
    elif args.mode == 'eval':
        trainer.evaluate()
    elif args.mode == 'render':
        trainer.render()

if __name__ == '__main__':
    main()
