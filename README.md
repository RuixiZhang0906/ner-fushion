## NeuFusion: A Unified and Efficient Neural Radiance Field Framework for Static and Dynamic Scenes
**This is my bachelor degree final project.**

### Project Archetecture
``` python
# 项目结构
# nerf_fusion/
# ├── config/
# │   └── config.py              # 配置文件
# ├── data/
# │   ├── dataset.py             # 数据集加载
# │   └── preprocessing.py       # 数据预处理
# ├── models/
# │   ├── feature_extraction.py  # 特征提取与对齐模块
# │   ├── optical_flow.py        # 光流估计模块
# │   ├── octree.py              # 八叉树构建
# │   ├── scene_decomposition.py # 动态场景解耦
# │   ├── nerf_model.py          # NeRF核心模型
# │   └── adaptive_sampling.py   # 自适应采样策略
# ├── rendering/
# │   ├── ray_sampling.py        # 光线采样
# │   ├── volume_rendering.py    # 体渲染
# │   └── pipeline.py            # 渲染管线
# ├── optimization/
# │   ├── loss.py                # 损失函数
# │   ├── training.py            # 训练逻辑
# │   └── scheduler.py           # 资源调度
# ├── utils/
# │   ├── metrics.py             # 评估指标
# │   ├── visualization.py       # 可视化工具
# │   └── resource_monitor.py    # 资源监控
# └── main.py                    # 主入口
```
