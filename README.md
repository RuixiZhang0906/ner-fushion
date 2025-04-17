## NeuFusion: A Unified and Efficient Neural Radiance Field Framework for Static and Dynamic Scenes
NeuFusion: 
This is my bachelor degree graduation design.

### 项目结构
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
### 功能与实现介绍
1. 多帧输入与特征提取对齐：通过MultiFrameDataset和FeatureExtractor实现。
2. 光流估计与时空一致性分析：通过OpticalFlowEstimator和ConsistencyAnalyzer实现。
3. 八叉树构建与动态场景解耦：通过OctreeBuilder和SceneDecomposer实现。
4. 梯度分析与自适应采样：通过AdaptiveSampler和各种采样策略实现。
5. 自适应渲染管线：通过RenderingPipeline和体渲染函数实现。
6. 损失计算与参数更新：通过compute_loss和Trainer实现。
7. 资源调度与反馈优化：通过ResourceMonitor实现。

### 算法与实现介绍
#### 1. 多帧输入
#### 2. 动态场景/静态场景
#### 3. 八叉树构建
#### 4. 自适应渲染管线
#### 5. 光流估计与时空一致性分析