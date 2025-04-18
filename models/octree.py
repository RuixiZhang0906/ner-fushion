import torch
import torch.nn as nn
import torch.nn.functional as F

class OctreeBuilder(nn.Module):
    def __init__(self, config):
        super(OctreeBuilder, self).__init__()
        
        self.max_depth = config.max_depth
        self.min_voxel_size = config.min_voxel_size
        self.feature_dim = config.feature_dim
        
        # 特征编码器
        self.feature_encoder = nn.Sequential(
            nn.Conv2d(config.input_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, self.feature_dim, kernel_size=1)
        )
        
    def forward(self, features, flows=None, consistency=None):
        """
        构建八叉树表示
        
        Args:
            features: 特征列表
            flows: 光流列表（可选）
            consistency: 一致性分数（可选）
        
        Returns:
            octree: 八叉树结构
        """
        # 编码特征
        encoded_features = [self.feature_encoder(feat) for feat in features]
        
        # 初始化八叉树
        octree = Octree(
            max_depth=self.max_depth,
            min_voxel_size=self.min_voxel_size,
            feature_dim=self.feature_dim
        )
        
        # 构建八叉树
        octree.build(encoded_features, flows, consistency)
        
        # 优化树结构
        if hasattr(config, 'optimize_tree') and config.optimize_tree:
            octree.optimize()
        
        return octree
    
    def integrate_flow_information(self, octree, flows, consistency):
        """将光流信息集成到八叉树中"""
        # 遍历所有叶节点
        for node_id, node in octree.nodes.items():
            if not node.is_leaf:
                continue
                
            # 将节点投影到2D特征空间
            projections = self._project_node_to_2d(node)
            
            # 收集该节点对应的光流和一致性信息
            node_flows = []
            node_consistency = []
            
            for proj in projections:
                frame_idx, x, y = proj
                if frame_idx < len(flows):
                    # 获取对应位置的光流
                    flow = flows[frame_idx][:, :, int(y), int(x)]
                    node_flows.append(flow)
                    
                    # 获取对应位置的一致性分数
                    cons = consistency[:, :, int(y), int(x)]
                    node_consistency.append(cons)
            
            if node_flows and node_consistency:
                # 计算平均光流和一致性
                avg_flow = torch.stack(node_flows).mean(dim=0)
                avg_cons = torch.stack(node_consistency).mean(dim=0)
                
                # 将信息存储到节点
                node.flow = avg_flow
                node.consistency = avg_cons
                
        return octree
    
    def _project_node_to_2d(self, node):
        """将3D节点投影到2D特征空间
        
        注意：这个方法需要根据您的相机参数和投影方式进行实际实现
        这里提供一个简化版本
        """
        # 假设我们有一组相机参数
        # 在实际应用中，这些应该从配置或输入中获取
        projections = []
        
        # 这里简化为随机投影点
        # 实际应用中应该使用真实的相机投影矩阵
        for frame_idx in range(3):  # 假设有3帧
            x = torch.randint(0, 64, (1,)).item()  # 假设特征图宽64
            y = torch.randint(0, 64, (1,)).item()  # 假设特征图高64
            projections.append((frame_idx, x, y))
            
        return projections

class Octree:
    def __init__(self, max_depth, min_voxel_size, feature_dim):
        self.max_depth = max_depth
        self.min_voxel_size = min_voxel_size
        self.feature_dim = feature_dim
        self.nodes = {}  # 存储节点信息
        self.root = None
        
        # 添加缓存来加速查询
        self.query_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # 用于统计树的信息
        self.num_leaf_nodes = 0
        self.max_actual_depth = 0
    
    def build(self, features, flows=None, consistency=None):
        """构建八叉树"""
        # 创建根节点
        self.root = OctreeNode(
            center=torch.zeros(3),
            size=2.0,  # 假设场景范围为[-1, 1]^3
            depth=0
        )
        
        # 递归构建八叉树
        self._build_recursive(self.root, features, flows, consistency)
        
        # 计算统计信息
        self._compute_statistics()
        
        print(f"Octree built with {len(self.nodes)} nodes, {self.num_leaf_nodes} leaf nodes, max depth: {self.max_actual_depth}")
    
    def _build_recursive(self, node, features, flows=None, consistency=None):
        """递归构建八叉树"""
        # 如果达到最大深度或体素足够小，则停止
        if node.depth >= self.max_depth or node.size <= self.min_voxel_size:
            # 为叶节点分配特征
            node.is_leaf = True
            node.features = self._compute_node_features(node, features, flows)
            self.nodes[node.id] = node
            return
        
        # 否则继续细分
        children = node.subdivide()
        
        # 递归处理子节点
        for child in children:
            # 检查子节点是否需要进一步细分
            if self._should_subdivide(child, features, flows, consistency):
                self._build_recursive(child, features, flows, consistency)
            else:
                # 不需要细分的节点成为叶节点
                child.is_leaf = True
                child.features = self._compute_node_features(child, features, flows)
                self.nodes[child.id] = child
    
    def _should_subdivide(self, node, features, flows=None, consistency=None):
        """判断节点是否需要继续细分"""
        # 基本条件检查
        if node.depth >= self.max_depth or node.size <= self.min_voxel_size:
            return False
            
        # 获取节点在特征空间的投影
        proj_features = self._project_node_to_features(node, features)
        
        # 计算特征方差
        if proj_features is not None and len(proj_features) > 0:
            feature_var = torch.var(torch.stack(proj_features), dim=0).mean().item()
            # 如果方差大于阈值，说明区域复杂度高，需要细分
            if feature_var > 0.01:  # 阈值可调
                return True
        
        # 如果有光流信息，检查光流梯度
        if flows is not None:
            flow_grad = self._compute_flow_gradient(node, flows)
            if flow_grad > 0.05:  # 阈值可调
                return True
        
        # 如果有一致性信息，检查一致性分数
        if consistency is not None:
            cons_score = self._compute_consistency_score(node, consistency)
            if cons_score < 0.7:  # 阈值可调，低一致性区域需要细分
                return True
                
        # 默认情况
        return node.depth < self.max_depth / 2  # 至少细分到一半深度
    
    def _project_node_to_features(self, node, features):
        """将节点投影到特征空间
        
        注意：这个方法需要根据您的场景设置和相机参数进行实际实现
        这里提供一个简化版本
        """
        # 在实际应用中，应该使用相机参数将3D节点投影到2D特征图
        # 这里简化为随机采样特征
        proj_features = []
        
        # 对每个特征图
        for feat in features:
            B, C, H, W = feat.shape
            
            # 简化：随机采样特征图上的点
            # 实际应用中应该使用投影矩阵
            x = torch.randint(0, W, (1,)).item()
            y = torch.randint(0, H, (1,)).item()
            
            # 获取特征
            sampled_feat = feat[0, :, y, x]  # 假设batch_size=1
            proj_features.append(sampled_feat)
            
        return proj_features
    
    def _compute_flow_gradient(self, node, flows):
        """计算节点对应区域的光流梯度
        
        注意：这个方法需要根据您的场景设置进行实际实现
        这里提供一个简化版本
        """
        # 简化：返回一个随机值
        # 实际应用中应该计算节点投影区域的光流梯度
        return torch.rand(1).item() * 0.1
    
    def _compute_consistency_score(self, node, consistency):
        """计算节点的一致性分数
        
        注意：这个方法需要根据您的场景设置进行实际实现
        这里提供一个简化版本
        """
        # 简化：返回一个随机值
        # 实际应用中应该计算节点投影区域的一致性分数
        return 0.5 + torch.rand(1).item() * 0.5  # 0.5-1.0之间
    
    def _compute_node_features(self, node, features, flows=None):
        """计算节点的特征表示"""
        # 将3D节点投影到每个特征图上
        projected_features = self._project_node_to_features(node, features)
        
        if projected_features and len(projected_features) > 0:
            # 聚合来自不同视图的特征
            node_feature = torch.stack(projected_features).mean(dim=0)
        else:
            # 如果无法投影，使用零向量
            node_feature = torch.zeros(self.feature_dim, device=features[0].device)
            
        return node_feature
    
    def query(self, points):
        """查询点的特征"""
        results = []
        for point in points:
            # 检查缓存
            point_key = tuple(point.tolist())
            if point_key in self.query_cache:
                self.cache_hits += 1
                results.append(self.query_cache[point_key])
            else:
                self.cache_misses += 1
                node = self._find_leaf_node(self.root, point)
                if node:
                    results.append(node.features)
                    # 更新缓存
                    self.query_cache[point_key] = node.features
                else:
                    # 如果找不到对应节点，返回零向量
                    zero_vec = torch.zeros(self.feature_dim, device=points.device)
                    results.append(zero_vec)
                    # 更新缓存
                    self.query_cache[point_key] = zero_vec
                    
        return torch.stack(results)
    
    def _find_leaf_node(self, node, point):
        """找到包含给定点的叶节点"""
        if node.is_leaf:
            return node
        
        # 确定点在哪个子节点中
        octant = 0
        if point[0] > node.center[0]: octant |= 1
        if point[1] > node.center[1]: octant |= 2
        if point[2] > node.center[2]: octant |= 4
        
        # 如果子节点存在，继续搜索
        if octant in node.children:
            return self._find_leaf_node(node.children[octant], point)
        
        return None
    
    def optimize(self):
        """优化树结构，合并相似的节点"""
        print("Optimizing octree structure...")
        num_merged = self._optimize_recursive(self.root)
        print(f"Optimization complete. Merged {num_merged} nodes.")
        
        # 更新统计信息
        self._compute_statistics()
        
    def _optimize_recursive(self, node):
        """递归优化树结构，返回合并的节点数"""
        if node.is_leaf:
            return 0
            
        # 先递归处理子节点
        merged_count = 0
        for child_key in list(node.children.keys()):
            child = node.children[child_key]
            merged_count += self._optimize_recursive(child)
        
        # 检查子节点是否都是叶节点
        all_children_leaves = all(child.is_leaf for child in node.children.values())
        
        if all_children_leaves:
            # 计算子节点特征的方差
            child_features = torch.stack([child.features for child in node.children.values()])
            feature_var = torch.var(child_features, dim=0).mean().item()
            
            # 如果方差小于阈值，合并节点
            if feature_var < 0.005:  # 阈值可调
                # 计算平均特征
                avg_feature = torch.mean(child_features, dim=0)
                
                # 将当前节点转为叶节点
                node.is_leaf = True
                node.features = avg_feature
                
                # 从节点字典中移除子节点
                for child in node.children.values():
                    if child.id in self.nodes:
                        del self.nodes[child.id]
                
                # 清空子节点
                num_children = len(node.children)
                node.children = {}
                
                return merged_count + num_children
        
        return merged_count
    
    def _compute_statistics(self):
        """计算树的统计信息"""
        self.num_leaf_nodes = 0
        self.max_actual_depth = 0
        
        for node in self.nodes.values():
            if node.is_leaf:
                self.num_leaf_nodes += 1
            self.max_actual_depth = max(self.max_actual_depth, node.depth)
    
    def visualize(self, max_depth=None):
        """可视化八叉树结构
        
        需要安装matplotlib：pip install matplotlib
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if max_depth is None:
                max_depth = self.max_depth
            
            self._visualize_node(self.root, ax, max_depth)
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([-1.1, 1.1])
            ax.set_ylim([-1.1, 1.1])
            ax.set_zlim([-1.1, 1.1])
            
            plt.tight_layout()
            plt.title(f'Octree Visualization (max depth shown: {max_depth})')
            plt.show()
        except ImportError:
            print("Visualization requires matplotlib. Please install it with 'pip install matplotlib'")
    
    def _visualize_node(self, node, ax, max_depth):
        """可视化单个节点"""
        if node.depth > max_depth:
            return
            
        if node.is_leaf:
            # 绘制叶节点为立方体
            self._plot_cube(ax, node.center, node.size)
        else:
            # 递归可视化子节点
            for child in node.children.values():
                self._visualize_node(child, ax, max_depth)
    
    def _plot_cube(self, ax, center, size, alpha=0.1):
        """在3D轴上绘制立方体"""
        # 立方体的8个顶点
        half_size = size / 2
        points = []
        for x in [-half_size, half_size]:
            for y in [-half_size, half_size]:
                for z in [-half_size, half_size]:
                    points.append([center[0] + x, center[1] + y, center[2] + z])
        
        # 立方体的6个面
        faces = [
            [0, 1, 3, 2],  # 底面
            [4, 5, 7, 6],  # 顶面
            [0, 1, 5, 4],  # 前面
            [2, 3, 7, 6],  # 后面
            [0, 2, 6, 4],  # 左面
            [1, 3, 7, 5]   # 右面
        ]
        
        # 绘制每个面
        for face in faces:
            xs = [points[i][0] for i in face]
            ys = [points[i][1] for i in face]
            zs = [points[i][2] for i in face]
            
            # 添加第一个点以闭合多边形
            xs.append(xs[0])
            ys.append(ys[0])
            zs.append(zs[0])
            
            ax.plot(xs, ys, zs, 'k-', alpha=0.5)
            
        # 填充面
        import numpy as np
        import matplotlib.tri as mtri
        
        for face in faces:
            xs = np.array([points[i][0] for i in face])
            ys = np.array([points[i][1] for i in face])
            zs = np.array([points[i][2] for i in face])
            
            # 创建三角剖分
            tri = mtri.Triangulation(xs, ys)
            
            # 绘制填充面
            ax.plot_trisurf(xs, ys, zs, triangles=tri.triangles, alpha=alpha)
    
    def save(self, filepath):
        """保存八叉树到文件"""
        data = {
            'max_depth': self.max_depth,
            'min_voxel_size': self.min_voxel_size,
            'feature_dim': self.feature_dim,
            'nodes': {}
        }
        
        # 保存节点信息
        for node_id, node in self.nodes.items():
            data['nodes'][node_id] = {
                'center': node.center.tolist(),
                'size': node.size,
                'depth': node.depth,
                'is_leaf': node.is_leaf,
                'features': node.features.tolist() if node.features is not None else None,
                'children': list(node.children.keys()) if node.children else []
            }
        
        # 保存根节点ID
        data['root_id'] = self.root.id
        
        # 保存到文件
        torch.save(data, filepath)
        print(f"Octree saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """从文件加载八叉树"""
        data = torch.load(filepath)
        
        # 创建八叉树实例
        octree = cls(
            max_depth=data['max_depth'],
            min_voxel_size=data['min_voxel_size'],
            feature_dim=data['feature_dim']
        )
        
        # 创建所有节点
        nodes = {}
        for node_id, node_data in data['nodes'].items():
            node = OctreeNode(
                center=torch.tensor(node_data['center']),
                size=node_data['size'],
                depth=node_data['depth']
            )
            node.id = int(node_id)  # 确保ID是整数
            node.is_leaf = node_data['is_leaf']
            
            if node_data['features'] is not None:
                node.features = torch.tensor(node_data['features'])
                
            nodes[node.id] = node
        
        # 建立节点之间的父子关系
        for node_id, node_data in data['nodes'].items():
            node = nodes[int(node_id)]
            for child_id in node_data['children']:
                node.children[child_id % 8] = nodes[int(child_id)]
        
        # 设置根节点
        octree.root = nodes[data['root_id']]
        octree.nodes = nodes
        
        # 计算统计信息
        octree._compute_statistics()
        
        print(f"Octree loaded from {filepath} with {len(octree.nodes)} nodes")
        return octree

class OctreeNode:
    """八叉树节点"""
    next_id = 0
    
    def __init__(self, center, size, depth):
        self.id = OctreeNode.next_id
        OctreeNode.next_id += 1
        
        self.center = center  # 中心坐标
        self.size = size      # 节点大小
        self.depth = depth    # 深度
        self.is_leaf = False  # 是否为叶节点
        self.features = None  # 节点特征
        self.children = {}    # 子节点
        
        # 额外属性
        self.flow = None      # 光流信息
        self.consistency = None  # 一致性分数
    
    def subdivide(self):
        """将节点细分为8个子节点"""
        half_size = self.size / 2
        quarter_size = half_size / 2
        
        children = []
        for i in range(8):
            # 计算子节点中心
            offset = torch.tensor([
                (i & 1) * half_size - quarter_size,
                ((i >> 1) & 1) * half_size - quarter_size,
                ((i >> 2) & 1) * half_size - quarter_size
            ])
            
            child_center = self.center + offset
            child = OctreeNode(
                center=child_center,
                size=half_size,
                depth=self.depth + 1
            )
            
            self.children[i] = child
            children.append(child)
        
        return children
    
    def get_bounds(self):
        """获取节点的边界框"""
        half_size = self.size / 2
        min_bound = self.center - half_size
        max_bound = self.center + half_size
        return min_bound, max_bound
    
    def contains_point(self, point):
        """检查节点是否包含给定点"""
        min_bound, max_bound = self.get_bounds()
        
        # 检查点是否在边界框内
        for i in range(3):
            if point[i] < min_bound[i] or point[i] > max_bound[i]:
                return False
        
        return True
