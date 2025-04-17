# models/octree.py
import torch
import torch.nn as nn

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
        
    def forward(self, features, flows=None):
        """
        构建八叉树表示
        
        Args:
            features: 特征列表
            flows: 光流列表（可选）
        
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
        octree.build(encoded_features, flows)
        
        return octree

class Octree:
    def __init__(self, max_depth, min_voxel_size, feature_dim):
        self.max_depth = max_depth
        self.min_voxel_size = min_voxel_size
        self.feature_dim = feature_dim
        self.nodes = {}  # 存储节点信息
        self.root = None
    
    def build(self, features, flows=None):
        """构建八叉树"""
        # 创建根节点
        self.root = OctreeNode(
            center=torch.zeros(3),
            size=2.0,  # 假设场景范围为[-1, 1]^3
            depth=0
        )
        
        # 递归构建八叉树
        self._build_recursive(self.root, features, flows)
    
    def _build_recursive(self, node, features, flows):
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
            if self._should_subdivide(child, features, flows):
                self._build_recursive(child, features, flows)
            else:
                # 不需要细分的节点成为叶节点
                child.is_leaf = True
                child.features = self._compute_node_features(child, features, flows)
                self.nodes[child.id] = child
    
    def _should_subdivide(self, node, features, flows):
        """判断节点是否需要继续细分"""
        # 这里可以基于特征方差、光流梯度等决定是否细分
        # 为简化，这里仅考虑深度
        return node.depth < self.max_depth
    
    def _compute_node_features(self, node, features, flows):
        """计算节点的特征表示"""
        # 实际实现中，应该根据节点在3D空间中的位置投影到2D特征上
        # 这里简化为随机特征
        return torch.randn(self.feature_dim)
    
    def query(self, points):
        """查询点的特征"""
        results = []
        for point in points:
            node = self._find_leaf_node(self.root, point)
            if node:
                results.append(node.features)
            else:
                # 如果找不到对应节点，返回零向量
                results.append(torch.zeros(self.feature_dim))
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
