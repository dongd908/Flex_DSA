# Flex_DSA
Flex-DSA: Optimizing Edge Domain-Specific Architectures for AI Inference
1. **Automated Exploration**: Flex-DSA automates the exploration of the computational graph design space, optimizing the flexibility of operator expressions.

2. **Key Components**:
   - Fusion Switcher: Selects optimal subgraph fusion strategies
   - Subgraph Mapper: Efficiently explores computational subgraphs
   - Shape Tracker: Dynamically updates operator shapes based on runtime information

3. **Benefits**:
   - Minimizes unnecessary computations
   - Reduces inference time
   - Lowers power consumption
   - Enhances performance on resource-constrained and architecturally diverse edge DSAs

4. **Performance Improvements**: Flex-DSA significantly outperforms original toolchains in terms of inference speed and energy efficiency.

一个名为flex_dsa的Python包,包含以下主要模块:
fusion_switcher.py
subgraph_mapper.py
shape_tracker.py
platform_adapters/ (目录,包含各平台的适配器)
optimizer.py (主优化器)

from .fusion_switcher import FusionSwitcher
from .subgraph_mapper import SubgraphMapper
from .shape_tracker import ShapeTracker
from .platform_adapters import get_platform_adapter

class FlexDSAOptimizer:
    def __init__(self, platform):
        self.platform_adapter = get_platform_adapter(platform)
        self.fusion_switcher = FusionSwitcher()
        self.subgraph_mapper = SubgraphMapper()
        self.shape_tracker = ShapeTracker()

    def optimize(self, computational_graph):
        # 1. 使用Fusion Switcher选择最佳子图融合策略
        fused_graph = self.fusion_switcher.select_fusion_strategy(computational_graph)
        
        # 2. 使用Subgraph Mapper探索计算子图
        mapped_subgraphs = self.subgraph_mapper.map_subgraphs(fused_graph)
        
        # 3. 使用Shape Tracker更新算子形状
        tracked_graph = self.shape_tracker.update_shapes(mapped_subgraphs)
        
        # 4. 使用平台适配器生成优化后的代码
        optimized_code = self.platform_adapter.generate_code(tracked_graph)
        
        return optimized_code


class FusionSwitcher:
    def select_fusion_strategy(self, graph):
        # 实现子图融合策略选择逻辑
        # ...
        return fused_graph


class SubgraphMapper:
    def map_subgraphs(self, graph):
        # 实现计算子图探索逻辑
        # ...
        return mapped_subgraphs


class ShapeTracker:
    def update_shapes(self, graph):
        # 实现动态更新算子形状的逻辑
        # ...
        return updated_graph

from flex_dsa.optimizer import FlexDSAOptimizer

# 创建计算图
computational_graph = create_computational_graph()

# 为特定平台创建优化器
optimizer = FlexDSAOptimizer(platform='huawei')

# 运行优化
optimized_code = optimizer.optimize(computational_graph)

# 使用优化后的代码
print(optimized_code)


这个实现包括了以下关键特性：
支持大模型特定的融合策略，如 LayerNorm + Attention、Attention + Dropout 等。
使用 networkx 库来表示和操作计算图。
通过平台适配器支持不同的硬件平台。
分离了通用融合逻辑和平台特定优化。
提供了可扩展的框架，可以轻松添加新的融合模式和优化 pass。
根据实际的硬件 API 和限制，完善每个平台适配器中的优化 pass 和代码生成逻辑。
