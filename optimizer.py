from .fusion_switcher import FusionSwitcher
from .platform_adapters import get_platform_adapter
import networkx as nx

class FlexDSAOptimizer:
    def __init__(self, platform: str):
        self.platform_adapter = get_platform_adapter(platform)
        self.fusion_switcher = FusionSwitcher(self.platform_adapter)

    def optimize(self, computational_graph: nx.DiGraph) -> str:
        # 应用融合策略
        optimized_graph = self.fusion_switcher.select_fusion_strategy(computational_graph)
        
        # 生成优化后的代码
        optimized_code = self.platform_adapter.generate_code(optimized_graph)
        
        return optimized_code
        # 应用硬件特定优化
        hardware_optimized_graph = self.apply_hardware_specific_optimizations(optimized_graph)
        
        # 应用内存优化
        memory_optimized_graph = self.optimize_memory_usage(hardware_optimized_graph)
        
        # 应用并行化优化
        parallel_optimized_graph = self.apply_parallelization(memory_optimized_graph)
        
        # 应用动态调度优化
        dynamic_optimized_graph = self.apply_dynamic_scheduling(parallel_optimized_graph)
        
        # 生成最终优化后的代码
        final_optimized_code = self.platform_adapter.generate_code(dynamic_optimized_graph)
        
        return final_optimized_code

    def apply_hardware_specific_optimizations(self, graph: nx.DiGraph) -> nx.DiGraph:
        # 应用硬件特定的优化，如指令级并行、向量化等
        return self.platform_adapter.apply_hardware_optimizations(graph)

    def optimize_memory_usage(self, graph: nx.DiGraph) -> nx.DiGraph:
        # 优化内存使用，如内存重用、数据布局优化等
        return self.platform_adapter.optimize_memory(graph)

    def apply_parallelization(self, graph: nx.DiGraph) -> nx.DiGraph:
        # 应用并行化优化，如任务并行、数据并行等
        return self.platform_adapter.apply_parallelization(graph)

    def apply_dynamic_scheduling(self, graph: nx.DiGraph) -> nx.DiGraph:
        # 应用动态调度优化，根据运行时信息动态调整执行策略
        return self.platform_adapter.apply_dynamic_scheduling(graph)