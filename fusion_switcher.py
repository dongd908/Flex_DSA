import networkx as nx
from typing import Dict, List, Tuple, Callable
from .platform_adapters import PlatformAdapter

class FusionSwitcher:
    def __init__(self, platform_adapter: PlatformAdapter):
        self.platform_adapter = platform_adapter
        self.fusion_patterns = self._initialize_fusion_patterns()

    def select_fusion_strategy(self, graph: nx.DiGraph) -> nx.DiGraph:
        """
        选择最优的子图融合策略
        
        Args:
            graph (nx.DiGraph): 输入的计算图
        
        Returns:
            nx.DiGraph: 融合后的计算图
        """
        fused_graph = graph.copy()
        
        # 遍历图中的所有节点
        for node in list(fused_graph.nodes()):
            # 检查是否可以应用融合模式
            for pattern, fusion_func in self.fusion_patterns.items():
                if self._match_pattern(fused_graph, node, pattern):
                    fused_graph = fusion_func(fused_graph, node)
                    break  # 每个节点只应用一次融合
        
        # 应用平台特定的优化
        fused_graph = self.platform_adapter.apply_passes(fused_graph)
        
        return fused_graph

    def _initialize_fusion_patterns(self) -> Dict[Tuple[str, ...], Callable]:
        """
        初始化融合模式字典
        
        Returns:
            Dict[Tuple[str, ...], callable]: 融合模式到融合函数的映射
        """
        return {
            ("conv", "relu"): self._fuse_conv_relu,
            ("conv", "bn", "relu"): self._fuse_conv_bn_relu,
            ("matmul", "add"): self._fuse_matmul_add,
            ("layer_norm", "attention"): self._fuse_layernorm_attention,
            ("attention", "dropout"): self._fuse_attention_dropout,
            ("ffn", "dropout"): self._fuse_ffn_dropout,
            ("matmul", "scale", "softmax"): self._fuse_matmul_scale_softmax,
            ("matmul", "bias_add", "gelu"): self._fuse_matmul_bias_gelu,
        }

    def _match_pattern(self, graph: nx.DiGraph, node: str, pattern: Tuple[str, ...]) -> bool:
        """
        检查给定节点是否匹配融合模式
        
        Args:
            graph (nx.DiGraph): 计算图
            node (str): 当前节点
            pattern (Tuple[str, ...]): 要匹配的模式
        
        Returns:
            bool: 是否匹配
        """
        if graph.nodes[node]['op'] != pattern[0]:
            return False
        
        current = node
        for op in pattern[1:]:
            successors = list(graph.successors(current))
            if not successors or graph.nodes[successors[0]]['op'] != op:
                return False
            current = successors[0]
        
        return True

    def _fuse_conv_relu(self, graph: nx.DiGraph, conv_node: str) -> nx.DiGraph:
        """融合卷积和ReLU操作"""
        relu_node = list(graph.successors(conv_node))[0]
        fused_node = f"{conv_node}_relu"
        graph.add_node(fused_node, op="conv_relu", **graph.nodes[conv_node])
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(relu_node))
        graph.remove_node(conv_node)
        graph.remove_node(relu_node)
        return graph

    def _fuse_conv_bn_relu(self, graph: nx.DiGraph, conv_node: str) -> nx.DiGraph:
        """融合卷积、批归一化和ReLU操作"""
        bn_node = list(graph.successors(conv_node))[0]
        relu_node = list(graph.successors(bn_node))[0]
        fused_node = f"{conv_node}_bn_relu"
        graph.add_node(fused_node, op="conv_bn_relu", **graph.nodes[conv_node])
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(relu_node))
        graph.remove_node(conv_node)
        graph.remove_node(bn_node)
        graph.remove_node(relu_node)
        return graph

    def _fuse_matmul_add(self, graph: nx.DiGraph, matmul_node: str) -> nx.DiGraph:
        """融合矩阵乘法和加法操作"""
        add_node = list(graph.successors(matmul_node))[0]
        fused_node = f"{matmul_node}_add"
        graph.add_node(fused_node, op="matmul_add", **graph.nodes[matmul_node])
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(add_node))
        graph.remove_node(matmul_node)
        graph.remove_node(add_node)
        return graph

    def _fuse_layernorm_attention(self, graph: nx.DiGraph, layernorm_node: str) -> nx.DiGraph:
        """融合层归一化和注意力操作"""
        attention_node = list(graph.successors(layernorm_node))[0]
        fused_node = f"{layernorm_node}_attention"
        fused_attrs = {
            'op': 'layernorm_attention',
            'layernorm_weight': graph.nodes[layernorm_node]['weight'],
            'layernorm_bias': graph.nodes[layernorm_node]['bias'],
            'attention_weight': graph.nodes[attention_node]['weight'],
            'attention_bias': graph.nodes[attention_node]['bias'],
        }
        graph.add_node(fused_node, **fused_attrs)
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(attention_node))
        graph.remove_node(layernorm_node)
        graph.remove_node(attention_node)
        return graph

    def _fuse_attention_dropout(self, graph: nx.DiGraph, attention_node: str) -> nx.DiGraph:
        """融合注意力和dropout操作"""
        dropout_node = list(graph.successors(attention_node))[0]
        fused_node = f"{attention_node}_dropout"
        graph.add_node(fused_node, op="attention_dropout", **graph.nodes[attention_node])
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(dropout_node))
        graph.remove_node(attention_node)
        graph.remove_node(dropout_node)
        return graph

    def _fuse_ffn_dropout(self, graph: nx.DiGraph, ffn_node: str) -> nx.DiGraph:
        """融合前馈网络和dropout操作"""
        dropout_node = list(graph.successors(ffn_node))[0]
        fused_node = f"{ffn_node}_dropout"
        graph.add_node(fused_node, op="ffn_dropout", **graph.nodes[ffn_node])
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(dropout_node))
        graph.remove_node(ffn_node)
        graph.remove_node(dropout_node)
        return graph

    def _fuse_matmul_scale_softmax(self, graph: nx.DiGraph, matmul_node: str) -> nx.DiGraph:
        """融合矩阵乘法、缩放和softmax操作（用于注意力计算）"""
        scale_node = list(graph.successors(matmul_node))[0]
        softmax_node = list(graph.successors(scale_node))[0]
        fused_node = f"{matmul_node}_scale_softmax"
        graph.add_node(fused_node, op="matmul_scale_softmax", **graph.nodes[matmul_node])
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(softmax_node))
        graph.remove_node(matmul_node)
        graph.remove_node(scale_node)
        graph.remove_node(softmax_node)
        return graph

    def _fuse_matmul_bias_gelu(self, graph: nx.DiGraph, matmul_node: str) -> nx.DiGraph:
        """融合矩阵乘法、偏置加法和GELU激活（用于前馈网络）"""
        bias_add_node = list(graph.successors(matmul_node))[0]
        gelu_node = list(graph.successors(bias_add_node))[0]
        fused_node = f"{matmul_node}_bias_gelu"
        graph.add_node(fused_node, op="matmul_bias_gelu", **graph.nodes[matmul_node])
        graph.add_edges_from((fused_node, succ) for succ in graph.successors(gelu_node))
        graph.remove_node(matmul_node)
        graph.remove_node(bias_add_node)
        graph.remove_node(gelu_node)
        return graph