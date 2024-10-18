from abc import ABC, abstractmethod
import networkx as nx

class PlatformAdapter(ABC):
    @abstractmethod
    def apply_passes(self, graph: nx.DiGraph) -> nx.DiGraph:
        pass

    @abstractmethod
    def generate_code(self, optimized_graph: nx.DiGraph) -> str:
        pass