import numpy as np
from typing import Dict, List, Tuple, Any
from .platform_adapters import get_platform_adapter

class ShapeTracker:
    def __init__(self):
        self.shape_cache: Dict[str, np.ndarray] = {}
        self.dynamic_shapes: Dict[str, callable] = {}
        self.hardware_adapters = {
            'cpu': CPUShapeAdapter(),
            'gpu': GPUShapeAdapter(),
            'tpu': TPUShapeAdapter(),
            'fpga': FPGAShapeAdapter()
        }

    def track_shape(self, node_id: str, shape: Tuple[int, ...]) -> None:
        self.shape_cache[node_id] = np.array(shape)

    def get_shape(self, node_id: str) -> np.ndarray:
        return self.shape_cache[node_id]

    def update_shape(self, node_id: str, operation: str, *args) -> np.ndarray:
        current_shape = self.get_shape(node_id)

        if operation == 'reshape':
            new_shape = args[0]
            if -1 in new_shape:
                inferred_dim = np.prod(current_shape) // np.prod([d for d in new_shape if d != -1])
                new_shape = tuple(inferred_dim if d == -1 else d for d in new_shape)
            updated_shape = np.array(new_shape)
        elif operation == 'transpose':
            axes = args[0] if args else range(len(current_shape))[::-1]
            updated_shape = np.array([current_shape[i] for i in axes])
        else:
            raise ValueError(f"Unsupported operation: {operation}")

        self.shape_cache[node_id] = updated_shape
        return updated_shape

    def infer_output_shape(self, op_type: str, input_shapes: List[np.ndarray], hardware: str, **kwargs) -> np.ndarray:
        adapter = self.hardware_adapters.get(hardware, self.hardware_adapters['cpu'])
        return adapter.infer_output_shape(op_type, input_shapes, **kwargs)

    def register_dynamic_shape(self, node_id: str, shape_func: callable) -> None:
        """
        Register a dynamic shape function for a node.

        Args:
            node_id (str): Unique identifier for the node.
            shape_func (callable): A function that takes runtime information and returns the shape.
        """
        self.dynamic_shapes[node_id] = shape_func

    def get_dynamic_shape(self, node_id: str, runtime_info: Dict[str, Any]) -> np.ndarray:
        """
        Get the dynamic shape of a node based on runtime information.

        Args:
            node_id (str): Unique identifier for the node.
            runtime_info (Dict[str, Any]): Runtime information used to determine the shape.

        Returns:
            np.ndarray: The dynamically determined shape.
        """
        if node_id in self.dynamic_shapes:
            return np.array(self.dynamic_shapes[node_id](runtime_info))
        else:
            return self.get_shape(node_id)

class HardwareShapeAdapter:
    def infer_output_shape(self, op_type: str, input_shapes: List[np.ndarray], **kwargs) -> np.ndarray:
        raise NotImplementedError

class CPUShapeAdapter(HardwareShapeAdapter):
    def infer_output_shape(self, op_type: str, input_shapes: List[np.ndarray], **kwargs) -> np.ndarray:
        if op_type == 'conv2d':
            return self._infer_conv2d_shape(input_shapes[0], input_shapes[1], **kwargs)
        elif op_type == 'max_pool':
            return self._infer_pool_shape(input_shapes[0], **kwargs)
        elif op_type == 'matmul':
            return self._infer_matmul_shape(input_shapes[0], input_shapes[1])
        else:
            raise ValueError(f"Unsupported operation type: {op_type}")

    def _infer_conv2d_shape(self, input_shape: np.ndarray, kernel_shape: np.ndarray,
                            stride: Tuple[int, int] = (1, 1), padding: str = 'valid') -> np.ndarray:
        batch, in_height, in_width, _ = input_shape
        kernel_height, kernel_width, _, out_channels = kernel_shape

        if padding == 'same':
            out_height = np.ceil(in_height / stride[0])
            out_width = np.ceil(in_width / stride[1])
        elif padding == 'valid':
            out_height = np.ceil((in_height - kernel_height + 1) / stride[0])
            out_width = np.ceil((in_width - kernel_width + 1) / stride[1])
        else:
            raise ValueError(f"Unsupported padding: {padding}")

        return np.array([batch, out_height, out_width, out_channels])

    def _infer_pool_shape(self, input_shape: np.ndarray, pool_size: Tuple[int, int],
                          stride: Tuple[int, int] = None, padding: str = 'valid') -> np.ndarray:
        batch, in_height, in_width, channels = input_shape
        if stride is None:
            stride = pool_size

        if padding == 'same':
            out_height = np.ceil(in_height / stride[0])
            out_width = np.ceil(in_width / stride[1])
        elif padding == 'valid':
            out_height = np.ceil((in_height - pool_size[0] + 1) / stride[0])
            out_width = np.ceil((in_width - pool_size[1] + 1) / stride[1])
        else:
            raise ValueError(f"Unsupported padding: {padding}")

        return np.array([batch, out_height, out_width, channels])

    def _infer_matmul_shape(self, shape_a: np.ndarray, shape_b: np.ndarray) -> np.ndarray:
        if len(shape_a) == 1:
            shape_a = np.array([1, shape_a[0]])
        if len(shape_b) == 1:
            shape_b = np.array([shape_b[0], 1])

        if shape_a[-1] != shape_b[-2]:
            raise ValueError(f"Incompatible shapes for matmul: {shape_a} and {shape_b}")

        batch_dims_a = shape_a[:-2]
        batch_dims_b = shape_b[:-2]
        batch_dims = np.maximum(batch_dims_a, batch_dims_b)

        return np.concatenate([batch_dims, [shape_a[-2], shape_b[-1]]])

class GPUShapeAdapter(HardwareShapeAdapter):
    def infer_output_shape(self, op_type: str, input_shapes: List[np.ndarray], **kwargs) -> np.ndarray:
        # GPU-specific shape inference
        # This could include considerations for GPU memory layout, CUDA cores, etc.
        if op_type == 'conv2d':
            return self._infer_gpu_conv2d_shape(input_shapes[0], input_shapes[1], **kwargs)
        # Implement other GPU-specific shape inferences...
        return super().infer_output_shape(op_type, input_shapes, **kwargs)

    def _infer_gpu_conv2d_shape(self, input_shape: np.ndarray, kernel_shape: np.ndarray, **kwargs) -> np.ndarray:
        # GPU-optimized conv2d shape inference
        # This could consider GPU-specific optimizations like im2col
        # For now, we'll use the same logic as CPU, but this could be optimized further
        return self._infer_conv2d_shape(input_shape, kernel_shape, **kwargs)

class TPUShapeAdapter(HardwareShapeAdapter):
    def infer_output_shape(self, op_type: str, input_shapes: List[np.ndarray], **kwargs) -> np.ndarray:
        # TPU-specific shape inference
        # This could include considerations for TPU's systolic array architecture
        if op_type == 'matmul':
            return self._infer_tpu_matmul_shape(input_shapes[0], input_shapes[1], **kwargs)
        # Implement other TPU-specific shape inferences...
        return super().infer_output_shape(op_type, input_shapes, **kwargs)

    def _infer_tpu_matmul_shape(self, shape_a: np.ndarray, shape_b: np.ndarray, **kwargs) -> np.ndarray:
        # TPU-optimized matmul shape inference
        # This could consider TPU's matrix multiplication units and memory hierarchy
        # For now, we'll use the same logic as CPU, but this could be optimized further
        return self._infer_matmul_shape(shape_a, shape_b)

class FPGAShapeAdapter(HardwareShapeAdapter):
    def infer_output_shape(self, op_type: str, input_shapes: List[np.ndarray], **kwargs) -> np.ndarray:
        # FPGA-specific shape inference
        # This could include considerations for FPGA's reconfigurable logic and DSP slices
        if op_type == 'conv2d':
            return self._infer_fpga_conv2d_shape(input_shapes[0], input_shapes[1], **kwargs)
        # Implement other FPGA-specific shape inferences...
        return super().infer_output_shape(op_type, input_shapes, **kwargs)

    def _infer_fpga_conv2d_shape(self, input_shape: np.ndarray, kernel_shape: np.ndarray, **kwargs) -> np.ndarray:
        # FPGA-optimized conv2d shape inference
        # This could consider FPGA-specific optimizations like systolic array implementations
        # For now, we'll use the same logic as CPU, but this could be optimized further
        return self._infer_conv2d_shape(input_shape, kernel_shape, **kwargs)

# Example usage of dynamic shape adjustment
def dynamic_reshape(runtime_info: Dict[str, Any]) -> Tuple[int, ...]:
    batch_size = runtime_info.get('batch_size', 1)
    input_size = runtime_info.get('input_size', 224)
    return (batch_size, input_size, input_size, 3)

# In your main code:
shape_tracker = ShapeTracker()
shape_tracker.register_dynamic_shape('input_node', dynamic_reshape)

# Later, when you need the shape:
runtime_info = {'batch_size': 32, 'input_size': 299}
dynamic_shape = shape_tracker.get_dynamic_shape('input_node', runtime_info)

# When performing operations:
hardware = 'gpu'  # or 'cpu', 'tpu', 'fpga'

output_shape = shape_tracker.get_hardware_adapter(hardware).infer_output_shape('conv2d', [dynamic_shape, kernel_shape], stride=(2,2), padding='same')



