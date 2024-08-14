# API Reference

## GPUInfonv

```python
class GPUInfo(BaseModel)
```

A class to represent information about available GPUs and manage GPU resources.

### Methods

#### `__init__(**data)`

Initialize the GPUInfo object and gather initial GPU information.

#### `initialize_gpu_info()`

Initialize GPU information by detecting GPUs and checking PyTorch availability.

#### `get_gpu_stats() -> List[GPUStat]`

Retrieve current statistics for all detected GPUs.

#### `check_torch_availability()`

Check if PyTorch is installed and if CUDA is available for PyTorch.

#### `print_gpu_info()`

Print detailed information about all detected GPUs and PyTorch availability.

#### `free_up_gpu(gpu_index: int = 0, terminate_processes: bool = True, clear_cache: bool = True, exempt_processes: List[str] = DEFAULT_EXEMPT_PROCESSES)`

Attempt to free up the specified GPU from any ongoing operations and bring it to a clean idle state.

Arguments:
- `gpu_index` (int): The index of the GPU to free up. Defaults to 0.
- `terminate_processes` (bool): Whether to terminate CUDA processes. Defaults to True.
- `clear_cache` (bool): Whether to clear the GPU cache. Defaults to True.
- `exempt_processes` (List[str]): List of process names to exempt from termination. Defaults to DEFAULT_EXEMPT_PROCESSES.

## GPUStat

```python
class GPUStat(BaseModel)
```

A class to represent statistics for a single GPU.

### Attributes

- `index` (int): The index of the GPU.
- `name` (str): The name of the GPU.
- `total_memory` (float): Total memory of the GPU in GB.
- `used_memory` (float): Used memory of the GPU in GB.
- `free_memory` (float): Free memory of the GPU in GB.
- `temperature` (float): Temperature of the GPU in Celsius.
- `gpu_utilization` (float): GPU utilization as a percentage.
- `memory_utilization` (float): Memory utilization as a percentage.
- `fan_speed` (Optional[float]): Fan speed as a percentage, if available.

## Constants

### `DEFAULT_EXEMPT_PROCESSES`

A list of process names that are exempt from termination by default when using the `free_up_gpu` method.

Default value: `["/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell", "warp-terminal"]`
