# Usage Guide

This guide provides examples of how to use the GPUInfo library effectively.

## Basic Usage

### Initializing GPUInfo

```python
from gpuinfo import GPUInfo

# Create a GPUInfo instance
gpu_info = GPUInfo()
```

### Printing GPU Information

```python
# Print information about all detected GPUs
gpu_info.print_gpu_info()
```

### Freeing Up GPU Resources

```python
# Free up GPU 0 (terminate processes and clear cache)
gpu_info.free_up_gpu(0)

# Only clear cache on GPU 1 without terminating processes
gpu_info.free_up_gpu(1, terminate_processes=False, clear_cache=True)

# Only terminate processes on GPU 2 without clearing cache
gpu_info.free_up_gpu(2, terminate_processes=True, clear_cache=False)
```

## Advanced Usage

### Customizing Exempt Processes

You can customize the list of processes that are exempt from termination:

```python
from gpuinfo import GPUInfo, DEFAULT_EXEMPT_PROCESSES

gpu_info = GPUInfo()

# Add a custom process to the default exempt list
custom_exempt = DEFAULT_EXEMPT_PROCESSES + ["my-custom-process"]
gpu_info.free_up_gpu(0, exempt_processes=custom_exempt)

# Use a completely custom list of exempt processes
gpu_info.free_up_gpu(0, exempt_processes=["process1", "process2"])

# Terminate all processes (use with caution!)
gpu_info.free_up_gpu(0, exempt_processes=[])
```

### Retrieving GPU Statistics

You can get detailed statistics for each GPU:

```python
gpu_stats = gpu_info.get_gpu_stats()
for stat in gpu_stats:
    print(f"GPU {stat.index}:")
    print(f"  Name: {stat.name}")
    print(f"  Total Memory: {stat.total_memory:.2f} GB")
    print(f"  Used Memory: {stat.used_memory:.2f} GB")
    print(f"  Free Memory: {stat.free_memory:.2f} GB")
    print(f"  Temperature: {stat.temperature}°C")
    print(f"  GPU Utilization: {stat.gpu_utilization}%")
    print(f"  Memory Utilization: {stat.memory_utilization}%")
    if stat.fan_speed is not None:
        print(f"  Fan Speed: {stat.fan_speed}%")
    print()
```

This will give you a detailed overview of each GPU's current state and usage.
