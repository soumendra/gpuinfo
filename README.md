# GPUInfoNV

GPUInfoNV is a Python library for gathering and managing information about NVIDIA GPUs. It provides functionality to retrieve GPU statistics, check PyTorch availability, and free up GPU resources.

## Features

- Detect and gather information about NVIDIA GPUs
- Check PyTorch and CUDA availability
- Retrieve detailed GPU statistics
- Free up GPU resources, including process termination and cache clearing

## Installation

You can install GPUInfoNV using pip:

```bash
pip install gpuinfonv
```

## Quick Start

Here's a simple example of how to use GPUInfo:

```python
from gpuinfonv import GPUInfo

# Create a GPUInfo instance
gpu_info = GPUInfo()

# Print information about all detected GPUs
gpu_info.print_gpu_info()

# Free up GPU resources
gpu_info.free_up_gpu(0)  # Free up GPU 0
```

## Documentation

For more detailed information, check out our [full documentation](https://your-documentation-url.com).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
