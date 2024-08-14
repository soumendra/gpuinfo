"""
This script demonstrates the usage of the GPUInfo class to retrieve and display GPU information.

It creates a GPUInfo object, prints detailed information about all detected GPUs,
and then retrieves and displays current GPU statistics.
"""

from gpuinfo import GPUInfo

# Create a GPUInfo object
gpu_info = GPUInfo()

# Print detailed information about all detected GPUs
gpu_info.print_gpu_info()

# Get and display current GPU statistics
print("\nGetting current GPU stats:")
current_stats = gpu_info.get_gpu_stats()
for stat in current_stats:
    print(f"GPU {stat.index} - Utilization: {stat.gpu_utilization}%, Memory Used: {stat.used_memory:.2f} GB")