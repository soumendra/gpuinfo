"""
This module provides functionality to gather and display information about NVIDIA GPUs,
as well as manage GPU resources.

It uses the pynvml library to interact with NVIDIA GPUs and can also check for PyTorch availability.
It also includes functionality to free up GPU resources.
"""

import importlib
from typing import List, Optional
import os
import subprocess
from pydantic import BaseModel, Field
import pynvml
import psutil

from .gpu_stat import GPUStat


# Default list of processes exempt from termination
DEFAULT_EXEMPT_PROCESSES = ["/usr/lib/xorg/Xorg", "/usr/bin/gnome-shell", "warp-terminal"]


class GPUInfo(BaseModel):
    """
    A class to represent information about available GPUs and manage GPU resources.

    This class initializes GPU information, retrieves GPU statistics,
    checks for PyTorch and CUDA availability, and provides methods to free up GPU resources.

    Attributes:
        gpu_count (int): The number of GPUs detected.
        gpus (List[GPUStat]): A list of GPUStat objects containing information about each GPU.
        torch_available (bool): Whether PyTorch is installed.
        torch_cuda_available (bool): Whether CUDA is available for PyTorch.
    """
    
    gpu_count: int = 0
    gpus: List[GPUStat] = []
    torch_available: bool = False
    torch_cuda_available: bool = False

    def __init__(self, **data):
        """
        Initialize the GPUInfo object and gather initial GPU information.
        """
        super().__init__(**data)
        self.initialize_gpu_info()

    def initialize_gpu_info(self):
        """
        Initialize GPU information by detecting GPUs and checking PyTorch availability.
        """
        try:
            pynvml.nvmlInit()
            self.gpu_count = pynvml.nvmlDeviceGetCount()
            self.gpus = self.get_gpu_stats()
        except pynvml.NVMLError as error:
            print(f"Error initializing NVML: {error}")
            print("NVIDIA GPUs may not be present or the NVIDIA driver may not be installed.")
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

        self.check_torch_availability()

    def get_gpu_stats(self) -> List[GPUStat]:
        """
        Retrieve current statistics for all detected GPUs.

        Returns:
            List[GPUStat]: A list of GPUStat objects containing information about each GPU.
        """
        gpu_stats = []
        try:
            pynvml.nvmlInit()
            for i in range(self.gpu_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8') # fix for older pynvml versions
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                try:
                    fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
                except pynvml.NVMLError:
                    fan_speed = None

                gpu_stats.append(GPUStat(
                    index=i,
                    name=name,
                    total_memory=memory_info.total / 1024**3,
                    used_memory=memory_info.used / 1024**3,
                    free_memory=memory_info.free / 1024**3,
                    temperature=temperature,
                    gpu_utilization=utilization.gpu,
                    memory_utilization=utilization.memory,
                    fan_speed=fan_speed
                ))
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass
        return gpu_stats

    def check_torch_availability(self):
        """
        Check if PyTorch is installed and if CUDA is available for PyTorch.
        """
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            import torch
            self.torch_available = True
            self.torch_cuda_available = torch.cuda.is_available()

    def print_gpu_info(self):
        """
        Print detailed information about all detected GPUs and PyTorch availability.
        """
        print(f"Number of GPUs detected: {self.gpu_count}")
        for gpu in self.gpus:
            print(f"\nGPU {gpu.index}:")
            print(f"  Name: {gpu.name}")
            print(f"  Total memory: {gpu.total_memory:.2f} GB")
            print(f"  Used memory: {gpu.used_memory:.2f} GB")
            print(f"  Free memory: {gpu.free_memory:.2f} GB")
            print(f"  GPU Utilization: {gpu.gpu_utilization}%")
            print(f"  Memory Utilization: {gpu.memory_utilization}%")
            print(f"  Temperature: {gpu.temperature}°C")
            if gpu.fan_speed is not None:
                print(f"  Fan Speed: {gpu.fan_speed}%")
            else:
                print("  Fan Speed: N/A")

        print(f"\nPyTorch installed: {self.torch_available}")
        print(f"PyTorch CUDA available: {self.torch_cuda_available}")

    def _terminate_gpu_processes(self, gpu_index: int, exempt_processes: List[str]):
        """
        Terminate CUDA processes running on the specified GPU, except for exempt processes.

        Args:
            gpu_index (int): The index of the GPU to free up.
            exempt_processes (List[str]): List of process names to exempt from termination.

        Note:
            This method requires administrative privileges to terminate processes.
            It may not be able to terminate all processes, especially those started by other users.
        """
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits', f'--id={gpu_index}'], 
                                    capture_output=True, text=True, check=True)
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid:
                    try:
                        process = psutil.Process(int(pid))
                        if process.name() not in exempt_processes:
                            process.terminate()
                            print(f"Terminated process {pid} ({process.name()})")
                        else:
                            print(f"Skipped termination of exempt process {pid} ({process.name()})")
                    except psutil.NoSuchProcess:
                        print(f"Process {pid} not found")
                    except psutil.AccessDenied:
                        print(f"Permission denied to terminate process {pid}")
        except subprocess.CalledProcessError as e:
            print(f"Error querying NVIDIA SMI: {e}")

    def _clear_gpu_cache(self, gpu_index: int):
        """
        Clear CUDA cache for the specified GPU if PyTorch is available.

        Args:
            gpu_index (int): The index of the GPU to clear cache for.
        """
        if self.torch_available and self.torch_cuda_available:
            import torch
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    print("Cleared CUDA cache")
                    torch.cuda.device(gpu_index).empty_cache()
                    torch.cuda.reset_peak_memory_stats(gpu_index)
                    torch.cuda.reset_max_memory_allocated(gpu_index)
                    torch.cuda.reset_max_memory_cached(gpu_index)
                    print(f"Reset CUDA device {gpu_index}")
                except RuntimeError as e:
                    print(f"Error resetting CUDA device: {e}")
        else:
            print("PyTorch or CUDA is not available. Unable to clear GPU cache.")

    def free_up_gpu(self, gpu_index: int = 0, terminate_processes: bool = True, clear_cache: bool = True, exempt_processes: List[str] = DEFAULT_EXEMPT_PROCESSES):
        """
        Attempt to free up the specified GPU from any ongoing operations and bring it to a clean idle state.

        This method can optionally terminate CUDA processes and clear the GPU cache.

        Args:
            gpu_index (int): The index of the GPU to free up. Defaults to 0.
            terminate_processes (bool): Whether to terminate CUDA processes. Defaults to True.
            clear_cache (bool): Whether to clear the GPU cache. Defaults to True.
            exempt_processes (List[str]): List of process names to exempt from termination. 
                                          Defaults to DEFAULT_EXEMPT_PROCESSES.

        Note:
            Process termination requires administrative privileges and may not be able to terminate
            all processes, especially those started by other users.
        """
        if gpu_index < 0 or gpu_index >= self.gpu_count:
            raise ValueError(f"Invalid GPU index. Must be between 0 and {self.gpu_count - 1}")

        print(f"Attempting to free up GPU {gpu_index}...")

        if terminate_processes:
            self._terminate_gpu_processes(gpu_index, exempt_processes)

        if clear_cache:
            self._clear_gpu_cache(gpu_index)

        print(f"Finished attempting to free up GPU {gpu_index}.")
        print("Note: Some processes may still be running if they were exempted, started by other users, or if you lack permissions to terminate them.")
        print("Please check 'nvidia-smi' to confirm the current state of the GPU.")

# Make DEFAULT_EXEMPT_PROCESSES available at the module level
__all__ = ['GPUInfo', 'DEFAULT_EXEMPT_PROCESSES']


