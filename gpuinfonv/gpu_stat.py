"""
This module defines the GPUStat class, which represents statistics for a single GPU.
"""

from typing import Optional
from pydantic import BaseModel

class GPUStat(BaseModel):
    """
    A class to represent statistics for a single GPU.

    Attributes:
        index (int): The index of the GPU.
        name (str): The name of the GPU.
        total_memory (float): Total memory of the GPU in GB.
        used_memory (float): Used memory of the GPU in GB.
        free_memory (float): Free memory of the GPU in GB.
        temperature (float): Temperature of the GPU in Celsius.
        gpu_utilization (float): GPU utilization as a percentage.
        memory_utilization (float): Memory utilization as a percentage.
        fan_speed (Optional[float]): Fan speed as a percentage, if available.
    """
    index: int
    name: str
    total_memory: float  # in GB
    used_memory: float  # in GB
    free_memory: float  # in GB
    temperature: float  # in Celsius
    gpu_utilization: float  # in percentage
    memory_utilization: float  # in percentage
    fan_speed: Optional[float] = None  # in percentage