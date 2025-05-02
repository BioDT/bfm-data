"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

import sys
import threading
import time

import psutil
import torch

from bfm_data.dataset_creation.batch import DataBatch


def print_memory_usage():
    """
    Print the current memory usage of the running process.

    This function retrieves and prints the Resident Set Size (RSS) memory usage of the current process,
    which is the amount of memory occupied by the process in RAM (excluding swap space). The memory usage
    is displayed in megabytes (MB).

    Returns:
        None
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    print(f"Memory used: {memory_info.rss / 1024 ** 2:.2f} MB")


def monitor_memory_usage(interval: int = 10):
    """
    Monitor and print the memory usage at regular intervals.

    Args:
        interval (int): Time in seconds between memory usage updates. Default is 10 seconds.

    Returns:
        None
    """
    try:
        while True:
            print_memory_usage()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Memory monitoring stopped.")


def start_memory_monitoring(interval: int = 10):
    """
    Start the memory monitoring in a separate thread.

    Args:
        interval (int): Time in seconds between memory usage updates. Default is 10 seconds.

    Returns:
        None
    """
    monitoring_thread = threading.Thread(
        target=monitor_memory_usage, args=(interval,), daemon=True
    )
    monitoring_thread.start()


def get_tensor_memory_size(tensor: torch.Tensor) -> int:
    """
    Calculate the memory size of a PyTorch tensor in bytes.

    This function computes the memory size by multiplying the size of each element
    in the tensor (in bytes) by the total number of elements.

    Args:
        tensor (torch.Tensor): The input tensor for which to calculate the memory size.

    Returns:
        int: The total memory size of the tensor in bytes.
    """
    return tensor.element_size() * tensor.nelement()


def get_object_memory_size(obj) -> int:
    """
    Calculate the memory size of a Python object in bytes.

    This function uses Python's `sys.getsizeof()` to determine the memory size
    of any object except PyTorch tensors. It provides a quick way to measure
    the memory footprint of objects in Python.

    Args:
        obj (object): The Python object for which to calculate the memory size.

    Returns:
        int: The memory size of the object in bytes.
    """
    return sys.getsizeof(obj)


def get_data_batch_memory_size(batch: DataBatch) -> int:
    """
    Calculates the approximate memory usage of a DataBatch object, including all tensors.

    Args:
        batch (DataBatch): The DataBatch object.

    Returns:
        int: Total memory size in bytes.
    """
    total_memory = 0

    for key, tensor in batch.surface_variables.items():
        total_memory += get_tensor_memory_size(tensor)

    for key, tensor in batch.single_variables.items():
        total_memory += get_tensor_memory_size(tensor)

    for key, tensor in batch.atmospheric_variables.items():
        total_memory += get_tensor_memory_size(tensor)

    for key, tensor in batch.species_variables.items():
        total_memory += get_tensor_memory_size(tensor)

    total_memory += get_tensor_memory_size(batch.batch_metadata.latitudes)
    total_memory += get_tensor_memory_size(batch.batch_metadata.longitudes)

    total_memory += get_object_memory_size(batch.batch_metadata.timestamp)

    return total_memory
