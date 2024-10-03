# src/monitoring_logging/monitoring/memory.py

import threading
import time

import psutil


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
