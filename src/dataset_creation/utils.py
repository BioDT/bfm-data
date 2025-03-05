import numpy as np


def get_lat_lon_ranges(
    min_lon: float = -30.0,
    max_lon: float = 50.0,
    min_lat: float = 34.0,
    max_lat: float = 72.0,
    lon_step: float = 0.25,
    lat_step: float = 0.25,
):
    """
    Get latitude and longitude ranges.

    Args:
        min_lon (float): The minimum longitude.
        max_lon (float): The maximum longitude.
        min_lat (float): The minimum latitude.
        max_lat (float): The maximum latitude.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The latitude and longitude ranges.
    """

    lat_range = np.arange(min_lat, max_lat + lat_step, lat_step)
    lon_range = np.arange(min_lon, max_lon + lon_step, lon_step)
    # reverse lat_range to go from North to South
    lat_range = lat_range[::-1]

    return lat_range, lon_range
