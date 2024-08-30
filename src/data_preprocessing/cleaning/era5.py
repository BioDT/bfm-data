# src/data_preprocessing/cleaning/era5.py

import xarray as xr

from src.data_preprocessing.batch import DataBatch


def handle_missing_values(data: xr.Dataset, method: str = "interpolate") -> xr.Dataset:
    """
    Handle missing values in the ERA5 dataset.

    Args:
        data (xr.Dataset): The input dataset containing ERA5 data.
        method (str, optional): The method used to handle missing values.
            Available options are 'interpolate', 'fill', and 'drop'. Defaults to 'interpolate'.

    Returns:
        xr.Dataset: The dataset with missing values handled according to the specified method.
    """
    if method == "interpolate":
        data = data.interpolate_na(dim="time", method="linear")
    elif method == "fill":
        data = data.fillna(data.mean())
    elif method == "drop":
        data = data.dropna(dim="time", how="any")
    return data


def crop(self, patch_size: int) -> "DataBatch":
    """
    Crop the variables in the batch to the specified patch size.

    Args:
        patch_size (int): The target patch size to crop the data to.

    Returns:
        Batch: A new batch with variables cropped to the specified patch size.

    Raises:
        ValueError: If the width of the data is not a multiple of the patch size.
        ValueError: If more than one latitude is not divisible by the patch size.
    """
    height, width = self.spatial_dimensions

    if width % patch_size != 0:
        raise ValueError(
            f"Data width ({width}) must be a multiple of the patch size ({patch_size})."
        )

    if height % patch_size == 0:
        return self

    elif height % patch_size == 1:
        cropped_surface = {
            key: value[..., :-1, :] for key, value in self.surface_variables.items()
        }
        cropped_static = {
            key: value[:-1, :] for key, value in self.single_variables.items()
        }
        cropped_atmospheric = {
            key: value[..., :-1, :] for key, value in self.atmospheric_variables.items()
        }

        return DataBatch(
            surface_variables=cropped_surface,
            static_variables=cropped_static,
            atmospheric_variables=cropped_atmospheric,
            batch_metadata=self.batch_metadata,
        )
    else:
        excess_rows = height % patch_size
        raise ValueError(
            f"Expected at most one extra latitude row, but found {excess_rows}."
        )


DataBatch.crop = crop
