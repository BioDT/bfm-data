# src/data_preprocessing/batch.py

from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from src.data_preprocessing.metadata import BatchMetadata


@dataclass
class DataBatch:
    """
    Represents a batch of data including surface, static, and atmospheric variables.

    Args:
        surface_variables (Dict[str, torch.Tensor]): Dictionary of surface-level variables, where each tensor has shape `(b, t, h, w)`.
        static_variables (Dict[str, torch.Tensor]): Dictionary of static variables, where each tensor has shape `(h, w)`.
        atmospheric_variables (Dict[str, torch.Tensor]): Dictionary of atmospheric variables, where each tensor has shape `(b, t, c, h, w)`.
        batch_metadata (Metadata): Metadata associated with this batch, containing information such as latitudes, longitudes, and time.
    """

    surface_variables: Dict[str, torch.Tensor]
    single_variables: Dict[str, torch.Tensor]
    atmospheric_variables: Dict[str, torch.Tensor]
    batch_metadata: BatchMetadata

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """
        Retrieve the spatial dimensions `(height, width)` from any surface-level variable in the batch.

        Returns:
            Tuple[int, int]: The spatial dimensions of the surface-level variables.
        """
        first_key = list(self.surface_variables.keys())[0]
        return self.surface_variables[first_key].shape[-2:]
