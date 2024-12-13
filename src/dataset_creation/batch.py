# src/dataset_creation/batch.py

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import torch

from src.data_preprocessing.transformation.era5 import (
    normalise_atmospheric_variables,
    normalise_surface_variable,
    unnormalise_atmospheric_variables,
    unnormalise_surface_variables,
)
from src.dataset_creation.metadata import BatchMetadata


@dataclass
class DataBatch:
    """
    Represents a batch of data including surface, static, and atmospheric variables.

    Args:
        surface_variables (Dict[str, torch.Tensor]): Dictionary of surface-level variables, where each tensor has shape `(t, h, w)`.
        single_variables (Dict[str, torch.Tensor]): Dictionary of single variables, where each tensor has shape `(t, h, w)`.
        atmospheric_variables (Dict[str, torch.Tensor]): Dictionary of atmospheric variables, where each tensor has shape `(t, c, h, w)`.
        species_variables (Dict[str, torch.Tensor]): Dictionary of species variables, where each tensor has shape `(t, h, w)`.
        species_distribution_variables (Dict[str, torch.Tensor]): Dictionary of species distribution variables, where each tensor has shape `(t, h, w)`.
        species_extinction_variables (Dict[str, torch.Tensor]): Dictionary of species extinction variables, where each tensor has shape `(t, h, w)`.
        land_variables (Dict[str, torch.Tensor]): Dictionary of land variables, where each tensor has shape `(t, h, w)`.
        agriculture_variables (Dict[str, torch.Tensor]): Dictionary of agriculture variables, where each tensor has shape `(t, h, w)`.
        forest_variables (Dict[str, torch.Tensor]): Dictionary of forest variables, where each tensor has shape `(t, h, w)`.
        batch_metadata (Metadata): Metadata associated with this batch, containing information such as latitudes, longitudes, and time.
    """

    surface_variables: Dict[str, torch.Tensor]
    single_variables: Dict[str, torch.Tensor]
    atmospheric_variables: Dict[str, torch.Tensor]
    species_variables: Dict[str, torch.Tensor]
    species_distribution_variables: Dict[str, torch.Tensor]
    species_extinction_variables: Dict[str, torch.Tensor]
    land_variables: Dict[str, torch.Tensor]
    agriculture_variables: Dict[str, torch.Tensor]
    forest_variables: Dict[str, torch.Tensor]
    batch_metadata: BatchMetadata

    @property
    def spatial_dimensions(self) -> Tuple[int, int]:
        """
        Retrieve the spatial dimensions `(height, width)` from any surface-level variable in the batch.

        Returns:
            Tuple[int, int]: The spatial dimensions of the surface-level variables.
        """
        return next(iter(self.surface_variables.values())).shape[-2:]

    def _fmap(self, f: Callable[[torch.Tensor], torch.Tensor]) -> "DataBatch":
        """
        Applies a function `f` to all tensor elements within the DataBatch.

        Args:
            f (Callable[[torch.Tensor], torch.Tensor]): A function that takes a torch.Tensor
            as input and returns a torch.Tensor, applied to each tensor in the DataBatch.

        Returns:
            DataBatch: A new DataBatch instance with the function `f` applied to its tensors.
        """
        surface_variables = {
            key: f(value) for key, value in self.surface_variables.items()
        }
        single_variables = {
            key: f(value) for key, value in self.single_variables.items()
        }
        atmospheric_variables = {
            key: f(value) for key, value in self.atmospheric_variables.items()
        }
        species_variables = {
            key: f(value) for key, value in self.species_variables.items()
        }
        species_distribution_variables = {
            key: f(value) for key, value in self.species_distribution_variables.items()
        }
        species_extinction_variables = {
            key: f(value) for key, value in self.species_extinction_variables.items()
        }
        land_variables = {key: f(value) for key, value in self.land_variables.items()}
        agriculture_variables = {
            key: f(value) for key, value in self.agriculture_variables.items()
        }
        forest_variables = {
            key: f(value) for key, value in self.forest_variables.items()
        }

        return DataBatch(
            surface_variables=surface_variables,
            single_variables=single_variables,
            atmospheric_variables=atmospheric_variables,
            species_variables=species_variables,
            species_distribution_variables=species_distribution_variables,
            species_extinction_variables=species_extinction_variables,
            land_variables=land_variables,
            agriculture_variables=agriculture_variables,
            forest_variables=forest_variables,
            batch_metadata=BatchMetadata(
                latitudes=f(self.batch_metadata.latitudes),
                longitudes=f(self.batch_metadata.longitudes),
                pressure_levels=self.batch_metadata.pressure_levels,
                timestamp=self.batch_metadata.timestamp,
                prediction_step=self.batch_metadata.prediction_step,
            ),
        )

    def to(self, device: str | torch.device) -> "DataBatch":
        """
        Moves all tensors within the DataBatch to the specified device.

        Args:
            device (str | torch.device): The device to which the tensors should be moved.

        Returns:
            DataBatch: A new DataBatch instance with all tensors moved to the specified device.
        """
        return self._fmap(lambda x: x.to(device))

    def type(self, t: type) -> "DataBatch":
        """
        Converts all tensors within the DataBatch to the specified data type.

        Args:
            t (type): The data type to which the tensors should be converted.

        Returns:
            DataBatch: A new DataBatch instance with all tensors converted to the specified type.
        """
        return self._fmap(lambda x: x.type(t))

    def normalize_data(self, locations, scales) -> "DataBatch":
        """
        Normalise all variables in the batch.

        Args:
            locations (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the mean for that variable.
            scales (Dict[str, float]): A dictionary where the key is the variable name
                                        and the value is the standard deviation for that variable.

        Returns:
            DataBatch: A new batch with all variables normalized.
        """
        normalized_surface_variables = {
            key: normalise_surface_variable(value, key, locations, scales)
            for key, value in self.surface_variables.items()
        }
        normalized_single_variables = {
            key: normalise_surface_variable(value, key, locations, scales)
            for key, value in self.single_variables.items()
        }
        normalized_atmospheric_variables = {
            key: normalise_atmospheric_variables(
                value, key, self.batch_metadata.pressure_levels, locations, scales
            )
            for key, value in self.atmospheric_variables.items()
        }

        return DataBatch(
            surface_variables=normalized_surface_variables,
            single_variables=normalized_single_variables,
            atmospheric_variables=normalized_atmospheric_variables,
            species_variables=self.species_variables,
            species_distribution_variables=self.species_distribution_variables,
            species_extinction_variables=self.species_extinction_variables,
            land_variables=self.land_variables,
            agriculture_variables=self.agriculture_variables,
            forest_variables=self.forest_variables,
            batch_metadata=self.batch_metadata,
        )

    def unnormalise_data(self, locations, scales) -> "DataBatch":
        """
        Unnormalise all variables in the batch.

        Args:
            locations (Dict[str, float]): A dictionary where the key is the variable name
                                    and the value is the mean for that variable.
            scales (Dict[str, float]): A dictionary where the key is the variable name
                                        and the value is the standard deviation for that variable.

        Returns:
            DataBatch: A new batch with all variables unnormalized.
        """
        unnormalized_surface_variables = {
            key: unnormalise_surface_variables(value, key, locations, scales)
            for key, value in self.surface_variables.items()
        }
        unnormalized_single_variables = {
            key: unnormalise_surface_variables(value, key, locations, scales)
            for key, value in self.single_variables.items()
        }
        unnormalized_atmospheric_variables = {
            key: unnormalise_atmospheric_variables(
                value, key, self.batch_metadata.pressure_levels, locations, scales
            )
            for key, value in self.atmospheric_variables.items()
        }

        return DataBatch(
            surface_variables=unnormalized_surface_variables,
            single_variables=unnormalized_single_variables,
            atmospheric_variables=unnormalized_atmospheric_variables,
            species_variables=self.species_variables,
            species_distribution_variables=self.species_distribution_variables,
            species_extinction_variables=self.species_extinction_variables,
            land_variables=self.land_variables,
            agriculture_variables=self.agriculture_variables,
            forest_variables=self.forest_variables,
            batch_metadata=self.batch_metadata,
        )

    def crop(self, patch_size: int, mode: str = "truncate") -> "DataBatch":
        """
        Crop or pad the variables in the batch to the specified patch size.

        Args:
            patch_size (int): The target patch size to crop or pad the data to. Default is 4.
            mode (str): The mode to adjust dimensions ('truncate' or 'pad'). Default is 'truncate'.

        Returns:
            DataBatch: A new batch with variables cropped or padded to the specified patch size.

        Raises:
            ValueError: If invalid mode is provided.
        """
        height, width = self.spatial_dimensions

        if mode == "truncate":
            new_height = (height // patch_size) * patch_size
            new_width = (width // patch_size) * patch_size
            cropped_surface = {
                key: value[..., :new_height, :new_width]
                for key, value in self.surface_variables.items()
            }
            cropped_static = {
                key: value[:new_height, :new_width]
                for key, value in self.single_variables.items()
            }
            cropped_atmospheric = {
                key: value[..., :new_height, :new_width]
                for key, value in self.atmospheric_variables.items()
            }
            cropped_species = {
                key: value[..., :new_height, :new_width]
                for key, value in self.species_variables.items()
            }
            cropped_species_distribution = {
                key: value[..., :new_height, :new_width]
                for key, value in self.species_distribution_variables.items()
            }
            cropped_species_extinction = {
                key: value[..., :new_height, :new_width]
                for key, value in self.species_extinction_variables.items()
            }
            cropped_land = {
                key: value[..., :new_height, :new_width]
                for key, value in self.land_variables.items()
            }
            cropped_agriculture = {
                key: value[..., :new_height, :new_width]
                for key, value in self.agriculture_variables.items()
            }
            cropped_forest = {
                key: value[..., :new_height, :new_width]
                for key, value in self.forest_variables.items()
            }
            return DataBatch(
                surface_variables=cropped_surface,
                single_variables=cropped_static,
                atmospheric_variables=cropped_atmospheric,
                species_variables=cropped_species,
                species_distribution_variables=cropped_species_distribution,
                species_extinction_variables=cropped_species_extinction,
                land_variables=cropped_land,
                agriculture_variables=cropped_agriculture,
                forest_variables=cropped_forest,
                batch_metadata=self.batch_metadata,
            )

        elif mode == "pad":
            new_height = ((height + patch_size - 1) // patch_size) * patch_size
            new_width = ((width + patch_size - 1) // patch_size) * patch_size
            pad_height = new_height - height
            pad_width = new_width - width
            padding = (0, pad_width, 0, pad_height)

            padded_surface = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.surface_variables.items()
            }
            padded_static = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.single_variables.items()
            }
            padded_atmospheric = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.atmospheric_variables.items()
            }
            padded_species = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.species_variables.items()
            }
            padded_species_distribution = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.species_distribution_variables.items()
            }
            padded_species_extinction = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.species_extinction_variables.items()
            }
            padded_land = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.land_variables.items()
            }
            padded_agriculture = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.agriculture_variables.items()
            }
            padded_forest = {
                key: torch.nn.functional.pad(value, padding)
                for key, value in self.forest_variables.items()
            }

            return DataBatch(
                surface_variables=padded_surface,
                single_variables=padded_static,
                atmospheric_variables=padded_atmospheric,
                species_variables=padded_species,
                species_distribution_variables=padded_species_distribution,
                species_extinction_variables=padded_species_extinction,
                land_variables=padded_land,
                agriculture_variables=padded_agriculture,
                forest_variables=padded_forest,
                batch_metadata=self.batch_metadata,
            )

        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'truncate' or 'pad'.")
