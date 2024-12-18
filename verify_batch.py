from typing import List

import torch
import typer

from src.dataset_creation.batch import DataBatch
from src.dataset_creation.metadata import BatchMetadata


def format_prefix(prefix: List[str]) -> str:
    return ".".join(prefix)


def visit_obj(obj, prefix: List[str] = []):
    if isinstance(obj, torch.Tensor):
        nan_values = torch.isnan(obj.view(-1)).sum().item()
        tot_values = obj.numel()
        print(format_prefix(prefix), obj.shape, f"NaN {nan_values/tot_values:.2%}%")
    if isinstance(obj, dict):
        for k, v in obj.items():
            visit_obj(v, prefix + [k])
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            visit_obj(v, prefix + [str(i)])
    if isinstance(obj, DataBatch):
        visit_obj(obj.surface_variables, ["surface_variables"])
        visit_obj(obj.single_variables, ["single_variables"])
        visit_obj(obj.atmospheric_variables, ["atmospheric_variables"])
        try:
            visit_obj(obj.species_variables, ["species_variables"])
        except:
            pass
        try:
            visit_obj(
                obj.species_distribution_variables, ["species_distribution_variables"]
            )
        except:
            pass
        visit_obj(obj.species_extinction_variables, ["species_extinction_variables"])
        visit_obj(obj.land_variables, ["land_variables"])
        visit_obj(obj.agriculture_variables, ["agriculture_variables"])
        visit_obj(obj.forest_variables, ["forest_variables"])
        visit_obj(obj.batch_metadata, ["batch_metadata"])
    if isinstance(obj, BatchMetadata):
        visit_obj(obj.latitudes, ["latitudes"])
        visit_obj(obj.longitudes, ["longitudes"])
        visit_obj(obj.timestamp, ["timestamp"])
        visit_obj(obj.pressure_levels, ["pressure_levels"])


def inspect_batch(file_path: str):
    data = torch.load(file_path)

    # Check the keys in the loaded dictionary
    # print("Keys in the loaded batch:", data.keys())

    # # Access individual components
    # surface_variables = data.get("surface_variables")
    # single_variables = data.get("single_variables")
    # atmospheric_variables = data.get("atmospheric_variables")
    # species_variables = data.get("species_variables")
    # species_extinction_variables = data.get("species_extinction_variables")
    # land_variables = data.get("land_variables")
    # agriculture_variables = data.get("agriculture_variables")
    # forest_variables = data.get("forest_variables")
    # batch_metadata = data.get("batch_metadata")

    # # Display metadata
    # if batch_metadata:
    #     print("Metadata latitudes:", batch_metadata.get("latitudes"))
    #     print("Metadata longitudes:", batch_metadata.get("longitudes"))
    #     print("Metadata timestamp:", batch_metadata.get("timestamp"))
    #     print("Metadata pressure levels:", batch_metadata.get("pressure_levels"))
    #     print("Metadata species list:", batch_metadata.get("species_list"))

    visit_obj(data)


if __name__ == "__main__":
    # /projects/prjs1134/data/projects/biodt/storage/batches/batch_2000-01-01_00-00-00_to_2000-01-01_06-00-00.pt
    # old batches
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_11_21/batch_2000-01-01_2000-01-04.pt
    ## sparse (broken???)
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_11_21/batch_2000-01-01_00-00-00_to_2000-01-01_06-00-00.pt
    typer.run(inspect_batch)
