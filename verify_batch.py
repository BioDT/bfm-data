"""Copyright (C) 2025 TNO, The Netherlands. Licensed under the MIT license."""

from typing import List

import torch
import typer

app = typer.Typer(pretty_exceptions_enable=False)


def format_prefix(prefix: List[str]) -> str:
    return ".".join(prefix)


def visit_obj(obj, prefix: List[str] = []):
    if isinstance(obj, torch.Tensor):
        nan_values = torch.isnan(obj.view(-1)).sum().item()
        inf_values = torch.isinf(obj.view(-1)).sum().item()
        tot_values = obj.numel()
        values_not_nan = obj[~torch.isnan(obj)]
        values_valid = values_not_nan[~torch.isinf(values_not_nan)]
        res_str = f"{format_prefix(prefix)} {obj.shape} NaN {nan_values/tot_values:.5%} Inf {inf_values/tot_values:.5%}"
        if values_not_nan.numel():
            min_not_nan = values_valid.min().item()
            max_not_nan = values_valid.max().item()
            mean = values_valid.mean().item()
            std = values_valid.std().item()
            res_str += f" min_max range: [{min_not_nan:.3f}, {max_not_nan:.3f}], mean: {mean:.3f}, std: {std:.3f}"
        print(res_str)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            visit_obj(v, prefix + [k])
    elif isinstance(obj, list):
        if len(obj):
            item = obj[0]
            if (
                isinstance(item, float)
                or isinstance(item, int)
                or isinstance(item, str)
            ):
                print(format_prefix(prefix), len(obj), "list", type(item))
            else:
                for i, v in enumerate(obj):
                    visit_obj(v, prefix + [str(i)])
        else:
            print(format_prefix(prefix), obj, "EMPTY LIST")
    else:
        print(format_prefix(prefix), "has type not supported:", type(obj))


@app.command()
def inspect_batch(file_path: str):
    data = torch.load(file_path, weights_only=True)
    visit_obj(data)


if __name__ == "__main__":
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_12_13/batch_2000-01-01_00-00-00_to_2000-01-01_06-00-00.pt
    # old batches
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_11_21/batch_2000-01-01_2000-01-04.pt
    ## sparse (broken???)
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_11_21/batch_2000-01-01_00-00-00_to_2000-01-01_06-00-00.pt
    app()
