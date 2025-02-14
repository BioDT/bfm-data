from typing import List

import torch
import typer


def format_prefix(prefix: List[str]) -> str:
    return ".".join(prefix)


def visit_obj(obj, prefix: List[str] = []):
    if isinstance(obj, torch.Tensor):
        nan_values = torch.isnan(obj.view(-1)).sum().item()
        tot_values = obj.numel()
        print(format_prefix(prefix), obj.shape, f"NaN {nan_values/tot_values:.5%}")
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


def inspect_batch(file_path: str):
    data = torch.load(file_path, weights_only=True)
    visit_obj(data)


if __name__ == "__main__":
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_12_13/batch_2000-01-01_00-00-00_to_2000-01-01_06-00-00.pt
    # old batches
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_11_21/batch_2000-01-01_2000-01-04.pt
    ## sparse (broken???)
    # /projects/prjs1134/data/projects/biodt/storage/batches_2024_11_21/batch_2000-01-01_00-00-00_to_2000-01-01_06-00-00.pt
    typer.run(inspect_batch)
