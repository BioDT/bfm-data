import glob
import json
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import typer
from tqdm import tqdm

from src.config.paths import *

# from sklearn import preprocessing

app = typer.Typer(pretty_exceptions_enable=False)


excluded_keys = ["batch_metadata", "metadata"]
dimensions_to_keep_by_key = {
    "species_variables": {
        "dynamic": {
            "Distribution": [3],  # [time, lat, lon, species]
        },
    },
    "atmospheric_variables": {
        "z": [1],  # [time, z, lat, lon]
        "t": [1],  # [time, t, lat, lon]
    },
}


class OnlineMeanAndVariance:
    """
    Welford's algorithm computes the sample variance incrementally.
    https://stackoverflow.com/questions/5543651/computing-standard-deviation-in-a-stream
    """

    def __init__(self, iterable=None, ddof=0):
        self.ddof, self.n, self.mean, self.M2 = ddof, 0, 0.0, 0.0
        self._min = float("inf")
        self._max = float("-inf")
        if iterable is not None:
            for datum in iterable:
                self.include(datum)

    def include(self, datum, exclude_nan: bool = True):
        self.n += 1
        self.delta = datum - self.mean
        self.mean += self.delta / self.n
        self.M2 += self.delta * (datum - self.mean)
        if datum < self._min:
            self._min = datum
        if datum > self._max:
            self._max = datum

    @property
    def variance(self):
        try:
            return self.M2 / (self.n - self.ddof)
        except ZeroDivisionError:
            return 0.0

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def min(self):
        if self._min == float("inf"):
            return 0.0
        return self._min

    @property
    def max(self):
        if self._max == float("-inf"):
            return 0.0
        return self._max


def flatten_dimensions(
    data: torch.Tensor, full_name: str, dimensions_not_to_collapse: List[int]
) -> np.ndarray:
    assert isinstance(dimensions_not_to_collapse, list)
    data_np = data.numpy()
    original_shape = data_np.shape
    wanted_shape = [
        dim for i, dim in enumerate(original_shape) if i in dimensions_not_to_collapse
    ] + [-1]
    data_reshaped = data_np.reshape(wanted_shape)
    return data_reshaped


def accumulate_by_key(
    nested_dicts: Dict[str, torch.Tensor | dict],
    accumulator_dicts: Dict,
    dimensions_to_keep_by_key: Dict,
    skip_nans: bool = True,
    skip_inf: bool = True,
):
    for key, value in nested_dicts.items():
        if key in excluded_keys:
            continue
        if isinstance(value, dict):
            if key not in accumulator_dicts:
                accumulator_dicts[key] = {}
            accumulate_by_key(
                value,
                accumulator_dicts=accumulator_dicts[key],
                dimensions_to_keep_by_key=dimensions_to_keep_by_key.get(key, {}),
                skip_nans=skip_nans,
                skip_inf=skip_inf,
            )
        else:
            flattened = flatten_dimensions(
                value, key, dimensions_to_keep_by_key.get(key, [])
            )
            shape = flattened.shape
            assert (
                len(shape) <= 2
            ), f"Expected 2D array, got {shape}"  # TODO: consider more dimensions
            # print(key, "flattened.shape", flattened.shape)
            if len(shape) == 2:
                if key not in accumulator_dicts:
                    accumulator_dicts[key] = [
                        OnlineMeanAndVariance() for _ in range(shape[0])
                    ]
                for i in range(shape[0]):
                    selected_values = flattened[i]
                    # print("selected_values.shape", selected_values.shape)
                    # flatten out all the rest
                    selected_values = selected_values.reshape(-1)
                    if skip_nans:
                        selected_values = selected_values[~np.isnan(selected_values)]
                    if skip_inf:
                        selected_values = selected_values[~np.isinf(selected_values)]
                    # convert to list
                    selected_values = selected_values.tolist()
                    # print(
                    #     key,
                    #     "len(selected_values)",
                    #     len(selected_values),
                    #     type(selected_values),
                    #     selected_values[:2],
                    # )
                    for val in selected_values:
                        accumulator_dicts[key][i].include(val)
            elif len(shape) == 1:
                if key not in accumulator_dicts:
                    accumulator_dicts[key] = OnlineMeanAndVariance()
                if skip_nans:
                    flattened = flattened[~np.isnan(flattened)]
                if skip_inf:
                    flattened = flattened[~np.isinf(flattened)]
                selected_values = flattened.tolist()
                # if key == "Land":
                #     print(key, "selected_values", selected_values)
                # print(
                #     key,
                #     "len(selected_values)",
                #     len(selected_values),
                #     type(selected_values),
                #     selected_values[:2],
                # )
                for val in selected_values:
                    accumulator_dicts[key].include(val)
            else:
                raise ValueError(f"Unexpected shape {shape}")


def get_mean_std_min_max_count_by_key(
    accumulator_dicts: Dict,
) -> Tuple[dict, dict, dict, dict, dict]:
    means_by_key = {}
    std_by_key = {}
    min_by_key = {}
    max_by_key = {}
    count_valid_by_key = {}
    for key, accumulator in accumulator_dicts.items():
        if isinstance(accumulator, dict):
            (
                means_by_key[key],
                std_by_key[key],
                min_by_key[key],
                max_by_key[key],
                count_valid_by_key[key],
            ) = get_mean_std_min_max_count_by_key(accumulator)
        elif isinstance(accumulator, list):
            means_by_key[key] = [acc.mean for acc in accumulator]
            std_by_key[key] = [acc.std.item() for acc in accumulator]
            min_by_key[key] = [acc.min for acc in accumulator]
            max_by_key[key] = [acc.max for acc in accumulator]
            count_valid_by_key[key] = [float(acc.n) for acc in accumulator]
        else:
            means_by_key[key] = accumulator.mean
            std_by_key[key] = accumulator.std.item()
            min_by_key[key] = accumulator.min
            max_by_key[key] = accumulator.max
            count_valid_by_key[key] = float(accumulator.n)
    return means_by_key, std_by_key, min_by_key, max_by_key, count_valid_by_key


def combine_dicts_by_key(
    nested_dicts: List[Dict],
    names: List[str],
) -> Dict:
    keys_sets = set([frozenset(d.keys()) for d in nested_dicts])
    assert len(keys_sets) == 1, f"multiple keys_sets: {keys_sets}"
    keys = keys_sets.pop()
    result = {}
    for key in keys:
        types = set([type(d[key]) for d in nested_dicts])
        assert len(types) == 1, f"types are different at key {key}: {types}"
        single_type = types.pop()
        if single_type == dict:
            result[key] = combine_dicts_by_key(
                [d[key] for d in nested_dicts],
                names=names,
            )
        elif single_type in [list, float, np.nan]:
            result[key] = {names[i]: d[key] for i, d in enumerate(nested_dicts)}
        else:
            raise ValueError(f"Unsupported type at key {key}: {single_type}")
    return result


@app.command()
def compute_and_save_stats(
    batches_dir: Path = BATCHES_DATA_DIR,
):
    print("batches_dir", batches_dir)
    all_batches = glob.glob(str(batches_dir / "*.pt"))
    print("len(all_batches)", len(all_batches))

    # first accumulate all values
    accumulator_dicts = {}
    [
        accumulate_by_key(
            torch.load(batch, weights_only=True),
            accumulator_dicts=accumulator_dicts,
            dimensions_to_keep_by_key=dimensions_to_keep_by_key,
        )
        for batch in tqdm(all_batches, desc="Calculating stats")
    ]
    # then get the values
    means_by_key, std_by_key, mins_by_key, maxs_by_key, counts_by_key = (
        get_mean_std_min_max_count_by_key(accumulator_dicts)
    )
    print("means_by_key", means_by_key)
    print("std_by_key", std_by_key)
    print("mins_by_key", mins_by_key)
    print("maxs_by_key", maxs_by_key)
    print("counts_by_key", counts_by_key)
    res = combine_dicts_by_key(
        [means_by_key, std_by_key, mins_by_key, maxs_by_key, counts_by_key],
        names=["mean", "std", "min", "max", "count_valid"],
    )
    print(res)
    output_file_path = batches_dir / "statistics.json"
    with open(output_file_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Saved statistics to {output_file_path}")


if __name__ == "__main__":
    app()
