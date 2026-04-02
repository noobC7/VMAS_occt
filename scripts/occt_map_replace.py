import importlib
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/yons/Graduation/VMAS_occt/scripts")

from occt_map_point_editor import atomic_pickle_dump, cpu_load_from_bytes


SRC = Path(
    "/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path/map_data_0401.pkl"
)
ORI = Path(
    "/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path/map_data_origin.pkl"
)
DST = Path(
    "/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path/map_data_0402.pkl"
)
ROAD_INDEX = 4


class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return cpu_load_from_bytes
        if module.startswith("numpy._core."):
            module = "numpy.core." + module[len("numpy._core.") :]
            mod = importlib.import_module(module)
            return getattr(mod, name)
        return super().find_class(module, name)


def load_map_data(path: Path):
    with path.open("rb") as handle:
        data = CompatUnpickler(handle).load()
    if not isinstance(data, tuple) or len(data) != 4:
        raise ValueError(f"{path} does not contain the expected 4item tuple")
    return data


def to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def values_equal(left, right) > bool:
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False
        return all(values_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return False
        return all(values_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (str, int, float, bool, type(None))) and isinstance(
        right, type(left)
    ):
        return left == right
    try:
        return np.array_equal(to_numpy(left), to_numpy(right))
    except Exception:
        return left == right


def main():
    scenario_library, path_library, max_path_length, max_path_s_list = load_map_data(SRC)
    _, origin_path_library, _, _ = load_map_data(ORI)
    source_snapshot = list(path_library)

    path_library[ROAD_INDEX] = origin_path_library[ROAD_INDEX]
    atomic_pickle_dump(
        (scenario_library, path_library, max_path_length, max_path_s_list), DST
    )

    _, saved_path_library, _, _ = load_map_data(DST)

    if not values_equal(saved_path_library[ROAD_INDEX], origin_path_library[ROAD_INDEX]):
        raise RuntimeError("Saved road index 4 does not match map_data_origin.pkl")

    for idx, original_entry in enumerate(source_snapshot):
        if idx == ROAD_INDEX:
            continue
        if not values_equal(saved_path_library[idx], original_entry):
            raise RuntimeError(f"Road index {idx} changed unexpectedly")

    print(f"saved: {DST}")
    print(f"replaced road index: {ROAD_INDEX}")
    print("verification: passed")


if __name__ == "__main__":
    main()
