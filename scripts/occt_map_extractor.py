
import sys
from pathlib import Path

sys.path.insert(0, "/home/yons/Graduation/VMAS_occt/scripts")
from occt_map_point_editor import load_map_data, atomic_pickle_dump

src = Path("/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path/map_data_edited3.pkl")
dst = Path("/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path/map_data_edited4.pkl")
keep = [1, 2, 6, 7, 9, 10]

scenario_library, path_library, max_path_length, max_path_s_list = load_map_data(src)
subset = [path_library[i] for i in keep]

def scalar_last_s(path):
    s = path["s"]
    last = s[-1]
    return float(last.item()) if hasattr(last, "item") else float(last)

longest = max(subset, key=scalar_last_s)

payload = (
    scenario_library,
    subset,
    longest["s"][-1],
    longest["s"],
)
atomic_pickle_dump(payload, dst)

print(f"saved: {dst}")
for new_i, old_i in enumerate(keep):
    print(f"new_index={new_i} <- old_index={old_i}, path_ids={subset[new_i].get('path_ids')}")

