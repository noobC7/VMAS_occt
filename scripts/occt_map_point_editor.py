#!/usr/bin/env python3
"""Interactive editor for OCCT path vertices stored in map_data.pkl."""

from __future__ import annotations

import argparse
import io
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = Path(tempfile.gettempdir()) / "mpl_config"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

try:
    import torch
except ImportError:
    torch = None


FIELD_ALIASES = {
    "center": "center_vertices",
    "center_vertices": "center_vertices",
    "left": "left_vertices",
    "left_vertices": "left_vertices",
    "right": "right_vertices",
    "right_vertices": "right_vertices",
}

FIELD_LABELS = {
    "center_vertices": "Center",
    "left_vertices": "Left boundary",
    "right_vertices": "Right boundary",
}

FIELD_COLORS = {
    "center_vertices": "#6f42c1",
    "left_vertices": "#d62728",
    "right_vertices": "#1f77b4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Edit one OCCT path vertex set in a map_data.pkl file."
    )
    parser.add_argument(
        "--input-pkl",
        required=True,
        type=Path,
        help="Path to the source map_data.pkl file.",
    )
    parser.add_argument(
        "--output-pkl",
        required=True,
        type=Path,
        help="Path to save the edited map_data.pkl file.",
    )
    parser.add_argument(
        "--road-index",
        required=True,
        type=int,
        help="Index into path_library.",
    )
    parser.add_argument(
        "--edit-target",
        required=True,
        choices=sorted(FIELD_ALIASES.keys()),
        help="Which vertex set to edit.",
    )
    parser.add_argument(
        "--pick-radius",
        default=14.0,
        type=float,
        help="Point selection radius in display pixels.",
    )
    parser.add_argument(
        "--point-size",
        default=26.0,
        type=float,
        help="Scatter point size for the editable series.",
    )
    return parser.parse_args()


def normalize_field_name(field_name: str) -> str:
    try:
        return FIELD_ALIASES[field_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported edit target: {field_name}") from exc


def is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def cpu_load_from_bytes(blob: bytes) -> Any:
    if torch is None:
        raise RuntimeError(
            "This pickle contains torch tensors, but torch is not installed in the current environment."
        )
    return torch.load(io.BytesIO(blob), map_location="cpu")


class CpuTensorUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch.storage" and name == "_load_from_bytes":
            return cpu_load_from_bytes
        return super().find_class(module, name)


def load_map_data(path: Path) -> Tuple[Any, list, Any, Any]:
    try:
        with path.open("rb") as handle:
            data = CpuTensorUnpickler(handle).load()
    except Exception as exc:
        raise RuntimeError(
            "Failed to load the pickle file. Please run this script in the same Python "
            "environment that can load your current map_data.pkl, including torch and "
            "the CommonRoad dependencies used to create it."
        ) from exc

    if not isinstance(data, tuple) or len(data) != 4:
        raise ValueError("Expected map_data.pkl to contain a 4-item tuple.")

    scenario_library, path_library, max_path_length, max_path_s_list = data
    if not isinstance(path_library, list):
        raise ValueError("Expected path_library to be a list.")
    return scenario_library, path_library, max_path_length, max_path_s_list


def to_numpy_array(value: Any, expected_last_dim: Optional[int] = None) -> np.ndarray:
    if is_torch_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float64)
    if expected_last_dim is not None:
        if array.ndim != 2 or array.shape[1] != expected_last_dim:
            raise ValueError(
                f"Expected an array with shape [N, {expected_last_dim}], got {array.shape}."
            )
    return array.copy()


def restore_points_like(template: Any, points: np.ndarray) -> Any:
    if is_torch_tensor(template):
        return torch.as_tensor(points, dtype=template.dtype, device=template.device)
    if isinstance(template, np.ndarray):
        return np.asarray(points, dtype=template.dtype)
    if isinstance(template, tuple):
        return tuple(tuple(float(coord) for coord in point) for point in points.tolist())
    if isinstance(template, list):
        return [[float(coord) for coord in point] for point in points.tolist()]
    return points.tolist()


def atomic_pickle_dump(payload: Tuple[Any, list, Any, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{output_path.stem}_",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(fd)
    temp_file = Path(temp_path)
    try:
        with temp_file.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temp_file, output_path)
    finally:
        if temp_file.exists():
            temp_file.unlink()


def format_path_title(path_entry: Dict[str, Any], road_index: int, edit_field: str) -> str:
    map_name = path_entry.get("map_name", "<unknown map>")
    path_ids = path_entry.get("path_ids", [])
    return (
        f"Road {road_index} | {map_name}\n"
        f"path_ids={path_ids} | editing={FIELD_LABELS[edit_field]}"
    )


class TrajectoryPointEditor:
    def __init__(
        self,
        vertices: Dict[str, np.ndarray],
        edit_field: str,
        title: str,
        pick_radius: float,
        point_size: float,
    ) -> None:
        self.vertices = {key: value.copy() for key, value in vertices.items()}
        self.original_vertices = {key: value.copy() for key, value in vertices.items()}
        self.edit_field = edit_field
        self.pick_radius = pick_radius
        self.point_size = point_size

        self.drag_index: Optional[int] = None
        self.drag_start: Optional[np.ndarray] = None
        self.linear_start_index: Optional[int] = None
        self.highlight_indices = []
        self.mode = "drag"
        self.undo_stack = []
        self.saved = False

        self.figure, self.ax = plt.subplots(figsize=(11, 8))
        self.figure.subplots_adjust(bottom=0.18, top=0.88)
        self.ax.set_title(title)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.grid(True, alpha=0.25)
        self.ax.set_aspect("equal", adjustable="box")

        self.lines = {}
        self.scatters = {}
        for field_name in ("center_vertices", "left_vertices", "right_vertices"):
            alpha = 1.0 if field_name == self.edit_field else 0.6
            linewidth = 2.0 if field_name == self.edit_field else 1.2
            linestyle = "--" if field_name == "center_vertices" else "-"
            (line,) = self.ax.plot(
                self.vertices[field_name][:, 0],
                self.vertices[field_name][:, 1],
                color=FIELD_COLORS[field_name],
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha,
                label=FIELD_LABELS[field_name],
            )
            scatter = self.ax.scatter(
                self.vertices[field_name][:, 0],
                self.vertices[field_name][:, 1],
                s=self.point_size if field_name == self.edit_field else max(self.point_size * 0.55, 10.0),
                c=FIELD_COLORS[field_name],
                alpha=alpha,
                edgecolors="black" if field_name == self.edit_field else "none",
                linewidths=0.5 if field_name == self.edit_field else 0.0,
                zorder=3,
            )
            self.lines[field_name] = line
            self.scatters[field_name] = scatter

        self.selected_point = self.ax.scatter(
            [], [],
            s=self.point_size * 3.4,
            facecolors="none",
            edgecolors="#ffbf00",
            linewidths=2.0,
            zorder=5,
        )
        self.ax.legend(loc="upper right")
        self.ax.axis('equal')
        self.status_text = self.figure.text(
            0.02,
            0.02,
            "",
            fontsize=10,
        )

        mode_ax = self.figure.add_axes([0.50, 0.07, 0.15, 0.06])
        save_ax = self.figure.add_axes([0.66, 0.07, 0.11, 0.06])
        undo_ax = self.figure.add_axes([0.78, 0.07, 0.09, 0.06])
        reset_ax = self.figure.add_axes([0.88, 0.07, 0.09, 0.06])
        self.mode_button = Button(mode_ax, "")
        self.save_button = Button(save_ax, "Save")
        self.undo_button = Button(undo_ax, "Undo")
        self.reset_button = Button(reset_ax, "Reset")

        self.mode_button.on_clicked(self._toggle_mode)
        self.save_button.on_clicked(self._save_clicked)
        self.undo_button.on_clicked(self._undo_clicked)
        self.reset_button.on_clicked(self._reset_clicked)

        self.figure.canvas.mpl_connect("button_press_event", self._on_press)
        self.figure.canvas.mpl_connect("button_release_event", self._on_release)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.figure.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("close_event", self._on_close)

        self._set_axis_limits()
        self._sync_mode_button()
        self._update_status()

    def show(self) -> Dict[str, np.ndarray]:
        plt.show()
        return self.vertices

    def _set_axis_limits(self) -> None:
        stacked = np.vstack(list(self.vertices.values()))
        if len(stacked) == 0:
            return
        xy_min = stacked.min(axis=0)
        xy_max = stacked.max(axis=0)
        span = np.maximum(xy_max - xy_min, 1.0)
        margin = span * 0.08
        self.ax.set_xlim(xy_min[0] - margin[0], xy_max[0] + margin[0])
        self.ax.set_ylim(xy_min[1] - margin[1], xy_max[1] + margin[1])

    def _refresh(self, expand_limits: bool = False) -> None:
        for field_name in ("center_vertices", "left_vertices", "right_vertices"):
            points = self.vertices[field_name]
            self.lines[field_name].set_data(points[:, 0], points[:, 1])
            self.scatters[field_name].set_offsets(points)

        if not self.highlight_indices:
            self.selected_point.set_offsets(np.empty((0, 2)))
        else:
            self.selected_point.set_offsets(self.vertices[self.edit_field][self.highlight_indices])

        if expand_limits:
            self._set_axis_limits()
        self.figure.canvas.draw_idle()

    def _default_status_message(self) -> str:
        if self.mode == "drag":
            return (
                "Mode: Drag | Left drag: move point | Wheel: zoom | "
                "Button/m: switch mode | s: save | u: undo | r: reset | q: quit"
            )
        return (
            "Mode: Linear | Click 2 points: flatten segment | Wheel: zoom | "
            "Button/m: switch mode | s: save | u: undo | r: reset | q: quit"
        )

    def _update_status(self, message: Optional[str] = None) -> None:
        status_message = self._default_status_message()
        if message:
            status_message = f"{message} | {status_message}"
        self.status_text.set_text(status_message)
        self.figure.canvas.draw_idle()

    def _sync_mode_button(self) -> None:
        button_label = "Mode: Drag" if self.mode == "drag" else "Mode: Linear"
        self.mode_button.label.set_text(button_label)

    def _clear_active_selection(self) -> None:
        self.drag_index = None
        self.drag_start = None
        self.linear_start_index = None
        self.highlight_indices = []

    def _toggle_mode(self, _event=None) -> None:
        self.mode = "linear" if self.mode == "drag" else "drag"
        self._clear_active_selection()
        self._sync_mode_button()
        self._refresh()
        self._update_status("Switched editor mode")

    def _record_undo(
        self,
        point_indices: np.ndarray,
        old_points: np.ndarray,
        new_points: np.ndarray,
        description: str,
    ) -> bool:
        if np.allclose(old_points, new_points):
            return False
        self.undo_stack.append(
            {
                "indices": np.asarray(point_indices, dtype=np.int64).copy(),
                "old_points": np.asarray(old_points, dtype=np.float64).copy(),
                "new_points": np.asarray(new_points, dtype=np.float64).copy(),
                "description": description,
            }
        )
        return True

    def _apply_linear_interpolation(self, first_index: int, second_index: int) -> None:
        if first_index == second_index:
            self.highlight_indices = [first_index]
            self._refresh()
            self._update_status("Choose a different second point for linear interpolation")
            return

        start_index, end_index = sorted((first_index, second_index))
        point_indices = np.arange(start_index, end_index + 1, dtype=np.int64)
        old_points = self.vertices[self.edit_field][point_indices].copy()
        new_points = np.linspace(old_points[0], old_points[-1], num=len(point_indices))
        self.vertices[self.edit_field][point_indices] = new_points

        self.highlight_indices = []
        self.linear_start_index = None
        self._refresh()

        if self._record_undo(
            point_indices,
            old_points,
            new_points,
            f"Linear interpolation {start_index}-{end_index}",
        ):
            self._update_status(
                f"Flattened points {start_index}-{end_index} with linear interpolation"
            )
        else:
            self._update_status(
                f"Points {start_index}-{end_index} were already aligned linearly"
            )

    def _find_nearest_point(self, event) -> Optional[int]:
        if event.inaxes != self.ax or event.x is None or event.y is None:
            return None
        points = self.vertices[self.edit_field]
        screen_points = self.ax.transData.transform(points)
        mouse = np.array([event.x, event.y], dtype=np.float64)
        distances = np.linalg.norm(screen_points - mouse, axis=1)
        nearest_index = int(np.argmin(distances))
        if distances[nearest_index] <= self.pick_radius:
            return nearest_index
        return None

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes != self.ax:
            return
        nearest_index = self._find_nearest_point(event)
        if nearest_index is None:
            return

        if self.mode == "drag":
            self.drag_index = nearest_index
            self.drag_start = self.vertices[self.edit_field][nearest_index].copy()
            self.highlight_indices = [nearest_index]
            self._refresh()
            self._update_status(f"Dragging point {nearest_index}")
            return

        if self.linear_start_index is None:
            self.linear_start_index = nearest_index
            self.highlight_indices = [nearest_index]
            self._refresh()
            self._update_status(f"Selected point {nearest_index}; choose the second point")
            return

        self._apply_linear_interpolation(self.linear_start_index, nearest_index)

    def _on_motion(self, event) -> None:
        if self.mode != "drag" or self.drag_index is None or event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.vertices[self.edit_field][self.drag_index] = [event.xdata, event.ydata]
        # Preserve the user's current zoom while dragging points.
        self._refresh()

    def _on_scroll(self, event) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        scale_base = 1.2
        if event.button == "up" or getattr(event, "step", 0) > 0:
            scale_factor = 1.0 / scale_base
        elif event.button == "down" or getattr(event, "step", 0) < 0:
            scale_factor = scale_base
        else:
            return

        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        current_width = current_xlim[1] - current_xlim[0]
        current_height = current_ylim[1] - current_ylim[0]
        if abs(current_width) < 1e-12 or abs(current_height) < 1e-12:
            return

        relative_x = (event.xdata - current_xlim[0]) / current_width
        relative_y = (event.ydata - current_ylim[0]) / current_height
        new_width = current_width * scale_factor
        new_height = current_height * scale_factor

        self.ax.set_xlim(
            event.xdata - new_width * relative_x,
            event.xdata + new_width * (1.0 - relative_x),
        )
        self.ax.set_ylim(
            event.ydata - new_height * relative_y,
            event.ydata + new_height * (1.0 - relative_y),
        )
        self.figure.canvas.draw_idle()

    def _on_release(self, event) -> None:
        if self.mode != "drag" or self.drag_index is None or self.drag_start is None:
            return
        new_point = self.vertices[self.edit_field][self.drag_index].copy()
        moved_index = self.drag_index
        if self._record_undo(
            np.array([moved_index], dtype=np.int64),
            self.drag_start.reshape(1, 2),
            new_point.reshape(1, 2),
            f"Move point {moved_index}",
        ):
            self._update_status(
                f"Moved point {moved_index} -> ({new_point[0]:.3f}, {new_point[1]:.3f})"
            )
        else:
            self._update_status(f"Point {moved_index} unchanged")
        self.drag_index = None
        self.drag_start = None
        self.highlight_indices = []
        self._refresh()

    def _undo_clicked(self, _event) -> None:
        self.undo()

    def undo(self) -> None:
        if not self.undo_stack:
            self._update_status("Undo stack is empty")
            return
        undo_item = self.undo_stack.pop()
        self.vertices[self.edit_field][undo_item["indices"]] = undo_item["old_points"]
        self._clear_active_selection()
        self._refresh()
        self._update_status(f"Undo {undo_item['description']}")

    def _reset_clicked(self, _event) -> None:
        self.reset()

    def reset(self) -> None:
        self.vertices = {key: value.copy() for key, value in self.original_vertices.items()}
        self.undo_stack.clear()
        self._clear_active_selection()
        self._refresh(expand_limits=True)
        self._update_status("Reset to original vertices")

    def _save_clicked(self, _event) -> None:
        self.saved = True
        plt.close(self.figure)

    def _on_key_press(self, event) -> None:
        if event.key in {"s", "ctrl+s", "cmd+s"}:
            self.saved = True
            plt.close(self.figure)
        elif event.key == "m":
            self._toggle_mode()
        elif event.key == "u":
            self.undo()
        elif event.key == "r":
            self.reset()
        elif event.key == "q":
            plt.close(self.figure)

    def _on_close(self, _event) -> None:
        if not self.saved:
            print("Window closed without saving. No file was written.")


def main() -> None:
    args = parse_args()
    edit_field = normalize_field_name(args.edit_target)

    scenario_library, path_library, max_path_length, max_path_s_list = load_map_data(args.input_pkl)

    road_index = args.road_index
    if road_index < 0:
        road_index += len(path_library)
    if road_index < 0 or road_index >= len(path_library):
        raise IndexError(f"road-index {args.road_index} is out of range for {len(path_library)} paths.")

    path_entry = path_library[road_index]
    for field_name in ("center_vertices", "left_vertices", "right_vertices"):
        if field_name not in path_entry:
            raise KeyError(f"Missing field {field_name} in path_library[{road_index}].")

    vertices = {
        field_name: to_numpy_array(path_entry[field_name], expected_last_dim=2)
        for field_name in ("center_vertices", "left_vertices", "right_vertices")
    }

    print(f"Loaded {len(path_library)} paths from {args.input_pkl}")
    print(f"Editing road index: {road_index}")
    print(f"Edit target: {FIELD_LABELS[edit_field]}")
    print(f"Map: {path_entry.get('map_name', '<unknown map>')}")
    print(f"path_ids: {path_entry.get('path_ids', [])}")
    print(
        "Controls: drag mode can move a single point, linear mode can flatten a segment "
        "between two clicked points, mouse wheel zooms, m/button switches mode, "
        "s saves, u undoes, r resets, q quits."
    )

    editor = TrajectoryPointEditor(
        vertices=vertices,
        edit_field=edit_field,
        title=format_path_title(path_entry, road_index, edit_field),
        pick_radius=args.pick_radius,
        point_size=args.point_size,
    )
    edited_vertices = editor.show()

    if not editor.saved:
        return

    updated_path_entry = dict(path_entry)
    updated_path_entry[edit_field] = restore_points_like(path_entry[edit_field], edited_vertices[edit_field])

    updated_path_library = list(path_library)
    updated_path_library[road_index] = updated_path_entry

    payload = (
        scenario_library,
        updated_path_library,
        max_path_length,
        max_path_s_list,
    )
    atomic_pickle_dump(payload, args.output_pkl)

    print(f"Saved edited pickle to {args.output_pkl}")
    print("Derived fields were kept unchanged by design.")


if __name__ == "__main__":
    main()
