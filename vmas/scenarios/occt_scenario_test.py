import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vmas import render_interactively
from vmas.scenarios.occt_scenario import AGENT_INDEX_FOCUS, MethodClass, Scenario


DEFAULT_METHODS = ("pid", "mppi")
DEFAULT_ROAD_IDS = list(range(6))
DEFAULT_RESULT_DIR = Path(__file__).resolve().parent / "occt_scenario_test_result"
SPECIAL_ALL_HINGED_SKIP_ROADS = (0, 1)


def parse_method(method: str) -> MethodClass:
    try:
        return MethodClass[method.strip().upper()]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported method '{method}'. Expected one of: pid, mppi."
        ) from exc


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z._-]+", "_", name.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "road"


def to_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return torch.as_tensor(value).detach().cpu()


def get_follower_ids(scenario: Scenario) -> List[int]:
    followers = scenario.FOLLOWER_SLICE
    if isinstance(followers, slice):
        start = 0 if followers.start is None else followers.start
        stop = scenario.n_agents if followers.stop is None else followers.stop
        step = 1 if followers.step is None else followers.step
        return list(range(start, stop, step))
    return [int(agent_id) for agent_id in followers]


class ValidationRecorder:
    def __init__(
        self,
        method: MethodClass,
        road_id: int,
        requested_episodes: int,
        result_dir: Path,
        capture_episode_videos: bool = False,
    ):
        self.method = method
        self.method_name = method.name.lower()
        self.road_id = int(road_id)
        self.requested_episodes = int(requested_episodes)
        self.result_dir = result_dir
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.capture_episode_videos = bool(capture_episode_videos)

        self.dt: Optional[float] = None
        self.n_agents: Optional[int] = None
        self.agent_names: Optional[List[str]] = None
        self.followers: Optional[List[int]] = None
        self.road_name: Optional[str] = None
        self.first_saved_episode_outcome: Optional[str] = None
        self.pending_opposite_episode_outcome: Optional[str] = None
        self.saved_render_paths: List[str] = []

        self.current_steps: List[Dict[str, Any]] = []
        self.episodes: List[Dict[str, Any]] = []

    def _get_episode_outcome_label(self, episode_summary: Dict[str, Any]) -> str:
        return "success" if bool(episode_summary["success"]) else "fail"

    def _build_episode_render_name(self, episode_summary: Dict[str, Any]) -> str:
        road_name = sanitize_filename(self.road_name or f"road_{self.road_id}")
        outcome = self._get_episode_outcome_label(episode_summary)
        episode_index = int(episode_summary["episode_index"])
        return str(
            self.result_dir
            / (
                f"traditional_{self.method_name}_road{self.road_id}_{road_name}_"
                f"ep{episode_index}_{outcome}"
            )
        )

    def _stack_agent_info(
        self,
        info: Dict[str, Dict[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        assert self.agent_names is not None
        info_keys = list(info[self.agent_names[0]].keys())
        stacked_info = {}
        for key in info_keys:
            stacked_info[key] = torch.stack(
                [to_cpu_tensor(info[agent_name][key]) for agent_name in self.agent_names],
                dim=0,
            )
        return stacked_info

    def _finalize_episode(self, episode_index: int) -> None:
        if not self.current_steps:
            return

        info_keys = list(self.current_steps[0]["info"].keys())
        episode_info = {
            key: torch.stack(
                [step_record["info"][key] for step_record in self.current_steps], dim=0
            )
            for key in info_keys
        }
        reward = torch.stack(
            [step_record["reward"] for step_record in self.current_steps], dim=0
        ).to(torch.float32)
        done = torch.stack(
            [step_record["done"] for step_record in self.current_steps], dim=0
        ).to(torch.bool)

        final_info = {key: value[-1] for key, value in episode_info.items()}
        episode_summary = {
            "episode_index": int(episode_index),
            "num_steps": int(reward.shape[0]),
            "reward": reward,
            "done": done,
            "step_compute_time_s": torch.tensor(
                [
                    float(step_record["step_compute_time_s"])
                    for step_record in self.current_steps
                ],
                dtype=torch.float64,
            ),
            "info": episode_info,
            "total_reward": reward.sum(dim=0),
            "success": bool(final_info["episode_success"][0].item()),
            "failure": bool(final_info["episode_failure"][0].item()),
            "collision_with_agents": bool(
                final_info["done_collision_with_agents"][0].item()
            ),
            "collision_with_lanelets": bool(
                final_info["done_collision_with_lanelets"][0].item()
            ),
            "collision_with_exit_segments": bool(
                final_info["done_collision_with_exit_segments"][0].item()
            ),
            "road_id": int(final_info["road_batch_id"][0].item()),
            "road_name": self.road_name,
        }
        episode_summary["mean_step_compute_time_s"] = float(
            episode_summary["step_compute_time_s"].mean().item()
        )
        episode_summary["max_step_compute_time_s"] = float(
            episode_summary["step_compute_time_s"].max().item()
        )
        episode_summary["min_step_compute_time_s"] = float(
            episode_summary["step_compute_time_s"].min().item()
        )
        self.episodes.append(episode_summary)
        self.current_steps = []

        print(
            f"[validation] method={self.method_name} road_id={self.road_id} "
            f"episode={len(self.episodes)}/{self.requested_episodes} "
            f"steps={episode_summary['num_steps']} success={episode_summary['success']} "
            f"step_time_mean={episode_summary['mean_step_compute_time_s']:.6f}s "
            f"step_time_max={episode_summary['max_step_compute_time_s']:.6f}s"
        )

    def on_step(
        self,
        *,
        env,
        obs,
        rew,
        done,
        info,
        episode_index: int,
        step_index: int,
        total_rew,
        step_compute_time_s: Optional[float] = None,
    ) -> None:
        del obs, step_index, total_rew
        scenario = env.unwrapped.scenario

        if self.dt is None:
            self.dt = float(env.unwrapped.world.dt)
        if self.agent_names is None:
            self.agent_names = [agent.name for agent in env.unwrapped.agents]
            self.n_agents = len(self.agent_names)
        if self.followers is None:
            self.followers = get_follower_ids(scenario)
        if self.road_name is None and hasattr(scenario.road, "batch_map_name"):
            self.road_name = str(scenario.road.batch_map_name[0])

        self.current_steps.append(
            {
                "reward": torch.as_tensor(rew, dtype=torch.float32),
                "done": torch.as_tensor(done, dtype=torch.bool),
                "step_compute_time_s": (
                    float("nan")
                    if step_compute_time_s is None
                    else float(step_compute_time_s)
                ),
                "info": self._stack_agent_info(info),
            }
        )

        if done:
            self._finalize_episode(episode_index)

    def save(self) -> Path:
        if self.current_steps:
            self._finalize_episode(len(self.episodes))

        if (
            self.capture_episode_videos
            and self.first_saved_episode_outcome is not None
            and self.pending_opposite_episode_outcome is not None
        ):
            print(
                f"[validation] method={self.method_name} road_id={self.road_id} "
                f"no episode with outcome={self.pending_opposite_episode_outcome} "
                f"found after the first saved episode"
            )

        road_name = self.road_name or f"road_{self.road_id}"
        file_name = (
            f"traditional_{self.method_name}_road{self.road_id}_"
            f"{sanitize_filename(road_name)}_n{self.requested_episodes}.pt"
        )
        save_path = self.result_dir / file_name

        payload = {
            "schema_version": 1,
            "format": "occt_traditional_validation",
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "method": self.method_name,
            "method_id": int(self.method),
            "requested_road_id": self.road_id,
            "road_name": road_name,
            "episodes_requested": self.requested_episodes,
            "episodes_completed": len(self.episodes),
            "dt": self.dt,
            "n_agents": self.n_agents,
            "agent_names": self.agent_names,
            "followers": self.followers,
            "saved_render_paths": list(self.saved_render_paths),
            "episodes": self.episodes,
        }
        torch.save(payload, save_path)
        return save_path

    def on_episode_end(
        self,
        *,
        env,
        obs,
        rew,
        done,
        info,
        episode_index: int,
        episode_length: int,
        total_rew,
    ) -> Dict[str, Any]:
        del env, obs, rew, done, info, episode_length, total_rew
        if not self.capture_episode_videos or not self.episodes:
            return {"should_continue": True, "save_render": False}

        episode_summary = self.episodes[-1]
        if int(episode_summary["episode_index"]) != int(episode_index):
            episode_summary = next(
                (
                    saved_episode
                    for saved_episode in reversed(self.episodes)
                    if int(saved_episode["episode_index"]) == int(episode_index)
                ),
                self.episodes[-1],
            )

        outcome = self._get_episode_outcome_label(episode_summary)
        should_save_render = False

        if self.first_saved_episode_outcome is None:
            self.first_saved_episode_outcome = outcome
            self.pending_opposite_episode_outcome = (
                "fail" if outcome == "success" else "success"
            )
            should_save_render = True
        elif self.pending_opposite_episode_outcome == outcome:
            should_save_render = True
            self.pending_opposite_episode_outcome = None

        if not should_save_render:
            return {"should_continue": True, "save_render": False}

        render_name = self._build_episode_render_name(episode_summary)
        render_path = render_name + ".mp4"
        self.saved_render_paths.append(render_path)
        print(
            f"[validation] method={self.method_name} road_id={self.road_id} "
            f"saved render episode={episode_index} outcome={outcome} path={render_path}"
        )
        return {
            "should_continue": True,
            "save_render": True,
            "render_name": render_name,
        }


def run_validation(
    method: MethodClass,
    road_id: int,
    episodes: int,
    result_dir: Path,
    seed: Optional[int],
    display_info: bool,
    control_two_agents: bool,
    save_render: bool,
    disable_all_hinged_done_road_ids: Optional[List[int]],
) -> Path:
    recorder = ValidationRecorder(
        method=method,
        road_id=road_id,
        requested_episodes=episodes,
        result_dir=result_dir,
        capture_episode_videos=save_render,
    )

    print(
        f"[validation] start method={method.name.lower()} road_id={road_id} "
        f"episodes={episodes}"
    )
    render_interactively(
        Scenario(),
        control_two_agents=control_two_agents,
        display_info=display_info,
        save_render=save_render,
        save_render_max_episodes=2 if save_render else None,
        seed=seed,
        agent_index_focus=AGENT_INDEX_FOCUS,
        step_callback=recorder.on_step,
        episode_end_callback=recorder.on_episode_end if save_render else None,
        max_episodes=episodes,
        traditional_control=method,
        target_road_id=road_id,
        disable_all_hinged_done_road_ids=disable_all_hinged_done_road_ids,
    )
    save_path = recorder.save()
    print(f"[validation] saved result to {save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interactive validation for PID and MPPI OCCT controllers."
    )
    parser.add_argument(
        "-n",
        "--episodes",
        type=int,
        required=True,
        help="Number of validation episodes to run for each road and method.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        help="Traditional methods to evaluate. Supported values: pid, mppi.",
    )
    parser.add_argument(
        "--roads",
        nargs="+",
        type=int,
        default=DEFAULT_ROAD_IDS,
        help="Road ids passed to target_road_id. Defaults to 0 1 2 3 4 5.",
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=DEFAULT_RESULT_DIR,
        help="Directory used to save validation result files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional environment seed.",
    )
    parser.add_argument(
        "--display-info",
        action="store_true",
        help="Display interactive info text while rendering.",
    )
    parser.add_argument(
        "--control-two-agents",
        action="store_true",
        help="Enable manual control bindings for two agents while rendering.",
    )
    parser.add_argument(
        "--save-render",
        action="store_true",
        help=(
            "For each road and method run, save the first episode mp4 and, if it "
            "appears later, the first episode with the opposite outcome."
        ),
    )
    parser.add_argument(
        "--disable-all-hinged-done-on-road01",
        action="store_true",
        help=(
            "When the road id is 0 or 1, do not end the episode only because all "
            "followers are hinged."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be a positive integer.")

    methods = [parse_method(method) for method in args.methods]
    road_ids = [int(road_id) for road_id in args.roads]

    invalid_roads = [road_id for road_id in road_ids if road_id < 0 or road_id > 6]
    if invalid_roads:
        raise ValueError(f"road ids must be in [0, 6], got {invalid_roads}.")

    saved_paths = []
    for method in methods:
        for road_id in road_ids:
            saved_paths.append(
                run_validation(
                    method=method,
                    road_id=road_id,
                    episodes=args.episodes,
                    result_dir=args.result_dir,
                    seed=args.seed,
                    display_info=args.display_info,
                    control_two_agents=args.control_two_agents,
                    save_render=args.save_render,
                    disable_all_hinged_done_road_ids=(
                        list(SPECIAL_ALL_HINGED_SKIP_ROADS)
                        if args.disable_all_hinged_done_on_road01
                        else None
                    ),
                )
            )

    print("[validation] completed files:")
    for save_path in saved_paths:
        print(f"  - {save_path}")


if __name__ == "__main__":
    main()
