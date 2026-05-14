import time
from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import World, Agent, Sphere, Box
from vmas.simulator.utils import Color
#from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle
from vmas.simulator.dynamics.dynamic_kinematic_bicycle import DynamicKinematicBicycle
from vmas.simulator.dynamics.delayed_steering_kinematic_bicycle import DelayedSteeringKinematicBicycle,KinematicBicycle
from vmas.simulator import rendering
from vmas.simulator.utils import Color, ScenarioUtils

from vmas.scenarios.road_traffic import get_perpendicular_distances,get_distances_between_agents,get_rectangle_vertices,\
    transform_from_global_to_local_coordinate,interX,exponential_decreasing_fcn,angle_eliminate_two_pi,\
    Collisions,CircularBuffer,Timer,StateBuffer
# 添加Road类导入
from vmas.scenarios.occt_map import OcctMap,OcctCRMap
from vmas.scenarios.occt_utils import OcctObservations,OcctRewards,OcctNormalizers,OcctReferencePathsAgentRelated,\
    OcctPenalties,OcctThresholds,OcctConstants,OcctDistances,check_validity,get_short_term_hinge_path_by_s,\
    get_short_term_reference_path_simple,get_short_term_reference_path_by_s,check_boolean_block,calibrate_agent_s_by_road_pts,\
    is_point_left_of_polyline,get_frenet_distances_between_agents
from vmas.scenarios.simple_mppi import SimpleMPPIController
from enum import IntEnum
class MethodClass(IntEnum):
    MARL = 0
    PID = 1
    MPPI = 2
DEFAULT_TRADITIONAL_CONTROL = MethodClass.MARL
AGENT_INDEX_FOCUS=0
class TaskClass(IntEnum):
    SIMPLE_PLATOON = 0 # without cargo
    OCCT_PLATOON = 1 # with cargo


def parse_traditional_control(value) -> MethodClass:
    if isinstance(value, MethodClass):
        return value
    if isinstance(value, str):
        normalized = value.strip().upper()
        try:
            return MethodClass[normalized]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported traditional control '{value}'. Expected one of: "
                f"{', '.join(method.name for method in MethodClass)}."
            ) from exc
    try:
        return MethodClass(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported traditional control '{value}'. Expected MethodClass, int, or str."
        ) from exc


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.device = device
        self.batch_dim = batch_dim
        self.init_params(batch_dim, device, **kwargs)
        world = self.init_world(batch_dim, device)
        self.init_agents(world, batch_dim, device)
        return world
    def get_tensor_by_distribution(self, dist_type="uniform", size=None, mean=0.0, std=1.0):
        """
        Generate a random tensor with specified distribution type.
        
        Args:
            dist_type: Distribution type, either "uniform" or "normal" (default: "uniform")
            size: Size of the tensor (default: (self.batch_dim,) for uniform, required for normal)
            mean: Mean for normal distribution (default: 0.0)
            std: Standard deviation for normal distribution (default: 1.0)
            
        Returns:
            Random tensor with specified distribution
        """
        import time
        seed = int(time.time() * 1000) % 1000000
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        if size is None:
            size = (self.batch_dim,)
        if dist_type == "uniform":
            tensor = torch.rand(size=size, device=self.device, generator=generator)
        elif dist_type == "normal":
            tensor = torch.normal(mean, std, size=size, device=self.device, generator=generator)
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}. Use 'uniform' or 'normal'.")
        return tensor
    
    def get_normal_tensor(self, mean, std, size=None):
        return self.get_tensor_by_distribution(dist_type="normal", size=size, mean=mean, std=std)
    
    def get_random_tensor(self, size=None):
        return self.get_tensor_by_distribution(dist_type="uniform", size=size)
    
    def get_platoon_space(self, platoon_vel):
        """
        Get the spacing of the platoon.
        Args:
            platoon_vel: Velocity of the platoon.
        Returns:
            platoon_space: Spacing of the platoon.
        """
        return self.still_space+self.platoon_tau*platoon_vel
    
    def init_params(self, batch_dim: int, device: torch.device, **kwargs):
        # 建议在类的__init__中初始化计时字典
        self.reset_total_time = 0.0
        self.reward_update_time=0.0
        self.reset_count=0
        self.time_records = {
            "total": 0.0,          # 函数总耗时
            "reset_agents_loop": 0.0,   # 10. 重置智能体循环（距离/碰撞）
        }
        # episode step 追踪（向量化，每个并行环境独立计数）
        self.env_current_step = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.env_total_step = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.success_count = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.failure_count = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.all_hinged_reward_granted = torch.zeros(
            batch_dim, device=device, dtype=torch.bool
        )
        self.agent_index_focus = kwargs.pop("agent_index_focus", AGENT_INDEX_FOCUS)
        self.enable_obs_audit = kwargs.pop("enable_obs_audit", False)
        self.obs_audit_interval = int(kwargs.pop("obs_audit_interval", 100))
        self.obs_audit_agent_index = int(
            kwargs.pop("obs_audit_agent_index", self.agent_index_focus)
        )
        self.obs_audit_small_threshold = float(
            kwargs.pop("obs_audit_small_threshold", 1e-2)
        )
        self.obs_audit_large_threshold = float(
            kwargs.pop("obs_audit_large_threshold", 3.0)
        )
        self.obs_audit_last_logged_step = -1
        self.obs_audit_prev_groups = {}
        self._tracking_error_last_synced_step = -1
        # world params
        self.device = device
        self.batch_dim = batch_dim
        self.traditional_control = parse_traditional_control(
            kwargs.pop("traditional_control", DEFAULT_TRADITIONAL_CONTROL)
        )
        target_road_id = kwargs.pop("target_road_id", 2)
        self.target_road_id = None if target_road_id is None else int(target_road_id)
        disable_all_hinged_done_road_ids = kwargs.pop(
            "disable_all_hinged_done_road_ids", ()
        )
        if disable_all_hinged_done_road_ids is None:
            disable_all_hinged_done_road_ids = ()
        if isinstance(disable_all_hinged_done_road_ids, int):
            disable_all_hinged_done_road_ids = (disable_all_hinged_done_road_ids,)
        self.disable_all_hinged_done_road_ids = tuple(
            sorted({int(road_id) for road_id in disable_all_hinged_done_road_ids})
        )
        self.disable_all_hinged_done_road_id_tensor = (
            torch.tensor(
                self.disable_all_hinged_done_road_ids,
                device=device,
                dtype=torch.long,
            )
            if self.disable_all_hinged_done_road_ids
            else torch.empty(0, device=device, dtype=torch.long)
        )
        self.task_class=kwargs.pop("task_class", TaskClass.OCCT_PLATOON)
        self.dt = float(kwargs.get("dt", 0.05))
        self.n_agents=kwargs.pop("n_agents", 5)
        self.logged_control_acc = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.float32
        )
        self.logged_control_steer = torch.zeros_like(self.logged_control_acc)
        self.logged_occt_drive_force = torch.zeros(
            (batch_dim, 2), device=device, dtype=torch.float32
        )
        self.logged_occt_internal_force = torch.zeros(
            batch_dim, device=device, dtype=torch.float32
        )
        self.logged_occt_resistance_force = torch.zeros(
            batch_dim, device=device, dtype=torch.float32
        )
        self.hinge_active_steps = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.long
        )
        self.reward_phase_weights = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.float32
        )
        self.reward_phase_transition_start = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.float32
        )
        self.reward_phase_transition_target = torch.zeros(
            (batch_dim, self.n_agents), device=device, dtype=torch.float32
        )
        self.reward_phase_transition_progress = torch.ones(
            (batch_dim, self.n_agents), device=device, dtype=torch.float32
        )
        self.obs_audit_agent_index = min(max(self.obs_audit_agent_index, 0), self.n_agents - 1)
        # platoon params
        self.is_loop=kwargs.pop("is_loop", False)
        # use agents_s to get ref pts for short term
        self.use_center_frenet_ref=kwargs.pop("use_center_frenet_ref", True)
        self.use_boundary_frenet_ref=kwargs.pop("use_boundary_frenet_ref", True)
        self.is_rand_arc_pos=kwargs.pop("is_rand_arc_pos", False)
        self.init_arc_pos = kwargs.pop("init_arc_pos", 0.0)
        self.init_vel_mean = kwargs.pop("init_vel_mean", 3)
        self.init_vel_std = kwargs.pop("init_vel_std", 0.0) 
        self.still_space = kwargs.pop("still_space", 6.0)
        self.platoon_tau = kwargs.pop("platoon_tau", 0.0)
        self.platoon_vel_batch = torch.zeros((self.batch_dim), device=device)
        if self.task_class == TaskClass.SIMPLE_PLATOON:
            self.n_followers = self.n_agents
            self.TRACTOR_SLICE = [0]
            self.FOLLOWER_SLICE=slice(0, self.n_agents)
        else:
            self.n_followers = self.n_agents - 2
            self.HINGE_FIRST_INDEX=0
            self.HINGE_LAST_INDEX=self.n_agents-1
            self.TRACTOR_SLICE = [self.HINGE_FIRST_INDEX,self.HINGE_LAST_INDEX]
            self.FOLLOWER_SLICE=slice(self.HINGE_FIRST_INDEX+1,self.HINGE_LAST_INDEX)
        self.n_nearing_agents_observed=kwargs.pop("n_nearing_agents_observed", 2)
        if self.n_nearing_agents_observed >= self.n_agents:
            raise ValueError("n_nearing_agents_observed must be less than n_agents")

        self.is_real_time_rendering=kwargs.pop("is_real_time_rendering", False)
        self.n_points_short_term=kwargs.pop("n_points_short_term", 4)
        self.agent_lookahead_idx = kwargs.pop("agent_lookahead_idx", 2) # lookahead index for agent tracking ref path
        self.hinge_lookahead_idx = kwargs.pop("hinge_lookahead_idx", 2) # lookahead index for hinge tracking agent path
        assert self.agent_lookahead_idx < self.n_points_short_term, "agent_lookahead_idx must be less than n_points_short_term"
        assert self.hinge_lookahead_idx < self.n_points_short_term, "hinge_lookahead_idx must be less than n_points_short_term"
        self.occt_drive_baseline = str(
            kwargs.pop("occt_drive_baseline", "4car")
        ).lower()
        drive_baseline_defaults = {
            "2car": (400.0, 5.0),
            "3car": (500.0, 7.5),
            "4car": (800.0, 9.0),
            "front_base": (800.0, 9.0),
            "rear_base": (800.0, 9.0),
        }
        if self.occt_drive_baseline not in drive_baseline_defaults:
            raise ValueError(
                "occt_drive_baseline must be one of "
                "{'2car', '3car', '4car', 'front_base', 'rear_base'}."
            )
        default_force_cap, default_power_cap_kw = drive_baseline_defaults[
            self.occt_drive_baseline
        ]
        self.occt_drive_force_cap = float(
            kwargs.pop("occt_drive_force_cap", default_force_cap)
        )
        self.occt_drive_power_cap_kw = float(
            kwargs.pop("occt_drive_power_cap_kw", default_power_cap_kw)
        )
        self.occt_drive_power_blend_speed = float(
            kwargs.pop("occt_drive_power_blend_speed", 0.5)
        )
        self.max_acceleration = float(kwargs.get("max_acceleration", 3.0))
        self.occt_kinematic_max_acceleration = float(
            kwargs.pop(
                "occt_kinematic_max_acceleration",
                0.1
                if self.occt_drive_baseline in {"front_base", "rear_base"}
                else self.max_acceleration,
            )
        )
        if self.occt_kinematic_max_acceleration <= 0.0:
            raise ValueError("occt_kinematic_max_acceleration must be positive.")
        self.mppi_horizon_steps = int(kwargs.pop("mppi_horizon_steps", 30))
        self.mppi_num_samples = int(kwargs.pop("mppi_num_samples", 256))
        self.mppi_lambda = float(kwargs.pop("mppi_lambda", 10.0))
        self.mppi_exploration = float(kwargs.pop("mppi_exploration", 0.1))
        self.mppi_debug_top_k = int(kwargs.pop("mppi_debug_top_k", 8))
        self.enable_mppi_debug_render = bool(
            kwargs.pop("enable_mppi_debug_render", True)
        )
        self.sample_interval=kwargs.pop("sample_interval", 2)
        self.boundary_offset=kwargs.pop("boundary_offset", -self.sample_interval)
        self.n_points_nearing_boundary=kwargs.pop("n_points_nearing_boundary", self.n_points_short_term+1)
        self.is_apply_mask=kwargs.pop("is_apply_mask", False)
        self.is_observe_vertices=kwargs.pop("is_observe_vertices", False)
        self.is_observe_distance_to_agents=kwargs.pop(
            "is_observe_distance_to_agents", True
        )
        self.is_add_noise=kwargs.pop("is_add_noise", False)
        self.is_observe_ref_path_other_agents=kwargs.pop(
            "is_observe_ref_path_other_agents", False
        )
        is_partial_observation=kwargs.pop("is_partial_observation", True)
        
        # Visualization
        self.visualize_semidims=True
        self.viewer_zoom = float(kwargs.pop("viewer_zoom", 20)) #7
        self.world_x_dim = kwargs.pop(
            "world_x_dim", 200
        )  # The x-dimension of the world in [m]
        self.world_y_dim = kwargs.pop(
            "world_y_dim", 150
        )  # The y-dimension of the world in [m]
        self.resolution_factor = kwargs.pop("resolution_factor", 5)  # Default 5
        self.render_origin = kwargs.pop(
            "render_origin", [self.world_x_dim / 2, self.world_y_dim / 2]
        )
        self.viewer_size = kwargs.pop(
            "viewer_size",
            (
                int(self.world_x_dim * self.resolution_factor),
                int(self.world_y_dim * self.resolution_factor),
            ),
        )
        # agent params
        self.max_speed = float(kwargs.pop("max_speed", 5))
        self.max_steering_angle = kwargs.pop(
            "max_steering_angle",
            torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)),
        )
        self.max_steering_rate = kwargs.pop(
            "max_steering_rate",
            torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)),
        )
        self.l_f = float(kwargs.get("l_f", 1.17))
        self.l_r = float(kwargs.get("l_r", 1.15))
        self.agent_length = self.l_f + self.l_r + 1.5
        self.agent_width = float(kwargs.get("agent_width", 1.5))
        self.mppi_horizon_steps = max(self.mppi_horizon_steps, 1)
        self.mppi_stage_cost_weight = torch.tensor(
            kwargs.pop("mppi_stage_cost_weight", [40.0, 8.0, 12.0, 0.05, 0.20]),
            device=device,
            dtype=torch.float32,
        )
        self.mppi_terminal_cost_weight = torch.tensor(
            kwargs.pop("mppi_terminal_cost_weight", [80.0, 12.0, 16.0]),
            device=device,
            dtype=torch.float32,
        )
        self.mppi_hinge_stage_cost_weight = torch.tensor(
            kwargs.pop("mppi_hinge_stage_cost_weight", [80.0, 12.0, 18.0, 0.05, 0.20]),
            device=device,
            dtype=torch.float32,
        )
        self.mppi_hinge_terminal_cost_weight = torch.tensor(
            kwargs.pop("mppi_hinge_terminal_cost_weight", [140.0, 18.0, 24.0]),
            device=device,
            dtype=torch.float32,
        )
        self.simple_mppi = None
        if self.traditional_control == MethodClass.MPPI:
            self.simple_mppi = SimpleMPPIController(
                num_agents=self.n_agents,
                device=device,
                dt=self.dt,
                l_f=self.l_f,
                l_r=self.l_r,
                max_steer_abs=self.max_steering_angle,
                max_accel_abs=self.max_acceleration,
                max_speed=self.max_speed,
                horizon_step_T=self.mppi_horizon_steps,
                number_of_samples_K=self.mppi_num_samples,
                param_exploration=self.mppi_exploration,
                param_lambda=self.mppi_lambda,
                stage_cost_weight=self.mppi_stage_cost_weight,
                terminal_cost_weight=self.mppi_terminal_cost_weight,
                debug_top_k=self.mppi_debug_top_k,
            )
        
        noise_level = kwargs.pop(
            "noise_level", 0.2 * self.agent_width
        )  # Noise will be generated by the standary normal distribution. This parameter controls the noise level
        n_stored_steps = kwargs.pop(
            "n_stored_steps",
            5,  # The number of steps to store (include the current step). At least one
        )
        n_observed_steps = kwargs.pop(
            "n_observed_steps", 5
        )  # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`
        use_history_observation = bool(kwargs.pop("use_history_observation", False))
        history_obs_len = int(kwargs.pop("history_obs_len", n_observed_steps))
        history_obs_dim = kwargs.pop("history_obs_dim", None)
        if history_obs_dim is not None:
            history_obs_dim = int(history_obs_dim)
            if history_obs_dim <= 0:
                raise ValueError("history_obs_dim must be a positive integer or None.")
        
        # map params
        B = batch_dim
        self.lane_width = 6  # 道路宽度
        if self.task_class == TaskClass.OCCT_PLATOON:
            self.hinge_edge_buffer=kwargs.pop("hinge_edge_buffer", self.agent_width/2*1.2) # hinge判断可用的最小边缘距离
            self.reward_transition_blend_enabled = bool(
                kwargs.pop("reward_transition_blend_enabled", False)
            )
            self.reward_transition_blend_mode = str(
                kwargs.pop("reward_transition_blend_mode", "ramp")
            ).lower()
            self.reward_transition_duration_sec = float(
                kwargs.pop("reward_transition_duration_sec", 0.2)
            )
            if self.reward_transition_duration_sec <= 0.0:
                self.reward_transition_step_delta = 1.0
            else:
                self.reward_transition_step_delta = min(
                    1.0, self.dt / self.reward_transition_duration_sec
                )
            if self.reward_transition_blend_mode not in {"ramp", "first_order", "s_curve"}:
                raise ValueError(
                    "reward_transition_blend_mode must be one of {'ramp', 'first_order', 's_curve'}."
                )
            self.rod_len = (self.n_followers+1) * self.still_space   # 货物长度 L
            self.hinge_side_width = float(kwargs.get("hinge_side_width", 5))
            self.corner_prepare_len = float(kwargs.get("corner_prepare_len", 40))
            self.hinge_relative_pos = torch.tensor(
                [
                    [0, i * self.rod_len / (self.n_agents - 1)] 
                    for i in range(self.n_agents - 1)
                ] + [[0, self.rod_len]],
                device=device,
                dtype=torch.float32
            )
            self.cargo_half_width = float(kwargs.pop("cargo_half_width", 2))
            self.n_hinges = self.hinge_relative_pos.size(0)
            self.dock_agent_when_hinged = kwargs.pop("dock_agent_when_hinged", True)
        # self.road = OcctMap(
        #     batch_dim=B,
        #     device=device,
        #     pts_gap=1.0,
        #     lane_width=self.lane_width
        # )
        self.road = OcctCRMap(
            batch_dim=B,
            device=device,
            cr_map_dir="/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/cr_maps/chapter4_6_path",
            max_ref_v=self.max_speed,
            is_constant_ref_v=True,
            rod_len=self.rod_len,
            n_agents=self.n_agents,
            target_road_id=self.target_road_id,
        )
        self.road_total_step = torch.zeros_like(self.road.batch_id.unique())
        self.lane_width = self.road.get_lane_width("mean")
        
        # 直接使用Road对象的边界点
        self.ref_paths_agent_related = OcctReferencePathsAgentRelated(
            long_term=self.road.get_road_center_pts().unsqueeze(1).expand(-1, self.n_agents, -1, -1),
            left_boundary=self.road.get_road_left_pts().unsqueeze(1).expand(-1, self.n_agents, -1, -1),
            right_boundary=self.road.get_road_right_pts().unsqueeze(1).expand(-1, self.n_agents, -1, -1),
            
            short_term=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_short_term, 3),
                device=device,
                dtype=torch.float32,
            ),  # [x, y, v]
            hinge_short_term=torch.zeros(
                (batch_dim, self.n_hinges, self.n_points_short_term, 5),
                device=device,
                dtype=torch.float32,
            ),  # [x, y, vx, vy, hinge_dis_boundary]
            short_term_indices=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_short_term),
                device=device,
                dtype=torch.int32,
            ),
            hinge_status=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.bool,
            ), # each hinge is ready to hinge or not
            hinge_heading_vel_angle_diff_deg=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            ),
            agent_heading_hinge_heading_angle_diff_deg=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            ),
            agent_hinge_status=CircularBuffer(
                torch.zeros(
                    (
                        2, # only for checking once hinge
                        batch_dim,
                        self.n_agents # ignore the self.TRACTOR_SLICE, dont use
                    ),
                    device=device,
                    dtype=torch.bool,
                )
            ), # hinge in each agent is occupied or not
            nearing_points_left_boundary=torch.zeros(
                (
                    batch_dim,
                    self.n_agents,
                    self.n_points_nearing_boundary,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),  # Nearing left boundary
            nearing_points_right_boundary=torch.zeros(
                (
                    batch_dim,
                    self.n_agents,
                    self.n_points_nearing_boundary,
                    2,
                ),
                device=device,
                dtype=torch.float32,
            ),  # Nearing right boundary
            exit=torch.zeros(
                (batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32
            ),
        )
        # Timer for the first env
        self.timer = Timer(
            start=time.time(),
            end=0,
            step=torch.zeros(
                batch_dim, device=device, dtype=torch.int32
            ),  # Each environment has its own time step
            step_begin=time.time(),
            render_begin=0,
        )
        self.constants = OcctConstants(
            env_idx_broadcasting=torch.arange(
                batch_dim, device=device, dtype=torch.int32
            ).unsqueeze(-1),
            empty_action_acc=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            empty_action_steering=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor(
                (self.agent_length) ** 2 + self.agent_width**2,
                device=device,
                dtype=torch.float32,
            ).sqrt()
            * 1.2,
        )

        obs_relative_velocity_scale = kwargs.pop(
            "obs_relative_velocity_scale", max(self.max_speed / 4, 1.0)
        )
        obs_relative_acceleration_scale = kwargs.pop(
            "obs_relative_acceleration_scale", max(self.max_acceleration, 0.5)
        )

        self.normalizers = OcctNormalizers(
            pos=torch.tensor(
                [self.agent_length * 5, self.agent_width * 5],
                device=device,
                dtype=torch.float32,
            ),
            error_pos=torch.tensor(
                self.agent_length,
                device=device,
                dtype=torch.float32,
            ),
            pos_world=torch.tensor(
                [self.world_x_dim, self.world_y_dim], device=device, dtype=torch.float32
            ),
            v=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            error_v=torch.tensor(
                obs_relative_velocity_scale,
                device=device,
                dtype=torch.float32,
            ),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            action_steering=self.max_steering_angle,
            action_vel=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            action_steering_rate=self.max_steering_rate,
            action_acc=torch.tensor(self.max_acceleration, device=device, dtype=torch.float32),
            distance_lanelet=torch.tensor(
                self.lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_ref=torch.tensor(
                self.lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_agent=torch.tensor(
                self.agent_length * 10, device=device, dtype=torch.float32
            ),
            hinge_step=torch.tensor(
                200, device=device, dtype=torch.float32
            ),
        )
        self.obs_relative_velocity_scale = torch.tensor(
            obs_relative_velocity_scale,
            device=device,
            dtype=torch.float32,
        )
        self.obs_relative_acceleration_scale = torch.tensor(
            obs_relative_acceleration_scale,
            device=device,
            dtype=torch.float32,
        )
        self.use_history_observation = use_history_observation
        self.history_obs_len = max(1, min(history_obs_len, n_observed_steps))
        self.history_obs_dim = history_obs_dim
        self.history_obs_raw_dim = None
        self.observations = OcctObservations(
            is_partial=torch.tensor(
                is_partial_observation, device=device, dtype=torch.bool
            ),
            n_nearing_agents=torch.tensor(
                self.n_nearing_agents_observed,
                device=device,
                dtype=torch.int32,
            ),
            noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32),
            n_stored_steps=torch.tensor(
                n_stored_steps, device=device, dtype=torch.int32
            ),
            n_observed_steps=torch.tensor(
                n_observed_steps, device=device, dtype=torch.int32
            ),
            platoon_error_vel=torch.zeros(
                (batch_dim, self.n_agents, 2), device=device, dtype=torch.float32
            ),
            hinge_error_vel=torch.zeros(
                (batch_dim, self.n_agents, 2), device=device, dtype=torch.float32
            ),
            past_platoon_error_vel=CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_hinge_error_vel=CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            self_platoon_error_space=CircularBuffer(
                torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, 2), 
                device=device, 
                dtype=torch.float32
                )
            ),
            agent_s=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            nearing_agents_indices=torch.zeros(
                    (batch_dim, self.n_agents, self.n_agents),
                    device=device, 
                    dtype=torch.int32,
            ),
            past_pos = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_rot = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_vertices = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 4, 2),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_vel = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_steering = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_relative_ref_info = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.n_points_short_term,
                        3, # [x, y, v]
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_relative_hinge_info = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_hinges,
                        self.n_points_short_term,
                        5,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            ), # hinge n_points_short_term agent i relative to hinge j
            past_left_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_right_boundary = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.n_points_nearing_boundary,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_action_acc = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_action_steering = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_distance_to_ref_path = CircularBuffer(
            torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )
            ),
            past_distance_to_boundaries = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_distance_to_left_boundary = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_distance_to_right_boundary = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
            past_distance_to_agents = CircularBuffer(
                torch.zeros(
                    (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                    device=device,
                    dtype=torch.float32,
                )
            ),
        )

        self.distances = OcctDistances(
            agents=torch.zeros(
                batch_dim, self.n_agents, self.n_agents, dtype=torch.float32,device=device
            ),
            agents_frenet=torch.zeros(
                batch_dim, self.n_agents, self.n_agents, dtype=torch.float32,device=device
            ),
            left_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32
            ),  # The first entry for the center, the last 4 entries for the four vertices
            right_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4), device=device, dtype=torch.float32
            ),
            boundaries=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            ref_paths=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            lookahead_pts=torch.zeros(
                (batch_dim, self.n_agents, 2), device=device, dtype=torch.float32
            ),
            closest_point_on_ref_path=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            closest_point_on_left_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
            closest_point_on_right_b=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.int32
            ),
        )
        n_agents=self.n_agents

        self.reward_details = {
            "reward_total": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_progress": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_vel": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_platoon_heading": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),

            "reward_goal": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_hinge": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_all_hinge": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_hinged_hold": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),

            "reward_platoon_vel": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_hinge_vel": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),

            "reward_platoon_ref": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_hinge_ref": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),

            "reward_platoon_space": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "reward_hinge_space": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),

            "reward_approach_hinge": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "penalty_hinge_time_cost": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),

            "penalty_near_boundary": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "penalty_near_other_agents": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "penalty_change_steering": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "penalty_change_acc": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "penalty_collide_with_agents": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "penalty_outside_boundaries": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            "penalty_backward": torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
        }

        # Penalty
        threshold_change_steering = kwargs.pop(
            "threshold_change_steering", 10
        )  # Threshold above which agents will be penalized for changing steering too quick [degree]
        threshold_change_acc = kwargs.pop(
            "threshold_change_acc", 10
        )  # Threshold above which agents will be penalized for changing acceleration too quick [m/s^2]

        threshold_near_boundary_high = kwargs.pop(
            "threshold_near_boundary_high", self.agent_width/2
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to lanelet boundaries
        threshold_near_boundary_low = kwargs.pop(
            "threshold_near_boundary_low", 0
        )  # Threshold above which agents will be penalized for being too close to lanelet boundaries

        threshold_near_other_agents_c2c_high = kwargs.pop(
            "threshold_near_other_agents_c2c_high", 1.8 * (self.agent_length**2 + self.agent_width**2)**0.5
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to other agents (for center-to-center distance)
        threshold_near_other_agents_c2c_low = kwargs.pop(
            "threshold_near_other_agents_c2c_low",
            (self.agent_length**2 + self.agent_width**2)**0.5,
        )  # Threshold above which agents will be penalized (for center-to-center distance,
        # If a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

        self.thresholds = OcctThresholds(
            near_boundary_low=torch.tensor(
                threshold_near_boundary_low, device=device, dtype=torch.float32
            ),
            near_boundary_high=torch.tensor(
                threshold_near_boundary_high, device=device, dtype=torch.float32
            ),
            near_other_agents_low=torch.tensor(
                threshold_near_other_agents_c2c_low, device=device, dtype=torch.float32
            ),
            near_other_agents_high=torch.tensor(
                threshold_near_other_agents_c2c_high, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(
                threshold_change_steering, device=device, dtype=torch.float32
            ).deg2rad(),
            change_acc=torch.tensor(
                threshold_change_acc, device=device, dtype=torch.float32
            ),
            distance_mask_agents=self.normalizers.pos[0],
        )
        # Initialize collision matrix
        self.collisions = Collisions(
            with_agents=torch.zeros(
                (batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.bool,
            ),
            with_lanelets=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
            with_exit_segments=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.bool
            ),
        )
        # Initialize agent-specific reference paths, which will be determined in `reset_world_at` function
        
        # The shape of each agent is considered a rectangle with 4 vertices.
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros(
            (batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32
        )

        weighting_ref_directions = torch.linspace(
            1,
            0.2,
            steps=self.n_points_short_term - 1, #251226 after revise the progress reward, the steps need -1
            device=device,
            dtype=torch.float32,
        )
        weighting_ref_directions /= weighting_ref_directions.sum()

        # init_Reward
        r_p_normalizer = (
            100  # This parameter normalizes rewards and penalties to [-1, 1].
        )
        # This is useful for RL algorithms with an actor-critic architecture where the critic's
        # output is limited to [-1, 1] (e.g., due to tanh activation function).
        reward_progress = (
            kwargs.pop("reward_progress", 10) / r_p_normalizer
        )  # Reward for moving along reference paths
        reward_vel = (
            kwargs.pop("reward_vel", 0) / r_p_normalizer
        )  # Reward for moving in high velocities.
        reward_goal = (
            kwargs.pop("reward_goal", 10) / r_p_normalizer
        )  # Goal-reaching reward
        reward_platoon_space = (
            kwargs.pop("reward_platoon_space", 20) / r_p_normalizer
        )
        reward_hinge_space = (
            kwargs.pop("reward_hinge_space", 50) / r_p_normalizer
        )
        reward_platoon_vel = (
            kwargs.pop("reward_platoon_vel", 20) / r_p_normalizer
        )
        reward_hinge_vel = (
            kwargs.pop("reward_hinge_vel", 0) / r_p_normalizer
        )
        reward_platoon_ref = (
            kwargs.pop("reward_platoon_ref", 50) / r_p_normalizer
        )
        reward_hinge_ref = (
            kwargs.pop("reward_hinge_ref", 50) / r_p_normalizer
        )
        reward_platoon_heading = (
            kwargs.pop("reward_platoon_heading", 50) / r_p_normalizer
        )
        reward_hinge = (
            kwargs.pop("reward_hinge", 100) / r_p_normalizer
        )
        reward_all_hinge = (
            kwargs.pop("reward_all_hinge", 0) / r_p_normalizer
        )
        reward_hinged_hold = (
            kwargs.pop("reward_hinged_hold", 0) / r_p_normalizer
        )
        self.reward_approach_hinge = torch.tensor(
            kwargs.pop("reward_approach_hinge", 100) / r_p_normalizer,
            device=device,
            dtype=torch.float32,
        )
        self.reward_all_hinge = torch.tensor(
            reward_all_hinge,
            device=device,
            dtype=torch.float32,
        )
        self.reward_hinged_hold = torch.tensor(
            reward_hinged_hold,
            device=device,
            dtype=torch.float32,
        )
        self.penalty_goal_incomplete_hinge = torch.tensor(
            kwargs.pop("penalty_goal_incomplete_hinge", -100) / r_p_normalizer,
            device=device,
            dtype=torch.float32,
        )
        self.goal_incomplete_hinge_penalty_mode = kwargs.pop(
            "goal_incomplete_hinge_penalty_mode", "scaled"
        )
        # +reward_hinge at the first hinge step, linearly decreasing to
        # -reward_hinge after `hinge_reward_zero_time` seconds in the hinge zone.
        self.hinge_reward_zero_time = float(
            kwargs.pop("hinge_reward_zero_time", 2.0)
        )
        self.rewards = OcctRewards(
            reward_progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions,  # Progress in the weighted directions (directions indicating by
            # closer short-term reference points have higher weights)
            reward_vel=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reward_goal=torch.tensor(reward_goal, device=device, dtype=torch.float32),
            reward_platoon_heading=torch.tensor(reward_platoon_heading, device=device, dtype=torch.float32),
            reward_platoon_space=torch.tensor(reward_platoon_space, device=device, dtype=torch.float32),
            reward_hinge_space=torch.tensor(reward_hinge_space, device=device, dtype=torch.float32),
            reward_platoon_vel=torch.tensor(reward_platoon_vel, device=device, dtype=torch.float32),
            reward_hinge_vel=torch.tensor(reward_hinge_vel, device=device, dtype=torch.float32),
            reward_platoon_ref=torch.tensor(reward_platoon_ref, device=device, dtype=torch.float32),
            reward_hinge_ref=torch.tensor(reward_hinge_ref, device=device, dtype=torch.float32),
            reward_hinge=torch.tensor(reward_hinge, device=device, dtype=torch.float32),
        )
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        penalty_near_boundary = (
            kwargs.pop("penalty_near_boundary", -20) / r_p_normalizer
        )
        penalty_near_other_agents = (
            kwargs.pop("penalty_near_other_agents", -20) / r_p_normalizer
        )
        penalty_collide_with_agents = (
            kwargs.pop("penalty_collide_with_agents", -100) / r_p_normalizer
        )
        penalty_outside_boundaries = (
            kwargs.pop("penalty_outside_boundaries", -100) / r_p_normalizer
        )
        penalty_change_steering = (
            kwargs.pop("penalty_change_steering", -20) / r_p_normalizer
        )
        penalty_change_acc = (
            kwargs.pop("penalty_change_acc", -20)
        )
        penalty_backward = (
            kwargs.pop("penalty_backward", -100) / r_p_normalizer
        ) 
        penalty_hinge_time_cost = (
            kwargs.pop("penalty_hinge_time_cost", -5) / r_p_normalizer
        ) 

        self.penalties = OcctPenalties(
            near_boundary=torch.tensor(penalty_near_boundary, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(
                penalty_near_other_agents, device=device, dtype=torch.float32
            ),
            collide_with_agents=torch.tensor(
                penalty_collide_with_agents, device=device, dtype=torch.float32
            ),
            collide_with_boundaries=torch.tensor(
                penalty_outside_boundaries, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(penalty_change_steering, device=device, dtype=torch.float32),
            change_acc=torch.tensor(penalty_change_acc, device=device, dtype=torch.float32),
            backward=torch.tensor(penalty_backward, device=device, dtype=torch.float32),
            hinge_time_cost=torch.tensor(penalty_hinge_time_cost, device=device, dtype=torch.float32),
        )

        self.enable_failure_replay_restore = bool(
            kwargs.pop("enable_failure_replay_restore", False)
        )
        self.failure_replay_pre_failure_seconds = float(
            kwargs.pop("failure_replay_pre_failure_seconds", 1.5)
        )
        self.failure_replay_margin_steps = int(
            kwargs.pop("failure_replay_margin_steps", 0)
        )
        self.failure_replay_k_steps = max(
            1, int(round(self.failure_replay_pre_failure_seconds / self.dt))
        )
        self.failure_replay_snapshot_dim = 8
        self.failure_curriculum_bank = None
        self.failure_curriculum_collect_enabled = False
        self.failure_curriculum_sampling_enabled = False
        self.failure_curriculum_replay_probability = 0.0
        self.failure_curriculum_min_bank_size = 0
        self.failure_curriculum_iteration = 0
        self.current_episode_replay_source = torch.zeros(
            batch_dim, device=device, dtype=torch.bool
        )
        self.current_episode_replay_entry_id = torch.full(
            (batch_dim,), -1, device=device, dtype=torch.long
        )
        self.failure_curriculum_events = []

        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.n_steps_before_recording=kwargs.pop("n_steps_before_recording", 10)

        self.state_buffer = StateBuffer(
            buffer=torch.zeros(
                (self.n_steps_before_recording, batch_dim, self.n_agents, 5),
                device=device,
                dtype=torch.float32,
            )  # [pos_x, pos_y, rot, vel_x, vel_y],
        )
        self.failure_replay_snapshot_buffer = CircularBuffer(
            torch.zeros(
                (
                    self.failure_replay_k_steps + 1,
                    batch_dim,
                    self.n_agents,
                    self.failure_replay_snapshot_dim,
                ),
                device=device,
                dtype=torch.float32,
            )
        )
    # ========== 2) 创建 World ==========
    def init_world(self, batch_dim: int, device: torch.device) -> World:
        # 创建世界，设置合适的边界
        world = World(batch_dim=batch_dim, device=device, dt=self.dt,
                        x_semidim=self.world_x_dim,
                        y_semidim=self.world_y_dim, dim_c=0)
        return world

    # ========== 3) 创建 Agent & 标注 policy_agents ==========
    def init_agents(self, world: World, *kwargs):
        self.followers = []
        i=0
        if self.task_class != TaskClass.SIMPLE_PLATOON:
            self.tractor_front = Agent(
                name=f"agent_{i}",
                shape=Box(length=self.agent_length, width=self.agent_width),
                color=Color.RED,
                collide=False,
                render_action=False,
                u_range=[
                    self.max_acceleration,
                    self.max_steering_angle,
                ],
                u_multiplier=[1, 1],
                max_speed=self.max_speed,
                # 禁用 drag 和 linear_friction
                drag = 0.0,
                linear_friction = 0.0,
                angular_friction = 0.0,
                # 禁用 movable 和 rotatable
                movable=False,
                rotatable=False,
                dynamics=KinematicBicycle(
                        world,
                        width=self.agent_width,
                        l_f=self.l_f,
                        l_r=self.l_r,
                        max_acceleration=self.max_acceleration,
                        max_steering_angle=self.max_steering_angle,
                        integration="rk4",  # one of {"euler", "rk4"}
                    ),
            )
            world.add_agent(self.tractor_front)
            i=1
        # 0-1 浮点数格式 (Matplotlib 常用)
        colors = [
            (31/255, 73/255, 125/255),    # 深蓝
            (123/255, 31/255, 162/255),   # 深紫红
            (0/255, 109/255, 119/255),    # 深翠绿
            (145/255, 30/255, 18/255),    # 深红棕
            (45/255, 48/255, 91/255),     # 深靛青
            (127/255, 80/255, 0/255)      # 深琥珀
        ]
        for _ in range(self.n_followers):
            a = Agent(
                    name=f"agent_{i}", 
                    shape=Box(length=self.agent_length, width=self.agent_width),
                    # color=tuple(
                    #     torch.rand(3, device=world.device, dtype=torch.float32).tolist()
                    # ),
                    color=colors[i%len(colors)],
                    collide=False,
                    render_action=False,
                    u_range=[
                        self.max_acceleration,
                        self.max_steering_angle,
                    ],
                    u_multiplier=[1, 1],
                    max_speed=self.max_speed,
                    # 禁用 drag 和 linear_friction
                    drag = 0.0,
                    linear_friction = 0.0,
                    angular_friction = 0.0,
                    movable=False if self.traditional_control!=MethodClass.MARL else True,
                    rotatable=False if self.traditional_control!=MethodClass.MARL else True,
                    dynamics=KinematicBicycle(
                        world,
                        width=self.agent_width,
                        l_f=self.l_f,
                        l_r=self.l_r,
                        max_acceleration=self.max_acceleration,
                        max_steering_angle=self.max_steering_angle,
                        integration="rk4",  # one of {"euler", "rk4"}
                    ),
                    # 260104 try control front wheel angle rate, but very difficult, 
                    # vehicle always collide with boundary, 
                    # lead to frequent reset which slow down the training time(2 times)\
                    # finally give up.
                    # dynamics=DynamicKinematicBicycle(
                    #     world,
                    #     width=self.agent_width,
                    #     l_f=self.l_f,
                    #     l_r=self.l_r,
                    #     max_steering_angle=self.max_steering_angle,
                    #     max_steering_rate=self.max_steering_rate,
                    #     max_acceleration=self.max_acceleration,
                    #     integration="rk4",  # one of {"euler", "rk4"}
                    # ),
                )
            world.add_agent(a)
            self.followers.append(a)
            i+=1
        
        if self.task_class != TaskClass.SIMPLE_PLATOON:
            self.tractor_rear  = Agent(
                name=f"agent_{i}",  
                shape=Box(length=self.agent_length, width=self.agent_width),
                color=Color.BLUE,
                collide=False,
                render_action=False,
                u_range=[
                    self.max_acceleration,
                    self.max_steering_angle,
                ],
                u_multiplier=[1, 1],
                max_speed=self.max_speed,
                # 禁用 drag 和 linear_friction
                drag = 0.0,
                linear_friction = 0.0,
                angular_friction = 0.0,
                # 禁用 movable 和 rotatable
                movable=False,
                rotatable=False,
                dynamics=KinematicBicycle(
                        world,
                        width=self.agent_width,
                        l_f=self.l_f,
                        l_r=self.l_r,
                        max_acceleration=self.max_acceleration,
                        max_steering_angle=self.max_steering_angle,
                        integration="rk4",  # one of {"euler", "rk4"}
                    ),
            )
            world.add_agent(self.tractor_rear)
            
    def get_occt_cr_path_num(self):
        """
        获取OCCT CR地图路径数量
        """
        return len(self.road.path_library)
    def get_front_rear_pts(self, front_rear_s: Tensor, env_index: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        获取牵引车和末尾车的坐标
        input:
            front_rear_s: [B,2] 牵引车和末尾车弧长
        return:
            p_front: [B,2] 牵引车坐标
            p_rear:  [B,2] 末尾车坐标
        """
        front_rear_pts = self.road.get_pts(front_rear_s, env_index)    # [B,2]
        p_front = front_rear_pts[:,0,:]    # [B,2,2]
        p_rear = front_rear_pts[:,1,:]    # [B,2,2]
        return p_front, p_rear
    
    def _set_pose(self, agent: Agent, pos: Tensor, theta: Tensor, vel: Tensor, idx_mask: Tensor):
        if hasattr(agent.state, "pos"):
            agent.state.pos[idx_mask] = pos[idx_mask]
        if hasattr(agent.state, "rot"):
            theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
            agent.state.rot[idx_mask] = theta_reshaped[idx_mask]
        elif hasattr(agent.state, "angle"):
            theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
            agent.state.angle[idx_mask] = theta_reshaped[idx_mask]
        if hasattr(agent.state, "vel"):
            vx = vel[idx_mask] * torch.cos(theta[idx_mask])
            vy = vel[idx_mask] * torch.sin(theta[idx_mask])
            agent.state.vel[idx_mask] = torch.stack([vx, vy], dim=-1)

    def configure_failure_curriculum(
        self,
        bank,
        collect_enabled: bool,
        enabled: bool,
        replay_probability: float,
        min_bank_size: int,
        iteration: int,
    ) -> None:
        self.failure_curriculum_bank = bank
        self.failure_curriculum_collect_enabled = bool(collect_enabled)
        self.failure_curriculum_sampling_enabled = bool(enabled)
        self.failure_curriculum_replay_probability = float(replay_probability)
        self.failure_curriculum_min_bank_size = int(min_bank_size)
        self.failure_curriculum_iteration = int(iteration)

    def drain_failure_curriculum_events(self) -> List[Dict]:
        events = self.failure_curriculum_events
        self.failure_curriculum_events = []
        return events

    def _get_current_cur_delta(self) -> Tensor:
        cur_delta = torch.zeros(
            (self.batch_dim, self.n_agents, 1),
            device=self.device,
            dtype=torch.float32,
        )
        for agent_idx, agent in enumerate(self.world.agents):
            if hasattr(agent.dynamics, "cur_delta") and agent.dynamics.cur_delta is not None:
                cur_delta[:, agent_idx, 0] = agent.dynamics.cur_delta.squeeze(-1)
        return cur_delta

    def _build_failure_replay_buffer_state(self) -> Tensor:
        return torch.cat(
            (
                torch.stack([a.state.pos for a in self.world.agents], dim=1),
                torch.stack([a.state.rot for a in self.world.agents], dim=1),
                torch.stack([a.state.vel for a in self.world.agents], dim=1),
                torch.stack([a.state.ang_vel for a in self.world.agents], dim=1),
                self._get_current_cur_delta(),
                self.observations.agent_s.unsqueeze(-1),
            ),
            dim=-1,
        )

    def _build_failure_replay_snapshot(self, env_index: int) -> Dict[str, Tensor]:
        snapshot_state = self.failure_replay_snapshot_buffer.get_latest(
            n=self.failure_replay_k_steps + 1
        )[env_index].detach().clone()
        return {
            "agent_state": snapshot_state,
        }

    def _get_failure_type_for_env(self, done_status: Dict[str, Tensor], env_index: int) -> str:
        if bool(done_status["is_collision_with_agents_env"][env_index].item()):
            return "collision_with_agents"
        if bool(done_status["is_collision_with_lanelets"][env_index].item()):
            return "collision_with_lanelets"
        if bool(done_status["is_collision_with_exit_segments"][env_index].item()):
            return "collision_with_exit_segments"
        return "unknown_failure"

    def _record_failure_curriculum_events(self, done_status: Dict[str, Tensor]) -> None:
        if not self.failure_curriculum_collect_enabled:
            return

        done_mask = done_status["is_done"]
        success_mask = done_status["is_success"]
        failure_mask = done_status["is_failure"]

        for env_index in torch.where(done_mask)[0].tolist():
            road_id = int(self.road.batch_id[env_index].item())
            source_entry_id = int(self.current_episode_replay_entry_id[env_index].item())
            is_replay_source = bool(self.current_episode_replay_source[env_index].item())
            failure_type = self._get_failure_type_for_env(done_status, env_index)
            has_valid_snapshot = int(self.env_current_step[env_index].item()) >= (
                self.failure_replay_k_steps + self.failure_replay_margin_steps
            )
            snapshot = (
                self._build_failure_replay_snapshot(env_index)
                if has_valid_snapshot and bool(failure_mask[env_index].item())
                else None
            )

            if is_replay_source:
                if bool(success_mask[env_index].item()):
                    self.failure_curriculum_events.append(
                        {
                            "event_type": "replay_success",
                            "source_entry_id": source_entry_id,
                            "road_id": road_id,
                        }
                    )
                elif bool(failure_mask[env_index].item()):
                    self.failure_curriculum_events.append(
                        {
                            "event_type": "replay_failure",
                            "source_entry_id": source_entry_id,
                            "road_id": road_id,
                            "failure_type": failure_type,
                            "snapshot": snapshot,
                        }
                    )
            elif bool(failure_mask[env_index].item()) and snapshot is not None:
                self.failure_curriculum_events.append(
                    {
                        "event_type": "new_failure",
                        "source_entry_id": -1,
                        "road_id": road_id,
                        "failure_type": failure_type,
                        "snapshot": snapshot,
                    }
                )

    def _sample_failure_curriculum_snapshot(self, env_index: int):
        if (
            not self.enable_failure_replay_restore
            or not self.failure_curriculum_sampling_enabled
            or self.failure_curriculum_bank is None
        ):
            return None
        if torch.rand(1, device=self.device).item() >= self.failure_curriculum_replay_probability:
            return None
        road_id = int(self.road.batch_id[env_index].item())
        return self.failure_curriculum_bank.sample(
            self.failure_curriculum_iteration,
            road_id=road_id,
        )

    def _restore_failure_replay_snapshot(
        self,
        env_index: int,
        snapshot: Dict[str, Tensor],
        entry_id: int,
        agents,
    ) -> None:
        agent_state = snapshot["agent_state"].to(self.device)
        pos = agent_state[:, 0:2]
        rot = agent_state[:, 2:3]
        vel = agent_state[:, 3:5]
        ang_vel = agent_state[:, 5:6]
        cur_delta = agent_state[:, 6:7]
        agent_s = agent_state[:, 7]

        for agent_idx, agent in enumerate(agents):
            agent.state.pos[env_index] = pos[agent_idx]
            agent.state.rot[env_index] = rot[agent_idx]
            agent.state.vel[env_index] = vel[agent_idx]
            if hasattr(agent.state, "ang_vel"):
                agent.state.ang_vel[env_index] = ang_vel[agent_idx]
            if hasattr(agent.dynamics, "cur_delta") and agent.dynamics.cur_delta is not None:
                agent.dynamics.cur_delta[env_index] = cur_delta[agent_idx]

        self.observations.agent_s[env_index] = agent_s
        for i_agent in range(self.n_agents):
            self.reset_init_distances_and_short_term_ref_path(
                env_index, i_agent, agents
            )
        if self.task_class == TaskClass.OCCT_PLATOON:
            self.reset_init_hinge_short_term(env_index, agents)

        mutual_distances = get_distances_between_agents(self=self, is_set_diagonal=True)
        mutual_frenet_distances = get_frenet_distances_between_agents(
            self.observations.agent_s
        )
        self.distances.agents[env_index, :, :] = mutual_distances[env_index, :, :]
        self.distances.agents_frenet[env_index, :, :] = mutual_frenet_distances[
            env_index, :, :
        ]
        self.collisions.with_agents[env_index, :, :] = False
        self.collisions.with_lanelets[env_index, :] = False
        self.collisions.with_exit_segments[env_index, :] = False
        self.current_episode_replay_source[env_index] = True
        self.current_episode_replay_entry_id[env_index] = int(entry_id)
            
    def reset_world_at(self, env_index: Optional[int] = None, agent_index: Optional[int] = None):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is given as None, the majority part of computation will be done in a vectorized manner.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        :param agent_index: index of the agent to reset. If None all agents in the specified environment will be reset.
        """
        # ============== 总计时开始 ==============
        total_start = time.time()
        B = self.batch_dim
        device = self.device
        assert agent_index==None,"agent_index must be None, not supported"
        # ============== 阶段1：初始化idx_mask ==============
        stage1_start = time.time()
        if env_index is None:
            idx_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            idx_mask = torch.zeros(B, dtype=torch.bool, device=device)
            idx_mask[env_index] = True
        self.current_episode_replay_source[idx_mask] = False
        self.current_episode_replay_entry_id[idx_mask] = -1
        self.hinge_active_steps[idx_mask] = 0
        self.all_hinged_reward_granted[idx_mask] = False
        self.reward_phase_weights[idx_mask] = 0.0
        self.reward_phase_transition_start[idx_mask] = 0.0
        self.reward_phase_transition_target[idx_mask] = 0.0
        self.reward_phase_transition_progress[idx_mask] = 1.0
        sampled_failure_replay = {}
        if self.simple_mppi is not None:
            self.simple_mppi.reset()
        if hasattr(self, "_last_vel_errors"):
            self._last_vel_errors.clear()
        if hasattr(self, "_last_pid_steering"):
            self._last_pid_steering.clear()

        # ============== 阶段2：生成车队速度/间距 ==============
        stage2_start = time.time()
        if self.task_class == TaskClass.OCCT_PLATOON:
            # init vel is normal distribution depend on road ref velocity, self.platoon_vel_batch is useless
            # and platoon_space_batch equals to self.still_space
            self.platoon_vel_batch =torch.zeros(B, dtype=torch.float32, device=device)
        else:
            platoon_vel_batch = self.get_normal_tensor(self.init_vel_mean, self.init_vel_std)
            #platoon_vel_batch = self.init_vel_min+(self.init_vel_max-self.init_vel_min)*self.get_random_tensor()
            self.platoon_vel_batch[idx_mask] = torch.clamp(platoon_vel_batch, min=0.0)[idx_mask]
        self.platoon_space_batch = self.get_platoon_space(self.platoon_vel_batch)
        spacing = self.platoon_space_batch

        # ============== 阶段3：计算最后一辆车弧长 ==============
        stage3_start = time.time()
        s_start_buffer = 0.0
        s_end_buffer = 10.0
        last_vehicle_s_max = torch.clamp(self.road.batch_corner_s_end-self.rod_len,min=0)
        if self.is_rand_arc_pos:
            last_vehicle_s = torch.ones(B,device=device) * self.init_arc_pos + self.get_random_tensor() * (last_vehicle_s_max - self.init_arc_pos) #260320 two stage
        else:
            last_vehicle_s = torch.ones(B,device=device) * self.init_arc_pos #260320 two stage
        last_vehicle_s = torch.clamp(last_vehicle_s, s_start_buffer * torch.ones(B,device=device),
                                    self.road.get_s_max() - (s_start_buffer + s_end_buffer) - (self.n_agents - 1) * torch.mean(spacing, dim=-1))
        init_vel_noise = self.get_normal_tensor(0,self.init_vel_std)
        # ============== 阶段4：OCCT_PLATOON核心逻辑 ==============
        stage4_start = time.time()
        if self.task_class == TaskClass.OCCT_PLATOON:
            # caculate the s of front tractor via last_vehicle_s and rod_len
            self.ref_paths_agent_related.agent_hinge_status.get_latest(n=1)[env_index, :] = False
            self.ref_paths_agent_related.agent_hinge_status.get_latest(n=2)[env_index, :] = False
            self.observations.agent_s[idx_mask, self.HINGE_LAST_INDEX] = last_vehicle_s[idx_mask]
            delta_s, infeasible = self.road.solve_delta_s(last_vehicle_s,self.rod_len,False)
            assert not infeasible.any(), "Infeasible delta_s"
            s_front_new = delta_s + last_vehicle_s
            assert (s_front_new[idx_mask] <= self.road.get_s_max()[idx_mask]).all(), "s_front_new out of range"
            self.observations.agent_s[idx_mask, self.HINGE_FIRST_INDEX] = s_front_new[idx_mask]
            
            p_front, p_rear = self.get_front_rear_pts(self.observations.agent_s[:,self.TRACTOR_SLICE], env_index)
            rod_vec = (p_front - p_rear)           # [B,2]
            rod_theta = torch.atan2(rod_vec[:, 1], rod_vec[:, 0])  # [B]
            
            # 计算道路切线方向而不是使用货物方向
            front_rear_theta = self.road.get_tangent_heading(self.observations.agent_s[:, self.TRACTOR_SLICE])
            front_theta = front_rear_theta[:,0]
            rear_theta = front_rear_theta[:,1]
            
            front_ref_v = self.road.get_ref_v(s_front_new[:,None])[:,0,0]+init_vel_noise # [B]
            front_ref_v = torch.clamp(front_ref_v,min=0,max=self.max_speed)
            rear_ref_v = self.road.get_ref_v(last_vehicle_s[:,None])[:,0,0]+init_vel_noise # [B]
            rear_ref_v = torch.clamp(rear_ref_v,min=0,max=self.max_speed)
            self._set_pose(self.tractor_front, p_front, front_theta, front_ref_v, idx_mask)
            self._set_pose(self.tractor_rear, p_rear, rear_theta, rear_ref_v, idx_mask)

        else:
            s_front_new = (self.n_followers-1)* self.still_space + last_vehicle_s
        # ============== 阶段5：生成横向偏移/航向误差 ==============
        stage5_start = time.time()
        F = self.n_followers
        # 3. 随机横向偏移（0-2）
        lateral_offset = torch.rand(B, F, device=device) * (self.lane_width-self.agent_width)/2 * 0  # [B, F]
        # 随机方向（左右）
        lateral_direction = torch.sign(torch.randn(B, F, device=device))  # [B, F]
        lateral_offset = lateral_offset * lateral_direction  # [B, F]
        
        # 4. 随机航向角误差（0-10度，转换为弧度）
        heading_error = (torch.rand(B, F, device=device) * 0.0) * (torch.pi / 180.0)  # [B, F] 弧度
        # 随机方向（正负）
        heading_direction = torch.sign(torch.randn(B, F, device=device))  # [B, F]
        heading_error = heading_error * heading_direction  # [B, F] 弧度

        # ============== 阶段6：计算所有车辆弧长 ==============
        stage6_start = time.time()
        # 计算每辆车的弧长位置
        vehicle_s = torch.zeros(B, F, device=device)  # [B, F]
        if self.task_class == TaskClass.OCCT_PLATOON:
            vehicle_s[:, 0] = s_front_new - spacing
        else:
            vehicle_s[:, 0] = s_front_new
        
        # 从后往前计算每辆车的位置
        for i in range(F-1):
            vehicle_s[:, i+1] = vehicle_s[:, i] - spacing
        
        # 确保所有车辆的弧长都在道路的有效范围内
        vehicle_s = torch.clamp(vehicle_s, max=self.road.get_s_max()[:, None].expand(-1, F) - 1e-6) #BUG FIXED: produce nan pos

        # ============== 阶段7：记录agent_s ==============
        stage7_start = time.time()
        # 251224: record arch of agents for closest ref pts
        if self.task_class == TaskClass.OCCT_PLATOON:
            self.observations.agent_s[env_index][...,self.FOLLOWER_SLICE] = vehicle_s[env_index] #BUG FIX： asynchronous update of agent_s instead of self.observations.agent_s = vehicle_s
        else:
            self.observations.agent_s[env_index] = vehicle_s[env_index]

        # ============== 阶段8：计算车辆位置/方向 ==============
        stage8_start = time.time()
        # 计算每辆车的位置和方向
        # 获取道路坐标
        vehicle_pos = self.road.get_pts(vehicle_s)  # [B, F, 2]
        
        # 获取道路切线方向
        road_theta = self.road.get_tangent_heading(vehicle_s)  # [B, F]
        
        # 获取道路法线向量，用于应用横向偏移
        normal_vec = self.road.get_normal_vector(vehicle_s)  # [B, F, 2]
        
        # 应用横向偏移
        vehicle_pos = vehicle_pos + lateral_offset.unsqueeze(-1) * normal_vec  # [B, F, 2]
        
        # 应用航向角误差
        vehicle_theta = road_theta + heading_error  # [B, F]

        # ============== 阶段9：设置跟随车状态 ==============
        stage9_start = time.time()
        # 设置车辆状态
        for i, ag in enumerate(self.followers):
            agent_s = self.observations.agent_s[...,self.FOLLOWER_SLICE][:,i]
            #ref_v = self.road.get_ref_v(agent_s[:,None])[:,0,0]+init_vel_noise # [B]
            ref_v = torch.clamp((rear_ref_v + front_ref_v)/2 ,min=0,max=self.max_speed)
            self._set_pose(ag, vehicle_pos[:,i,:], vehicle_theta[:,i], ref_v, idx_mask)

        # ============== 阶段10：重置智能体循环（距离/碰撞） ==============
        stage10_start = time.time()
        agents = self.world.agents

        is_reset_single_agent = agent_index is not None
        # refresh platoon vel and space
        for env_i in (
            [env_index] if env_index is not None else range(self.world.batch_dim)
        ):
            # Begining of a new simulation (only record for the first env)
            if env_i == 0:
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0

            if not is_reset_single_agent:
                # Each time step of a simulation
                self.timer.step[env_i] = 0
            
            # The operations below can be done for all envs in parallel
            if env_index is None:
                if env_i == (self.world.batch_dim - 1):
                    env_j = slice(None)  # `slice(None)` is equivalent to `:`
                else:
                    continue
            else:
                env_j = env_i

            tmp_t=time.time()
            for i_agent in (
                range(self.n_agents) #251226 revise: old version is self.n_agents
                if not is_reset_single_agent
                else agent_index.unsqueeze(0)
            ):
                assert torch.isnan(agents[i_agent].state.pos[env_j, :]).any() == False, f"agent {i_agent} pos is nan"
                self.reset_init_distances_and_short_term_ref_path(
                    env_j, i_agent, agents
                )
                agents[i_agent].dynamics.cur_delta[env_j] = 0.0
            if self.task_class == TaskClass.OCCT_PLATOON:
                self.reset_init_hinge_short_term(env_j, agents)
            # Compute mutual distances between agents
            mutual_distances = get_distances_between_agents(
                self=self, is_set_diagonal=True
            )
            mutual_frenet_distances = get_frenet_distances_between_agents(self.observations.agent_s)
            # Reset mutual distances of all envs
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]
            self.distances.agents_frenet[env_j, :, :] = mutual_frenet_distances[env_j, :, :]
            

            # Reset the collision matrix
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False
            self.collisions.with_exit_segments[env_j, :] = False
            if self.enable_failure_replay_restore:
                sampled_result = self._sample_failure_curriculum_snapshot(env_i)
                if sampled_result is not None:
                    sampled_failure_replay[env_i] = sampled_result
        self.time_records["reset_agents_loop"] = time.time() - stage10_start

        if sampled_failure_replay:
            for replay_env_index, (entry_id, snapshot) in sampled_failure_replay.items():
                self._restore_failure_replay_snapshot(
                    replay_env_index,
                    snapshot,
                    entry_id,
                    agents,
                )

        # ============== 阶段11：重置状态缓冲区 ==============
        stage11_start = time.time()
        # Reset the state buffer
        self.state_buffer.reset()
        state_add = torch.cat(
            (
                torch.stack([a.state.pos for a in agents], dim=1),
                torch.stack([a.state.rot for a in agents], dim=1),
                torch.stack([a.state.vel for a in agents], dim=1),
            ),
            dim=-1,
        )
        self.state_buffer.add(state_add)  # Add new state
        self.failure_replay_snapshot_buffer.add(
            self._build_failure_replay_buffer_state()
        )
        
        self.time_records["state_buffer"] = time.time() - stage11_start

        # ============== 总计时结束 + 输出耗时报告 ==============
        self.time_records["total"] = time.time() - total_start
        self.reset_total_time += self.time_records["total"]
        # 输出各阶段耗时（建议每调用10次函数输出一次，避免刷屏）
        # if self.reset_count % self.batch_dim == 0 or self.time_records["total"]>0.5:
        #     print(f"reset_world_at time:{time.time() - total_start:.6f},total time:{self.reset_total_time:.6f}s")
        self.reset_count += 1
        # if env_index is None:
        #     self._print_time_report()
    def _print_time_report(self):
        """输出各阶段耗时报告，按耗时从高到低排序"""
        print("\n========== reset_world_at 耗时分析 ==========")
        # 按耗时降序排序
        sorted_records = sorted(self.time_records.items(), key=lambda x: x[1], reverse=True)
        total = self.time_records["total"]
        for stage, cost in sorted_records:
            ratio = (cost / total) * 100 if total > 0 else 0
            print(f"{stage:20s}: {cost:.6f}s ({ratio:.2f}%)")
        print("=============================================\n")
    def reset_init_hinge_short_term(self, env_j, agents):
        """
        This function resets the short-term reference paths for all agents in the environment.
        """
        self.ref_paths_agent_related.hinge_short_term[env_j, :] = get_short_term_hinge_path_by_s(
            occt_map=self.road,
            agents=agents,
            agent_s=self.observations.agent_s,
            n_points_to_return=self.n_points_short_term,
            tractor_slice=self.TRACTOR_SLICE,
            device=self.world.device,
            sample_ds=self.sample_interval,
            env_j=env_j,
            hinge_relative_pos=self.hinge_relative_pos,
        )[env_j]
        self.ref_paths_agent_related.hinge_status[env_j, :] = False
        for agent_i in range(self.n_agents):
            if agent_i not in self.TRACTOR_SLICE:
                self.ref_paths_agent_related.hinge_status[
                    env_j, agent_i
                ] = self.get_target_hinge_status(agent_i)[env_j]
        
    def reset_init_distances_and_short_term_ref_path(self, env_j, i_agent, agents):
        """
        This function calculates the distances from the agent's center of gravity (CG) to its reference path and boundaries,
        and computes the positions of the four vertices of the agent. It also determines the short-term reference paths
        for the agent based on the long-term reference paths and the agent's current position.
        """
        tmp_t=time.time()
        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[env_j, i_agent],
            self.distances.closest_point_on_ref_path[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
            n_points_long_term=None
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
            n_points_long_term=None
        )
        self.distances.left_boundaries[env_j, i_agent, 0] = center_2_left_b - (
            agents[i_agent].shape.width / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
            n_points_long_term=None
        )
        self.distances.right_boundaries[env_j, i_agent, 0] = center_2_right_b - (
            agents[i_agent].shape.width / 2
        )
        assert torch.isnan(agents[i_agent].state.pos[env_j, :]).any() == False, f"agent {i_agent} pos is nan"
        # Calculate the positions of the four vertices of the agents
        self.vertices[env_j, i_agent] = get_rectangle_vertices(
            center=agents[i_agent].state.pos[env_j, :],
            yaw=agents[i_agent].state.rot[env_j, :],
            width=agents[i_agent].shape.width,
            length=agents[i_agent].shape.length,
            is_close_shape=True,
        )
        #print(f"get_rectangle_vertices, time_cost: {time.time()-tmp_t:.6f}s")
        tmp_t=time.time()

        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                n_points_long_term=None
            )
            (
                self.distances.right_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                n_points_long_term=None
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[env_j, i_agent], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[env_j, i_agent],
                    self.distances.right_boundaries[env_j, i_agent],
                )
            ),
            dim=-1,
        )
        #print(f"get_perpendicular_distances, time_cost: {time.time()-tmp_t:.6f}s")
        tmp_t = time.time()
        # Get the short-term reference paths
        if self.use_center_frenet_ref:
            self.ref_paths_agent_related.short_term[env_j, i_agent] = \
                get_short_term_reference_path_by_s(
                    self.road,
                    self.observations.agent_s[env_j, i_agent],
                    n_points_to_return=self.n_points_short_term,
                    device=self.world.device,
                    sample_interval=self.sample_interval,
                    return_ref_v=True,
                    env_j=env_j
                )
            if self.task_class==TaskClass.SIMPLE_PLATOON and i_agent!=0:
                self.ref_paths_agent_related.short_term[env_j, i_agent,:,-1] = self.ref_paths_agent_related.short_term[env_j, 0,:,-1]
        else:
            (
                self.ref_paths_agent_related.short_term[env_j, i_agent, :, 0:2],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_ref_path[
                    env_j, i_agent
                ],
                n_points_to_return=self.n_points_short_term,
                device=self.world.device,
                sample_interval=self.sample_interval,
                n_points_shift=1,
            )
            self.ref_paths_agent_related.short_term[env_j, i_agent, :, 2] = self.init_vel_mean

        #print(f"get_short_term time_cost: {time.time()-tmp_t:.6f}s")
        tmp_t = time.time()
        # Get nearing points on boundaries
        if self.use_boundary_frenet_ref:
            self.ref_paths_agent_related.nearing_points_left_boundary[env_j, i_agent] = \
                get_short_term_reference_path_by_s(
                    self.road,
                    self.observations.agent_s[env_j, i_agent]+self.boundary_offset,
                    n_points_to_return=self.n_points_nearing_boundary,
                    device=self.world.device,
                    sample_interval=self.sample_interval,
                    return_ref_v=False,
                    env_j=env_j,
                    line="left",
                )
            self.ref_paths_agent_related.nearing_points_right_boundary[env_j, i_agent] = \
                get_short_term_reference_path_by_s(
                    self.road,
                    self.observations.agent_s[env_j, i_agent]+self.boundary_offset,
                    n_points_to_return=self.n_points_nearing_boundary,
                    device=self.world.device,
                    sample_interval=self.sample_interval,
                    return_ref_v=False,
                    env_j=env_j,
                    line="right",
                )
        else:
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_j, i_agent
                ],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_left_b[
                    env_j, i_agent
                ],
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=self.sample_interval,
                n_points_shift=1,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_j, i_agent
                ],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_right_b[
                    env_j, i_agent
                ],
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=self.sample_interval,
                n_points_shift=1,
            )
        # 260115
        theta = agents[i_agent].state.rot[env_j, :]
        for idx in range(self.agent_lookahead_idx):
            lookahead_pts = agents[i_agent].state.pos[env_j, :] + (idx)*self.sample_interval * torch.hstack([torch.cos(theta), torch.sin(theta)])
            self.distances.lookahead_pts[env_j, i_agent, idx] = \
                torch.linalg.norm(self.ref_paths_agent_related.short_term[env_j, i_agent, idx, :2] - lookahead_pts, dim=-1)
        #251231 exit segment initialization
        s_max_idx = self.road.get_s_max_idx()[env_j]
        if s_max_idx.dim():
            last_pts_idx = s_max_idx[:, None, None].expand(-1, -1, 2)
        else:
            # single env_j
            last_pts_idx = s_max_idx[None, None].expand(-1, 2)
        self.ref_paths_agent_related.exit[env_j, i_agent, 0, :] = torch.gather(
            self.ref_paths_agent_related.left_boundary[env_j, i_agent], 
            dim=-2,
            index=last_pts_idx
        ).squeeze(-2)
        self.ref_paths_agent_related.exit[env_j, i_agent, 1, :] = torch.gather(
            self.ref_paths_agent_related.right_boundary[env_j, i_agent], 
            dim=-2,
            index=last_pts_idx
        ).squeeze(-2)
    
    def get_front_rear_v_use_front(self):
        s_front=self.observations.agent_s[:, self.HINGE_FIRST_INDEX].clone()
        ref_v = self.road.get_ref_v(s_front[:,None])[:,0,0] # [B]
        v_front = torch.linalg.norm(self.tractor_front.state.vel, dim=-1)
        error_v = v_front - ref_v
        last_v = torch.linalg.norm(self.state_buffer.get_latest(n=2)[:,self.HINGE_FIRST_INDEX,-2:], dim=-1)
        cur_acc = (v_front-last_v)/self.dt
        Kp=-1
        Kd=-0
        desire_acc = torch.clamp(Kp * error_v + Kd * cur_acc, 
                                 min=-self.max_acceleration,
                                 max=self.occt_kinematic_max_acceleration \
                                    if self.occt_drive_baseline in \
                                          {"front_base", "rear_base"} \
                                            else self.max_acceleration)
        v_front += desire_acc * self.dt
        s_front += v_front * self.dt # [B]
        delta_s, infeasible = self.road.solve_delta_s(s_front, self.rod_len*torch.ones_like(s_front))
        assert not infeasible.any(), "Infeasible delta_s"
        s_rear = s_front - delta_s
        v_rear = (s_rear - self.observations.agent_s[:, self.HINGE_LAST_INDEX])/self.dt
        return v_front, v_rear

    def get_front_rear_v_use_rear(self):
        s_rear=self.observations.agent_s[:, self.HINGE_LAST_INDEX].clone()
        ref_v = self.road.get_ref_v(s_rear[:,None])[:,0,0] # [B]
        v_rear = torch.linalg.norm(self.tractor_rear.state.vel, dim=-1)
        error_v = v_rear - ref_v
        last_v = torch.linalg.norm(self.state_buffer.get_latest(n=2)[:,self.HINGE_LAST_INDEX,-2:], dim=-1)
        cur_acc = (v_rear-last_v)/self.dt
        Kp=-1
        Kd=-0
        desire_acc = torch.clamp(Kp * error_v + Kd * cur_acc, 
                                 min=-self.max_acceleration,
                                 max=self.occt_kinematic_max_acceleration \
                                    if self.occt_drive_baseline in \
                                          {"front_base", "rear_base"} \
                                            else self.max_acceleration)
        v_rear += desire_acc * self.dt
        s_rear += v_rear * self.dt # [B]
        delta_s, infeasible = self.road.solve_delta_s(s_rear, self.rod_len*torch.ones_like(s_rear), backward=False)
        assert not infeasible.any(), "Infeasible delta_s"
        s_front = s_rear + delta_s
        v_front = (s_front - self.observations.agent_s[:, self.HINGE_FIRST_INDEX])/self.dt
        return v_front, v_rear

    def _get_occt_drive_force_limit(self, speed: Tensor) -> Tensor:
        effective_speed = torch.clamp(
            speed, min=self.occt_drive_power_blend_speed
        )
        power_limited_force = (
            self.occt_drive_power_cap_kw * 1000.0 / effective_speed
        )
        return torch.minimum(
            torch.full_like(speed, self.occt_drive_force_cap),
            power_limited_force,
        )

    def _pre_step_occt_kinematic(self, v_front: Tensor) -> None:
        s_front_prev = self.observations.agent_s[:, self.HINGE_FIRST_INDEX].clone()
        s_rear_prev = self.observations.agent_s[:, self.HINGE_LAST_INDEX].clone()

        s_front = v_front * self.dt + s_front_prev
        s_max = self.road.get_s_max() - 1e-6
        s_min = torch.zeros_like(s_max)
        s_front = torch.clamp(s_front, min=s_min, max=s_max)

        delta_s, infeasible = self.road.solve_delta_s(
            s_front, self.rod_len * torch.ones_like(s_front)
        )
        assert not infeasible.any(), "Infeasible delta_s"
        s_rear = torch.clamp(s_front - delta_s, min=s_min, max=s_max)

        v_front_actual = (s_front - s_front_prev) / self.dt
        v_rear_actual = (s_rear - s_rear_prev) / self.dt
        v_front_prev = torch.linalg.norm(self.tractor_front.state.vel, dim=-1)
        v_rear_prev = torch.linalg.norm(self.tractor_rear.state.vel, dim=-1)

        self.observations.agent_s[:, self.HINGE_FIRST_INDEX] = s_front
        self.observations.agent_s[:, self.HINGE_LAST_INDEX] = s_rear
        self.logged_control_acc[:, self.HINGE_FIRST_INDEX] = (
            (v_front_actual - v_front_prev) / self.dt
        ).detach().clone()
        self.logged_control_acc[:, self.HINGE_LAST_INDEX] = (
            (v_rear_actual - v_rear_prev) / self.dt
        ).detach().clone()

        front_rear_theta = self.road.get_tangent_heading(
            self.observations.agent_s[:, self.TRACTOR_SLICE]
        )
        front_theta = front_rear_theta[:, 0]
        rear_theta = front_rear_theta[:, 1]
        p_front, p_rear = self.get_front_rear_pts(
            self.observations.agent_s[:, self.TRACTOR_SLICE]
        )
        idx_mask = torch.ones(self.batch_dim, dtype=torch.bool, device=self.device)
        self._set_pose(
            self.tractor_front, p_front, front_theta, v_front_actual, idx_mask
        )
        self._set_pose(
            self.tractor_rear, p_rear, rear_theta, v_rear_actual, idx_mask
        )
    
    def pre_step(self):
        self.logged_control_acc.zero_()
        self.logged_control_steer.zero_()
        self.logged_occt_drive_force.zero_()
        self.logged_occt_internal_force.zero_()
        self.logged_occt_resistance_force.zero_()
        if self.task_class == TaskClass.SIMPLE_PLATOON:
            return
        self.M_total = 5000.0  # 总质量 (2辆小车 + 扇叶)
        self.L_cargo = self.rod_len     # 扇叶长度
        self.K_rigid = 1000.0  # 虚拟刚性系数 (弹簧系数)，越大越接近刚体
        self.D_rigid = 1000.0   # 虚拟阻尼系数，防止震荡
        self.K_drive = 10000.0 
        # 1. 获取运动学建议速度 (Kinematic candidates)
        v_front1, v_rear1 = self.get_front_rear_v_use_front()
        v_front2, v_rear2 = self.get_front_rear_v_use_rear()
        
        if self.occt_drive_baseline == "front_base":
            self._pre_step_occt_kinematic(v_front1)
        elif self.occt_drive_baseline == "rear_base":
            self._pre_step_occt_kinematic(v_front2)
        else:
            # 依然保留你的保守速度选择逻辑，作为“驱动力”的输入
            select_idx = torch.max(v_front1, v_rear1) < torch.max(v_front2, v_rear2)
            v_target_f = torch.where(select_idx, v_front1, v_front2)
            v_target_r = torch.where(select_idx, v_rear1, v_rear2)

            # 2. 获取当前状态
            s_f_curr = self.observations.agent_s[:, self.HINGE_FIRST_INDEX]
            s_r_curr = self.observations.agent_s[:, self.HINGE_LAST_INDEX]
            
            # 3. 计算刚体动力学约束力
            # 计算当前实际弦长距离 (可以通过 get_front_rear_pts 得到欧式距离)
            p_f, p_r = self.get_front_rear_pts(self.observations.agent_s[:, self.TRACTOR_SLICE])
            current_dist = torch.norm(p_f - p_r, dim=1)
            dist_error = current_dist - self.L_cargo
            
            # 计算距离变化率 (用于阻尼)
            v_f_curr =  torch.linalg.norm(self.tractor_front.state.vel, dim=-1)
            v_r_curr = torch.linalg.norm(self.tractor_rear.state.vel, dim=-1)
            dist_rate = v_f_curr - v_r_curr # 简化表达

            # 虚拟内部约束力 (Internal Force)
            f_internal = self.K_rigid * dist_error + self.D_rigid * dist_rate
            # 4. 计算驱动力 (Driving Force)
            # 基于理想运动学速度与当前速度的偏差来产生驱动力
            f_drive_f = self.K_drive * (v_target_f - v_f_curr)
            f_drive_r = self.K_drive * (v_target_r - v_r_curr)
            drive_limit_f = self._get_occt_drive_force_limit(v_f_curr)
            drive_limit_r = self._get_occt_drive_force_limit(v_r_curr)
            f_drive_f = torch.minimum(f_drive_f, drive_limit_f)
            f_drive_r = torch.minimum(f_drive_r, drive_limit_r)
            self.logged_occt_drive_force[:, 0] = f_drive_f.detach().clone()
            self.logged_occt_drive_force[:, 1] = f_drive_r.detach().clone()
            self.logged_occt_internal_force = f_internal.detach().clone()
            self.logged_occt_resistance_force = f_internal.abs().detach().clone()

            # 5. 应用牛顿定律更新加速度 (考虑巨大的质量 M)
            # a = (F_drive + F_internal) / M
            # 注意：首车受向后的拉力，尾车受向前的拉力
            a_f = (f_drive_f - f_internal) / (self.M_total / 2) 
            a_r = (f_drive_r + f_internal) / (self.M_total / 2)
            self.logged_control_acc[:, self.HINGE_FIRST_INDEX] = a_f.detach().clone()
            self.logged_control_acc[:, self.HINGE_LAST_INDEX] = a_r.detach().clone()

            # 6. 积分得到新速度和新位移
            v_f_new = v_f_curr + a_f * self.dt
            v_r_new = v_r_curr + a_r * self.dt
            s_f_new = s_f_curr + v_f_new * self.dt
            s_r_new = s_r_curr + v_r_new * self.dt

            # 7. 更新状态与位姿 (保持你的道路映射逻辑)
            s_max = self.road.get_s_max() - 1e-6
            self.observations.agent_s[:, self.HINGE_FIRST_INDEX] = torch.clamp(s_f_new, max = s_max)
            self.observations.agent_s[:, self.HINGE_LAST_INDEX] = torch.clamp(s_r_new, max = s_max)
            
            p_front_dyn, p_rear_dyn = self.get_front_rear_pts(self.observations.agent_s[:, self.TRACTOR_SLICE])
            front_rear_theta_dyn = self.road.get_tangent_heading(
                self.observations.agent_s[:, self.TRACTOR_SLICE]
            )
            front_theta_dyn = front_rear_theta_dyn[:, 0]
            rear_theta_dyn = front_rear_theta_dyn[:, 1]

            # ---- 3. 改写 _set_pose 调用 ----
            # 使用动力学积分得到的速度 v_f_new 和 v_r_new，而不是运动学解算的建议速度
            idx_mask = torch.ones(self.batch_dim, dtype=torch.bool, device=self.device)
            self._set_pose(
                self.tractor_front, 
                p_front_dyn, 
                front_theta_dyn, 
                v_f_new,       # 动力学平滑后的速度
                idx_mask
            )

            self._set_pose(
                self.tractor_rear, 
                p_rear_dyn, 
                rear_theta_dyn, 
                v_r_new,       # 动力学平滑后的速度
                idx_mask
            )

        if self.traditional_control!=MethodClass.MARL:
            if self.task_class==TaskClass.SIMPLE_PLATOON:
                self._pure_pursuit_control_platoon()
            else:
                if self.traditional_control==MethodClass.PID:
                    self._pure_pursuit_control_occt()
                elif self.traditional_control==MethodClass.MPPI:
                    self._mppi_control_occt()
                elif self.traditional_control==MethodClass.MARL:
                    pass
                else:
                    raise TypeError
    def _mppi_control_occt(self):
        """
        基于 MPPI 的编队及自主铰接控制器 
        - 用于batch_size=1时的手动控制，
        在对应的hinge不可用时保持横向路径跟踪以及纵向编队速度控制，
        在hinge可用时跟踪hinge进行自主铰接

        对每个agent：
        1. 使用纯跟踪算法计算前轮转角
        2. 使用MPPI控制器前向采样轨迹（要求采样n*sample_interval长度的轨迹，n可以设置为2）
        3. 计算每一个轨迹的reward（在编队阶段是与ref_path对应n线段的差距，在铰接阶段是与hinge_short_term对应n线段的差距）
        4. 使用MPPI的更新轨迹权重获得最优控制序列，取第一个控制量作为输出
        5. 调用_set_pose直接设置状态（绕过VMAS的dynamics）
        """
        if self.batch_dim != 1:
            raise NotImplementedError(
                "MPPI controller currently only supports batch_dim=1."
            )
        if self.simple_mppi is None:
            raise RuntimeError("Simple MPPI controller is not initialized.")
        target_agent_s, _ = self.get_dynamic_target_arc_positions()
        mppi_ref_point_count = self.simple_mppi.T + 1
        hinge_ref_sample_ds = torch.clamp(
            0.5
            * (
                torch.linalg.norm(self.world.agents[self.HINGE_FIRST_INDEX].state.vel[0])
                + torch.linalg.norm(self.world.agents[self.HINGE_LAST_INDEX].state.vel[0])
            )
            * self.dt,
            min=1e-3,
        )

        for agent_idx, agent in enumerate(self.world.agents):
            if agent_idx in self.TRACTOR_SLICE:
                continue

            current_pos = agent.state.pos[0]
            current_theta = agent.state.rot[0, 0]
            current_vel = agent.state.vel[0]

            v_current = torch.linalg.norm(current_vel)
            vel_dir = current_vel / (v_current + 1e-8)
            heading_vec = torch.stack(
                [torch.cos(current_theta), torch.sin(current_theta)]
            )
            direction_sign = torch.sign(torch.sum(vel_dir * heading_vec))
            current_speed = torch.clamp(
                v_current * direction_sign, min=0.0, max=self.max_speed
            )

            hinge_status = self.ref_paths_agent_related.hinge_status[:, agent_idx]
            hinge_ready = bool(hinge_status[0].item())

            if hinge_ready:
                ref_path = get_short_term_hinge_path_by_s(
                    occt_map=self.road,
                    agents=self.world.agents,
                    agent_s=self.observations.agent_s,
                    n_points_to_return=mppi_ref_point_count,
                    tractor_slice=self.TRACTOR_SLICE,
                    device=self.world.device,
                    sample_ds=hinge_ref_sample_ds,
                    env_j=slice(None),
                    hinge_relative_pos=self.hinge_relative_pos,
                )[0, agent_idx]
                ref_points = ref_path[:, :2]
                ref_speeds = torch.linalg.norm(ref_path[:, 2:4], dim=-1)
                stage_weight = self.mppi_hinge_stage_cost_weight
                terminal_weight = self.mppi_hinge_terminal_cost_weight
            else:
                front_vehicle_speed = torch.linalg.norm(
                    self.world.agents[agent_idx - 1].state.vel[0]
                )
                ref_sample_interval = torch.clamp(
                    front_vehicle_speed * self.dt,
                    min=1e-3,
                )
                ref_path = get_short_term_reference_path_by_s(
                    self.road,
                    target_agent_s[:, agent_idx],
                    n_points_to_return=mppi_ref_point_count,
                    device=self.world.device,
                    sample_interval=ref_sample_interval,
                    return_ref_v=False,
                    env_j=None,
                )[0]
                ref_points = ref_path[:, :2]
                ref_speeds = torch.ones(
                    ref_points.shape[0],
                    device=self.device,
                    dtype=torch.float32,
                ) * front_vehicle_speed
                stage_weight = self.mppi_stage_cost_weight
                terminal_weight = self.mppi_terminal_cost_weight
            observed_x = torch.stack(
                [current_pos[0], current_pos[1], current_theta, current_speed]
            )
            action, _, _ = self.simple_mppi.command(
                agent_idx=agent_idx,
                observed_x=observed_x,
                ref_points=ref_points,
                ref_speeds=ref_speeds,
                stage_cost_weight=stage_weight,
                terminal_cost_weight=terminal_weight,
            )

            steering_angle = torch.clamp(
                action[0],
                min=-self.max_steering_angle,
                max=self.max_steering_angle,
            )
            acceleration = torch.clamp(
                action[1],
                min=-self.max_acceleration,
                max=self.max_acceleration,
            )
            self.logged_control_steer[0, agent_idx] = steering_angle.detach()
            self.logged_control_acc[0, agent_idx] = acceleration.detach()

            beta = torch.atan2(
                torch.tan(steering_angle) * self.l_r / (self.l_f + self.l_r),
                torch.tensor(1.0, device=self.device),
            )
            dx = current_speed * torch.cos(current_theta + beta)
            dy = current_speed * torch.sin(current_theta + beta)
            dtheta = (
                (current_speed / (self.l_f + self.l_r))
                * torch.cos(beta)
                * torch.tan(steering_angle)
            )

            next_pos = current_pos + torch.stack([dx, dy]) * self.dt
            next_theta = current_theta + dtheta * self.dt
            next_v = torch.clamp(
                current_speed + acceleration * self.dt,
                min=0.0,
                max=self.max_speed,
            )

            idx_mask = torch.ones(
                self.batch_dim, dtype=torch.bool, device=self.device
            )
            self._set_pose(
                agent,
                next_pos.unsqueeze(0),
                next_theta.unsqueeze(0).unsqueeze(0),
                next_v.unsqueeze(0),
                idx_mask,
            )

    def _pure_pursuit_control_occt(self):
        """
        纯跟踪编队及自主铰接控制器 
        - 用于batch_size=1时的手动控制，
        在对应的hinge不可用时保持横向路径跟踪以及纵向编队速度控制，
        在hinge可用时跟踪hinge进行自主铰接

        对每个agent：
        1. 使用纯跟踪算法计算前轮转角
        2. 使用PD控制器计算加速度
        3. 手动积分计算下一个状态
        4. 调用_set_pose直接设置状态（绕过VMAS的dynamics）
        """
        LOOKAHEAD_DIST = 5.0
        WHEELBASE = self.l_f + self.l_r

        KP_VEL = 1.0
        KD_VEL = 0.0
        KP_HINGE_DIST = 8
        KP_GAP = 0.8
        pure_pursuit_ref_point_count = max(
            self.n_points_short_term,
            int(
                (LOOKAHEAD_DIST + max(self.still_space, float(self.sample_interval)))
                / max(float(self.sample_interval), 1e-3)
            )
            + 2,
        )

        if not hasattr(self, "_last_vel_errors"):
            self._last_vel_errors = {}
        if not hasattr(self, "_last_pid_steering"):
            self._last_pid_steering = {}
        target_agent_s, _ = self.get_dynamic_target_arc_positions()

        for agent_idx, agent in enumerate(self.world.agents):
            if agent_idx in self.TRACTOR_SLICE:
                continue

            current_pos = agent.state.pos[0]
            current_theta = agent.state.rot[0, 0]
            current_vel = agent.state.vel[0]
            heading_vec = torch.stack(
                [torch.cos(current_theta), torch.sin(current_theta)]
            )
            v_forward = torch.clamp(
                torch.dot(current_vel, heading_vec),
                min=0.0,
                max=self.max_speed,
            )

            hinge_status = self.ref_paths_agent_related.hinge_status[:, agent_idx]
            hinge_ready = bool(hinge_status[0].item())

            # 未进入可铰接区时跟踪编队短期参考路径；进入后切到 hinge 参考轨迹。
            if hinge_ready:
                ref_path = self.ref_paths_agent_related.hinge_short_term[0, agent_idx]
                ref_points = ref_path[:, :2]
            else:
                ref_path = get_short_term_reference_path_by_s(
                    self.road,
                    target_agent_s[:, agent_idx],
                    n_points_to_return=pure_pursuit_ref_point_count,
                    device=self.world.device,
                    sample_interval=self.sample_interval,
                    return_ref_v=False,
                    env_j=None,
                )[0]
                ref_points = ref_path[:, :2]

            cos_theta = torch.cos(current_theta)
            sin_theta = torch.sin(current_theta)
            rel_points = ref_points - current_pos.unsqueeze(0)
            target_x_vehicle_all = rel_points[:, 0] * cos_theta + rel_points[:, 1] * sin_theta
            target_y_vehicle_all = -rel_points[:, 0] * sin_theta + rel_points[:, 1] * cos_theta
            dists = torch.sqrt(target_x_vehicle_all**2 + target_y_vehicle_all**2)
            forward_mask = target_x_vehicle_all > 0.0
            if forward_mask.any():
                lookahead_cost = torch.where(
                    forward_mask,
                    torch.abs(dists - LOOKAHEAD_DIST),
                    torch.full_like(dists, float("inf")),
                )
                target_idx = torch.argmin(lookahead_cost)
            else:
                target_idx = torch.argmax(target_x_vehicle_all)
            target_x_vehicle = target_x_vehicle_all[target_idx]
            target_y_vehicle = target_y_vehicle_all[target_idx]

            ld = dists[target_idx]
            ly = target_y_vehicle
            ld = torch.clamp(ld, min=0.1)

            curvature = 2.0 * ly / (ld**2)
            steering_angle = torch.atan(curvature * WHEELBASE)
            steering_angle = torch.clamp(
                steering_angle,
                min=-self.max_steering_angle,
                max=self.max_steering_angle,
            )
            last_steering = self._last_pid_steering.get(
                agent_idx, steering_angle.detach().clone()
            )
            max_steering_delta = self.max_steering_rate * self.dt
            steering_angle = torch.clamp(
                steering_angle,
                min=last_steering - max_steering_delta,
                max=last_steering + max_steering_delta,
            )
            self._last_pid_steering[agent_idx] = steering_angle.detach().clone()

            front_neighbor_speed = torch.linalg.norm(
                self.world.agents[agent_idx - 1].state.vel[0]
            )

            if hinge_ready:
                hinge_pos = self.get_target_hinge_pos(agent_idx)[0]
                hinge_vel = self.get_target_hinge_vel(agent_idx)[0]
                hinge_speed = torch.linalg.norm(hinge_vel)
                if hinge_speed > 1e-6:
                    hinge_tangent = hinge_vel / hinge_speed
                else:
                    hinge_tangent = torch.stack(
                        [torch.cos(current_theta), torch.sin(current_theta)]
                    )

                hinge_distance_vec = hinge_pos - current_pos
                hinge_distance_error = torch.dot(hinge_distance_vec, hinge_tangent)
                v_ref = torch.clamp(
                    hinge_speed + KP_HINGE_DIST * hinge_distance_error,
                    min=0.0,
                    max=self.max_speed,
                )
            else:
                actual_gap_s = torch.clamp(
                    self.observations.agent_s[0, agent_idx - 1]
                    - self.observations.agent_s[0, agent_idx],
                    min=0.0,
                )
                desired_gap_s = self.platoon_space_batch[0]
                gap_error = actual_gap_s - desired_gap_s
                v_ref = torch.clamp(
                    front_neighbor_speed + KP_GAP * gap_error,
                    min=0.0,
                    max=self.max_speed,
                )

            vel_error = v_ref - v_forward
            last_vel_error = self._last_vel_errors.get(
                agent_idx, torch.tensor(0.0, device=self.device)
            )
            derivative = (vel_error - last_vel_error) / self.dt
            acceleration = KP_VEL * vel_error + KD_VEL * derivative
            acceleration = torch.clamp(
                acceleration,
                min=-self.max_acceleration,
                max=self.max_acceleration,
            )
            self.logged_control_steer[0, agent_idx] = steering_angle.detach()
            self.logged_control_acc[0, agent_idx] = acceleration.detach()
            self._last_vel_errors[agent_idx] = vel_error.detach().clone()

            beta = torch.atan2(
                torch.tan(steering_angle) * self.l_r / (self.l_f + self.l_r),
                torch.tensor(1.0, device=self.device),
            )

            dx = v_forward * torch.cos(current_theta + beta)
            dy = v_forward * torch.sin(current_theta + beta)
            dtheta = (
                (v_forward / (self.l_f + self.l_r))
                * torch.cos(beta)
                * torch.tan(steering_angle)
            )
            dv = acceleration

            next_pos = current_pos + torch.stack([dx, dy]) * self.dt
            next_theta = current_theta + dtheta * self.dt
            next_v = torch.clamp(v_forward + dv * self.dt, min=0.0, max=self.max_speed)

            idx_mask = torch.ones(
                self.batch_dim, dtype=torch.bool, device=self.device
            )
            self._set_pose(
                agent,
                next_pos.unsqueeze(0),
                next_theta.unsqueeze(0).unsqueeze(0),
                next_v.unsqueeze(0),
                idx_mask,
            )

    def _pure_pursuit_control_platoon(self):
        """
        纯跟踪编队控制器 
        - 用于batch_size=1时的手动控制，
        保持横向路径跟踪以及纵向编队速度控制

        对每个agent：
        1. 使用纯跟踪算法计算前轮转角
        2. 使用PD控制器计算加速度
        3. 手动积分计算下一个状态
        4. 调用_set_pose直接设置状态（绕过VMAS的dynamics）
        """
        # 纯跟踪控制器参数
        LOOKAHEAD_DIST = 5.0  # 前瞻距离（米）
        WHEELBASE = self.l_f + self.l_r  # 轴距

        # PD控制器参数
        KP_VEL = 1.0   # 速度比例增益
        KD_VEL = 0.0   # 速度微分增益

        # 初始化速度误差存储（如果不存在）
        if not hasattr(self, '_last_vel_errors'):
            self._last_vel_errors = {}

        for agent_idx, agent in enumerate(self.world.agents):
            # 跳过牵引车（它们有自己的控制）
            if self.task_class == TaskClass.OCCT_PLATOON and agent_idx in self.TRACTOR_SLICE:
                continue

            # ========== 1) 获取当前状态 ==========
            # 当前位置、航向、速度 [B=1, 2] -> [2]
            current_pos = agent.state.pos[0]           # [2]
            current_theta = agent.state.rot[0, 0]      # scalar
            current_vel = agent.state.vel[0]           # [2]

            # 当前速度大小
            v_current = torch.linalg.norm(current_vel)  # scalar

            # 速度方向（用于倒车判断）
            vel_dir = current_vel / (v_current + 1e-8)
            heading_vec = torch.stack([torch.cos(current_theta), torch.sin(current_theta)])
            direction_sign = torch.sign(torch.sum(vel_dir * heading_vec))
            v_signed = v_current * direction_sign

            # ========== 2) 获取参考路径信息 ==========
            # short_term: [B, n_agents, n_points_short_term, 3]
            # 最后维度: [x, y, ref_v]
            ref_path = self.ref_paths_agent_related.short_term[0, agent_idx]  # [n_points, 3]

            # 提取参考点和参考速度
            ref_points = ref_path[:, :2]  # [n_points, 2]

            # ========== 3) 纯跟踪算法 - 选择lookahead点 ==========
            # 计算当前点到所有参考点的距离
            dists = torch.linalg.norm(ref_points - current_pos, dim=-1)  # [n_points]

            # 找到最接近lookahead距离的参考点
            target_idx = torch.argmin(torch.abs(dists - LOOKAHEAD_DIST))
            target_point = ref_points[target_idx]  # [2]

            # ========== 4) 计算前轮转角（纯跟踪算法）==========
            # 将目标点转换到车辆坐标系
            dx = target_point[0] - current_pos[0]
            dy = target_point[1] - current_pos[1]

            # 旋转到车辆坐标系
            cos_theta = torch.cos(current_theta)
            sin_theta = torch.sin(current_theta)

            # 目标点在车辆坐标系中的位置
            target_x_vehicle = dx * cos_theta + dy * sin_theta
            target_y_vehicle = -dx * sin_theta + dy * cos_theta

            # 纯跟踪算法计算曲率：kappa = 2 * ly / ld^2
            # 其中: ly是横向偏差，ld是lookahead距离
            ld = torch.sqrt(target_x_vehicle**2 + target_y_vehicle**2)
            ly = target_y_vehicle

            # 避免除零
            ld = torch.clamp(ld, min=0.1)

            # 计算曲率
            curvature = 2.0 * ly / (ld**2)

            # 计算前轮转角：delta = arctan(kappa * L)
            steering_angle = torch.atan(curvature * WHEELBASE)

            # 限制前轮转角范围
            steering_angle = torch.clamp(
                steering_angle,
                min=-self.max_steering_angle,
                max=self.max_steering_angle
            )

            # ========== 5) PD控制器 - 计算加速度 ==========
            # 获取目标速度（使用lookahead点的参考速度）
            v_ref = torch.linalg.norm(self.world.agents[0].state.vel)

            # 速度误差
            vel_error = v_ref - v_signed

            # 获取上一时刻的速度误差
            last_vel_error = self._last_vel_errors.get(agent_idx, torch.tensor(0.0, device=self.device))

            # PD控制器计算加速度
            # acc = Kp * error + Kd * (error - last_error) / dt
            derivative = (vel_error - last_vel_error) / self.dt
            acceleration = KP_VEL * vel_error + KD_VEL * derivative

            # 限制加速度范围
            acceleration = torch.clamp(
                acceleration,
                min=-self.max_acceleration,
                max=self.max_acceleration
            )
            self.logged_control_steer[0, agent_idx] = steering_angle.detach().clone()
            self.logged_control_acc[0, agent_idx] = acceleration.detach().clone()

            # 保存当前速度误差供下次使用
            self._last_vel_errors[agent_idx] = vel_error.detach().clone()

            # ========== 6) 手动积分计算下一个状态 ==========
            # 使用自行车模型积分
            # 参考DelayedSteeringKinematicBicycle的f()函数

            # 滑移角 beta = atan(l_r / (l_f + l_r) * tan(delta))
            beta = torch.atan2(
                torch.tan(steering_angle) * self.l_r / (self.l_f + self.l_r),
                torch.tensor(1.0, device=self.device)
            )

            # 状态导数
            dx = v_signed * torch.cos(current_theta + beta)
            dy = v_signed * torch.sin(current_theta + beta)
            dtheta = (v_signed / (self.l_f + self.l_r)) * torch.cos(beta) * torch.tan(steering_angle)
            dv = acceleration

            # Euler积分
            next_pos = current_pos + torch.stack([dx, dy]) * self.dt
            next_theta = current_theta + dtheta * self.dt
            next_v = v_signed + dv * self.dt

            # 确保速度为正（自行车模型限制）
            next_v = torch.clamp(next_v, min=0.0)

            # ========== 7) 直接设置状态 ==========
            idx_mask = torch.ones(self.batch_dim, dtype=torch.bool, device=self.device)
            self._set_pose(agent, next_pos.unsqueeze(0), next_theta.unsqueeze(0).unsqueeze(0), next_v.unsqueeze(0), idx_mask)

            # ========== 8) 调试输出 ==========
            # if agent_idx == 1:  # 只打印第一个follower
            #     print(f"Agent {agent_idx}: v={v_current:.2f}m/s, v_ref={v_ref:.2f}m/s, "
            #           f"acc={acceleration:.2f}m/s², steer={torch.rad2deg(steering_angle):.1f}°, "
            #           f"next_v={next_v:.2f}m/s")
    def _sync_agent_s_from_world(self):
        """Project current agent poses back to Frenet s after physics or hard projection."""
        B = self.batch_dim
        F = len(self.world.agents)
        agents_pos = torch.zeros((B, F, 2), device=self.device)
        for i, agent in enumerate(self.world.agents):
            agents_pos[:, i, :2] = agent.state.pos

        agent_vel_vector = torch.stack(
            [
                torch.linalg.norm(self.world.agents[i].state.vel, dim=-1)
                for i in range(self.n_agents)
            ],
            dim=-1,
        )
        desire_agent_ds = agent_vel_vector * self.dt
        new_agent_s = calibrate_agent_s_by_road_pts(
            agent_pos=agents_pos,
            ref_agent_s=self.observations.agent_s.clone() + desire_agent_ds,
            road_get_pts_func=self.road.get_pts,
            interval=0.25,
            precision=0.005,
            forward_search=False,
            device=self.observations.agent_s.device,
        )
        self.observations.agent_s[..., self.FOLLOWER_SLICE] = new_agent_s[
            :, self.FOLLOWER_SLICE
        ]

    def _refresh_hinge_targets_and_status(self):
        """Refresh hinge targets using the latest world state before reward consumption."""
        if self.task_class != TaskClass.OCCT_PLATOON:
            return

        self.ref_paths_agent_related.hinge_short_term = get_short_term_hinge_path_by_s(
            occt_map=self.road,
            agents=self.world.agents,
            agent_s=self.observations.agent_s,
            n_points_to_return=self.n_points_short_term,
            tractor_slice=self.TRACTOR_SLICE,
            device=self.world.device,
            sample_ds=self.sample_interval,
            env_j=slice(None),
            hinge_relative_pos=self.hinge_relative_pos,
        )

        self.ref_paths_agent_related.hinge_status.zero_()
        for agent_i in range(self.n_agents):
            if agent_i not in self.TRACTOR_SLICE:
                self.ref_paths_agent_related.hinge_status[
                    :, agent_i
                ] = self.get_target_hinge_status(agent_i)

        hinge_info = self.ref_paths_agent_related.hinge_short_term
        # Dock uses the geometric secant between the first two hinge preview points.
        # Track its mismatch with the instantaneous hinge velocity direction for debugging.
        hinge_heading = hinge_info[..., 1, :2] - hinge_info[..., 0, :2]
        hinge_heading_mag = torch.linalg.norm(hinge_heading, dim=-1, keepdim=True)
        hinge_heading_tangent = hinge_heading / torch.clamp(hinge_heading_mag, min=1e-6)
        hinge_vel_tangent = hinge_info[..., 0, 2:4] / torch.clamp(
            torch.linalg.norm(hinge_info[..., 0, 2:4], dim=-1, keepdim=True),
            min=1e-6,
        )
        hinge_heading_vel_cross = (
            hinge_heading_tangent[..., 0] * hinge_vel_tangent[..., 1]
            - hinge_heading_tangent[..., 1] * hinge_vel_tangent[..., 0]
        )
        hinge_heading_vel_dot = (hinge_heading_tangent * hinge_vel_tangent).sum(
            dim=-1
        ).clamp(-1.0, 1.0)
        self.ref_paths_agent_related.hinge_heading_vel_angle_diff_deg = (
            torch.atan2(torch.abs(hinge_heading_vel_cross), hinge_heading_vel_dot)
            * (180.0 / torch.pi)
        )
        hinge_pos = hinge_info[..., 0, :2]
        hinge_vel = hinge_info[..., 0, 2:4]
        hinge_vel_mag = torch.linalg.norm(hinge_vel, dim=-1, keepdim=True)
        target_hinge_status = self.ref_paths_agent_related.hinge_status
        agent_pos = torch.stack(
            [self.world.agents[i].state.pos for i in range(self.n_agents)], dim=1
        )
        agent_theta = torch.stack(
            [self.world.agents[i].state.rot.squeeze(-1) for i in range(self.n_agents)],
            dim=1,
        )
        agent_heading_tangent = torch.stack(
            [torch.cos(agent_theta), torch.sin(agent_theta)], dim=-1
        )
        agent_heading_hinge_heading_cross = (
            hinge_heading_tangent[..., 0] * agent_heading_tangent[..., 1]
            - hinge_heading_tangent[..., 1] * agent_heading_tangent[..., 0]
        )
        agent_heading_hinge_heading_dot = (
            hinge_heading_tangent * agent_heading_tangent
        ).sum(dim=-1).clamp(-1.0, 1.0)
        self.ref_paths_agent_related.agent_heading_hinge_heading_angle_diff_deg = (
            torch.atan2(
                torch.abs(agent_heading_hinge_heading_cross),
                agent_heading_hinge_heading_dot,
            )
            * (180.0 / torch.pi)
        )
        agent_vel = torch.stack(
            [self.world.agents[i].state.vel for i in range(self.n_agents)], dim=1
        )
        agent_vel_mag = torch.linalg.norm(agent_vel, dim=-1, keepdim=True)
        agent_pos_legal = torch.linalg.norm(hinge_pos - agent_pos, dim=-1) < 0.15
        agent_heading_legal = agent_heading_hinge_heading_dot > torch.cos(
            torch.tensor(2 / 180 * torch.pi, device=self.world.device)
        )
        agent_vel_legal = (
            torch.abs(agent_vel_mag - hinge_vel_mag) < 0.75
        ).squeeze(-1)
        prev_agent_hinge_status = self.ref_paths_agent_related.agent_hinge_status.get_latest()
        agent_legal_to_hinge = (
            agent_pos_legal & agent_heading_legal & agent_vel_legal
        ) | prev_agent_hinge_status
        agent_hinge_status = agent_legal_to_hinge & target_hinge_status
        self.ref_paths_agent_related.agent_hinge_status.add(agent_hinge_status)

    def _project_hinged_followers(self):
        """Hard-project hinged followers onto the latest hinge targets."""
        follower_idx_mask = self.ref_paths_agent_related.agent_hinge_status.get_latest()
        target_hinge_info = self.ref_paths_agent_related.hinge_short_term
        for i, agent in enumerate(self.world.agents):
            if i in self.TRACTOR_SLICE:
                continue
            target_hinge_pos = target_hinge_info[:, i, 0, :2]
            target_hinge_heading = (
                target_hinge_info[:, i, 1, :2] - target_hinge_info[:, i, 0, :2]
            )
            target_hinge_theta = torch.atan2(
                target_hinge_heading[:, 1], target_hinge_heading[:, 0]
            )
            target_hinge_speed = torch.linalg.norm(
                target_hinge_info[:, i, 0, 2:4], dim=1
            )
            self._set_pose(
                agent,
                target_hinge_pos,
                target_hinge_theta,
                target_hinge_speed,
                follower_idx_mask[:, i],
            )

    def post_step(self):
        """
        每次 world.step() 之后：
        - 先把最新 world 状态同步回 Frenet s
        - 再根据当前状态刷新 hinge 目标和铰接状态
        - 最后对已经铰接的随动车辆做硬投影
        """
        self._sync_agent_s_from_world()

        if self.task_class == TaskClass.OCCT_PLATOON:
            self._refresh_hinge_targets_and_status()
            if self.dock_agent_when_hinged:
                self._project_hinged_followers()
                self._sync_agent_s_from_world()

    def get_scenario_info(self):
        """获取场景信息，用于调试和验证"""
        return {
            "batch_dim": self.batch_dim,
            "n_agents": self.n_agents,
            "n_followers": self.n_followers,
            "rod_len": self.rod_len,
            "dt": self.dt,
            "device": str(self.device),
        }

    def _compute_tracking_error_space(self):
        error_space = torch.zeros(
            (self.world.batch_dim, self.n_agents, 2),
            device=self.world.device,
            dtype=torch.float32,
        )
        if self.task_class == TaskClass.OCCT_PLATOON:
            target_agent_s, _ = self.get_dynamic_target_arc_positions()
            error_space[:, :, 0] = self.observations.agent_s - target_agent_s
        else:
            for agent_idx in range(self.n_agents):
                if agent_idx > 0:
                    actual_distance = self.distances.agents[:, agent_idx, agent_idx - 1]
                    error_space[:, agent_idx, 0] = actual_distance - self.platoon_space_batch
                if agent_idx < self.n_agents - 1:
                    actual_distance = self.distances.agents[:, agent_idx, agent_idx + 1]
                    error_space[:, agent_idx, 1] = actual_distance - self.platoon_space_batch
        return error_space

    def _compute_tracking_error_vel_components(self):
        platoon_error_vel = torch.zeros(
            (self.world.batch_dim, self.n_agents, 2),
            device=self.world.device,
            dtype=torch.float32,
        )
        hinge_error_vel = torch.zeros_like(platoon_error_vel)
        agent_speed = torch.stack(
            [torch.linalg.norm(agent.state.vel, dim=1) for agent in self.world.agents],
            dim=1,
        )
        leader_speed = agent_speed[:, 0]
        leader_ref_speed = self.ref_paths_agent_related.short_term[:, 0, 0, -1]
        for agent_idx in range(self.n_agents):
            ego_speed = agent_speed[:, agent_idx]
            if self.task_class == TaskClass.OCCT_PLATOON:
                if agent_idx in self.TRACTOR_SLICE:
                    continue
                target_hinge_speed = torch.linalg.norm(
                    self.get_target_hinge_vel(agent_idx, 0),
                    dim=-1,
                )
                front_vehicle_speed_error = agent_speed[:, agent_idx - 1] - ego_speed
                rear_vehicle_speed_error = agent_speed[:, agent_idx + 1] - ego_speed
                platoon_error_vel[:, agent_idx, 0] = front_vehicle_speed_error
                platoon_error_vel[:, agent_idx, 1] = rear_vehicle_speed_error
                hinge_error_vel[:, agent_idx, 0] = ego_speed - target_hinge_speed
                hinge_error_vel[:, agent_idx, 1] = hinge_error_vel[:, agent_idx, 0]
            else:
                if agent_idx == 0:
                    speed_error = ego_speed - leader_ref_speed
                else:
                    speed_error = ego_speed - leader_speed
                platoon_error_vel[:, agent_idx, 0] = speed_error
                platoon_error_vel[:, agent_idx, 1] = speed_error
                hinge_error_vel[:, agent_idx] = platoon_error_vel[:, agent_idx]
        return platoon_error_vel, hinge_error_vel
    def update_observation_and_normalize(self, agent, agent_index):
        """Update observation and normalize them."""
        if agent_index == 0:  # Avoid repeated computations
            positions_global = torch.stack(
                [a.state.pos for a in self.world.agents], dim=0
            ).transpose(0, 1)
            rotations_global = (
                torch.stack([a.state.rot for a in self.world.agents], dim=0)
                .transpose(0, 1)
                .squeeze(-1)
            )
            # Add new observation & normalize
            self.observations.past_distance_to_agents.add(
                self.distances.agents / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_ref_path.add(
                self.distances.ref_paths / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_left_boundary.add(
                torch.min(self.distances.left_boundaries, dim=-1)[0]
                / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_right_boundary.add(
                torch.min(self.distances.right_boundaries, dim=-1)[0]
                / self.normalizers.distance_lanelet
            )
            self.observations.past_distance_to_boundaries.add(
                self.distances.boundaries / self.normalizers.distance_lanelet
            )

            platoon_error_space = self._compute_tracking_error_space()
            platoon_error_vel, hinge_error_vel = self._compute_tracking_error_vel_components()
            self.observations.self_platoon_error_space.add(platoon_error_space)
            self.observations.platoon_error_vel = platoon_error_vel
            self.observations.hinge_error_vel = hinge_error_vel
            self.observations.past_platoon_error_vel.add(platoon_error_vel)
            self.observations.past_hinge_error_vel.add(hinge_error_vel)
            if True:
                pos_i_others = torch.zeros(
                    (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                    device=self.world.device,
                    dtype=torch.float32,
                )  # Positions of other agents relative to agent i
                rot_i_others = torch.zeros(
                    (self.world.batch_dim, self.n_agents, self.n_agents),
                    device=self.world.device,
                    dtype=torch.float32,
                )  # Rotations of other agents relative to agent i
                vel_i_others = torch.zeros(
                    (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                    device=self.world.device,
                    dtype=torch.float32,
                )  # Velocities of other agents relative to agent i
                ref_i_others = torch.zeros_like(
                    (self.observations.past_relative_ref_info.get_latest())
                )  # Reference paths of other agents relative to agent i
                hinge_i_others = torch.zeros_like(
                    (self.observations.past_relative_hinge_info.get_latest())
                )  # Reference paths of hinge points relative to agent i
                l_b_i_others = torch.zeros_like(
                    (self.observations.past_left_boundary.get_latest())
                )  # Left boundaries of other agents relative to agent i
                r_b_i_others = torch.zeros_like(
                    (self.observations.past_right_boundary.get_latest())
                )  # Right boundaries of other agents relative to agent i
                ver_i_others = torch.zeros_like(
                    (self.observations.past_vertices.get_latest())
                )  # Vertices of other agents relative to agent i
                steering_agents = torch.zeros(
                    (self.world.batch_dim, self.n_agents),
                    device=self.world.device,
                    dtype=torch.float32,
                )  # Steering of other agents relative to agent i
                for a_i in range(self.n_agents):
                    pos_i = self.world.agents[a_i].state.pos
                    rot_i = self.world.agents[a_i].state.rot
                    steering_agents[:, a_i] = self.world.agents[a_i].dynamics.cur_delta.squeeze(-1)
                    # Store new observation - position
                    pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=positions_global,
                        rot_i=rot_i,
                    )
                    # Store new observation - rotation
                    rot_i_others[:, a_i] = angle_eliminate_two_pi(
                        rotations_global - rot_i
                    )

                    for a_j in range(self.n_agents):
                        # Store new observation - velocities
                        rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                        vel_abs = torch.norm(
                            self.world.agents[a_j].state.vel, dim=1
                        ).unsqueeze(1)
                        vel_i_others[:, a_i, a_j] = torch.hstack(
                            (vel_abs * torch.cos(rot_rel), vel_abs * torch.sin(rot_rel))
                        )

                        # Store new observation - reference paths
                        ref_i_others[
                            :, a_i, a_j, :, 0:2
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.short_term[:, a_j, :, 0:2],
                            rot_i=rot_i,
                        )
                        ref_i_others[
                            :, a_i, a_j, :, 2
                        ] = self.ref_paths_agent_related.short_term[:, a_j, :, 2]
                        
                        # Store new observation - left boundary
                        l_b_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.nearing_points_left_boundary[
                                :, a_j
                            ],
                            rot_i=rot_i,
                        )

                        # Store new observation - right boundary
                        r_b_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.nearing_points_right_boundary[
                                :, a_j
                            ],
                            rot_i=rot_i,
                        )

                        # Store new observation - vertices
                        ver_i_others[
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.vertices[:, a_j, 0:4, :],
                            rot_i=rot_i,
                        )
                    # j-th hinge short term relative to agent i
                    for a_j in range(self.n_hinges):
                        hinge_i_others[
                            :, a_i, a_j, :, 0:2
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.hinge_short_term[:, a_j, :, 0:2],
                            rot_i=rot_i,
                        )
                        hinge_i_others[
                            :, a_i, a_j, :, 2:
                        ] = self.ref_paths_agent_related.hinge_short_term[:, a_j, :, 2:]
                
                assert not torch.isnan(self.observations.platoon_error_vel).any()
                assert not torch.isnan(self.observations.hinge_error_vel).any()

                # Add new observations & normalize
                self.observations.past_pos.add(
                    pos_i_others/self.normalizers.pos
                )
                self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
                self.observations.past_vel.add(vel_i_others / self.normalizers.v)
                self.observations.past_steering.add(steering_agents / self.normalizers.action_steering)
                self.observations.past_relative_ref_info.add(
                    ref_i_others / torch.hstack((self.normalizers.pos, self.normalizers.v.unsqueeze(0)))
                )
                if self.task_class == TaskClass.OCCT_PLATOON:
                    self.observations.past_relative_hinge_info.add(
                        hinge_i_others / torch.tensor([self.normalizers.pos[0], 
                                                       self.normalizers.pos[1], 
                                                       self.normalizers.v,
                                                       self.normalizers.v,
                                                       1],device=self.device,dtype=torch.float32)
                    )
                        
                self.observations.past_left_boundary.add(
                    l_b_i_others
                    / self.normalizers.pos
                )
                self.observations.past_right_boundary.add(
                    r_b_i_others
                    / self.normalizers.pos
                )
                self.observations.past_vertices.add(
                    ver_i_others
                    / self.normalizers.pos
                )

            # Add new observation - actions & normalize
            if agent.action.u is None:
                self.observations.past_action_acc.add(self.constants.empty_action_acc)
                self.observations.past_action_steering.add(
                    self.constants.empty_action_steering
                )
            else:
                self.observations.past_action_acc.add(
                    torch.stack([a.action.u[:, 0] for a in self.world.agents], dim=1)
                    / self.normalizers.action_acc
                )
                self.observations.past_action_steering.add(
                    torch.stack([a.action.u[:, 1] for a in self.world.agents], dim=1)
                    / self.normalizers.action_steering
                )

    def _filter_obs_group_dict(self, obs_groups: List[Tuple[str, Optional[Tensor]]]) -> Dict[str, Tensor]:
        return {
            name: tensor
            for name, tensor in obs_groups
            if tensor is not None
        }

    def _stack_history_obs_dicts(self, history_obs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if not history_obs:
            return {}
        return {
            key: torch.stack([obs_dict[key] for obs_dict in history_obs], dim=1)
            for key in history_obs[0].keys()
        }

    def _flatten_obs_dict_for_audit(self, obs_dict: Dict[str, Tensor]) -> Tensor:
        if not obs_dict:
            return torch.zeros((self.world.batch_dim, 0), device=self.world.device)
        return torch.cat(
            [value.reshape(self.world.batch_dim, -1) for value in obs_dict.values()],
            dim=-1,
        )

    def _add_noise_to_obs_dict(self, obs_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if not self.is_add_noise:
            return obs_dict
        return {
            key: value
            + self.observations.noise_level
            * torch.rand_like(value, device=self.world.device, dtype=torch.float32)
            for key, value in obs_dict.items()
        }

    def _finalize_observation_dict(
        self,
        agent_index: int,
        obs_dict: Dict[str, Tensor],
        obs_groups: List[Tuple[str, Tensor]],
    ) -> Dict[str, Tensor]:
        if self.enable_obs_audit:
            self._maybe_print_obs_audit(
                agent_index,
                obs_groups + [("obs_total", self._flatten_obs_dict_for_audit(obs_dict))],
            )
        check_validity(self.observations)
        check_validity(self.ref_paths_agent_related)
        return self._add_noise_to_obs_dict(obs_dict)

    def _get_history_step_indices(self) -> List[int]:
        obs_len = min(self.history_obs_len, int(self.observations.n_observed_steps.item()))
        return list(range(1, obs_len + 1))

    def _observe_self_at_step(self, agent_index: int, step: int = 1) -> Dict[str, Tensor]:
        batch_size = self.world.batch_dim
        indexing_tuple_vel = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index, 0))
        )
        self_short_term = self.observations.past_relative_ref_info.get_latest(step)[
            :, agent_index, agent_index
        ]

        observed_hinge_info = None
        observed_hinge_first_point_info = None
        self_hinge_status = None
        if self.task_class == TaskClass.OCCT_PLATOON:
            self_hinge_status = self.ref_paths_agent_related.hinge_status[
                :, agent_index
            ].to(torch.float32)
            relative_hinge_info = self.observations.past_relative_hinge_info.get_latest(
                step
            )[:, agent_index, agent_index]
            hinge_pos = torch.clamp(relative_hinge_info[..., :2], min=-1, max=1)
            hinge_speed = torch.linalg.norm(
                relative_hinge_info[..., 2:4],
                dim=-1,
                keepdim=True,
            )
            hinge_dis_boundary = torch.clamp(
                relative_hinge_info[..., 4:5] / (self.lane_width / 2),
                min=-1,
                max=1,
            )
            observed_hinge_info = torch.cat(
                (hinge_pos, hinge_speed, hinge_dis_boundary),
                dim=-1,
            )
            observed_hinge_first_point_info = observed_hinge_info[:, 0, :]
            if (
                observed_hinge_info is not None
                and observed_hinge_info.max() > self.obs_audit_large_threshold
            ):
                print(
                    f"[OBS_AUDIT_DEBUG] HINGE_INFO_ABNORMAL "
                    f"step={self._get_obs_audit_step()} agent={agent_index} "
                    f"max_abs={observed_hinge_info.max():.3e}"
                )

        self_left_boundary_pts = self.observations.past_left_boundary.get_latest(step)[
            :, agent_index, agent_index, 1:, :
        ]
        self_right_boundary_pts = self.observations.past_right_boundary.get_latest(step)[
            :, agent_index, agent_index, 1:, :
        ]
        self_left_dis = torch.linalg.norm(
            self_left_boundary_pts - self_short_term[..., :2], dim=-1
        ).unsqueeze(-1)
        self_right_dis = torch.linalg.norm(
            self_right_boundary_pts - self_short_term[..., :2], dim=-1
        ).unsqueeze(-1)
        vel = self.observations.past_vel.get_latest(step)[indexing_tuple_vel].reshape(
            batch_size, 1
        )
        vel_mag = torch.linalg.norm(vel, dim=-1, keepdim=True)

        return self._filter_obs_group_dict(
            [
                ("self_vel", vel),
                ("self_speed", vel_mag),
                (
                    "self_steering",
                    self.observations.past_steering.get_latest(step)[:, agent_index].reshape(
                        batch_size, 1
                    ),
                ),
                (
                    "self_acc",
                    self.observations.past_action_acc.get_latest(step)[:, agent_index].reshape(
                        batch_size, 1
                    ),
                ),
                (
                    "self_ref_velocity",
                    self_short_term[..., 2:3],
                ),
                ("self_ref_points", self_short_term[..., :2]),
                ("self_left_boundary_distance", self_left_dis),
                ("self_right_boundary_distance", self_right_dis),
                (
                    "self_hinge_status",
                    self_hinge_status.unsqueeze(-1)
                    if self.task_class == TaskClass.OCCT_PLATOON
                    else None,
                ),
                (
                    "self_hinge_preview_info",
                    observed_hinge_info
                    if self.task_class == TaskClass.OCCT_PLATOON
                    else None,
                ),
                (
                    "self_hinge_past_info",
                    observed_hinge_first_point_info
                    if self.task_class == TaskClass.OCCT_PLATOON
                    else None,
                ),
                (
                    "self_hinge_error_vel",
                    (
                        self.observations.past_hinge_error_vel.get_latest(step)[:, agent_index,0]
                        / self.normalizers.error_v
                    ).reshape(batch_size, 1),
                ),
                (
                    "self_distance_to_ref",
                    torch.linalg.norm(self_short_term[:, 0, :2], dim=-1, keepdim=True),
                ),
                (
                    "self_distance_to_left_boundary",
                    self.observations.past_distance_to_left_boundary.get_latest(step)[
                        :, agent_index
                    ].reshape(batch_size, 1),
                ),
                (
                    "self_distance_to_right_boundary",
                    self.observations.past_distance_to_right_boundary.get_latest(step)[
                        :, agent_index
                    ].reshape(batch_size, 1),
                ),
                (
                    "self_platoon_error_vel",
                    (
                        self.observations.past_platoon_error_vel.get_latest(step)[:, agent_index]
                        / self.normalizers.error_v
                    ).reshape(batch_size, 2),
                ),
                (
                    "self_platoon_error_space",
                    self.observations.self_platoon_error_space.get_latest(step)[:, agent_index, :]
                    / self.normalizers.error_pos,
                ),
            ]
        )

    def observe_self(self, agent_index):
        return self._observe_self_at_step(agent_index, step=1)

    def observe_self_history(self, agent_index: int) -> Dict[str, Tensor]:
        step_indices = self._get_history_step_indices()
        obs_self_history = [
            self._observe_self_at_step(agent_index, step=step) for step in step_indices
        ]
        obs_self_history_dict = self._stack_history_obs_dicts(obs_self_history)
        return obs_self_history_dict

    def _get_nearing_agents_indices(self, agent_index: int):
        nearing_agents_distances, nearing_agents_indices = torch.topk(
            self.distances.agents[:, agent_index],
            k=self.observations.n_nearing_agents,
            largest=False,
        )
        nearing_agents_indices, sorted_pos = torch.sort(nearing_agents_indices, dim=1)
        nearing_agents_distances = torch.gather(
            nearing_agents_distances, dim=1, index=sorted_pos
        )
        return nearing_agents_distances, nearing_agents_indices

    def _observe_other_agents_platoon_at_step(
        self,
        agent_index: int,
        *,
        step: int = 1,
        nearing_agents_indices: Optional[Tensor] = None,
        relative_longitudinal_velocity: Optional[Tensor] = None,
        relative_acceleration: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        if nearing_agents_indices is None:
            _, nearing_agents_indices = self._get_nearing_agents_indices(agent_index)

        indexing_tuple_1 = (
            (self.constants.env_idx_broadcasting,)
            + ((agent_index,))
            + (nearing_agents_indices,)
        )
        obs_pos = self.observations.past_pos.get_latest(step)[indexing_tuple_1]
        obs_rot = self.observations.past_rot.get_latest(step)[indexing_tuple_1].unsqueeze(-1)
        obs_distance = self.observations.past_distance_to_agents.get_latest(step)[
            self.constants.env_idx_broadcasting,
            agent_index,
            nearing_agents_indices,
        ].unsqueeze(-1)

        if relative_longitudinal_velocity is None or relative_acceleration is None:
            relative_longitudinal_velocity_history = (
                self.get_local_relative_longitudinal_velocity_history(
                    agent_index, indexing_tuple_1
                )
            )
            relative_longitudinal_velocity = relative_longitudinal_velocity_history[..., :1]
            if relative_longitudinal_velocity_history.shape[-1] > 1:
                relative_acceleration = (
                    relative_longitudinal_velocity
                    - relative_longitudinal_velocity_history[..., 1:2]
                ) / self.dt
            else:
                relative_acceleration = torch.zeros_like(relative_longitudinal_velocity)

        return self._filter_obs_group_dict(
            [
                ("others_pos", obs_pos),
                ("others_rot", obs_rot),
                (
                    "others_relative_longitudinal_velocity",
                    relative_longitudinal_velocity / self.obs_relative_velocity_scale,
                ),
                (
                    "others_relative_acceleration",
                    relative_acceleration / self.obs_relative_acceleration_scale,
                ),
                ("others_distance", obs_distance),
            ]
        )

    def observe_other_agents_history(self, agent_index: int) -> Dict[str, Tensor]:
        step_indices = self._get_history_step_indices()
        _, nearing_agents_indices = self._get_nearing_agents_indices(agent_index)
        indexing_tuple_1 = (
            (self.constants.env_idx_broadcasting,)
            + ((agent_index,))
            + (nearing_agents_indices,)
        )

        other_local_vel_history = torch.stack(
            [
                self.observations.past_vel.get_latest(step)[indexing_tuple_1]
                for step in step_indices
            ],
            dim=1,
        ) * self.normalizers.v
        ego_local_vel_history = torch.stack(
            [
                self.observations.past_vel.get_latest(step)[:, agent_index, agent_index]
                for step in step_indices
            ],
            dim=1,
        ) * self.normalizers.v
        relative_longitudinal_velocity_history = (
            other_local_vel_history[..., 0]
            - ego_local_vel_history[..., 0].unsqueeze(-1)
        )
        relative_acceleration_history = torch.zeros_like(
            relative_longitudinal_velocity_history
        )
        if len(step_indices) > 1:
            relative_acceleration_history[:, :-1] = (
                relative_longitudinal_velocity_history[:, :-1]
                - relative_longitudinal_velocity_history[:, 1:]
            ) / self.dt

        obs_other_history = []
        for time_index, step in enumerate(step_indices):
            obs_other_history.append(
                self._observe_other_agents_platoon_at_step(
                    agent_index,
                    step=step,
                    nearing_agents_indices=nearing_agents_indices,
                    relative_longitudinal_velocity=relative_longitudinal_velocity_history[
                        :, time_index
                    ].unsqueeze(-1),
                    relative_acceleration=relative_acceleration_history[
                        :, time_index
                    ].unsqueeze(-1),
                )
            )

        obs_other_history_dict = self._stack_history_obs_dicts(obs_other_history)
        return obs_other_history_dict

    def observe_other_agents(self, agent_index):
        return self.observe_other_agents_platoon(agent_index)

    def observe_other_agents_platoon(self, agent_index):
        obs_other_agents = self._observe_other_agents_platoon_at_step(
            agent_index, step=1
        )
        return obs_other_agents

    def _collect_observation_dict(self, agent_index: int) -> Dict[str, Tensor]:
        if self.use_history_observation:
            obs_self = self.observe_self_history(agent_index)
            obs_other = self.observe_other_agents_history(agent_index)
        else:
            obs_self = self.observe_self(agent_index)
            obs_other = self.observe_other_agents_platoon(agent_index)
        return {**obs_self, **obs_other}

    def observation(self, agent: Agent):
        agent_index = self.world.agents.index(agent)

        self.update_observation_and_normalize(agent, agent_index)
        obs = self._collect_observation_dict(agent_index)
        return self._finalize_observation_dict(
            agent_index,
            obs,
            list(obs.items()),
        )
    def _get_obs_audit_step(self) -> int:
        if hasattr(self, "timer") and hasattr(self.timer, "step"):
            return int(self.timer.step.max().item())
        return int(self.env_current_step.max().item())
    def _format_obs_audit_stats(self, obs_tensor: Tensor, previous_tensor: Optional[Tensor]):
        flat = obs_tensor.detach().reshape(-1).to(dtype=torch.float32)
        abs_flat = flat.abs()
        flat_cpu = flat.cpu()
        quantiles = torch.quantile(
            flat_cpu,
            torch.tensor([0.01, 0.50, 0.99], dtype=torch.float32),
        )
        stats = {
            "dim": int(obs_tensor.shape[-1]),
            "mean": flat.mean().item(),
            "std": flat.std(unbiased=False).item() if flat.numel() > 1 else 0.0,
            "mean_abs": abs_flat.mean().item(),
            "max_abs": abs_flat.max().item(),
            "p01": quantiles[0].item(),
            "p50": quantiles[1].item(),
            "p99": quantiles[2].item(),
            "small_frac": (
                (abs_flat < self.obs_audit_small_threshold).to(dtype=torch.float32).mean().item()
            ),
            "large_frac": (
                (abs_flat > self.obs_audit_large_threshold).to(dtype=torch.float32).mean().item()
            ),
            "delta_std": float("nan"),
        }
        if previous_tensor is not None and previous_tensor.shape == obs_tensor.shape:
            delta = (obs_tensor.detach() - previous_tensor).reshape(-1).to(dtype=torch.float32)
            stats["delta_std"] = delta.std(unbiased=False).item() if delta.numel() > 1 else 0.0
        return stats
    def _maybe_print_obs_audit(self, agent_index: int, observation_groups: List[Tuple[str, Tensor]]):
        if agent_index != self.obs_audit_agent_index:
            return

        current_step = self._get_obs_audit_step()
        should_log = (
            current_step > 0
            and self.obs_audit_interval > 0
            and current_step % self.obs_audit_interval == 0
            and current_step != self.obs_audit_last_logged_step
        )

        if should_log:
            print(
                f"\n[OBS_AUDIT] step={current_step} agent={agent_index} "
                f"small<{self.obs_audit_small_threshold:g} large>{self.obs_audit_large_threshold:g}"
            )
            for name, tensor in observation_groups:
                previous_tensor = self.obs_audit_prev_groups.get(name)
                stats = self._format_obs_audit_stats(tensor, previous_tensor)
                print(
                    f"  - {name:30s} d={stats['dim']:3d} "
                    f"mu={stats['mean']:+.2e} sd={stats['std']:.2e} "
                    f"q=[{stats['p01']:+.2e}|{stats['p50']:+.2e}|{stats['p99']:+.2e}] "
                    f"s={stats['small_frac']:.1%} l={stats['large_frac']:.1%} "
                    f"ds={stats['delta_std']:.2e}"
                )
            print("[OBS_AUDIT] end\n")
            self.obs_audit_last_logged_step = current_step

        for name, tensor in observation_groups:
            self.obs_audit_prev_groups[name] = tensor.detach().clone()
    def get_target_hinge_status(self, agent_index):
        hinge_dis_boundary = self.ref_paths_agent_related.hinge_short_term[:, agent_index, :, -1] # [B, n_points, 5]
        hinge_ready = hinge_dis_boundary > self.hinge_edge_buffer
        is_block, block_order = check_boolean_block(hinge_ready)
        # 0=先0后1(过完弯），2=纯1(直道) is_block(不是反复可铰接)
        is_after_corner = self.observations.agent_s[:,agent_index] > self.road.batch_corner_s_begin
        if self.traditional_control==MethodClass.MARL:
            ready_to_hinge = (((block_order==0) | (block_order==2)) & is_block) & is_after_corner
        else:
            ready_to_hinge = (((block_order==2)) & is_block) & is_after_corner
        return ready_to_hinge

    def get_dynamic_target_arc_positions(self):
        """Get equally spaced target arc positions between front and rear tractors."""
        s_front = self.observations.agent_s[:, self.HINGE_FIRST_INDEX]
        s_rear = self.observations.agent_s[:, self.HINGE_LAST_INDEX]
        if self.n_agents <= 1:
            return self.observations.agent_s.clone(), torch.zeros_like(s_front)

        desired_gap_s = (s_front - s_rear) / (self.n_agents - 1)
        agent_indices = torch.arange(
            self.n_agents, device=self.device, dtype=s_front.dtype
        )
        target_agent_s = s_front.unsqueeze(-1) - desired_gap_s.unsqueeze(-1) * agent_indices
        return target_agent_s, desired_gap_s

    def get_all_followers_hinged(self):
        if self.task_class != TaskClass.OCCT_PLATOON or self.n_followers <= 0:
            return torch.ones(self.batch_dim, device=self.device, dtype=torch.bool)
        return self.ref_paths_agent_related.agent_hinge_status.get_latest(n=1)[
            :, self.FOLLOWER_SLICE
        ].all(dim=-1)

    def _get_all_hinged_done_disabled_mask(self):
        if self.disable_all_hinged_done_road_id_tensor.numel() == 0:
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.bool)
        return (
            self.road.batch_id.unsqueeze(-1)
            == self.disable_all_hinged_done_road_id_tensor.unsqueeze(0)
        ).any(dim=-1)

    def get_hinged_followers_ratio(self):
        if self.task_class != TaskClass.OCCT_PLATOON or self.n_followers <= 0:
            return torch.ones(self.batch_dim, device=self.device, dtype=torch.float32)
        return self.ref_paths_agent_related.agent_hinge_status.get_latest(n=1)[
            :, self.FOLLOWER_SLICE
        ].to(torch.float32).mean(dim=-1)

    def get_all_hinged_reward_mask(self):
        if self.task_class != TaskClass.OCCT_PLATOON:
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.bool)
        return self.get_all_followers_hinged() & ~self.all_hinged_reward_granted

    def get_hinge_approach_progress(self, agent_index, is_simple=False):
        if self.observations.past_relative_hinge_info.valid_size < 2:
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)

        hinge_status = self.ref_paths_agent_related.hinge_status[:,agent_index]
        if not hinge_status.any():
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)

        prev_relative_hinge = self.observations.past_relative_hinge_info.get_latest(n=2)[
            :, agent_index, agent_index
        ]
        prev_hinge_distance = torch.linalg.norm(
            prev_relative_hinge[:, 0, :2] * self.normalizers.pos,
            dim=-1,
        )
        current_hinge_distance = torch.linalg.norm(
            self.get_target_hinge_pos(agent_index) - self.world.agents[agent_index].state.pos,
            dim=-1,
        )
        hinge_approach_progress = prev_hinge_distance - current_hinge_distance
        if is_simple:
            return torch.sign(hinge_approach_progress)
        return torch.where(
            hinge_status,
            hinge_approach_progress,
            torch.zeros_like(hinge_approach_progress),
        )
    def get_local_relative_longitudinal_velocity_history(self, agent_index, indexing_tuple_1):
        n_observed_steps = int(self.observations.n_observed_steps.item())
        other_local_vel_history = torch.stack(
            [
                self.observations.past_vel.get_latest(i + 1)[indexing_tuple_1]
                for i in range(n_observed_steps)
            ],
            dim=-1,
        ) * self.normalizers.v
        ego_local_vel_history = torch.stack(
            [
                self.observations.past_vel.get_latest(i + 1)[:, agent_index, agent_index]
                for i in range(n_observed_steps)
            ],
            dim=-1,
        ) * self.normalizers.v
        return other_local_vel_history[..., 0, :] - ego_local_vel_history[:, None, 0, :]
    def clamp_error_reward(self,weight,error,offset:float=1.0,norm:int=2,max:float=1.0,min:float=None):
        return (
            offset - torch.clamp(
                weight * error**norm,
                max=max,
                min=min,
            )
        )
    def _reward_conditional_goal(self):
        all_followers_hinged = self.get_all_followers_hinged()
        hinged_followers_ratio = self.get_hinged_followers_ratio()
        if self.goal_incomplete_hinge_penalty_mode == "constant":
            goal_penalty = self.penalty_goal_incomplete_hinge.expand(self.batch_dim)
        elif self.goal_incomplete_hinge_penalty_mode == "scaled":
            goal_penalty = self.penalty_goal_incomplete_hinge * (1 - hinged_followers_ratio)
        else:
            raise ValueError(
                f"Unsupported goal_incomplete_hinge_penalty_mode: "
                f"{self.goal_incomplete_hinge_penalty_mode}"
            )
        goal_mask=all_followers_hinged.to(torch.float32)
        is_reach_goal = self.collisions.with_exit_segments.any(dim=-1).to(torch.float32)
        reward_goal = is_reach_goal * (
            goal_mask * self.rewards.reward_goal + (1 - goal_mask) * goal_penalty
        )
        return reward_goal

    def _reward_agent_heading(
        self,
        ref_points_vecs,
        move_vec,
    ):
        ref_vector = torch.mean(ref_points_vecs, dim=1)
        ref_vector_normalized = ref_vector / (torch.norm(ref_vector, dim=-1, keepdim=True) + 1e-8)
        move_vector = move_vec[:, 0, :]
        move_vector_normalized = move_vector / (torch.norm(move_vector, dim=-1, keepdim=True) + 1e-8)
        max_delta_angle = torch.deg2rad(torch.tensor(15, device=self.device, dtype=torch.float32))
        constant_k = 1 / (1 - torch.cos(max_delta_angle))
        costant_b = 1 - constant_k
        heading_alignment = torch.clamp(
            constant_k * torch.sum(ref_vector_normalized * move_vector_normalized, dim=-1)
            + costant_b,
            min=0.0,
            max=1.0,
        )
        reward_platoon_heading = self.clamp_error_reward(
            self.rewards.reward_platoon_heading,
            (1 - heading_alignment),
            norm=1,
        )
        return reward_platoon_heading

    def reward_simple_platoon(self, reward_details, agent_index, ref_points_vecs, move_vec):
        space_errors_sq = self.observations.self_platoon_error_space.get_latest(n=1)[:, agent_index, 0] ** 2
        track_ref_mask = torch.ones(self.batch_dim, device=self.device, dtype=torch.bool)

        # reward_details = self._apply_reference_tracking_rewards(
        #     reward_details=reward_details,
        #     agent_index=agent_index,
        #     ref_points_vecs=ref_points_vecs,
        #     move_vec=move_vec,
        #     goal_mask=torch.ones(self.batch_dim, device=self.device, dtype=torch.float32),
        # )
        return reward_details

    def _update_reward_phase_weight(
        self, agent_index: int, hinge_status: Tensor
    ) -> Tensor:
        target_weight = hinge_status.to(torch.float32)
        if not self.reward_transition_blend_enabled:
            self.reward_phase_weights[:, agent_index] = target_weight
            self.reward_phase_transition_start[:, agent_index] = target_weight
            self.reward_phase_transition_target[:, agent_index] = target_weight
            self.reward_phase_transition_progress[:, agent_index] = 1.0
            return target_weight

        previous_weight = self.reward_phase_weights[:, agent_index]
        previous_target = self.reward_phase_transition_target[:, agent_index]
        target_changed = (target_weight - previous_target).abs() > 1e-6
        if target_changed.any():
            self.reward_phase_transition_start[:, agent_index] = torch.where(
                target_changed,
                previous_weight,
                self.reward_phase_transition_start[:, agent_index],
            )
            self.reward_phase_transition_target[:, agent_index] = torch.where(
                target_changed,
                target_weight,
                previous_target,
            )
            self.reward_phase_transition_progress[:, agent_index] = torch.where(
                target_changed,
                torch.zeros_like(target_weight),
                self.reward_phase_transition_progress[:, agent_index],
            )

        if self.reward_transition_blend_mode == "ramp":
            delta = torch.clamp(
                target_weight - previous_weight,
                min=-self.reward_transition_step_delta,
                max=self.reward_transition_step_delta,
            )
            updated_weight = previous_weight + delta
        elif self.reward_transition_blend_mode in {"s_curve", "sigmoid"}:
            progress = (
                self.reward_phase_transition_progress[:, agent_index]
                + self.reward_transition_step_delta
            ).clamp(0.0, 1.0)
            self.reward_phase_transition_progress[:, agent_index] = progress
            smooth_progress = progress * progress * (3.0 - 2.0 * progress)
            transition_start = self.reward_phase_transition_start[:, agent_index]
            transition_target = self.reward_phase_transition_target[:, agent_index]
            updated_weight = transition_start + (
                transition_target - transition_start
            ) * smooth_progress
        else:
            # First-order smoothing: update magnitude scales with the current gap.
            updated_weight = previous_weight + self.reward_transition_step_delta * (
                target_weight - previous_weight
            )
        updated_weight = updated_weight.clamp(0.0, 1.0)
        self.reward_phase_weights[:, agent_index] = updated_weight
        return updated_weight

    def reward_occt_platoon(self, reward_details, agent_index, ref_points_vecs, move_vec):
        hinge_status = self.ref_paths_agent_related.hinge_status[:,agent_index]
        hinge_weight = self._update_reward_phase_weight(agent_index, hinge_status)
        platoon_weight = 1.0 - hinge_weight
        self.hinge_active_steps[:, agent_index] = torch.where(
            hinge_status,
            self.hinge_active_steps[:, agent_index] + 1,
            torch.zeros_like(self.hinge_active_steps[:, agent_index]),
        )
        platoon_s_errors = self.observations.self_platoon_error_space.get_latest(n=1)[:, agent_index, 0]
        reward_platoon_space = (
            self.clamp_error_reward(self.rewards.reward_platoon_space, platoon_s_errors)
            * platoon_weight
        )
        reward_details["reward_platoon_space"][:, agent_index] = reward_platoon_space

        agent_desire_pos = self.get_lookahead_agent_pos(agent_index)
        hinge_desire_pos = self.get_target_hinge_pos(agent_index)
        hinge_pos_errors = torch.norm(hinge_desire_pos - agent_desire_pos, dim=-1)
        reward_hinge_space = (
            self.clamp_error_reward(self.rewards.reward_hinge_space, hinge_pos_errors, offset = 2.5, max = 2.5,)
            * hinge_weight
        )
        reward_details["reward_hinge_space"][:, agent_index] = reward_hinge_space
        
        platoon_vel_error = torch.max(
            torch.abs(self.observations.platoon_error_vel[:, agent_index]), dim=-1
        )[0]
        reward_platoon_vel = (
            self.clamp_error_reward(self.rewards.reward_platoon_vel, platoon_vel_error)
            * platoon_weight
        )
        reward_details["reward_platoon_vel"][:, agent_index] = reward_platoon_vel

        hinge_vel_error = torch.max(
            torch.abs(self.observations.hinge_error_vel[:, agent_index]), dim=-1
        )[0]
        reward_hinge_vel = (
            self.clamp_error_reward(self.rewards.reward_hinge_vel, hinge_vel_error)
            * hinge_weight
        )
        reward_details["reward_hinge_vel"][:, agent_index] = reward_hinge_vel

        weighted_hinge_ref_path_error = torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)
        for idx, ratio in enumerate((0.2, 0.8)):
            agent_desire_pos = self.get_lookahead_agent_pos(agent_index, idx)
            hinge_desire_pos = self.get_target_hinge_pos(agent_index, idx)
            weighted_hinge_ref_path_error += ratio * torch.norm(hinge_desire_pos - agent_desire_pos, dim=-1)
        reward_hinge_ref = (
            self.clamp_error_reward(
                self.rewards.reward_hinge_ref, weighted_hinge_ref_path_error
                ,offset = 2.5, max = 2.5,
            )
            * hinge_weight
        )
        reward_details["reward_hinge_ref"][:, agent_index] = reward_hinge_ref

        weighted_platoon_ref_path_error = torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)
        for idx, ratio in enumerate((0.2, 0.8)):
            weighted_platoon_ref_path_error += ratio * self.distances.lookahead_pts[:, agent_index, idx]
        reward_platoon_ref = (
            self.clamp_error_reward(
                self.rewards.reward_platoon_ref, weighted_platoon_ref_path_error
            )
            * platoon_weight
        )
        reward_details["reward_platoon_ref"][:, agent_index] = reward_platoon_ref

        reward_platoon_heading = self._reward_agent_heading(
                ref_points_vecs=ref_points_vecs,
                move_vec=move_vec,
            ) * platoon_weight
        reward_details["reward_platoon_heading"][:, agent_index] = reward_platoon_heading

        approach_mask = hinge_weight # & (hinge_pos_errors>0.5)
        hinge_approach_progress = self.get_hinge_approach_progress(agent_index,is_simple=True)
        reward_approach_hinge = torch.clamp(
            self.reward_approach_hinge * hinge_approach_progress,
            min=-1,
            max=1,
        ) * approach_mask
        reward_details["reward_approach_hinge"][:, agent_index] = reward_approach_hinge
        
        current_agent_hinge_status = self.ref_paths_agent_related.agent_hinge_status.get_latest(n=1)[:, agent_index]
        last_agent_hinge_status = self.ref_paths_agent_related.agent_hinge_status.get_latest(n=2)[:, agent_index]
        hinge_once = current_agent_hinge_status & ~last_agent_hinge_status
        
        reward_hinge = self.rewards.reward_hinge * (hinge_once & hinge_status).to(torch.float32)
            
        reward_details["reward_hinge"][:, agent_index] = reward_hinge
        reward_details["reward_all_hinge"][:, agent_index] = (
            self.reward_all_hinge * self.get_all_hinged_reward_mask().to(torch.float32)
        )
        hinge_steps = self.hinge_active_steps[:, agent_index]
        #print(f"hinge_steps-[{hinge_steps.min().cpu(),hinge_steps.mean().cpu(),hinge_steps.max().cpu()}]")
        #hinge_time_cost = -torch.clamp(hinge_steps/self.normalizers.hinge_step,max=2)
        reward_details["penalty_hinge_time_cost"][:, agent_index] = (
            self.penalties.hinge_time_cost
            * ((hinge_status & ~current_agent_hinge_status).to(torch.float32))
            * hinge_weight
        )
        #reward_goal = self._reward_conditional_goal()
        reward_details["reward_goal"][:, agent_index] = 0
        return reward_details
    
    def reward(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        if agent_index == 0:
            self.env_current_step += 1
        # Initialize
        reward_details=self.reward_details
        self.rew[:] = 0
        # we exclude the front vehicle and end vehicle
        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        t0=time.time()
        self.update_state_before_rewarding(agent, agent_index)
        t1=time.time()
        #print(f"update_state_before_rewarding, agent_index: {agent_index}, time: {t1-t0:.6f}s")
    
        if self.task_class == TaskClass.OCCT_PLATOON and agent_index in self.TRACTOR_SLICE:
            # copy reward from follower to self.TRACTOR_SLICE agent
            for r in reward_details.keys():
                mean_reward = torch.mean(reward_details[r][:,self.HINGE_FIRST_INDEX+1:self.HINGE_LAST_INDEX],dim=-1)
                reward_details[r][:,0] = mean_reward
                reward_details[r][:,self.n_agents-1] = mean_reward
            self.update_state_after_rewarding(agent_index)
            return self.rew
        # [penalty] close to other agents
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :],
            x0=self.thresholds.near_other_agents_low,
            x1=self.thresholds.near_other_agents_high,
        )
        penalty_near_other_agents = (
            torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        )
        reward_details["penalty_near_other_agents"][:,agent_index] = penalty_near_other_agents


        # [penalty] changing steering too quick
        steering_current = self.observations.past_action_steering.get_latest(n=1)[
            :, agent_index
        ]
        steering_past = self.observations.past_action_steering.get_latest(n=2)[
            :, agent_index
        ]
        steering_change = torch.clamp(
            (steering_current - steering_past).abs() * self.normalizers.action_steering
            - self.thresholds.change_steering,  # Not forget to denormalize
            min=0,
        )
        if self.observations.past_action_steering.valid_size==self.observations.n_stored_steps:
            penalty_change_steering = (
                (steering_change/torch.deg2rad(torch.tensor(3,device=self.device)))**2 * self.penalties.change_steering
            )
            penalty_change_steering = torch.clamp(penalty_change_steering,min=-5,max=0)
        else:
            penalty_change_steering = 0.0
        reward_details["penalty_change_steering"][:,agent_index] = penalty_change_steering


        # [penalty] changing acc too quick
        acc_current = self.observations.past_action_acc.get_latest(n=1)[
            :, agent_index
        ]
        acc_past = self.observations.past_action_acc.get_latest(n=2)[
            :, agent_index
        ]

        acc_change = torch.clamp(
            (acc_current - acc_past).abs() * self.normalizers.action_acc
            - self.thresholds.change_acc,  # Not forget to denormalize
            min=0,
        )
        acc_nor=0.1
        if self.observations.past_action_acc.valid_size==self.observations.n_stored_steps:
            penalty_change_acc = (
                (acc_change/acc_nor)**2 * self.penalties.change_acc
            )
            penalty_change_acc = torch.clamp(penalty_change_acc,min=-5,max=0)
        else:
            penalty_change_acc = 0.0
        reward_details["penalty_change_acc"][:,agent_index] = penalty_change_acc

        # [penalty] colliding with other agents
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]
        penalty_collide_with_agents = (
            is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        )
        reward_details["penalty_collide_with_agents"][:,agent_index] = penalty_collide_with_agents

        # [penalty] colliding with lanelet boundaries
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_outside_boundaries = (
            is_collide_with_lanelets * self.penalties.collide_with_boundaries
        )
        reward_details["penalty_outside_boundaries"][:,agent_index] = penalty_outside_boundaries

        # [penalty] close to lanelet boundaries
        current_lane_width = torch.linalg.norm(self.ref_paths_agent_related.nearing_points_left_boundary[:, agent_index, 1] -\
              self.ref_paths_agent_related.nearing_points_right_boundary[:, agent_index, 1],dim=-1)
        penalty_near_boundary = (
            torch.max(exponential_decreasing_fcn(
                x=self.distances.boundaries[:, agent_index]/current_lane_width,
                x0=self.thresholds.near_boundary_low,
                x1=self.thresholds.near_boundary_high,
            ),is_collide_with_lanelets.float())
            * self.penalties.near_boundary
        )
        reward_details["penalty_near_boundary"][:,agent_index] = penalty_near_boundary

        ref_points_vecs = self.ref_paths_agent_related.short_term[:, agent_index, 1:, 0:2] -\
              self.ref_paths_agent_related.short_term[:, agent_index, :-1, 0:2] 
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(
            -1
        )
        backward_penalty = (
            torch.where(v_proj <= 0, 1, 0)
            * self.penalties.backward
        )
        reward_details["penalty_backward"][:,agent_index] = backward_penalty

        # [reward] forward movement
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(
            1
        )  # Vector of the current movement

        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(
            move_projected, self.rewards.weighting_ref_directions
        )  # Put more weights on nearing reference points
        # [reward] hinge tracking
        reward_progress = (
            move_projected_weighted
            / (agent.max_speed * self.world.dt)
            * self.rewards.reward_progress
        )
        reward_details["reward_progress"][:,agent_index] = reward_progress

        # [reward] high velocity
        reward_vel = v_proj / agent.max_speed * self.rewards.reward_vel
        reward_details["reward_vel"][:,agent_index] = reward_vel

        if self.task_class == TaskClass.SIMPLE_PLATOON:
            reward_details = self.reward_simple_platoon(
                reward_details=reward_details,
                agent_index=agent_index,
                ref_points_vecs=ref_points_vecs,
                move_vec=move_vec,
            )
        elif self.task_class == TaskClass.OCCT_PLATOON:
            reward_details = self.reward_occt_platoon(
                reward_details=reward_details,
                agent_index=agent_index,
                ref_points_vecs=ref_points_vecs,
                move_vec=move_vec,
            )
        else:
            raise ValueError(f"Unsupported task class: {self.task_class}")
        t2=time.time()
        # hinge之后就屏蔽奖励
        if self.task_class==TaskClass.OCCT_PLATOON:
            last_hinge_status = self.ref_paths_agent_related.agent_hinge_status.get_latest(n=2)[:, agent_index]
            current_hinge_status = self.ref_paths_agent_related.agent_hinge_status.get_latest(n=1)[:, agent_index]
            agent_is_fixed = last_hinge_status & current_hinge_status
            self.rew = self.rew * ~agent_is_fixed
            reward_details["reward_hinged_hold"][:, agent_index] = (
                self.reward_hinged_hold * agent_is_fixed.to(torch.float32)
            )
            for r in reward_details.keys():
                if r in {"reward_all_hinge", "reward_hinged_hold"}:
                    continue
                reward_details[r][:,agent_index] = reward_details[r][:,agent_index] * ~agent_is_fixed
        #print(f"reward calc, agent_index: {agent_index}, time: {t2-t1:.6f}s")
        # [update] previous positions and short-term reference paths
        self.update_state_after_rewarding(agent_index)
        t3=time.time()
        #print(f"update_state_after_rewarding, agent_index: {agent_index}, time: {t3-t2:.6f}s")
        for r in reward_details.keys():
            if r!="reward_total":
                self.rew+=reward_details[r][:,agent_index]
        reward_details["reward_total"][:,agent_index] = self.rew
        if (
            self.task_class == TaskClass.OCCT_PLATOON
            and agent_index == (self.HINGE_LAST_INDEX - 1)
        ):
            self.all_hinged_reward_granted |= self.get_all_followers_hinged()
        self.reward_update_time += t3-t0
        #print(f"reward_update_time_total: {self.reward_update_time:.6f}s")
        return self.rew
    

    def update_state_before_rewarding(self, agent, agent_index):
        """Update some states (such as mutual distances between agents, vertices of each agent, and
        collision matrices) that will be used before rewarding agents.
        """
        if agent_index == 0:  # Avoid repeated computations
            # Timer
            self.timer.step_begin = (
                time.time()
            )  # Set to the current time as the begin of the current time step
            self.timer.step += 1  # Increment step by 1
            assert torch.isnan(agent.state.pos).any() == False, f"agent {agent_index} pos is nan"
            # Update distances between agents
            self.distances.agents = get_distances_between_agents(
                self=self, is_set_diagonal=True
            )
            self.distances.agents_frenet = get_frenet_distances_between_agents(self.observations.agent_s)
            self.collisions.with_agents[:] = False  # Reset
            self.collisions.with_lanelets[:] = False  # Reset
            self.collisions.with_exit_segments[:] = False  # Reset
            
            for a_i in range(self.n_agents):
                self.vertices[:, a_i] = get_rectangle_vertices(
                    center=self.world.agents[a_i].state.pos,
                    yaw=self.world.agents[a_i].state.rot,
                    width=self.world.agents[a_i].shape.width,
                    length=self.world.agents[a_i].shape.length,
                    is_close_shape=True,
                )
                # Update the collision matrices
                for a_j in range(a_i + 1, self.n_agents):
                    # Check for collisions between agents using the interX function
                    collision_batch_index = interX(
                        self.vertices[:, a_i], self.vertices[:, a_j], False
                    )
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index), a_i, a_j
                    ] = True
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index), a_j, a_i
                    ] = True
                
                # Check for collisions with entry segments
                if not self.is_loop:
                    self.collisions.with_exit_segments[:, a_i] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.exit[:, a_i],
                        is_return_points=False,
                    )
                # ignore the front and rear vehicle collision
                if (self.task_class == TaskClass.OCCT_PLATOON and a_i not in self.TRACTOR_SLICE) or\
                    self.task_class == TaskClass.SIMPLE_PLATOON:
                    # Check for collisions between agents and lanelet boundaries
                    collision_with_left_boundary = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.left_boundary[:, a_i],
                        is_return_points=False,
                    ).to(self.device)  # [batch_dim]
                    collision_with_right_boundary = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                        is_return_points=False,
                    ).to(self.device)  # [batch_dim]
                    is_left_outside_boundary = is_point_left_of_polyline(
                        point=self.world.agents[a_i].state.pos,
                        polyline=self.ref_paths_agent_related.nearing_points_left_boundary[:, a_i],
                    ).to(self.device)
                    is_right_outside_boundary = ~is_point_left_of_polyline(
                        point=self.world.agents[a_i].state.pos,
                        polyline=self.ref_paths_agent_related.nearing_points_right_boundary[:, a_i],
                    ).to(self.device)
                    self.collisions.with_lanelets[
                        ((collision_with_left_boundary | is_left_outside_boundary) | \
                        (collision_with_right_boundary | is_right_outside_boundary)), a_i
                    ] = True
                assert self.use_center_frenet_ref, "use_center_frenet_ref must be True"
                # agent short term reference path and nearing points on boundaries
                self.ref_paths_agent_related.short_term[:, a_i] = \
                    get_short_term_reference_path_by_s(
                        self.road,
                        self.observations.agent_s[:, a_i],
                        n_points_to_return=self.n_points_short_term,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        return_ref_v=True,
                        line='center',
                    )
                if self.task_class==TaskClass.SIMPLE_PLATOON and a_i!=0:
                    self.ref_paths_agent_related.short_term[:, a_i,:,-1] = self.ref_paths_agent_related.short_term[:, 0,:,-1]

                # Get nearing points on boundaries
                if self.use_boundary_frenet_ref:
                    self.ref_paths_agent_related.nearing_points_left_boundary[:, a_i] = \
                            get_short_term_reference_path_by_s(
                        self.road,
                        self.observations.agent_s[:, a_i]+self.boundary_offset,
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        return_ref_v=False,
                        line='left',
                        )
                    self.ref_paths_agent_related.nearing_points_right_boundary[:, a_i] = \
                        get_short_term_reference_path_by_s(
                        self.road,
                        self.observations.agent_s[:, a_i]+self.boundary_offset,
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        return_ref_v=False,
                        line='right',
                    )
                else:
                    (
                        self.ref_paths_agent_related.nearing_points_left_boundary[
                            :, a_i
                        ],
                        _,
                    ) = get_short_term_reference_path_simple(
                        polyline=self.ref_paths_agent_related.left_boundary[:, a_i],
                        index_closest_point=self.distances.closest_point_on_left_b[
                            :, a_i
                        ],
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        n_points_shift=1,
                    )
                    (
                        self.ref_paths_agent_related.nearing_points_right_boundary[
                            :, a_i
                        ],
                        _,
                    ) = get_short_term_reference_path_simple(
                        polyline=self.ref_paths_agent_related.right_boundary[:, a_i],
                        index_closest_point=self.distances.closest_point_on_right_b[
                            :, a_i
                        ],
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        n_points_shift=1,
                    )
        # Distance from the center of gravity (CG) of the agent to its reference path
        (
            self.distances.ref_paths[:, agent_index],
            self.distances.closest_point_on_ref_path[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos,
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=None
        )
        # Distances from CG to left boundary
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=None
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - (
            agent.shape.width / 2
        )
        # Distances from CG to right boundary
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=None
        )
        self.distances.right_boundaries[:, agent_index, 0] = center_2_right_b - (
            agent.shape.width / 2
        )
        # Distances from the four vertices of the agent to its left and right lanelet boundary
        for c_i in range(4):
            (
                self.distances.left_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=None
            )
            (
                self.distances.right_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=None
            )
        # Distance from agent to its left/right lanelet boundary is defined as the minimum distance among five distances (four vertices, CG)
        self.distances.boundaries[:, agent_index], _ = torch.min(
            torch.hstack(
                (
                    self.distances.left_boundaries[:, agent_index],
                    self.distances.right_boundaries[:, agent_index],
                )
            ),
            dim=-1,
        )
        for idx in range(self.agent_lookahead_idx):
            if idx==0:
                lookahead_pts = agent.state.pos
            else:
                # dist_travelled=torch.ones_like(agent.action.u[:, 1])*self.sample_interval*idx
                # lookahead_pts = self.compute_lookahead_kinematics(agent, agent.action.u[:, 1], dist_travelled)
                lookahead_pts = agent.state.pos + idx*self.sample_interval * torch.hstack([torch.cos(agent.state.rot), torch.sin(agent.state.rot)])
            self.distances.lookahead_pts[:, agent_index, idx] = \
                torch.linalg.norm(self.ref_paths_agent_related.short_term[:, agent_index, idx, :2] - lookahead_pts, dim=-1)
    def compute_lookahead_kinematics(self, agent, delta, dist_travelled):
        """
        使用匀速圆弧模型预测车辆在 dt 时间后的位置
        delta: agent.action[:, 1], 弧度单位
        dt: 预测的时间跨度 (sample_dt * idx)
        """
        theta = agent.state.rot.squeeze(-1)
        L = agent.dynamics.l_f + agent.dynamics.l_r
        kappa = torch.tan(delta) / L
        delta_theta = dist_travelled * kappa
        is_straight = torch.abs(kappa) < 1e-4
        inv_kappa = 1.0 / (kappa + 1e-8)
        lookahead_pos_curve = agent.state.pos + inv_kappa.unsqueeze(-1) * torch.stack([
            torch.sin(theta + delta_theta) - torch.sin(theta),
            -(torch.cos(theta + delta_theta) - torch.cos(theta))
        ], dim=-1)
        lookahead_pos_straight = agent.state.pos + dist_travelled.unsqueeze(-1) * torch.stack([
            torch.cos(theta), torch.sin(theta)
        ], dim=-1)
        lookahead_pts = torch.where(is_straight.unsqueeze(-1), lookahead_pos_straight, lookahead_pos_curve)
        return lookahead_pts
    def update_state_after_rewarding(self, agent_index):
        """Update some states (such as previous positions and short-term reference paths) after rewarding agents."""
        if agent_index == (self.n_agents - 1):  # Avoid repeated updating
            state_add = torch.cat(
                (
                    torch.stack([a.state.pos for a in self.world.agents], dim=1),
                    torch.stack([a.state.rot for a in self.world.agents], dim=1),
                    torch.stack([a.state.vel for a in self.world.agents], dim=1),
                ),
                dim=-1,
            )
            self.state_buffer.add(state_add)
            self.failure_replay_snapshot_buffer.add(
                self._build_failure_replay_buffer_state()
            )
    def _get_done_status(self) -> Dict[str, Tensor]:
        """Compute environment-level done reasons and success/failure labels."""
        is_collision_with_agents_env = self.collisions.with_agents.view(
            self.world.batch_dim, -1
        ).any(dim=-1)
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        is_collision_with_exit_segments = self.collisions.with_exit_segments.any(dim=-1)
        is_agent_all_hinged = self.get_all_followers_hinged()
        is_all_hinged_done_disabled = self._get_all_hinged_done_disabled_mask()
        is_done_by_all_hinged = is_agent_all_hinged & ~is_all_hinged_done_disabled
        is_done = (
            #is_collision_with_agents_env
             is_collision_with_exit_segments
            #| is_collision_with_lanelets
            | is_done_by_all_hinged
        )
        is_success = is_done_by_all_hinged & ~(
            is_collision_with_agents_env
            | is_collision_with_exit_segments
            | is_collision_with_lanelets
        )
        is_failure = is_done & ~is_success
        return {
            "is_done": is_done,
            "is_success": is_success,
            "is_failure": is_failure,
            "is_collision_with_agents_env": is_collision_with_agents_env,
            "is_collision_with_lanelets": is_collision_with_lanelets,
            "is_collision_with_exit_segments": is_collision_with_exit_segments,
            "is_agent_all_hinged": is_agent_all_hinged,
        }
    def done(self):
        """
        This function computes the done flag for each env in a vectorized way.
        """
        done_status = self._get_done_status()
        is_done = done_status["is_done"]
        self._record_failure_curriculum_events(done_status)
        self.success_count += done_status["is_success"].float()
        self.failure_count += done_status["is_failure"].float()
        self.env_total_step[is_done] = self.env_current_step[is_done]
        self.env_current_step[is_done] = 0
        return is_done
    def get_lookahead_agent_pos(self, agent_index, lookahead_idx = 0):
        """
        Get the current agent position of the agent.
        """
        current_pos = self.world.agents[agent_index].state.pos
        theta = self.world.agents[agent_index].state.rot
        lookahead_pts = current_pos + lookahead_idx*self.sample_interval * torch.hstack([torch.cos(theta), torch.sin(theta)])
        return lookahead_pts
    def get_target_hinge_pos(self, agent_index, lookahead_idx = 0):
        """
        Get the current hinge position of the agent.
        """
        # leader_hinge_pos = self.world.agents[self.TRACTOR_SLICE[0]].state.pos
        # latter_hinge_pos = self.world.agents[self.TRACTOR_SLICE[-1]].state.pos
        # current_hinge_pos = leader_hinge_pos + (latter_hinge_pos - leader_hinge_pos) * agent_index / (self.n_agents - 1)
        current_hinge_pos = self.ref_paths_agent_related.hinge_short_term[:, agent_index, lookahead_idx, :2]
        return current_hinge_pos
    def get_target_hinge_vel(self, agent_index, lookahead_idx = 0):
        """
        Get the current hinge velocity of the agent.
        """
        return self.ref_paths_agent_related.hinge_short_term[:, agent_index, lookahead_idx, 2:4]

    def _compute_agent_command_acceleration(self, agent_index: int) -> Tensor:
        if self.observations.past_action_acc.valid_size == 0:
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)
        return (
            self.observations.past_action_acc.get_latest(n=1)[:, agent_index]
            * self.normalizers.action_acc
        ).to(torch.float32)

    def _compute_agent_command_jerk(self, agent_index: int) -> Tensor:
        if self.observations.past_action_acc.valid_size < 2:
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)
        acc_current = self.observations.past_action_acc.get_latest(n=1)[:, agent_index]
        acc_previous = self.observations.past_action_acc.get_latest(n=2)[:, agent_index]
        return (
            (acc_current - acc_previous) * self.normalizers.action_acc / self.dt
        ).to(torch.float32)

    def _compute_agent_steering_rate_deg(self, agent_index: int) -> Tensor:
        if self.observations.past_action_steering.valid_size < 2:
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)
        steering_current = (
            self.observations.past_action_steering.get_latest(n=1)[:, agent_index]
            * self.normalizers.action_steering
        )
        steering_previous = (
            self.observations.past_action_steering.get_latest(n=2)[:, agent_index]
            * self.normalizers.action_steering
        )
        delta = steering_current - steering_previous
        return (delta * (180.0 / torch.pi) / self.dt).to(torch.float32)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        agent_index = self.world.agents.index(agent)  # Index of the current agent

        is_action_empty = agent.action.u is None
        done_status = self._get_done_status()
        total_finished = self.success_count + self.failure_count
        running_success_rate = self.success_count / total_finished.clamp_min(1.0)

        is_collision_with_agents = self.collisions.with_agents[:, agent_index].any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        agent_error_space=self.observations.self_platoon_error_space.get_latest()[:, agent_index,:]
        command_acceleration = self._compute_agent_command_acceleration(agent_index)
        command_jerk = self._compute_agent_command_jerk(agent_index)
        steering_rate_deg = self._compute_agent_steering_rate_deg(agent_index)
        agent_reward_details = {}
        for reward_name, reward_tensor in self.reward_details.items():
            # reward_tensor 维度：(batch_dim, n_agents) → 提取当前智能体的列
            # 结果维度：(batch_dim,)，与info中其他字段（如pos/vel）维度对齐
            agent_reward_details[reward_name] = reward_tensor[:, agent_index]
        hinge_dict = {}
        if self.task_class == TaskClass.OCCT_PLATOON:
            hinge_pos = self.get_target_hinge_pos(agent_index)
            hinge_vel = self.ref_paths_agent_related.hinge_short_term[:,agent_index,0,2:4]
            hinge_status = self.ref_paths_agent_related.hinge_status[:,agent_index]
            agent_hinge_status = self.ref_paths_agent_related.agent_hinge_status.get_latest()[:,agent_index] # [B, n_agents]
            hinge_heading_vel_angle_diff_deg = (
                self.ref_paths_agent_related.hinge_heading_vel_angle_diff_deg[
                    :, agent_index
                ]
            )
            hinge_gate_angle_diff_deg = (
                self.ref_paths_agent_related.agent_heading_hinge_heading_angle_diff_deg[
                    :, agent_index
                ]
            )
            hinge_dis = torch.norm(hinge_pos - agent.state.pos, dim=-1)*hinge_status  # [batch_dim, n_points]
            hinged_followers_ratio = self.get_hinged_followers_ratio()*hinge_status
            hinge_steps = self.hinge_active_steps[:, agent_index]
            hinge_dict = {
                "hinge_pos": hinge_pos,
                "hinge_vel": hinge_vel,
                "hinge_status": hinge_status,
                "agent_hinge_status": agent_hinge_status,
                "hinge_heading_vel_angle_diff_deg": hinge_heading_vel_angle_diff_deg,
                "hinge_gate_angle_diff_deg": hinge_gate_angle_diff_deg,
                "hinge_dis": hinge_dis,
                "hinged_followers_ratio": hinged_followers_ratio,
                "hinge_steps": hinge_steps,
            }
        #print(f"agent_index: {agent_index}, hinge_status: {hinge_status}, hinge_dis: {hinge_dis}")
        info = {
            "pos": agent.state.pos,
            "s":self.observations.agent_s[:, agent_index],
            "rot": angle_eliminate_two_pi(agent.state.rot),
            "vel": agent.state.vel,
            "vel_norm": torch.norm(agent.state.vel, dim=-1),
            "act_acc": (
                (agent.action.u[:, 0])
                if (self.traditional_control == MethodClass.MARL and not is_action_empty)
                else self.logged_control_acc[:, agent_index]
            ),
            "act_steer": (
                (agent.action.u[:, 1])
                if (self.traditional_control == MethodClass.MARL and not is_action_empty)
                else self.logged_control_steer[:, agent_index]
            ),
            "command_acceleration": command_acceleration,
            "command_acceleration_abs": command_acceleration.abs(),
            "command_jerk": command_jerk,
            "command_jerk_abs": command_jerk.abs(),
            "steering_rate_deg": steering_rate_deg,
            "steering_rate_abs_deg": steering_rate_deg.abs(),
            "distance_ref": self.distances.ref_paths[:, agent_index],
            "distance_lookahead_pts": torch.mean(self.distances.lookahead_pts[:, agent_index], dim=-1),
            "distance_left_b": self.distances.left_boundaries[:, agent_index].min(
                dim=-1
            )[0],
            "distance_right_b": self.distances.right_boundaries[:, agent_index].min(
                dim=-1
            )[0],
            "is_collision_with_agents": is_collision_with_agents,
            "is_collision_with_lanelets": is_collision_with_lanelets,
            "mean_error_space": agent_error_space.mean(-1),
            "error_space": agent_error_space,
            "platoon_error_vel": self.observations.platoon_error_vel[:, agent_index],
            "hinge_error_vel": self.observations.hinge_error_vel[:, agent_index],
            "ref_vel": self.ref_paths_agent_related.short_term[:, agent_index, 0, 2],
            "episode_done": done_status["is_done"].float(),
            "episode_success": done_status["is_success"].float(),
            "episode_failure": done_status["is_failure"].float(),
            "episode_replay_source": self.current_episode_replay_source.float(),
            "episode_replay_entry_id": self.current_episode_replay_entry_id.to(torch.float32),
            "done_all_hinged": done_status["is_agent_all_hinged"].float(),
            "done_collision_with_agents": done_status["is_collision_with_agents_env"].float(),
            "done_collision_with_lanelets": done_status["is_collision_with_lanelets"].float(),
            "done_collision_with_exit_segments": done_status["is_collision_with_exit_segments"].float(),
            "scenario_success_count": self.success_count,
            "scenario_failure_count": self.failure_count,
            "running_scenario_success_rate": running_success_rate,
            "env_total_step": self.env_total_step,
            "road_batch_id": self.road.batch_id,
            **hinge_dict,
            **agent_reward_details,
            }
        
        return info

    def extra_render(self, env_index: int = 0):
        from vmas.simulator import rendering

        if self.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0  # Not sure how long should the simulation be paused at time step 0, so rather 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)

            self.timer.render_begin = time.time()  # Update
        # map rendering
        geoms = []
        map_geoms = self.extra_render_map(env_index)
        geoms.extend(map_geoms)
        extend_road_polygons = self.extra_render_extend_road(env_index)
        geoms.extend(extend_road_polygons)
        # target road rendering
        if hasattr(self, "road"):
            s_max_idx=self.road.get_s_max_idx(env_index)
            center_pts = self.road.get_road_center_pts()[env_index]  # [N,2]
            center_pts = center_pts[:s_max_idx+1]
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in center_pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.PURPLE.value, alpha=1.0)
            geom.set_linewidth(3.0)  # 设置左边界线宽度
            geoms.append(geom)
        
            left_pts = self.road.get_road_left_pts()[env_index]  # [N,2]
            left_pts = left_pts[:s_max_idx+1]
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in left_pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(1.0)  # 设置左边界线宽度
            geoms.append(geom)
        
            right_pts = self.road.get_road_right_pts()[env_index]  # [N,2]
            right_pts = right_pts[:s_max_idx+1]
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in right_pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(1.0)  # 设置右边界线宽度
            geoms.append(geom)
        # for x_val in range(-50, 51, 10):
        #     for y_val in range(-50, 51, 10):
        #         pos = [x_val, y_val]
        #         geom = rendering.TextLine(
        #             text=f"({x_val},{y_val})",  # 显示实际的坐标点文本
        #             x=2*pos[0] * self.resolution_factor + self.viewer_size[0]/2,
        #             y=2*pos[1] * self.resolution_factor + self.viewer_size[1]/2,
        #             font_size=10,  # 字体稍微调小一点，防止 121 个字体重叠
        #         )
        #         xform = rendering.Transform()
        #         geom.add_attr(xform)
        #         geoms.append(geom)
        
        # agent rendering
        pos_origin = self.world.agents[self.agent_index_focus].state.pos[env_index, :]
        last_state = self.state_buffer.get_latest(n=2)[env_index,:,:]
        for agent_i, ag in enumerate(self.world.agents):
            pos = ag.state.pos[env_index].detach().cpu().tolist()
            target_hinge_idx = agent_i if self.ref_paths_agent_related.hinge_status[env_index,agent_i] else -1
            v = torch.linalg.norm(ag.state.vel[env_index]).detach().cpu()
            action = ag.action.u[env_index].detach().cpu().tolist() if ag.action.u is not None else [0,0,0.0]
            last_v = torch.linalg.norm(last_state[agent_i,3:5]).detach().cpu()
            acc = (v-last_v)/self.dt
            #acc,steering = action[0],action[1]
            # text info render
            # [x,y,v,a]
            agent_desire_pos = self.get_lookahead_agent_pos(agent_i)
            hinge_desire_pos = self.get_target_hinge_pos(agent_i)
            desire_distance = torch.norm(hinge_desire_pos - agent_desire_pos, dim=-1)
            desire_distance = desire_distance.detach().cpu().tolist()[env_index]
            space_errors = torch.abs(self.observations.self_platoon_error_space.get_latest(n=1)[env_index, agent_i, :]).detach().cpu()
            front_space_errors = space_errors[0]
            rear_space_errors = space_errors[1]
            geom = rendering.TextLine(
                #text=f"a{agent_i}->h{target_hinge_idx}:[{pos[0]:.1f},{pos[1]:.1f},{v:.1f},{acc:.1f},{space_errors:.1f}]",
                text=f"a{agent_i}->h{target_hinge_idx}:[v:{v:.1f},a:{acc:.1f}]",
                #text=f"a{agent_i} to h{target_hinge_idx}",
                x=4*(pos[0] - pos_origin[0]) * self.resolution_factor + self.viewer_size[0]/2,
                y=4*(pos[1] - pos_origin[1]) * self.resolution_factor + self.viewer_size[1]/2+2.2*5.2*self.resolution_factor,
                font_size=int(2*self.resolution_factor),
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
            geom = rendering.TextLine(
                #text=f"a{agent_i}->h{target_hinge_idx}:[{pos[0]:.1f},{pos[1]:.1f},{v:.1f},{acc:.1f},{space_errors:.1f}]",
                text=f"[se:({front_space_errors:.1f},{rear_space_errors:.1f}),he:{desire_distance:.2f}]",
                #text=f"a{agent_i} to h{target_hinge_idx}",
                x=4*(pos[0] - pos_origin[0]) * self.resolution_factor + self.viewer_size[0]/2,
                y=4*(pos[1] - pos_origin[1]) * self.resolution_factor + self.viewer_size[1]/2+2.2*4*self.resolution_factor,
                font_size=int(2*self.resolution_factor),
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
            # TODO: steering render

            # agent_hinge_status render
            agent_hinge_status = self.ref_paths_agent_related.agent_hinge_status.get_latest()[env_index, agent_i]
            if agent_hinge_status or agent_i in self.TRACTOR_SLICE:
                dot = rendering.make_circle(radius=3, filled=False)
                xf = rendering.Transform()
                dot.add_attr(xf)
                xf.set_translation(float(pos[0]), float(pos[1]))
                dot.set_color(*Color.RED.value)  # 黑点
                geoms.append(dot)
            if hasattr(self, "ref_paths_agent_related"):
                if hasattr(self.ref_paths_agent_related, "short_term"):
                    short_term_path = self.ref_paths_agent_related.short_term[env_index, agent_i]
                    geom = rendering.PolyLine(
                        v=[(float(p[0]), float(p[1])) for p in short_term_path.detach().cpu().tolist()],
                        close=False
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(*self.world.agents[agent_i].color)
                    geoms.append(geom)
                    for p in short_term_path:
                        circle = rendering.make_circle(radius=0.2, filled=True)
                        xform = rendering.Transform()
                        circle.add_attr(xform)
                        xform.set_translation(float(p[0]), float(p[1]))
                        circle.set_color(*self.world.agents[agent_i].color)
                        geoms.append(circle)
            if (
                self.traditional_control == MethodClass.MPPI
                and self.enable_mppi_debug_render
                and self.simple_mppi is not None
                and agent_i == self.agent_index_focus
            ):
                mppi_debug = self.simple_mppi.last_debug.get(agent_i)
                if mppi_debug is not None:
                    ref_points = mppi_debug["ref_points"]
                    sampled_trajs = mppi_debug["sampled_trajs"]
                    optimal_traj = mppi_debug["optimal_traj"]

                    for traj in sampled_trajs:
                        geom = rendering.PolyLine(
                            v=[
                                (float(p[0]), float(p[1]))
                                for p in traj[:, :2].detach().cpu().tolist()
                            ],
                            close=False,
                        )
                        xform = rendering.Transform()
                        geom.add_attr(xform)
                        geom.set_color(0.10, 0.70, 0.95, alpha=0.12)
                        geom.set_linewidth(1.0)
                        geoms.append(geom)

                    geom = rendering.PolyLine(
                        v=[
                            (float(p[0]), float(p[1]))
                            for p in ref_points.detach().cpu().tolist()
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(0.95, 0.78, 0.15, alpha=1.0)
                    geom.set_linewidth(3.0)
                    geoms.append(geom)
                    for p in ref_points:
                        circle = rendering.make_circle(radius=0.25, filled=True)
                        xform = rendering.Transform()
                        circle.add_attr(xform)
                        xform.set_translation(float(p[0]), float(p[1]))
                        circle.set_color(0.95, 0.78, 0.15)
                        geoms.append(circle)

                    geom = rendering.PolyLine(
                        v=[
                            (float(p[0]), float(p[1]))
                            for p in optimal_traj[:, :2].detach().cpu().tolist()
                        ],
                        close=False,
                    )
                    xform = rendering.Transform()
                    geom.add_attr(xform)
                    geom.set_color(0.15, 0.95, 0.35, alpha=1.0)
                    geom.set_linewidth(3.0)
                    geoms.append(geom)
                    for p in optimal_traj[:, :2]:
                        circle = rendering.make_circle(radius=0.18, filled=False)
                        xform = rendering.Transform()
                        circle.add_attr(xform)
                        xform.set_translation(float(p[0]), float(p[1]))
                        circle.set_color(0.15, 0.95, 0.35)
                        geoms.append(circle)
            if hasattr(self, "ref_paths_agent_related"):
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_left_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_linewidth(2)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)

                # Left boundary
                for i_p in self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.2, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)

                # Right boundary
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_right_boundary[
                        env_index, agent_i
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_linewidth(2)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)

                for i_p in self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_index, agent_i
                ]:
                    circle = rendering.make_circle(radius=0.2, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)
        # hinge short term rendering
        if self.task_class==TaskClass.OCCT_PLATOON and \
            hasattr(self, "ref_paths_agent_related") and\
            hasattr(self.ref_paths_agent_related, "hinge_short_term"):
            for hinge_i in range(self.n_hinges):
                hinge_short_term = self.ref_paths_agent_related.hinge_short_term[env_index, hinge_i]
                hinge_status = self.ref_paths_agent_related.hinge_status[env_index, hinge_i]
                pos = hinge_short_term[0, :2].detach().cpu().tolist()
                geom = rendering.TextLine(
                    #text=f"h{hinge_i}_✔" if hinge_status else f"h{hinge_i}_✘",
                    text=f"h{hinge_i}-V" if hinge_status else f"h{hinge_i}-X",
                    x=4*(pos[0] - pos_origin[0]) * self.resolution_factor + self.viewer_size[0]/2,
                    y=4*(pos[1] - pos_origin[1]) * self.resolution_factor + self.viewer_size[1]/2-2.2*self.resolution_factor,
                    font_size=int(2*self.resolution_factor),
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geoms.append(geom)
                if hinge_i in self.TRACTOR_SLICE:
                    continue
                hinge_status = hinge_short_term[...,-1]
                geom = rendering.PolyLine(
                    v=[(float(p[0]), float(p[1])) for p in hinge_short_term.detach().cpu().tolist()],
                    close=False
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_color(*Color.BLACK.value)
                geoms.append(geom)
                for hinge_status in hinge_short_term:
                    p = hinge_status[:2]
                    status = hinge_status[-1]
                    diamond_poly = [(0, 0.2), (0.2, 0), (0, -0.2), (-0.2, 0)]
                    if status > 0.0:
                        circle = rendering.make_polygon(diamond_poly, filled=True)
                    else:
                        circle = rendering.make_polygon(diamond_poly, filled=False)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(float(p[0]), float(p[1]))
                    circle.set_color(*Color.BLACK.value, 0.6)  # 半透明
                    geoms.append(circle)        
        # cargo rendering
        if self.task_class==TaskClass.OCCT_PLATOON:
            pf = self.tractor_front.state.pos[env_index].detach().cpu()
            pr = self.tractor_rear.state.pos[env_index].detach().cpu()
            rod = pf - pr
            rod_len = torch.linalg.norm(rod).item() + 1e-9
            t_hat = (rod / rod_len)            # 切向
            n_hat = torch.tensor([-t_hat[1], t_hat[0]])  # 法向（左法向）
            hinge_gap = rod_len / (self.n_agents-1)
            cargo_half_w = self.cargo_half_width
            edge_width = 1.7
            rear_left   = (pr + n_hat * cargo_half_w - t_hat * edge_width).tolist()
            rear_right  = (pr - n_hat * cargo_half_w - t_hat * edge_width).tolist()
            front_left  = (pf + n_hat * cargo_half_w + t_hat * edge_width).tolist()
            front_right = (pf - n_hat * cargo_half_w + t_hat * edge_width).tolist()
            cargo_outline = rendering.PolyLine(
                v=[tuple(rear_left), tuple(front_left), tuple(front_right), tuple(rear_right)],
                close=True
            )
            cargo_outline.set_color(*Color.BLACK.value, alpha=0.9)
            geoms.append(cargo_outline)

            # cargo compartment lines (横向分割线)
            for i in range(1, self.n_agents - 1):
                # 计算第 i 个 hinge 点的位置（从后往前）
                hinge_pos = pr + t_hat * (hinge_gap * i)
                # 计算该 hinge 点处的左右边界
                left_pt = hinge_pos + n_hat * cargo_half_w
                right_pt = hinge_pos - n_hat * cargo_half_w
                # 画分割线
                compartment_line = rendering.PolyLine(
                    v=[tuple(left_pt.tolist()), tuple(right_pt.tolist())],
                    close=False
                )
                compartment_line.set_color(*Color.BLACK.value, alpha=0.9)
                geoms.append(compartment_line)
        return geoms
    def extra_render_extend_road(self, env_index: int = 0):
        """
        绘制扩展的道路区域（灰色填充）
        """
        left_pts1 = self.road.get_pts(torch.tensor(0,device=self.device),env_index,"left")
        left_pts2 = self.road.get_pts(torch.tensor(self.rod_len+1,device=self.device),env_index,"left")
        right_pts1 = self.road.get_pts(torch.tensor(0,device=self.device),env_index,"right")
        right_pts2 = self.road.get_pts(torch.tensor(self.rod_len+1,device=self.device),env_index,"right")
        extend_road_pts = [left_pts1,left_pts2,right_pts2,right_pts1]
        extend_road_polygon1 = rendering.make_polygon(extend_road_pts, draw_border=False)
        extend_road_polygon1.set_color(0.7, 0.7, 0.7, alpha=1.0)  # 灰色填充
        s_max=self.road.get_s_max()[env_index]
        left_pts3 = self.road.get_pts(s_max-self.rod_len-1,env_index,"left")
        left_pts4 = self.road.get_pts(s_max,env_index,"left")
        right_pts3 = self.road.get_pts(s_max-self.rod_len-1,env_index,"right")
        right_pts4 = self.road.get_pts(s_max,env_index,"right")
        extend_road_pts = [left_pts3,left_pts4,right_pts4,right_pts3]
        extend_road_polygon2 = rendering.make_polygon(extend_road_pts, draw_border=False)
        extend_road_polygon2.set_color(0.7, 0.7, 0.7, alpha=1.0)  # 灰色填充
        return [extend_road_polygon1,extend_road_polygon2]
    def extra_render_map(self, env_index: int = 0):
        """
        绘制道路地图：
        1) 道路中心线（黑色，较细）
        2) 道路左右边界线（黑色）
        3) 左右边界构成的多边形区域（灰色填充）
        """
        geoms = []
        try:
            scenario = self.road.get_scenario_by_env_index(env_index)
        except:
            return geoms
        
        # 获取所有车道段
        lanelets = scenario.lanelet_network.lanelets
        
        # 遍历每个车道段
        for lanelet in lanelets:

            left_vertices = lanelet.left_vertices
            right_vertices = lanelet.right_vertices
            # ---------- 1) 道路多边形区域填充（灰色） - 分段绘制 ----------
            if left_vertices is not None and right_vertices is not None:
                # 分段参数：每段道路的顶点数（约10米一段，假设顶点间隔约1-2米）
                SEGMENT_VERTEX_COUNT = 3  # 每段约10-20米

                n_left = len(left_vertices)
                n_right = len(right_vertices)

                # 确保左右边界顶点数量一致
                n_vertices = min(n_left, n_right)

                # 分段绘制道路，每段创建一个小四边形
                for i in range(0, n_vertices - 1, SEGMENT_VERTEX_COUNT):
                    # 当前段的结束索引
                    end_idx = min(i + SEGMENT_VERTEX_COUNT, n_vertices - 1)

                    # 构建当前段的多边形：左边界(前→后) + 右边界(后→前)
                    segment_pts = []

                    # 添加当前段的左边界点（从前往后）
                    for j in range(i, end_idx + 1):
                        x, y = left_vertices[j]
                        segment_pts.append((float(x), float(y)))

                    # 添加当前段的右边界点（从后往前，形成闭合）
                    for j in range(end_idx, i - 1, -1):
                        x, y = right_vertices[j]
                        segment_pts.append((float(x), float(y)))

                    # 创建当前段的多边形并填充灰色
                    road_polygon = rendering.make_polygon(segment_pts, draw_border=False)
                    road_polygon.set_color(0.7, 0.7, 0.7, alpha=1.0)  # 灰色填充
                    geoms.append(road_polygon)

            # ---------- 2) 道路中心线 ----------
            center_vertices = lanelet.center_vertices
            if center_vertices is not None:
                center_line = rendering.PolyLine(
                    v=[(float(x), float(y)) for x, y in center_vertices],
                    close=False
                )
                center_line.set_color(*Color.BLACK.value, alpha=1.0)
                center_line.set_linewidth(1.0)  # 中心线稍微细一点
                # 添加虚线效果，使用0x00FF图案（短划线）
                center_line.add_attr(rendering.LineStyle(0x00FF))
                geoms.append(center_line)
            
            # ---------- 3) 道路左右边界线 ----------
            
            if left_vertices is not None:
                left_line = rendering.PolyLine(
                    v=[(float(x), float(y)) for x, y in left_vertices],
                    close=False
                )
                left_line.set_color(*Color.BLACK.value, alpha=1.0)
                left_line.set_linewidth(2.0)
                geoms.append(left_line)
            
            if right_vertices is not None:
                right_line = rendering.PolyLine(
                    v=[(float(x), float(y)) for x, y in right_vertices],
                    close=False
                )
                right_line.set_color(*Color.BLACK.value, alpha=1.0)
                right_line.set_linewidth(2.0)
                geoms.append(right_line)
        return geoms
if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
        display_info=False,
        seed=None,
        agent_index_focus=AGENT_INDEX_FOCUS,
    )
