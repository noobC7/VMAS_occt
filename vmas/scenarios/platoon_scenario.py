import time
from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor
from vmas import render_interactively
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import World, Agent, Sphere, Box
from vmas.simulator.utils import Color
from vmas.simulator.dynamics.dynamic_kinematic_bicycle import DynamicKinematicBicycle
from vmas.simulator.dynamics.delayed_steering_kinematic_bicycle import (
    DelayedSteeringKinematicBicycle,
    KinematicBicycle,
)
from vmas.simulator import rendering
from vmas.simulator.utils import Color, ScenarioUtils
from vmas.scenarios.road_traffic import (
    get_perpendicular_distances,
    get_distances_between_agents,
    get_rectangle_vertices,
    transform_from_global_to_local_coordinate,
    interX,
    exponential_decreasing_fcn,
    angle_eliminate_two_pi,
    Collisions,
    CircularBuffer,
    Timer,
    StateBuffer,
)
from vmas.scenarios.occt_map import OcctMap, OcctCRMap
from vmas.scenarios.occt_utils import (
    OcctObservations,
    OcctRewards,
    OcctNormalizers,
    OcctReferencePathsAgentRelated,
    OcctPenalties,
    OcctThresholds,
    OcctConstants,
    OcctDistances,
    check_validity,
    get_short_term_reference_path_simple,
    get_short_term_reference_path_by_s,
    calibrate_agent_s_by_road_pts,
    is_point_left_of_polyline,
    get_frenet_distances_between_agents,
)
from vmas.scenarios.simple_mppi import SimpleMPPIController
from enum import IntEnum

class MethodClass(IntEnum):
    MARL = 0
    PID = 1
    MPPI = 2
DEFAULT_TRADITIONAL_CONTROL = MethodClass.MARL
AGENT_INDEX_FOCUS = 2

class TaskClass(IntEnum):
    SIMPLE_PLATOON = 0

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
                f"{', '.join(method.name for method in MethodClass)}.",
            ) from exc
    try:
        return MethodClass(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported traditional control '{value}'. Expected MethodClass, int, or str.",
        ) from exc

class Scenario(BaseScenario):

    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.device = device
        self.batch_dim = batch_dim
        self.init_params(batch_dim, device, **kwargs)
        world = self.init_world(batch_dim, device)
        self.init_agents(world, batch_dim, device)
        return world

    def get_tensor_by_distribution(self, dist_type='uniform', size=None, mean=0.0, std=1.0):
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
        if dist_type == 'uniform':
            tensor = torch.rand(size=size, device=self.device, generator=generator)
        elif dist_type == 'normal':
            tensor = torch.normal(mean, std, size=size, device=self.device, generator=generator)
        else:
            raise ValueError(
                f"Unsupported distribution type: {dist_type}. Use 'uniform' or 'normal'.",
            )
        return tensor

    def get_normal_tensor(self, mean, std, size=None):
        return self.get_tensor_by_distribution(dist_type='normal', size=size, mean=mean, std=std)

    def get_random_tensor(self, size=None):
        return self.get_tensor_by_distribution(dist_type='uniform', size=size)

    def get_platoon_space(self, platoon_vel):
        """
        Get the spacing of the platoon.
        Args:
            platoon_vel: Velocity of the platoon.
        Returns:
            platoon_space: Spacing of the platoon.
        """
        return self.still_space + self.platoon_tau * platoon_vel

    def init_params(self, batch_dim: int, device: torch.device, **kwargs):
        self.reset_total_time = 0.0
        self.reward_update_time = 0.0
        self.reset_count = 0
        self.time_records = {'total': 0.0, 'reset_agents_loop': 0.0}
        self.env_current_step = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.env_total_step = torch.zeros(batch_dim, device=device, dtype=torch.long)
        self.success_count = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.failure_count = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        self.agent_index_focus = kwargs.pop('agent_index_focus', AGENT_INDEX_FOCUS)
        self.enable_obs_audit = kwargs.pop('enable_obs_audit', True)
        self.obs_audit_interval = int(kwargs.pop('obs_audit_interval', 100))
        self.obs_audit_agent_index = int(kwargs.pop(
            'obs_audit_agent_index',
            self.agent_index_focus,
        ))
        self.obs_audit_small_threshold = float(kwargs.pop('obs_audit_small_threshold', 0.01))
        self.obs_audit_large_threshold = float(kwargs.pop('obs_audit_large_threshold', 3.0))
        self.obs_audit_last_logged_step = -1
        self.obs_audit_prev_groups = {}
        self._tracking_error_last_synced_step = -1
        self.device = device
        self.batch_dim = batch_dim
        self.traditional_control = parse_traditional_control(kwargs.pop(
            'traditional_control',
            DEFAULT_TRADITIONAL_CONTROL,
        ))
        target_road_id = kwargs.pop('target_road_id', 0)
        self.target_road_id = None if target_road_id is None else int(target_road_id)
        self.task_class = 0
        self.dt = float(kwargs.get('dt', 0.05))
        self.n_agents = kwargs.pop('n_agents', 5)
        self.logged_control_acc = torch.zeros(
            (batch_dim, self.n_agents),
            device=device,
            dtype=torch.float32,
        )
        self.logged_control_steer = torch.zeros_like(self.logged_control_acc)
        self.obs_audit_agent_index = min(max(self.obs_audit_agent_index, 0), self.n_agents - 1)
        self.is_loop = kwargs.pop('is_loop', False)
        self.use_center_frenet_ref = kwargs.pop('use_center_frenet_ref', True)
        self.use_boundary_frenet_ref = kwargs.pop('use_boundary_frenet_ref', True)
        self.is_rand_arc_pos = kwargs.pop('is_rand_arc_pos', False)
        self.init_arc_pos = kwargs.pop('init_arc_pos', 0.0)
        self.init_vel_mean = kwargs.pop('init_vel_mean', 3)
        self.init_vel_std = kwargs.pop('init_vel_std', 1.0)
        self.still_space = kwargs.pop('still_space', 6.0)
        self.platoon_tau = kwargs.pop('platoon_tau', 0.0)
        self.platoon_vel_batch = torch.zeros(self.batch_dim, device=device)
        self.n_followers = self.n_agents
        self.FOLLOWER_SLICE = slice(0, self.n_agents)
        self.n_nearing_agents_observed = kwargs.pop('n_nearing_agents_observed', 2)
        if self.n_nearing_agents_observed >= self.n_agents:
            raise ValueError('n_nearing_agents_observed must be less than n_agents')
        self.is_real_time_rendering = kwargs.pop('is_real_time_rendering', False)
        self.n_points_short_term = kwargs.pop('n_points_short_term', 4)
        self.agent_lookahead_idx = kwargs.pop('agent_lookahead_idx', 2)
        assert self.agent_lookahead_idx < self.n_points_short_term, (
            'agent_lookahead_idx must be less than n_points_short_term'
        )
        self.mppi_horizon_steps = int(kwargs.pop('mppi_horizon_steps', 30))
        self.mppi_num_samples = int(kwargs.pop('mppi_num_samples', 256))
        self.mppi_lambda = float(kwargs.pop('mppi_lambda', 10.0))
        self.mppi_exploration = float(kwargs.pop('mppi_exploration', 0.1))
        self.mppi_debug_top_k = int(kwargs.pop('mppi_debug_top_k', 8))
        self.enable_mppi_debug_render = bool(kwargs.pop('enable_mppi_debug_render', True))
        self.sample_interval = kwargs.pop('sample_interval', 2)
        self.boundary_offset = kwargs.pop('boundary_offset', -self.sample_interval)
        self.n_points_nearing_boundary = kwargs.pop(
            'n_points_nearing_boundary',
            self.n_points_short_term + 1,
        )
        self.is_apply_mask = kwargs.pop('is_apply_mask', False)
        self.is_observe_vertices = kwargs.pop('is_observe_vertices', False)
        self.is_observe_distance_to_agents = kwargs.pop('is_observe_distance_to_agents', True)
        self.is_add_noise = kwargs.pop('is_add_noise', False)
        self.is_observe_ref_path_other_agents = kwargs.pop(
            'is_observe_ref_path_other_agents',
            False,
        )
        is_partial_observation = kwargs.pop('is_partial_observation', True)
        self.visualize_semidims = True
        self.viewer_zoom = float(kwargs.pop('viewer_zoom', 20))
        self.world_x_dim = kwargs.pop('world_x_dim', 200)
        self.world_y_dim = kwargs.pop('world_y_dim', 150)
        self.resolution_factor = kwargs.pop('resolution_factor', 3)
        self.render_origin = kwargs.pop(
            'render_origin',
            [self.world_x_dim / 2, self.world_y_dim / 2],
        )
        self.viewer_size = kwargs.pop(
            'viewer_size',
            (
                int(self.world_x_dim * self.resolution_factor),
                int(self.world_y_dim * self.resolution_factor),
            ),
        )
        self.max_speed = float(kwargs.pop('max_speed', 5))
        self.max_steering_angle = kwargs.pop(
            'max_steering_angle',
            torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)),
        )
        self.max_acceleration = float(kwargs.get('max_acceleration', 3.0))
        self.max_steering_rate = kwargs.pop(
            'max_steering_rate',
            torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)),
        )
        self.l_f = float(kwargs.get('l_f', 1.17))
        self.l_r = float(kwargs.get('l_r', 1.15))
        self.agent_length = self.l_f + self.l_r + 1.5
        self.agent_width = float(kwargs.get('agent_width', 1.5))
        self.mppi_horizon_steps = max(self.mppi_horizon_steps, 1)
        self.mppi_stage_cost_weight = torch.tensor(
            kwargs.pop('mppi_stage_cost_weight', [40.0, 8.0, 12.0, 0.05, 0.2]),
            device=device,
            dtype=torch.float32,
        )
        self.mppi_terminal_cost_weight = torch.tensor(
            kwargs.pop('mppi_terminal_cost_weight', [80.0, 12.0, 16.0]),
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
        noise_level = kwargs.pop('noise_level', 0.2 * self.agent_width)
        n_stored_steps = kwargs.pop('n_stored_steps', 5)
        n_observed_steps = kwargs.pop('n_observed_steps', 5)
        use_history_observation = bool(kwargs.pop('use_history_observation', False))
        history_obs_len = int(kwargs.pop('history_obs_len', n_observed_steps))
        history_obs_dim = kwargs.pop('history_obs_dim', None)
        if history_obs_dim is not None:
            history_obs_dim = int(history_obs_dim)
            if history_obs_dim <= 0:
                raise ValueError('history_obs_dim must be a positive integer or None.')
        B = batch_dim
        self.lane_width = 6
        self.rod_len = (self.n_followers + 1) * self.still_space
        self.road = OcctCRMap(
            batch_dim=B,
            device=device,
            cr_map_dir=(
                '/home/yons/Graduation/VMAS_occt/vmas/scenarios_data/'
                'cr_maps/chapter3_2_path'
            ),
            max_ref_v=self.max_speed,
            is_constant_ref_v=True,
            rod_len=self.rod_len,
            n_agents=self.n_agents,
            target_road_id=self.target_road_id,
        )
        self.road_total_step = torch.zeros_like(self.road.batch_id.unique())
        self.lane_width = self.road.get_lane_width('mean')
        self.ref_paths_agent_related = OcctReferencePathsAgentRelated(
            long_term=self.road.get_road_center_pts().unsqueeze(1).expand(
                -1,
                self.n_agents,
                -1,
                -1,
            ),
            left_boundary=self.road.get_road_left_pts().unsqueeze(1).expand(
                -1,
                self.n_agents,
                -1,
                -1,
            ),
            right_boundary=self.road.get_road_right_pts().unsqueeze(1).expand(
                -1,
                self.n_agents,
                -1,
                -1,
            ),
            short_term=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_short_term, 3),
                device=device,
                dtype=torch.float32,
            ),
            short_term_indices=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_short_term),
                device=device,
                dtype=torch.int32,
            ),
            nearing_points_left_boundary=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_nearing_boundary, 2),
                device=device,
                dtype=torch.float32,
            ),
            nearing_points_right_boundary=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_nearing_boundary, 2),
                device=device,
                dtype=torch.float32,
            ),
            exit=torch.zeros((batch_dim, self.n_agents, 2, 2), device=device, dtype=torch.float32),
        )
        self.timer = Timer(
            start=time.time(),
            end=0,
            step=torch.zeros(batch_dim, device=device, dtype=torch.int32),
            step_begin=time.time(),
            render_begin=0,
        )
        self.constants = OcctConstants(
            env_idx_broadcasting=torch.arange(
                batch_dim,
                device=device,
                dtype=torch.int32,
            ).unsqueeze(-1),
            empty_action_acc=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            ),
            empty_action_steering=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            ),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor(
                self.agent_length ** 2 + self.agent_width ** 2,
                device=device,
                dtype=torch.float32,
            ).sqrt() * 1.2,
        )
        obs_relative_velocity_scale = kwargs.pop(
            'obs_relative_velocity_scale',
            max(self.max_speed / 4, 1.0),
        )
        obs_relative_acceleration_scale = kwargs.pop(
            'obs_relative_acceleration_scale',
            max(self.max_acceleration, 0.5),
        )
        self.normalizers = OcctNormalizers(
            pos=torch.tensor(
                [self.agent_length * 5, self.agent_width * 5],
                device=device,
                dtype=torch.float32,
            ),
            error_pos=torch.tensor(self.agent_length, device=device, dtype=torch.float32),
            pos_world=torch.tensor(
                [self.world_x_dim, self.world_y_dim],
                device=device,
                dtype=torch.float32,
            ),
            v=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            error_v=torch.tensor(obs_relative_velocity_scale, device=device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            action_steering=self.max_steering_angle,
            action_vel=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            action_steering_rate=self.max_steering_rate,
            action_acc=torch.tensor(self.max_acceleration, device=device, dtype=torch.float32),
            distance_lanelet=torch.tensor(self.lane_width * 3, device=device, dtype=torch.float32),
            distance_ref=torch.tensor(self.lane_width * 3, device=device, dtype=torch.float32),
            distance_agent=torch.tensor(self.agent_length * 10, device=device, dtype=torch.float32),
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
            is_partial=torch.tensor(is_partial_observation, device=device, dtype=torch.bool),
            n_nearing_agents=torch.tensor(
                self.n_nearing_agents_observed,
                device=device,
                dtype=torch.int32,
            ),
            noise_level=torch.tensor(noise_level, device=device, dtype=torch.float32),
            n_stored_steps=torch.tensor(n_stored_steps, device=device, dtype=torch.int32),
            n_observed_steps=torch.tensor(n_observed_steps, device=device, dtype=torch.int32),
            platoon_error_vel=torch.zeros(
                (batch_dim, self.n_agents, 2),
                device=device,
                dtype=torch.float32,
            ),
            past_platoon_error_vel=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, 2),
                device=device,
                dtype=torch.float32,
            )),
            self_platoon_error_space=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, 2),
                device=device,
                dtype=torch.float32,
            )),
            agent_s=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            nearing_agents_indices=torch.zeros(
                (batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.int32,
            ),
            past_pos=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                device=device,
                dtype=torch.float32,
            )),
            past_rot=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_vertices=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 4, 2),
                device=device,
                dtype=torch.float32,
            )),
            past_vel=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, self.n_agents, 2),
                device=device,
                dtype=torch.float32,
            )),
            past_steering=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_relative_ref_info=CircularBuffer(torch.zeros(
                (
                    n_stored_steps,
                    batch_dim,
                    self.n_agents,
                    self.n_agents,
                    self.n_points_short_term,
                    3,
                ),
                device=device,
                dtype=torch.float32,
            )),
            past_left_boundary=CircularBuffer(torch.zeros(
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
            )),
            past_right_boundary=CircularBuffer(torch.zeros(
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
            )),
            past_action_acc=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_action_steering=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_distance_to_ref_path=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_distance_to_boundaries=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_distance_to_left_boundary=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_distance_to_right_boundary=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
            past_distance_to_agents=CircularBuffer(torch.zeros(
                (n_stored_steps, batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.float32,
            )),
        )
        self.distances = OcctDistances(
            agents=torch.zeros(
                batch_dim,
                self.n_agents,
                self.n_agents,
                dtype=torch.float32,
                device=device,
            ),
            agents_frenet=torch.zeros(
                batch_dim,
                self.n_agents,
                self.n_agents,
                dtype=torch.float32,
                device=device,
            ),
            left_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4),
                device=device,
                dtype=torch.float32,
            ),
            right_boundaries=torch.zeros(
                (batch_dim, self.n_agents, 1 + 4),
                device=device,
                dtype=torch.float32,
            ),
            boundaries=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            ref_paths=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.float32),
            lookahead_pts=torch.zeros(
                (batch_dim, self.n_agents, 2),
                device=device,
                dtype=torch.float32,
            ),
            closest_point_on_ref_path=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.int32,
            ),
            closest_point_on_left_b=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.int32,
            ),
            closest_point_on_right_b=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.int32,
            ),
        )
        n_agents = self.n_agents
        self.reward_details = {
            'reward_total': torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            'reward_progress': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'reward_vel': torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            'reward_platoon_heading': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'reward_goal': torch.zeros((batch_dim, n_agents), device=device, dtype=torch.float32),
            'reward_platoon_vel': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'reward_platoon_ref': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'reward_platoon_space': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'penalty_near_boundary': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'penalty_near_other_agents': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'penalty_change_steering': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'penalty_change_acc': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'penalty_collide_with_agents': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'penalty_outside_boundaries': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
            'penalty_backward': torch.zeros(
                (batch_dim, n_agents),
                device=device,
                dtype=torch.float32,
            ),
        }
        threshold_change_steering = kwargs.pop('threshold_change_steering', 10)
        threshold_change_acc = kwargs.pop('threshold_change_acc', 10)
        threshold_near_boundary_high = kwargs.pop(
            'threshold_near_boundary_high',
            self.agent_width / 2,
        )
        threshold_near_boundary_low = kwargs.pop('threshold_near_boundary_low', 0)
        threshold_near_other_agents_c2c_high = kwargs.pop(
            'threshold_near_other_agents_c2c_high',
            1.8 * (self.agent_length ** 2 + self.agent_width ** 2) ** 0.5,
        )
        threshold_near_other_agents_c2c_low = kwargs.pop(
            'threshold_near_other_agents_c2c_low',
            (self.agent_length ** 2 + self.agent_width ** 2) ** 0.5,
        )
        self.thresholds = OcctThresholds(
            near_boundary_low=torch.tensor(
                threshold_near_boundary_low,
                device=device,
                dtype=torch.float32,
            ),
            near_boundary_high=torch.tensor(
                threshold_near_boundary_high,
                device=device,
                dtype=torch.float32,
            ),
            near_other_agents_low=torch.tensor(
                threshold_near_other_agents_c2c_low,
                device=device,
                dtype=torch.float32,
            ),
            near_other_agents_high=torch.tensor(
                threshold_near_other_agents_c2c_high,
                device=device,
                dtype=torch.float32,
            ),
            change_steering=torch.tensor(
                threshold_change_steering,
                device=device,
                dtype=torch.float32,
            ).deg2rad(),
            change_acc=torch.tensor(threshold_change_acc, device=device, dtype=torch.float32),
            distance_mask_agents=self.normalizers.pos[0],
        )
        self.collisions = Collisions(
            with_agents=torch.zeros(
                (batch_dim, self.n_agents, self.n_agents),
                device=device,
                dtype=torch.bool,
            ),
            with_lanelets=torch.zeros((batch_dim, self.n_agents), device=device, dtype=torch.bool),
            with_exit_segments=torch.zeros(
                (batch_dim, self.n_agents),
                device=device,
                dtype=torch.bool,
            ),
        )
        self.vertices = torch.zeros(
            (batch_dim, self.n_agents, 5, 2),
            device=device,
            dtype=torch.float32,
        )
        weighting_ref_directions = torch.linspace(
            1,
            0.2,
            steps=self.n_points_short_term - 1,
            device=device,
            dtype=torch.float32,
        )
        weighting_ref_directions /= weighting_ref_directions.sum()
        r_p_normalizer = 100
        reward_progress = kwargs.pop('reward_progress', 10) / r_p_normalizer
        reward_vel = kwargs.pop('reward_vel', 0) / r_p_normalizer
        reward_goal = kwargs.pop('reward_goal', 10) / r_p_normalizer
        reward_platoon_space = kwargs.pop('reward_platoon_space', 20) / r_p_normalizer
        reward_platoon_vel = kwargs.pop('reward_platoon_vel', 20) / r_p_normalizer
        reward_platoon_ref = kwargs.pop('reward_platoon_ref', 50) / r_p_normalizer
        reward_platoon_heading = kwargs.pop('reward_platoon_heading', 50) / r_p_normalizer
        self.rewards = OcctRewards(
            reward_progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions,
            reward_vel=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reward_goal=torch.tensor(reward_goal, device=device, dtype=torch.float32),
            reward_platoon_heading=torch.tensor(
                reward_platoon_heading,
                device=device,
                dtype=torch.float32,
            ),
            reward_platoon_space=torch.tensor(
                reward_platoon_space,
                device=device,
                dtype=torch.float32,
            ),
            reward_platoon_vel=torch.tensor(reward_platoon_vel, device=device, dtype=torch.float32),
            reward_platoon_ref=torch.tensor(reward_platoon_ref, device=device, dtype=torch.float32),
        )
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)
        penalty_near_boundary = kwargs.pop('penalty_near_boundary', -20) / r_p_normalizer
        penalty_near_other_agents = kwargs.pop('penalty_near_other_agents', -20) / r_p_normalizer
        penalty_collide_with_agents = kwargs.pop(
            'penalty_collide_with_agents',
            -100,
        ) / r_p_normalizer
        penalty_outside_boundaries = kwargs.pop('penalty_outside_boundaries', -100) / r_p_normalizer
        penalty_change_steering = kwargs.pop('penalty_change_steering', -20) / r_p_normalizer
        penalty_change_acc = kwargs.pop('penalty_change_acc', -20)
        penalty_backward = kwargs.pop('penalty_backward', -100) / r_p_normalizer
        self.penalties = OcctPenalties(
            near_boundary=torch.tensor(penalty_near_boundary, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(
                penalty_near_other_agents,
                device=device,
                dtype=torch.float32,
            ),
            collide_with_agents=torch.tensor(
                penalty_collide_with_agents,
                device=device,
                dtype=torch.float32,
            ),
            collide_with_boundaries=torch.tensor(
                penalty_outside_boundaries,
                device=device,
                dtype=torch.float32,
            ),
            change_steering=torch.tensor(
                penalty_change_steering,
                device=device,
                dtype=torch.float32,
            ),
            change_acc=torch.tensor(penalty_change_acc, device=device, dtype=torch.float32),
            backward=torch.tensor(penalty_backward, device=device, dtype=torch.float32),
        )
        self.enable_failure_replay_restore = bool(kwargs.pop(
            'enable_failure_replay_restore',
            False,
        ))
        self.failure_replay_pre_failure_seconds = float(kwargs.pop(
            'failure_replay_pre_failure_seconds',
            1.5,
        ))
        self.failure_replay_margin_steps = int(kwargs.pop('failure_replay_margin_steps', 0))
        self.failure_replay_k_steps = max(
            1,
            int(round(self.failure_replay_pre_failure_seconds / self.dt)),
        )
        self.failure_replay_snapshot_dim = 8
        self.failure_curriculum_bank = None
        self.failure_curriculum_collect_enabled = False
        self.failure_curriculum_sampling_enabled = False
        self.failure_curriculum_replay_probability = 0.0
        self.failure_curriculum_min_bank_size = 0
        self.failure_curriculum_iteration = 0
        self.current_episode_replay_source = torch.zeros(batch_dim, device=device, dtype=torch.bool)
        self.current_episode_replay_entry_id = torch.full(
            (batch_dim,),
            -1,
            device=device,
            dtype=torch.long,
        )
        self.failure_curriculum_events = []
        self.n_steps_before_recording = kwargs.pop('n_steps_before_recording', 10)
        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.state_buffer = StateBuffer(buffer=torch.zeros(
            (self.n_steps_before_recording, batch_dim, self.n_agents, 5),
            device=device,
            dtype=torch.float32,
        ))
        self.failure_replay_snapshot_buffer = CircularBuffer(torch.zeros(
            (
                self.failure_replay_k_steps + 1,
                batch_dim,
                self.n_agents,
                self.failure_replay_snapshot_dim,
            ),
            device=device,
            dtype=torch.float32,
        ))

    def init_world(self, batch_dim: int, device: torch.device) -> World:
        world = World(
            batch_dim=batch_dim,
            device=device,
            dt=self.dt,
            x_semidim=self.world_x_dim,
            y_semidim=self.world_y_dim,
            dim_c=0,
        )
        return world

    def init_agents(self, world: World, *kwargs):
        self.followers = []
        i = 0
        colors = [
            (31 / 255, 73 / 255, 125 / 255),
            (123 / 255, 31 / 255, 162 / 255),
            (0 / 255, 109 / 255, 119 / 255),
            (145 / 255, 30 / 255, 18 / 255),
            (45 / 255, 48 / 255, 91 / 255),
            (127 / 255, 80 / 255, 0 / 255),
        ]
        for _ in range(self.n_followers):
            a = Agent(
                name=f'agent_{i}',
                shape=Box(length=self.agent_length, width=self.agent_width),
                color=colors[i % len(colors)],
                collide=False,
                render_action=False,
                u_range=[self.max_acceleration, self.max_steering_angle],
                u_multiplier=[1, 1],
                max_speed=self.max_speed,
                drag=0.0,
                linear_friction=0.0,
                angular_friction=0.0,
                movable=False if self.traditional_control != MethodClass.MARL else True,
                rotatable=False if self.traditional_control != MethodClass.MARL else True,
                dynamics=KinematicBicycle(
                    world,
                    width=self.agent_width,
                    l_f=self.l_f,
                    l_r=self.l_r,
                    max_acceleration=self.max_acceleration,
                    max_steering_angle=self.max_steering_angle,
                    integration='rk4',
                ),
            )
            world.add_agent(a)
            self.followers.append(a)
            i += 1

    def get_occt_cr_path_num(self):
        """
        Get the number of available OCCT CR map paths.
        """
        return len(self.road.path_library)

    def _set_pose(self, agent: Agent, pos: Tensor, theta: Tensor, vel: Tensor, idx_mask: Tensor):
        if hasattr(agent.state, 'pos'):
            agent.state.pos[idx_mask] = pos[idx_mask]
        if hasattr(agent.state, 'rot'):
            theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
            agent.state.rot[idx_mask] = theta_reshaped[idx_mask]
        elif hasattr(agent.state, 'angle'):
            theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
            agent.state.angle[idx_mask] = theta_reshaped[idx_mask]
        if hasattr(agent.state, 'vel'):
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
        for (agent_idx, agent) in enumerate(self.world.agents):
            if hasattr(agent.dynamics, 'cur_delta') and agent.dynamics.cur_delta is not None:
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
            n=self.failure_replay_k_steps + 1,
        )[env_index].detach().clone()
        return {'agent_state': snapshot_state}

    def _get_failure_type_for_env(self, done_status: Dict[str, Tensor], env_index: int) -> str:
        if bool(done_status['is_collision_with_agents_env'][env_index].item()):
            return 'collision_with_agents'
        if bool(done_status['is_collision_with_lanelets'][env_index].item()):
            return 'collision_with_lanelets'
        if bool(done_status['is_collision_with_exit_segments'][env_index].item()):
            return 'collision_with_exit_segments'
        return 'unknown_failure'

    def _record_failure_curriculum_events(self, done_status: Dict[str, Tensor]) -> None:
        if not self.failure_curriculum_collect_enabled:
            return
        done_mask = done_status['is_done']
        success_mask = done_status['is_success']
        failure_mask = done_status['is_failure']
        for env_index in torch.where(done_mask)[0].tolist():
            road_id = int(self.road.batch_id[env_index].item())
            source_entry_id = int(self.current_episode_replay_entry_id[env_index].item())
            is_replay_source = bool(self.current_episode_replay_source[env_index].item())
            failure_type = self._get_failure_type_for_env(done_status, env_index)
            has_valid_snapshot = int(
                self.env_current_step[env_index].item(),
            ) >= self.failure_replay_k_steps + self.failure_replay_margin_steps
            snapshot = self._build_failure_replay_snapshot(
                env_index,
            ) if has_valid_snapshot and bool(failure_mask[env_index].item()) else None
            if is_replay_source:
                if bool(success_mask[env_index].item()):
                    self.failure_curriculum_events.append({
                        'event_type': 'replay_success',
                        'source_entry_id': source_entry_id,
                        'road_id': road_id,
                    })
                elif bool(failure_mask[env_index].item()):
                    self.failure_curriculum_events.append({
                        'event_type': 'replay_failure',
                        'source_entry_id': source_entry_id,
                        'road_id': road_id,
                        'failure_type': failure_type,
                        'snapshot': snapshot,
                    })
            elif bool(failure_mask[env_index].item()) and snapshot is not None:
                self.failure_curriculum_events.append({
                    'event_type': 'new_failure',
                    'source_entry_id': -1,
                    'road_id': road_id,
                    'failure_type': failure_type,
                    'snapshot': snapshot,
                })

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
        agent_state = snapshot['agent_state'].to(self.device)
        pos = agent_state[:, 0:2]
        rot = agent_state[:, 2:3]
        vel = agent_state[:, 3:5]
        ang_vel = agent_state[:, 5:6]
        cur_delta = agent_state[:, 6:7]
        agent_s = agent_state[:, 7]
        for (agent_idx, agent) in enumerate(agents):
            agent.state.pos[env_index] = pos[agent_idx]
            agent.state.rot[env_index] = rot[agent_idx]
            agent.state.vel[env_index] = vel[agent_idx]
            if hasattr(agent.state, 'ang_vel'):
                agent.state.ang_vel[env_index] = ang_vel[agent_idx]
            if hasattr(agent.dynamics, 'cur_delta') and agent.dynamics.cur_delta is not None:
                agent.dynamics.cur_delta[env_index] = cur_delta[agent_idx]
        self.observations.agent_s[env_index] = agent_s
        for i_agent in range(self.n_agents):
            self.reset_init_distances_and_short_term_ref_path(env_index, i_agent, agents)
        mutual_distances = get_distances_between_agents(self=self, is_set_diagonal=True)
        mutual_frenet_distances = get_frenet_distances_between_agents(self.observations.agent_s)
        self.distances.agents[env_index, :, :] = mutual_distances[env_index, :, :]
        self.distances.agents_frenet[env_index, :, :] = mutual_frenet_distances[env_index, :, :]
        self.collisions.with_agents[env_index, :, :] = False
        self.collisions.with_lanelets[env_index, :] = False
        self.collisions.with_exit_segments[env_index, :] = False
        self.current_episode_replay_source[env_index] = True
        self.current_episode_replay_entry_id[env_index] = int(entry_id)

    def reset_world_at(self, env_index: Optional[int]=None, agent_index: Optional[int]=None):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is None, most computation is done in a vectorized way.

        Args:
        :param env_index: environment index to reset. ``None`` means vectorized reset.
        :param agent_index: agent index to reset. ``None`` means reset all agents.
        """
        total_start = time.time()
        B = self.batch_dim
        device = self.device
        assert agent_index == None, 'agent_index must be None, not supported'
        stage1_start = time.time()
        if env_index is None:
            idx_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            idx_mask = torch.zeros(B, dtype=torch.bool, device=device)
            idx_mask[env_index] = True
        self.current_episode_replay_source[idx_mask] = False
        self.current_episode_replay_entry_id[idx_mask] = -1
        sampled_failure_replay = {}
        if self.simple_mppi is not None:
            self.simple_mppi.reset()
        if hasattr(self, '_last_vel_errors'):
            self._last_vel_errors.clear()
        if hasattr(self, '_last_pid_steering'):
            self._last_pid_steering.clear()
        stage2_start = time.time()
        platoon_vel_batch = self.get_normal_tensor(self.init_vel_mean, self.init_vel_std)
        self.platoon_vel_batch[idx_mask] = torch.clamp(platoon_vel_batch, min=0.0)[idx_mask]
        self.platoon_space_batch = self.get_platoon_space(self.platoon_vel_batch)
        spacing = self.platoon_space_batch
        stage3_start = time.time()
        s_start_buffer = 0.0
        s_end_buffer = 10.0
        last_vehicle_s_max = torch.clamp(self.road.batch_corner_s_end - self.rod_len, min=0)
        if self.is_rand_arc_pos:
            last_vehicle_s = torch.ones(
                B,
                device=device,
            ) * self.init_arc_pos + self.get_random_tensor() * (
                last_vehicle_s_max - self.init_arc_pos
            )
        else:
            last_vehicle_s = torch.ones(B, device=device) * self.init_arc_pos
        last_vehicle_s = torch.clamp(
            last_vehicle_s,
            s_start_buffer * torch.ones(B, device=device),
            self.road.get_s_max()
            - (s_start_buffer + s_end_buffer)
            - (self.n_agents - 1) * torch.mean(
                spacing,
                dim=-1,
            ),
        )
        init_vel_noise = self.get_normal_tensor(0, self.init_vel_std)
        stage4_start = time.time()
        s_front_new = (self.n_followers - 1) * self.still_space + last_vehicle_s
        stage5_start = time.time()
        F = self.n_followers
        lateral_offset = torch.rand(
            B,
            F,
            device=device,
        ) * (self.lane_width - self.agent_width) / 2 * 0
        lateral_direction = torch.sign(torch.randn(B, F, device=device))
        lateral_offset = lateral_offset * lateral_direction
        heading_error = torch.rand(B, F, device=device) * 0.0 * (torch.pi / 180.0)
        heading_direction = torch.sign(torch.randn(B, F, device=device))
        heading_error = heading_error * heading_direction
        stage6_start = time.time()
        vehicle_s = torch.zeros(B, F, device=device)
        vehicle_s[:, 0] = s_front_new
        for i in range(F - 1):
            vehicle_s[:, i + 1] = vehicle_s[:, i] - spacing
        vehicle_s = torch.clamp(vehicle_s, max=self.road.get_s_max()[:, None].expand(-1, F) - 1e-06)
        stage7_start = time.time()
        self.observations.agent_s[env_index] = vehicle_s[env_index]
        stage8_start = time.time()
        vehicle_pos = self.road.get_pts(vehicle_s)
        road_theta = self.road.get_tangent_heading(vehicle_s)
        normal_vec = self.road.get_normal_vector(vehicle_s)
        vehicle_pos = vehicle_pos + lateral_offset.unsqueeze(-1) * normal_vec
        vehicle_theta = road_theta + heading_error
        stage9_start = time.time()
        for (i, ag) in enumerate(self.followers):
            agent_s = self.observations.agent_s[..., self.FOLLOWER_SLICE][:, i]
            ref_v = self.road.get_ref_v(agent_s[:, None])[:, 0, 0] + init_vel_noise
            ref_v = torch.clamp(ref_v, min=0, max=self.max_speed)
            self._set_pose(ag, vehicle_pos[:, i, :], vehicle_theta[:, i], ref_v, idx_mask)
        stage10_start = time.time()
        agents = self.world.agents
        is_reset_single_agent = agent_index is not None
        for env_i in [env_index] if env_index is not None else range(self.world.batch_dim):
            if env_i == 0:
                self.timer.start = time.time()
                self.timer.step_begin = time.time()
                self.timer.end = 0
            if not is_reset_single_agent:
                self.timer.step[env_i] = 0
            if env_index is None:
                if env_i == self.world.batch_dim - 1:
                    env_j = slice(None)
                else:
                    continue
            else:
                env_j = env_i
            tmp_t = time.time()
            for i_agent in range(
                self.n_agents,
            ) if not is_reset_single_agent else agent_index.unsqueeze(0):
                assert torch.isnan(agents[i_agent].state.pos[
                    env_j,
                    :,
                ]).any() == False, f'agent {i_agent} pos is nan'
                self.reset_init_distances_and_short_term_ref_path(env_j, i_agent, agents)
                agents[i_agent].dynamics.cur_delta[env_j] = 0.0
            mutual_distances = get_distances_between_agents(self=self, is_set_diagonal=True)
            mutual_frenet_distances = get_frenet_distances_between_agents(self.observations.agent_s)
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]
            self.distances.agents_frenet[env_j, :, :] = mutual_frenet_distances[env_j, :, :]
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False
            self.collisions.with_exit_segments[env_j, :] = False
            if self.enable_failure_replay_restore:
                sampled_result = self._sample_failure_curriculum_snapshot(env_i)
                if sampled_result is not None:
                    sampled_failure_replay[env_i] = sampled_result
        self.time_records['reset_agents_loop'] = time.time() - stage10_start
        if sampled_failure_replay:
            for (replay_env_index, (entry_id, snapshot)) in sampled_failure_replay.items():
                self._restore_failure_replay_snapshot(replay_env_index, snapshot, entry_id, agents)
        stage11_start = time.time()
        self.state_buffer.reset()
        state_add = torch.cat(
            (
                torch.stack([a.state.pos for a in agents], dim=1),
                torch.stack([a.state.rot for a in agents], dim=1),
                torch.stack([a.state.vel for a in agents], dim=1),
            ),
            dim=-1,
        )
        self.state_buffer.add(state_add)
        self.failure_replay_snapshot_buffer.add(self._build_failure_replay_buffer_state())
        self.time_records['state_buffer'] = time.time() - stage11_start
        self.time_records['total'] = time.time() - total_start
        self.reset_total_time += self.time_records['total']
        self.reset_count += 1

    def _print_time_report(self):
        """Print per-stage timing statistics sorted by runtime."""
        print('\n========== reset_world_at timing breakdown ==========')
        sorted_records = sorted(self.time_records.items(), key=lambda x: x[1], reverse=True)
        total = self.time_records['total']
        for (stage, cost) in sorted_records:
            ratio = cost / total * 100 if total > 0 else 0
            print(f'{stage:20s}: {cost:.6f}s ({ratio:.2f}%)')
        print('===============================================\n')

    def reset_init_distances_and_short_term_ref_path(self, env_j, i_agent, agents):
        """
        Calculate distances from the agent center to the reference path and
        boundaries, update the vehicle vertices, and rebuild the short-term
        reference path for the current state.
        """
        tmp_t = time.time()
        (
            self.distances.ref_paths[env_j, i_agent],
            self.distances.closest_point_on_ref_path[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
            n_points_long_term=None,
        )
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
            n_points_long_term=None,
        )
        self.distances.left_boundaries[
            env_j,
            i_agent,
            0,
        ] = center_2_left_b - agents[i_agent].shape.width / 2
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[env_j, i_agent],
        ) = get_perpendicular_distances(
            point=agents[i_agent].state.pos[env_j, :],
            polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
            n_points_long_term=None,
        )
        self.distances.right_boundaries[
            env_j,
            i_agent,
            0,
        ] = center_2_right_b - agents[i_agent].shape.width / 2
        assert torch.isnan(agents[i_agent].state.pos[
            env_j,
            :,
        ]).any() == False, f'agent {i_agent} pos is nan'
        self.vertices[
            env_j,
            i_agent,
        ] = get_rectangle_vertices(
            center=agents[i_agent].state.pos[env_j, :],
            yaw=agents[i_agent].state.rot[env_j, :],
            width=agents[i_agent].shape.width,
            length=agents[i_agent].shape.length,
            is_close_shape=True,
        )
        tmp_t = time.time()
        for c_i in range(4):
            (
                self.distances.left_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                n_points_long_term=None,
            )
            (
                self.distances.right_boundaries[env_j, i_agent, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[env_j, i_agent, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                n_points_long_term=None,
            )
        (
            self.distances.boundaries[env_j, i_agent],
            _,
        ) = torch.min(
            torch.hstack((
                self.distances.left_boundaries[env_j, i_agent],
                self.distances.right_boundaries[env_j, i_agent],
            )),
            dim=-1,
        )
        tmp_t = time.time()
        if self.use_center_frenet_ref:
            self.ref_paths_agent_related.short_term[
                env_j,
                i_agent,
            ] = get_short_term_reference_path_by_s(
                self.road,
                self.observations.agent_s[env_j, i_agent],
                n_points_to_return=self.n_points_short_term,
                device=self.world.device,
                sample_interval=self.sample_interval,
                return_ref_v=True,
                env_j=env_j,
            )
            if i_agent != 0:
                self.ref_paths_agent_related.short_term[
                    env_j,
                    i_agent,
                    :,
                    -1,
                ] = self.ref_paths_agent_related.short_term[env_j, 0, :, -1]
        else:
            (
                self.ref_paths_agent_related.short_term[env_j, i_agent, :, 0:2],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_ref_path[env_j, i_agent],
                n_points_to_return=self.n_points_short_term,
                device=self.world.device,
                sample_interval=self.sample_interval,
                n_points_shift=1,
            )
            self.ref_paths_agent_related.short_term[env_j, i_agent, :, 2] = self.init_vel_mean
        tmp_t = time.time()
        if self.use_boundary_frenet_ref:
            self.ref_paths_agent_related.nearing_points_left_boundary[
                env_j,
                i_agent,
            ] = get_short_term_reference_path_by_s(
                self.road,
                self.observations.agent_s[env_j, i_agent] + self.boundary_offset,
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=self.sample_interval,
                return_ref_v=False,
                env_j=env_j,
                line='left',
            )
            self.ref_paths_agent_related.nearing_points_right_boundary[
                env_j,
                i_agent,
            ] = get_short_term_reference_path_by_s(
                self.road,
                self.observations.agent_s[env_j, i_agent] + self.boundary_offset,
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=self.sample_interval,
                return_ref_v=False,
                env_j=env_j,
                line='right',
            )
        else:
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[env_j, i_agent],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.left_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_left_b[env_j, i_agent],
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=self.sample_interval,
                n_points_shift=1,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[env_j, i_agent],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.right_boundary[env_j, i_agent],
                index_closest_point=self.distances.closest_point_on_right_b[env_j, i_agent],
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=self.sample_interval,
                n_points_shift=1,
            )
        theta = agents[i_agent].state.rot[env_j, :]
        for idx in range(self.agent_lookahead_idx):
            lookahead_pts = agents[i_agent].state.pos[
                env_j,
                :,
            ] + idx * self.sample_interval * torch.hstack([torch.cos(theta), torch.sin(theta)])
            self.distances.lookahead_pts[
                env_j,
                i_agent,
                idx,
            ] = torch.linalg.norm(
                self.ref_paths_agent_related.short_term[env_j, i_agent, idx, :2] - lookahead_pts,
                dim=-1,
            )
        s_max_idx = self.road.get_s_max_idx()[env_j]
        if s_max_idx.dim():
            last_pts_idx = s_max_idx[:, None, None].expand(-1, -1, 2)
        else:
            last_pts_idx = s_max_idx[None, None].expand(-1, 2)
        self.ref_paths_agent_related.exit[
            env_j,
            i_agent,
            0,
            :,
        ] = torch.gather(
            self.ref_paths_agent_related.left_boundary[env_j, i_agent],
            dim=-2,
            index=last_pts_idx,
        ).squeeze(-2)
        self.ref_paths_agent_related.exit[
            env_j,
            i_agent,
            1,
            :,
        ] = torch.gather(
            self.ref_paths_agent_related.right_boundary[env_j, i_agent],
            dim=-2,
            index=last_pts_idx,
        ).squeeze(-2)

    def pre_step(self):
        self.logged_control_acc.zero_()
        self.logged_control_steer.zero_()
        if self.traditional_control != MethodClass.MARL:
            self._pure_pursuit_control_platoon()

    def _pure_pursuit_control_platoon(self):
        """
        Pure pursuit platoon controller.

        Used for manual control when ``batch_size == 1``.
        It keeps lateral path tracking and longitudinal platoon speed control.

        For each agent:
        1. Compute the front wheel steering angle with pure pursuit.
        2. Compute acceleration with a PD controller.
        3. Integrate the next state manually.
        4. Apply the state with ``_set_pose`` to bypass VMAS dynamics.
        """
        LOOKAHEAD_DIST = 5.0
        WHEELBASE = self.l_f + self.l_r
        KP_VEL = 1.0
        KD_VEL = 0.0
        if not hasattr(self, '_last_vel_errors'):
            self._last_vel_errors = {}
        for (agent_idx, agent) in enumerate(self.world.agents):
            current_pos = agent.state.pos[0]
            current_theta = agent.state.rot[0, 0]
            current_vel = agent.state.vel[0]
            v_current = torch.linalg.norm(current_vel)
            vel_dir = current_vel / (v_current + 1e-08)
            heading_vec = torch.stack([torch.cos(current_theta), torch.sin(current_theta)])
            direction_sign = torch.sign(torch.sum(vel_dir * heading_vec))
            v_signed = v_current * direction_sign
            ref_path = self.ref_paths_agent_related.short_term[0, agent_idx]
            ref_points = ref_path[:, :2]
            dists = torch.linalg.norm(ref_points - current_pos, dim=-1)
            target_idx = torch.argmin(torch.abs(dists - LOOKAHEAD_DIST))
            target_point = ref_points[target_idx]
            dx = target_point[0] - current_pos[0]
            dy = target_point[1] - current_pos[1]
            cos_theta = torch.cos(current_theta)
            sin_theta = torch.sin(current_theta)
            target_x_vehicle = dx * cos_theta + dy * sin_theta
            target_y_vehicle = -dx * sin_theta + dy * cos_theta
            ld = torch.sqrt(target_x_vehicle ** 2 + target_y_vehicle ** 2)
            ly = target_y_vehicle
            ld = torch.clamp(ld, min=0.1)
            curvature = 2.0 * ly / ld ** 2
            steering_angle = torch.atan(curvature * WHEELBASE)
            steering_angle = torch.clamp(
                steering_angle,
                min=-self.max_steering_angle,
                max=self.max_steering_angle,
            )
            v_ref = torch.linalg.norm(self.world.agents[0].state.vel)
            vel_error = v_ref - v_signed
            last_vel_error = self._last_vel_errors.get(
                agent_idx,
                torch.tensor(0.0, device=self.device),
            )
            derivative = (vel_error - last_vel_error) / self.dt
            acceleration = KP_VEL * vel_error + KD_VEL * derivative
            acceleration = torch.clamp(
                acceleration,
                min=-self.max_acceleration,
                max=self.max_acceleration,
            )
            self.logged_control_steer[0, agent_idx] = steering_angle.detach().clone()
            self.logged_control_acc[0, agent_idx] = acceleration.detach().clone()
            self._last_vel_errors[agent_idx] = vel_error.detach().clone()
            beta = torch.atan2(
                torch.tan(steering_angle) * self.l_r / (self.l_f + self.l_r),
                torch.tensor(1.0, device=self.device),
            )
            dx = v_signed * torch.cos(current_theta + beta)
            dy = v_signed * torch.sin(current_theta + beta)
            dtheta = v_signed / (self.l_f + self.l_r) * torch.cos(beta) * torch.tan(steering_angle)
            dv = acceleration
            next_pos = current_pos + torch.stack([dx, dy]) * self.dt
            next_theta = current_theta + dtheta * self.dt
            next_v = v_signed + dv * self.dt
            next_v = torch.clamp(next_v, min=0.0)
            idx_mask = torch.ones(self.batch_dim, dtype=torch.bool, device=self.device)
            self._set_pose(
                agent,
                next_pos.unsqueeze(0),
                next_theta.unsqueeze(0).unsqueeze(0),
                next_v.unsqueeze(0),
                idx_mask,
            )

    def _sync_agent_s_from_world(self):
        """Project current agent poses back to Frenet s after physics or hard projection."""
        B = self.batch_dim
        F = len(self.world.agents)
        agents_pos = torch.zeros((B, F, 2), device=self.device)
        for (i, agent) in enumerate(self.world.agents):
            agents_pos[:, i, :2] = agent.state.pos
        agent_vel_vector = torch.stack(
            [torch.linalg.norm(
                self.world.agents[i].state.vel,
                dim=-1,
            ) for i in range(self.n_agents)],
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
        self.observations.agent_s[..., self.FOLLOWER_SLICE] = new_agent_s[:, self.FOLLOWER_SLICE]

    def post_step(self):
        """Synchronize the latest world state back to Frenet s."""
        self._sync_agent_s_from_world()

    def get_scenario_info(self):
        """Return scenario metadata for debugging and validation."""
        return {
            'batch_dim': self.batch_dim,
            'n_agents': self.n_agents,
            'n_followers': self.n_followers,
            'rod_len': self.rod_len,
            'dt': self.dt,
            'device': str(self.device),
        }

    def _compute_tracking_error_space(self):
        error_space = torch.zeros(
            (self.world.batch_dim, self.n_agents, 2),
            device=self.world.device,
            dtype=torch.float32,
        )
        for agent_idx in range(self.n_agents):
            if agent_idx > 0:
                actual_distance = self.distances.agents[:, agent_idx, agent_idx - 1]
                error_space[:, agent_idx, 0] = actual_distance - self.platoon_space_batch
            if agent_idx < self.n_agents - 1:
                actual_distance = self.distances.agents[:, agent_idx, agent_idx + 1]
                error_space[:, agent_idx, 1] = actual_distance - self.platoon_space_batch
        return error_space

    def _compute_tracking_error_vel(self):
        platoon_error_vel = torch.zeros(
            (self.world.batch_dim, self.n_agents, 2),
            device=self.world.device,
            dtype=torch.float32,
        )
        agent_speed = torch.stack(
            [torch.linalg.norm(agent.state.vel, dim=1) for agent in self.world.agents],
            dim=1,
        )
        leader_speed = agent_speed[:, 0]
        leader_ref_speed = self.ref_paths_agent_related.short_term[:, 0, 0, -1]
        for agent_idx in range(self.n_agents):
            ego_speed = agent_speed[:, agent_idx]
            if agent_idx == 0:
                speed_error = ego_speed - leader_ref_speed
            else:
                speed_error = ego_speed - leader_speed
            platoon_error_vel[:, agent_idx, 0] = speed_error
            platoon_error_vel[:, agent_idx, 1] = speed_error
        return platoon_error_vel

    def update_observation_and_normalize(self, agent, agent_index):
        """Update observation and normalize them."""
        if agent_index == 0:
            positions_global = torch.stack(
                [a.state.pos for a in self.world.agents],
                dim=0,
            ).transpose(0, 1)
            rotations_global = torch.stack(
                [a.state.rot for a in self.world.agents],
                dim=0,
            ).transpose(0, 1).squeeze(-1)
            self.observations.past_distance_to_agents.add(
                self.distances.agents / self.normalizers.distance_lanelet,
            )
            self.observations.past_distance_to_ref_path.add(
                self.distances.ref_paths / self.normalizers.distance_lanelet,
            )
            self.observations.past_distance_to_left_boundary.add(torch.min(
                self.distances.left_boundaries,
                dim=-1,
            )[0] / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_right_boundary.add(torch.min(
                self.distances.right_boundaries,
                dim=-1,
            )[0] / self.normalizers.distance_lanelet)
            self.observations.past_distance_to_boundaries.add(
                self.distances.boundaries / self.normalizers.distance_lanelet,
            )
            platoon_error_space = self._compute_tracking_error_space()
            platoon_error_vel = self._compute_tracking_error_vel()
            self.observations.self_platoon_error_space.add(platoon_error_space)
            self.observations.platoon_error_vel = platoon_error_vel
            self.observations.past_platoon_error_vel.add(platoon_error_vel)
            pos_i_others = torch.zeros(
                (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                device=self.world.device,
                dtype=torch.float32,
            )
            rot_i_others = torch.zeros(
                (self.world.batch_dim, self.n_agents, self.n_agents),
                device=self.world.device,
                dtype=torch.float32,
            )
            vel_i_others = torch.zeros(
                (self.world.batch_dim, self.n_agents, self.n_agents, 2),
                device=self.world.device,
                dtype=torch.float32,
            )
            ref_i_others = torch.zeros_like(self.observations.past_relative_ref_info.get_latest())
            l_b_i_others = torch.zeros_like(self.observations.past_left_boundary.get_latest())
            r_b_i_others = torch.zeros_like(self.observations.past_right_boundary.get_latest())
            ver_i_others = torch.zeros_like(self.observations.past_vertices.get_latest())
            steering_agents = torch.zeros(
                (self.world.batch_dim, self.n_agents),
                device=self.world.device,
                dtype=torch.float32,
            )
            for a_i in range(self.n_agents):
                pos_i = self.world.agents[a_i].state.pos
                rot_i = self.world.agents[a_i].state.rot
                steering_agents[:, a_i] = self.world.agents[a_i].dynamics.cur_delta.squeeze(-1)
                pos_i_others[
                    :,
                    a_i,
                ] = transform_from_global_to_local_coordinate(
                    pos_i=pos_i,
                    pos_j=positions_global,
                    rot_i=rot_i,
                )
                rot_i_others[:, a_i] = angle_eliminate_two_pi(rotations_global - rot_i)
                for a_j in range(self.n_agents):
                    rot_rel = rot_i_others[:, a_i, a_j].unsqueeze(1)
                    vel_abs = torch.norm(self.world.agents[a_j].state.vel, dim=1).unsqueeze(1)
                    vel_i_others[
                        :,
                        a_i,
                        a_j,
                    ] = torch.hstack((vel_abs * torch.cos(rot_rel), vel_abs * torch.sin(rot_rel)))
                    ref_i_others[
                        :,
                        a_i,
                        a_j,
                        :,
                        0:2,
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.ref_paths_agent_related.short_term[:, a_j, :, 0:2],
                        rot_i=rot_i,
                    )
                    ref_i_others[
                        :,
                        a_i,
                        a_j,
                        :,
                        2,
                    ] = self.ref_paths_agent_related.short_term[:, a_j, :, 2]
                    l_b_i_others[
                        :,
                        a_i,
                        a_j,
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.ref_paths_agent_related.nearing_points_left_boundary[:, a_j],
                        rot_i=rot_i,
                    )
                    r_b_i_others[
                        :,
                        a_i,
                        a_j,
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.ref_paths_agent_related.nearing_points_right_boundary[:, a_j],
                        rot_i=rot_i,
                    )
                    ver_i_others[
                        :,
                        a_i,
                        a_j,
                    ] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=self.vertices[:, a_j, 0:4, :],
                        rot_i=rot_i,
                    )
            assert not torch.isnan(self.observations.platoon_error_vel).any()
            self.observations.past_pos.add(pos_i_others / self.normalizers.pos)
            self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
            self.observations.past_vel.add(vel_i_others / self.normalizers.v)
            self.observations.past_steering.add(steering_agents / self.normalizers.action_steering)
            self.observations.past_relative_ref_info.add(ref_i_others / torch.hstack((
                self.normalizers.pos,
                self.normalizers.v.unsqueeze(0),
            )))
            self.observations.past_left_boundary.add(l_b_i_others / self.normalizers.pos)
            self.observations.past_right_boundary.add(r_b_i_others / self.normalizers.pos)
            self.observations.past_vertices.add(ver_i_others / self.normalizers.pos)
            if agent.action.u is None:
                self.observations.past_action_acc.add(self.constants.empty_action_acc)
                self.observations.past_action_steering.add(self.constants.empty_action_steering)
            else:
                self.observations.past_action_acc.add(torch.stack(
                    [a.action.u[:, 0] for a in self.world.agents],
                    dim=1,
                ) / self.normalizers.action_acc)
                self.observations.past_action_steering.add(torch.stack(
                    [a.action.u[:, 1] for a in self.world.agents],
                    dim=1,
                ) / self.normalizers.action_steering)

    def _filter_obs_group_dict(
        self,
        obs_groups: List[Tuple[str, Optional[Tensor]]],
    ) -> Dict[str, Tensor]:
        return {name: tensor for (name, tensor) in obs_groups if tensor is not None}

    def _stack_history_obs_dicts(self, history_obs: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if not history_obs:
            return {}
        return {key: torch.stack(
            [obs_dict[key] for obs_dict in history_obs],
            dim=1,
        ) for key in history_obs[0].keys()}

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
        return {key: value + self.observations.noise_level * torch.rand_like(
            value,
            device=self.world.device,
            dtype=torch.float32,
        ) for (key, value) in obs_dict.items()}

    def _finalize_observation_dict(
        self,
        agent_index: int,
        obs_dict: Dict[str, Tensor],
        obs_groups: List[Tuple[str, Tensor]],
    ) -> Dict[str, Tensor]:
        if self.enable_obs_audit:
            self._maybe_print_obs_audit(
                agent_index,
                obs_groups + [('obs_total', self._flatten_obs_dict_for_audit(obs_dict))],
            )
        check_validity(self.observations)
        check_validity(self.ref_paths_agent_related)
        return self._add_noise_to_obs_dict(obs_dict)

    def _get_history_step_indices(self) -> List[int]:
        obs_len = min(self.history_obs_len, int(self.observations.n_observed_steps.item()))
        return list(range(1, obs_len + 1))

    def _observe_self_at_step(self, agent_index: int, step: int=1) -> Dict[str, Tensor]:
        batch_size = self.world.batch_dim
        indexing_tuple_vel = (self.constants.env_idx_broadcasting,) + (agent_index,) + (
            agent_index,
            0,
        )
        self_short_term = self.observations.past_relative_ref_info.get_latest(step)[
            :,
            agent_index,
            agent_index,
        ]
        self_left_boundary_pts = self.observations.past_left_boundary.get_latest(step)[
            :,
            agent_index,
            agent_index,
            1:,
            :,
        ]
        self_right_boundary_pts = self.observations.past_right_boundary.get_latest(step)[
            :,
            agent_index,
            agent_index,
            1:,
            :,
        ]
        self_left_dis = torch.linalg.norm(
            self_left_boundary_pts - self_short_term[..., :2],
            dim=-1,
        ).unsqueeze(-1)
        self_right_dis = torch.linalg.norm(
            self_right_boundary_pts - self_short_term[..., :2],
            dim=-1,
        ).unsqueeze(-1)
        vel = self.observations.past_vel.get_latest(step)[indexing_tuple_vel].reshape(batch_size, 1)
        vel_mag = torch.linalg.norm(vel, dim=-1, keepdim=True)
        return self._filter_obs_group_dict([
            ('self_vel', vel),
            ('self_speed', vel_mag),
            (
                'self_steering',
                self.observations.past_steering.get_latest(step)[
                    :,
                    agent_index,
                ].reshape(batch_size, 1),
            ),
            (
                'self_acc',
                self.observations.past_action_acc.get_latest(step)[
                    :,
                    agent_index,
                ].reshape(batch_size, 1),
            ),
            ('self_ref_velocity', self_short_term[..., 2:3]),
            ('self_ref_points', self_short_term[..., :2]),
            ('self_left_boundary_distance', self_left_dis),
            ('self_right_boundary_distance', self_right_dis),
            (
                'self_distance_to_ref',
                torch.linalg.norm(self_short_term[:, 0, :2], dim=-1, keepdim=True),
            ),
            (
                'self_distance_to_left_boundary',
                self.observations.past_distance_to_left_boundary.get_latest(step)[
                    :,
                    agent_index,
                ].reshape(batch_size, 1),
            ),
            (
                'self_distance_to_right_boundary',
                self.observations.past_distance_to_right_boundary.get_latest(step)[
                    :,
                    agent_index,
                ].reshape(batch_size, 1),
            ),
            (
                'self_platoon_error_vel',
                (self.observations.past_platoon_error_vel.get_latest(step)[
                    :,
                    agent_index,
                ] / self.normalizers.error_v).reshape(batch_size, 2),
            ),
            (
                'self_platoon_error_space',
                self.observations.self_platoon_error_space.get_latest(step)[
                    :,
                    agent_index,
                    :,
                ] / self.normalizers.error_pos,
            ),
        ])

    def observe_self(self, agent_index):
        return self._observe_self_at_step(agent_index, step=1)

    def observe_self_history(self, agent_index: int) -> Dict[str, Tensor]:
        step_indices = self._get_history_step_indices()
        obs_self_history = [self._observe_self_at_step(
            agent_index,
            step=step,
        ) for step in step_indices]
        obs_self_history_dict = self._stack_history_obs_dicts(obs_self_history)
        return obs_self_history_dict

    def _get_nearing_agents_indices(self, agent_index: int):
        (
            nearing_agents_distances,
            nearing_agents_indices,
        ) = torch.topk(
            self.distances.agents[:, agent_index],
            k=self.observations.n_nearing_agents,
            largest=False,
        )
        (nearing_agents_indices, sorted_pos) = torch.sort(nearing_agents_indices, dim=1)
        nearing_agents_distances = torch.gather(nearing_agents_distances, dim=1, index=sorted_pos)
        return (nearing_agents_distances, nearing_agents_indices)

    def _observe_other_agents_platoon_at_step(
        self,
        agent_index: int,
        *,
        step: int=1,
        nearing_agents_indices: Optional[Tensor]=None,
        relative_longitudinal_velocity: Optional[Tensor]=None,
        relative_acceleration: Optional[Tensor]=None,
    ) -> Dict[str, Tensor]:
        if nearing_agents_indices is None:
            (_, nearing_agents_indices) = self._get_nearing_agents_indices(agent_index)
        indexing_tuple_1 = (
            self.constants.env_idx_broadcasting,
        ) + (agent_index,) + (nearing_agents_indices,)
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
                    agent_index,
                    indexing_tuple_1,
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
        return self._filter_obs_group_dict([
            ('others_pos', obs_pos),
            ('others_rot', obs_rot),
            (
                'others_relative_longitudinal_velocity',
                relative_longitudinal_velocity / self.obs_relative_velocity_scale,
            ),
            (
                'others_relative_acceleration',
                relative_acceleration / self.obs_relative_acceleration_scale,
            ),
            ('others_distance', obs_distance),
        ])

    def observe_other_agents_history(self, agent_index: int) -> Dict[str, Tensor]:
        step_indices = self._get_history_step_indices()
        (_, nearing_agents_indices) = self._get_nearing_agents_indices(agent_index)
        indexing_tuple_1 = (
            self.constants.env_idx_broadcasting,
        ) + (agent_index,) + (nearing_agents_indices,)
        other_local_vel_history = torch.stack(
            [
                self.observations.past_vel.get_latest(step)[indexing_tuple_1]
                for step in step_indices
            ],
            dim=1,
        ) * self.normalizers.v
        ego_local_vel_history = torch.stack(
            [
                self.observations.past_vel.get_latest(step)[
                    :,
                    agent_index,
                    agent_index,
                ]
                for step in step_indices
            ],
            dim=1,
        ) * self.normalizers.v
        relative_longitudinal_velocity_history = other_local_vel_history[
            ...,
            0,
        ] - ego_local_vel_history[..., 0].unsqueeze(-1)
        relative_acceleration_history = torch.zeros_like(relative_longitudinal_velocity_history)
        if len(step_indices) > 1:
            relative_acceleration_history[
                :,
                :-1,
            ] = (relative_longitudinal_velocity_history[
                :,
                :-1,
            ] - relative_longitudinal_velocity_history[:, 1:]) / self.dt
        obs_other_history = []
        for (time_index, step) in enumerate(step_indices):
            obs_other_history.append(self._observe_other_agents_platoon_at_step(
                agent_index,
                step=step,
                nearing_agents_indices=nearing_agents_indices,
                relative_longitudinal_velocity=relative_longitudinal_velocity_history[
                    :,
                    time_index,
                ].unsqueeze(-1),
                relative_acceleration=relative_acceleration_history[:, time_index].unsqueeze(-1),
            ))
        obs_other_history_dict = self._stack_history_obs_dicts(obs_other_history)
        return obs_other_history_dict

    def observe_other_agents(self, agent_index):
        return self.observe_other_agents_platoon(agent_index)

    def observe_other_agents_platoon(self, agent_index):
        obs_other_agents = self._observe_other_agents_platoon_at_step(agent_index, step=1)
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
        return self._finalize_observation_dict(agent_index, obs, list(obs.items()))

    def _get_obs_audit_step(self) -> int:
        if hasattr(self, 'timer') and hasattr(self.timer, 'step'):
            return int(self.timer.step.max().item())
        return int(self.env_current_step.max().item())

    def _format_obs_audit_stats(self, obs_tensor: Tensor, previous_tensor: Optional[Tensor]):
        flat = obs_tensor.detach().reshape(-1).to(dtype=torch.float32)
        abs_flat = flat.abs()
        flat_cpu = flat.cpu()
        quantiles = torch.quantile(flat_cpu, torch.tensor([0.01, 0.5, 0.99], dtype=torch.float32))
        stats = {
            'dim': int(obs_tensor.shape[-1]),
            'mean': flat.mean().item(),
            'std': flat.std(unbiased=False).item() if flat.numel() > 1 else 0.0,
            'mean_abs': abs_flat.mean().item(),
            'max_abs': abs_flat.max().item(),
            'p01': quantiles[0].item(),
            'p50': quantiles[1].item(),
            'p99': quantiles[2].item(),
            'small_frac': (
                abs_flat < self.obs_audit_small_threshold,
            ).to(dtype=torch.float32).mean().item(),
            'large_frac': (
                abs_flat > self.obs_audit_large_threshold,
            ).to(dtype=torch.float32).mean().item(),
            'delta_std': float('nan'),
        }
        if previous_tensor is not None and previous_tensor.shape == obs_tensor.shape:
            delta = (obs_tensor.detach() - previous_tensor).reshape(-1).to(dtype=torch.float32)
            stats['delta_std'] = delta.std(unbiased=False).item() if delta.numel() > 1 else 0.0
        return stats

    def _maybe_print_obs_audit(
        self,
        agent_index: int,
        observation_groups: List[Tuple[str, Tensor]],
    ):
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
                f'\n[OBS_AUDIT] step={current_step} agent={agent_index} '
                f'small<{self.obs_audit_small_threshold:g} '
                f'large>{self.obs_audit_large_threshold:g}',
            )
            for (name, tensor) in observation_groups:
                previous_tensor = self.obs_audit_prev_groups.get(name)
                stats = self._format_obs_audit_stats(tensor, previous_tensor)
                print(
                    f"  - {name:30s} d={stats['dim']:3d} "
                    f"mu={stats['mean']:+.2e} sd={stats['std']:.2e} "
                    f"q=[{stats['p01']:+.2e}|{stats['p50']:+.2e}|"
                    f"{stats['p99']:+.2e}] s={stats['small_frac']:.1%} "
                    f"l={stats['large_frac']:.1%} "
                    f"ds={stats['delta_std']:.2e}",
                )
            print('[OBS_AUDIT] end\n')
            self.obs_audit_last_logged_step = current_step
        for (name, tensor) in observation_groups:
            self.obs_audit_prev_groups[name] = tensor.detach().clone()

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
                self.observations.past_vel.get_latest(i + 1)[
                    :,
                    agent_index,
                    agent_index,
                ]
                for i in range(n_observed_steps)
            ],
            dim=-1,
        ) * self.normalizers.v
        return other_local_vel_history[..., 0, :] - ego_local_vel_history[:, None, 0, :]

    def clamp_error_reward(
        self,
        weight,
        error,
        offset: float=1.0,
        norm: int=2,
        max: float=1.0,
        min: float=None,
    ):
        return offset - torch.clamp(weight * error ** norm, max=max, min=min)

    def _reward_agent_heading(self, ref_points_vecs, move_vec):
        ref_vector = torch.mean(ref_points_vecs, dim=1)
        ref_vector_normalized = ref_vector / (torch.norm(ref_vector, dim=-1, keepdim=True) + 1e-08)
        move_vector = move_vec[:, 0, :]
        move_vector_normalized = move_vector / (torch.norm(
            move_vector,
            dim=-1,
            keepdim=True,
        ) + 1e-08)
        max_delta_angle = torch.deg2rad(torch.tensor(15, device=self.device, dtype=torch.float32))
        constant_k = 1 / (1 - torch.cos(max_delta_angle))
        costant_b = 1 - constant_k
        heading_alignment = torch.clamp(
            constant_k * torch.sum(
                ref_vector_normalized * move_vector_normalized,
                dim=-1,
            ) + costant_b,
            min=0.0,
            max=1.0,
        )
        reward_platoon_heading = self.clamp_error_reward(
            self.rewards.reward_platoon_heading,
            1 - heading_alignment,
            norm=1,
        )
        return reward_platoon_heading

    def reward_simple_platoon(self, reward_details, agent_index, ref_points_vecs, move_vec):
        reward_goal = (
            self.collisions.with_exit_segments[:, agent_index].to(torch.float32)
            * self.rewards.reward_goal
        )
        reward_details['reward_goal'][:, agent_index] = reward_goal

        ref_vector = torch.mean(ref_points_vecs, dim=1)
        ref_vector_normalized = ref_vector / (
            torch.norm(ref_vector, dim=-1, keepdim=True) + 1e-8
        )
        move_vector = move_vec[:, 0, :]
        move_vector_normalized = move_vector / (
            torch.norm(move_vector, dim=-1, keepdim=True) + 1e-8
        )

        max_delta_angle = torch.deg2rad(
            torch.tensor(15, device=self.device, dtype=torch.float32)
        )
        constant_k = 1 / (1 - torch.cos(max_delta_angle))
        constant_b = 1 - constant_k
        reward_platoon_heading = torch.clamp(
            self.rewards.reward_platoon_heading
            * (
                constant_k
                * torch.sum(
                    ref_vector_normalized * move_vector_normalized,
                    dim=-1,
                )
                + constant_b
            ),
            min=-1.0,
            max=1.0,
        )
        reward_details['reward_platoon_heading'][:, agent_index] = (
            reward_platoon_heading
        )

        agent_raw_vel = torch.linalg.norm(
            self.world.agents[agent_index].state.vel,
            dim=-1,
        )
        agent_vel = agent_raw_vel * torch.sum(
            ref_vector_normalized * move_vector_normalized,
            dim=-1,
        )
        ref_vel = self.ref_paths_agent_related.short_term[:, agent_index, 0, 2]
        error_vel = agent_vel - ref_vel
        reward_platoon_vel = torch.clamp(
            1 - self.rewards.reward_platoon_vel * error_vel**2,
            min=0,
        )
        reward_details['reward_platoon_vel'][:, agent_index] = reward_platoon_vel

        space_errors = self.observations.self_platoon_error_space.get_latest(n=1)[
            :, agent_index, 0
        ]
        reward_platoon_space = torch.clamp(
            1 - self.rewards.reward_platoon_space * space_errors**2,
            min=0,
        )
        reward_details['reward_platoon_space'][:, agent_index] = reward_platoon_space

        ratio = 0.7
        if self.distances.lookahead_pts.shape[-1] > 1:
            weighted_ref_dis = (
                ratio * self.distances.lookahead_pts[:, agent_index, 0]
                + (1 - ratio) * self.distances.lookahead_pts[:, agent_index, 1]
            )
        else:
            weighted_ref_dis = self.distances.lookahead_pts[:, agent_index, 0]
        reward_platoon_ref = 1 - self.rewards.reward_platoon_ref * weighted_ref_dis**2
        reward_details['reward_platoon_ref'][:, agent_index] = reward_platoon_ref

        return reward_details

    def reward(self, agent: Agent):
        agent_index = self.world.agents.index(agent)
        if agent_index == 0:
            self.env_current_step += 1
        reward_details = self.reward_details
        self.rew[:] = 0
        t0 = time.time()
        self.update_state_before_rewarding(agent, agent_index)
        t1 = time.time()
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :],
            x0=self.thresholds.near_other_agents_low,
            x1=self.thresholds.near_other_agents_high,
        )
        penalty_near_other_agents = torch.sum(
            mutual_distance_exp_fcn,
            dim=1,
        ) * self.penalties.near_other_agents
        reward_details['penalty_near_other_agents'][:, agent_index] = penalty_near_other_agents
        steering_current = self.observations.past_action_steering.get_latest(n=1)[:, agent_index]
        steering_past = self.observations.past_action_steering.get_latest(n=2)[:, agent_index]
        steering_change = torch.clamp(
            (
                steering_current - steering_past
            ).abs() * self.normalizers.action_steering - self.thresholds.change_steering,
            min=0,
        )
        if self.observations.past_action_steering.valid_size == self.observations.n_stored_steps:
            penalty_change_steering = (steering_change / torch.deg2rad(torch.tensor(
                3,
                device=self.device,
            ))) ** 2 * self.penalties.change_steering
            penalty_change_steering = torch.clamp(penalty_change_steering, min=-5, max=0)
        else:
            penalty_change_steering = 0.0
        reward_details['penalty_change_steering'][:, agent_index] = penalty_change_steering
        acc_current = self.observations.past_action_acc.get_latest(n=1)[:, agent_index]
        acc_past = self.observations.past_action_acc.get_latest(n=2)[:, agent_index]
        acc_change = torch.clamp(
            (
                acc_current - acc_past
            ).abs() * self.normalizers.action_acc - self.thresholds.change_acc,
            min=0,
        )
        acc_nor = 0.1
        if self.observations.past_action_acc.valid_size == self.observations.n_stored_steps:
            penalty_change_acc = (acc_change / acc_nor) ** 2 * self.penalties.change_acc
            penalty_change_acc = torch.clamp(penalty_change_acc, min=-5, max=0)
        else:
            penalty_change_acc = 0.0
        reward_details['penalty_change_acc'][:, agent_index] = penalty_change_acc
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]
        penalty_collide_with_agents = is_collide_with_agents.any(
            dim=-1,
        ) * self.penalties.collide_with_agents
        reward_details['penalty_collide_with_agents'][:, agent_index] = penalty_collide_with_agents
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_outside_boundaries = (
            is_collide_with_lanelets * self.penalties.collide_with_boundaries
        )
        reward_details['penalty_outside_boundaries'][:, agent_index] = penalty_outside_boundaries
        current_lane_width = torch.linalg.norm(
            self.ref_paths_agent_related.nearing_points_left_boundary[
                :,
                agent_index,
                1,
            ] - self.ref_paths_agent_related.nearing_points_right_boundary[:, agent_index, 1],
            dim=-1,
        )
        penalty_near_boundary = torch.max(
            exponential_decreasing_fcn(
                x=self.distances.boundaries[:, agent_index] / current_lane_width,
                x0=self.thresholds.near_boundary_low,
                x1=self.thresholds.near_boundary_high,
            ),
            is_collide_with_lanelets.float(),
        ) * self.penalties.near_boundary
        reward_details['penalty_near_boundary'][:, agent_index] = penalty_near_boundary
        ref_points_vecs = self.ref_paths_agent_related.short_term[
            :,
            agent_index,
            1:,
            0:2,
        ] - self.ref_paths_agent_related.short_term[:, agent_index, :-1, 0:2]
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(-1)
        backward_penalty = torch.where(v_proj <= 0, 1, 0) * self.penalties.backward
        reward_details['penalty_backward'][:, agent_index] = backward_penalty
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(1)
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(
            move_projected,
            self.rewards.weighting_ref_directions,
        )
        reward_progress = move_projected_weighted / (
            agent.max_speed * self.world.dt
        ) * self.rewards.reward_progress
        reward_details['reward_progress'][:, agent_index] = reward_progress
        reward_vel = v_proj / agent.max_speed * self.rewards.reward_vel
        reward_details['reward_vel'][:, agent_index] = reward_vel
        reward_details = self.reward_simple_platoon(
            reward_details=reward_details,
            agent_index=agent_index,
            ref_points_vecs=ref_points_vecs,
            move_vec=move_vec,
        )
        t2 = time.time()
        self.update_state_after_rewarding(agent_index)
        t3 = time.time()
        for r in reward_details.keys():
            if r != 'reward_total':
                self.rew += reward_details[r][:, agent_index]
        reward_details['reward_total'][:, agent_index] = self.rew
        self.reward_update_time += t3 - t0
        return self.rew

    def update_state_before_rewarding(self, agent, agent_index):
        """Update some states (such as mutual distances between agents, vertices of each agent, and
        collision matrices) that will be used before rewarding agents.
        """
        if agent_index == 0:
            self.timer.step_begin = time.time()
            self.timer.step += 1
            assert torch.isnan(agent.state.pos).any() == False, f'agent {agent_index} pos is nan'
            self.distances.agents = get_distances_between_agents(self=self, is_set_diagonal=True)
            self.distances.agents_frenet = get_frenet_distances_between_agents(
                self.observations.agent_s,
            )
            self.collisions.with_agents[:] = False
            self.collisions.with_lanelets[:] = False
            self.collisions.with_exit_segments[:] = False
            for a_i in range(self.n_agents):
                self.vertices[
                    :,
                    a_i,
                ] = get_rectangle_vertices(
                    center=self.world.agents[a_i].state.pos,
                    yaw=self.world.agents[a_i].state.rot,
                    width=self.world.agents[a_i].shape.width,
                    length=self.world.agents[a_i].shape.length,
                    is_close_shape=True,
                )
                for a_j in range(a_i + 1, self.n_agents):
                    collision_batch_index = interX(
                        self.vertices[:, a_i],
                        self.vertices[:, a_j],
                        False,
                    )
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index),
                        a_i,
                        a_j,
                    ] = True
                    self.collisions.with_agents[
                        torch.nonzero(collision_batch_index),
                        a_j,
                        a_i,
                    ] = True
                if not self.is_loop:
                    self.collisions.with_exit_segments[
                        :,
                        a_i,
                    ] = interX(
                        L1=self.vertices[:, a_i],
                        L2=self.ref_paths_agent_related.exit[:, a_i],
                        is_return_points=False,
                    )
                collision_with_left_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.left_boundary[:, a_i],
                    is_return_points=False,
                ).to(self.device)
                collision_with_right_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                    is_return_points=False,
                ).to(self.device)
                is_left_outside_boundary = is_point_left_of_polyline(
                    point=self.world.agents[a_i].state.pos,
                    polyline=self.ref_paths_agent_related.nearing_points_left_boundary[:, a_i],
                ).to(self.device)
                is_right_outside_boundary = ~is_point_left_of_polyline(
                    point=self.world.agents[a_i].state.pos,
                    polyline=self.ref_paths_agent_related.nearing_points_right_boundary[:, a_i],
                ).to(self.device)
                self.collisions.with_lanelets[
                    collision_with_left_boundary | is_left_outside_boundary | (
                        collision_with_right_boundary | is_right_outside_boundary
                    ),
                    a_i,
                ] = True
                assert self.use_center_frenet_ref, 'use_center_frenet_ref must be True'
                self.ref_paths_agent_related.short_term[
                    :,
                    a_i,
                ] = get_short_term_reference_path_by_s(
                    self.road,
                    self.observations.agent_s[:, a_i],
                    n_points_to_return=self.n_points_short_term,
                    device=self.world.device,
                    sample_interval=self.sample_interval,
                    return_ref_v=True,
                    line='center',
                )
                if a_i != 0:
                    self.ref_paths_agent_related.short_term[
                        :,
                        a_i,
                        :,
                        -1,
                    ] = self.ref_paths_agent_related.short_term[:, 0, :, -1]
                if self.use_boundary_frenet_ref:
                    self.ref_paths_agent_related.nearing_points_left_boundary[
                        :,
                        a_i,
                    ] = get_short_term_reference_path_by_s(
                        self.road,
                        self.observations.agent_s[:, a_i] + self.boundary_offset,
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        return_ref_v=False,
                        line='left',
                    )
                    self.ref_paths_agent_related.nearing_points_right_boundary[
                        :,
                        a_i,
                    ] = get_short_term_reference_path_by_s(
                        self.road,
                        self.observations.agent_s[:, a_i] + self.boundary_offset,
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        return_ref_v=False,
                        line='right',
                    )
                else:
                    (
                        self.ref_paths_agent_related.nearing_points_left_boundary[:, a_i],
                        _,
                    ) = get_short_term_reference_path_simple(
                        polyline=self.ref_paths_agent_related.left_boundary[:, a_i],
                        index_closest_point=self.distances.closest_point_on_left_b[:, a_i],
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        n_points_shift=1,
                    )
                    (
                        self.ref_paths_agent_related.nearing_points_right_boundary[:, a_i],
                        _,
                    ) = get_short_term_reference_path_simple(
                        polyline=self.ref_paths_agent_related.right_boundary[:, a_i],
                        index_closest_point=self.distances.closest_point_on_right_b[:, a_i],
                        n_points_to_return=self.n_points_nearing_boundary,
                        device=self.world.device,
                        sample_interval=self.sample_interval,
                        n_points_shift=1,
                    )
        (
            self.distances.ref_paths[:, agent_index],
            self.distances.closest_point_on_ref_path[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos,
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            n_points_long_term=None,
        )
        (
            center_2_left_b,
            self.distances.closest_point_on_left_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
            n_points_long_term=None,
        )
        self.distances.left_boundaries[:, agent_index, 0] = center_2_left_b - agent.shape.width / 2
        (
            center_2_right_b,
            self.distances.closest_point_on_right_b[:, agent_index],
        ) = get_perpendicular_distances(
            point=agent.state.pos[:, :],
            polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
            n_points_long_term=None,
        )
        self.distances.right_boundaries[
            :,
            agent_index,
            0,
        ] = center_2_right_b - agent.shape.width / 2
        for c_i in range(4):
            (
                self.distances.left_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                n_points_long_term=None,
            )
            (
                self.distances.right_boundaries[:, agent_index, c_i + 1],
                _,
            ) = get_perpendicular_distances(
                point=self.vertices[:, agent_index, c_i, :],
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                n_points_long_term=None,
            )
        (
            self.distances.boundaries[:, agent_index],
            _,
        ) = torch.min(
            torch.hstack((
                self.distances.left_boundaries[:, agent_index],
                self.distances.right_boundaries[:, agent_index],
            )),
            dim=-1,
        )
        for idx in range(self.agent_lookahead_idx):
            if idx == 0:
                lookahead_pts = agent.state.pos
            else:
                lookahead_pts = agent.state.pos + idx * self.sample_interval * torch.hstack([
                    torch.cos(agent.state.rot),
                    torch.sin(agent.state.rot),
                ])
            self.distances.lookahead_pts[
                :,
                agent_index,
                idx,
            ] = torch.linalg.norm(
                self.ref_paths_agent_related.short_term[:, agent_index, idx, :2] - lookahead_pts,
                dim=-1,
            )

    def compute_lookahead_kinematics(self, agent, delta, dist_travelled):
        """
        Predict the vehicle position with a constant-speed arc model.

        delta: ``agent.action[:, 1]`` in radians.
        dt: prediction horizon, usually ``sample_dt * idx``.
        """
        theta = agent.state.rot.squeeze(-1)
        L = agent.dynamics.l_f + agent.dynamics.l_r
        kappa = torch.tan(delta) / L
        delta_theta = dist_travelled * kappa
        is_straight = torch.abs(kappa) < 0.0001
        inv_kappa = 1.0 / (kappa + 1e-08)
        lookahead_pos_curve = agent.state.pos + inv_kappa.unsqueeze(-1) * torch.stack(
            [
                torch.sin(theta + delta_theta) - torch.sin(theta),
                -(torch.cos(theta + delta_theta) - torch.cos(theta)),
            ],
            dim=-1,
        )
        lookahead_pos_straight = agent.state.pos + dist_travelled.unsqueeze(-1) * torch.stack(
            [torch.cos(theta), torch.sin(theta)],
            dim=-1,
        )
        lookahead_pts = torch.where(
            is_straight.unsqueeze(-1),
            lookahead_pos_straight,
            lookahead_pos_curve,
        )
        return lookahead_pts

    def update_state_after_rewarding(self, agent_index):
        """Update cached state after rewarding all agents in the environment."""
        if agent_index == self.n_agents - 1:
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
            self.world.batch_dim,
            -1,
        ).any(dim=-1)
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        is_collision_with_exit_segments = self.collisions.with_exit_segments.any(dim=-1)
        is_done = (
            is_collision_with_agents_env
            | is_collision_with_exit_segments
            | is_collision_with_lanelets
        )
        is_success = is_collision_with_exit_segments & ~(
            is_collision_with_agents_env | is_collision_with_lanelets
        )
        is_failure = is_done & ~is_success
        return {
            'is_done': is_done,
            'is_success': is_success,
            'is_failure': is_failure,
            'is_collision_with_agents_env': is_collision_with_agents_env,
            'is_collision_with_lanelets': is_collision_with_lanelets,
            'is_collision_with_exit_segments': is_collision_with_exit_segments,
        }

    def done(self):
        """
        This function computes the done flag for each env in a vectorized way.
        """
        done_status = self._get_done_status()
        is_done = done_status['is_done']
        self._record_failure_curriculum_events(done_status)
        self.success_count += done_status['is_success'].float()
        self.failure_count += done_status['is_failure'].float()
        self.env_total_step[is_done] = self.env_current_step[is_done]
        self.env_current_step[is_done] = 0
        return is_done

    def get_lookahead_agent_pos(self, agent_index, lookahead_idx=0):
        """
        Get the current agent position of the agent.
        """
        current_pos = self.world.agents[agent_index].state.pos
        theta = self.world.agents[agent_index].state.rot
        lookahead_pts = current_pos + lookahead_idx * self.sample_interval * torch.hstack([
            torch.cos(theta),
            torch.sin(theta),
        ])
        return lookahead_pts

    def _compute_agent_command_acceleration(self, agent_index: int) -> Tensor:
        if self.observations.past_action_acc.valid_size == 0:
            return torch.zeros(self.batch_dim, device=self.device, dtype=torch.float32)
        return (self.observations.past_action_acc.get_latest(n=1)[
            :,
            agent_index,
        ] * self.normalizers.action_acc).to(torch.float32)

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
        steering_current = self.observations.past_action_steering.get_latest(n=1)[
            :,
            agent_index,
        ] * self.normalizers.action_steering
        steering_previous = self.observations.past_action_steering.get_latest(n=2)[
            :,
            agent_index,
        ] * self.normalizers.action_steering
        delta = steering_current - steering_previous
        return (delta * (180.0 / torch.pi) / self.dt).to(torch.float32)

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        agent_index = self.world.agents.index(agent)
        is_action_empty = agent.action.u is None
        done_status = self._get_done_status()
        total_finished = self.success_count + self.failure_count
        running_success_rate = self.success_count / total_finished.clamp_min(1.0)
        is_collision_with_agents = self.collisions.with_agents[:, agent_index].any(dim=-1)
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        agent_error_space = self.observations.self_platoon_error_space.get_latest()[
            :,
            agent_index,
            :,
        ]
        command_acceleration = self._compute_agent_command_acceleration(agent_index)
        command_jerk = self._compute_agent_command_jerk(agent_index)
        steering_rate_deg = self._compute_agent_steering_rate_deg(agent_index)
        agent_reward_details = {}
        for (reward_name, reward_tensor) in self.reward_details.items():
            agent_reward_details[reward_name] = reward_tensor[:, agent_index]
        info = {
            'pos': agent.state.pos,
            's': self.observations.agent_s[:, agent_index],
            'rot': angle_eliminate_two_pi(agent.state.rot),
            'vel': agent.state.vel,
            'vel_norm': torch.norm(agent.state.vel, dim=-1),
            'act_acc': agent.action.u[
                :,
                0,
            ] if self.traditional_control == MethodClass.MARL and (
                not is_action_empty
            ) else self.logged_control_acc[
                :,
                agent_index,
            ],
            'act_steer': agent.action.u[
                :,
                1,
            ] if self.traditional_control == MethodClass.MARL and (
                not is_action_empty
            ) else self.logged_control_steer[
                :,
                agent_index,
            ],
            'command_acceleration': command_acceleration,
            'command_acceleration_abs': command_acceleration.abs(),
            'command_jerk': command_jerk,
            'command_jerk_abs': command_jerk.abs(),
            'steering_rate_deg': steering_rate_deg,
            'steering_rate_abs_deg': steering_rate_deg.abs(),
            'distance_ref': self.distances.ref_paths[:, agent_index],
            'distance_lookahead_pts': torch.mean(
                self.distances.lookahead_pts[:, agent_index],
                dim=-1,
            ),
            'distance_left_b': self.distances.left_boundaries[:, agent_index].min(dim=-1)[0],
            'distance_right_b': self.distances.right_boundaries[:, agent_index].min(dim=-1)[0],
            'is_collision_with_agents': is_collision_with_agents,
            'is_collision_with_lanelets': is_collision_with_lanelets,
            'mean_error_space': agent_error_space.mean(-1),
            'error_space': agent_error_space,
            'platoon_error_vel': self.observations.platoon_error_vel[:, agent_index],
            'ref_vel': self.ref_paths_agent_related.short_term[:, agent_index, 0, 2],
            'episode_done': done_status['is_done'].float(),
            'episode_success': done_status['is_success'].float(),
            'episode_failure': done_status['is_failure'].float(),
            'episode_replay_source': self.current_episode_replay_source.float(),
            'episode_replay_entry_id': self.current_episode_replay_entry_id.to(torch.float32),
            'done_collision_with_agents': done_status['is_collision_with_agents_env'].float(),
            'done_collision_with_lanelets': done_status['is_collision_with_lanelets'].float(),
            'done_collision_with_exit_segments': done_status[
                'is_collision_with_exit_segments'
            ].float(),
            'scenario_success_count': self.success_count,
            'scenario_failure_count': self.failure_count,
            'running_scenario_success_rate': running_success_rate,
            'env_total_step': self.env_total_step,
            'road_batch_id': self.road.batch_id,
            **agent_reward_details,
        }
        return info

    def extra_render(self, env_index: int=0):
        from vmas.simulator import rendering
        if self.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)
            self.timer.render_begin = time.time()
        geoms = []
        map_geoms = self.extra_render_map(env_index)
        geoms.extend(map_geoms)
        extend_road_polygons = self.extra_render_extend_road(env_index)
        geoms.extend(extend_road_polygons)
        if hasattr(self, 'road'):
            s_max_idx = self.road.get_s_max_idx(env_index)
            center_pts = self.road.get_road_center_pts()[env_index]
            center_pts = center_pts[:s_max_idx + 1]
            geom = rendering.PolyLine(
                v=[(float(x), float(y)) for (x, y) in center_pts.detach().cpu().tolist()],
                close=False,
            )
            geom.set_color(*Color.PURPLE.value, alpha=1.0)
            geom.set_linewidth(3.0)
            geoms.append(geom)
            left_pts = self.road.get_road_left_pts()[env_index]
            left_pts = left_pts[:s_max_idx + 1]
            geom = rendering.PolyLine(
                v=[(float(x), float(y)) for (x, y) in left_pts.detach().cpu().tolist()],
                close=False,
            )
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(1.0)
            geoms.append(geom)
            right_pts = self.road.get_road_right_pts()[env_index]
            right_pts = right_pts[:s_max_idx + 1]
            geom = rendering.PolyLine(
                v=[(float(x), float(y)) for (x, y) in right_pts.detach().cpu().tolist()],
                close=False,
            )
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(1.0)
            geoms.append(geom)
        pos_origin = self.world.agents[self.agent_index_focus].state.pos[env_index, :]
        last_state = self.state_buffer.get_latest(n=2)[env_index, :, :]
        for (agent_i, ag) in enumerate(self.world.agents):
            pos = ag.state.pos[env_index].detach().cpu().tolist()
            v = torch.linalg.norm(ag.state.vel[env_index]).detach().cpu()
            last_v = torch.linalg.norm(last_state[agent_i, 3:5]).detach().cpu()
            acc = (v - last_v) / self.dt
            space_errors = torch.abs(self.observations.self_platoon_error_space.get_latest(n=1)[
                env_index,
                agent_i,
                :,
            ]).detach().cpu()
            front_space_errors = space_errors[0]
            rear_space_errors = space_errors[1]
            geom = rendering.TextLine(
                text=f'a{agent_i}:[v:{v:.1f},a:{acc:.1f}]',
                x=4 * (pos[0] - pos_origin[0]) * self.resolution_factor + self.viewer_size[0] / 2,
                y=4 * (
                    pos[1] - pos_origin[1]
                ) * self.resolution_factor + self.viewer_size[
                    1
                ] / 2 + 2.2 * 5.2 * self.resolution_factor,
                font_size=int(2 * self.resolution_factor),
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
            geom = rendering.TextLine(
                text=f'[gap:({front_space_errors:.1f},{rear_space_errors:.1f})]',
                x=4 * (pos[0] - pos_origin[0]) * self.resolution_factor + self.viewer_size[0] / 2,
                y=4 * (
                    pos[1] - pos_origin[1]
                ) * self.resolution_factor + self.viewer_size[
                    1
                ] / 2 + 2.2 * 4 * self.resolution_factor,
                font_size=int(2 * self.resolution_factor),
            )
            xform = rendering.Transform()
            geom.add_attr(xform)
            geoms.append(geom)
            if hasattr(self, 'ref_paths_agent_related'):
                if hasattr(self.ref_paths_agent_related, 'short_term'):
                    short_term_path = self.ref_paths_agent_related.short_term[env_index, agent_i]
                    geom = rendering.PolyLine(
                        v=[(
                            float(p[0]),
                            float(p[1]),
                        ) for p in short_term_path.detach().cpu().tolist()],
                        close=False,
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
            if self.traditional_control == MethodClass.MPPI and self.enable_mppi_debug_render and (
                self.simple_mppi is not None,
            ) and (agent_i == self.agent_index_focus):
                mppi_debug = self.simple_mppi.last_debug.get(agent_i)
                if mppi_debug is not None:
                    ref_points = mppi_debug['ref_points']
                    sampled_trajs = mppi_debug['sampled_trajs']
                    optimal_traj = mppi_debug['optimal_traj']
                    for traj in sampled_trajs:
                        geom = rendering.PolyLine(
                            v=[(
                                float(p[0]),
                                float(p[1]),
                            ) for p in traj[:, :2].detach().cpu().tolist()],
                            close=False,
                        )
                        xform = rendering.Transform()
                        geom.add_attr(xform)
                        geom.set_color(0.1, 0.7, 0.95, alpha=0.12)
                        geom.set_linewidth(1.0)
                        geoms.append(geom)
                    geom = rendering.PolyLine(
                        v=[(float(p[0]), float(p[1])) for p in ref_points.detach().cpu().tolist()],
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
                        v=[(
                            float(p[0]),
                            float(p[1]),
                        ) for p in optimal_traj[:, :2].detach().cpu().tolist()],
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
            if hasattr(self, 'ref_paths_agent_related'):
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_left_boundary[env_index, agent_i],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_linewidth(2)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)
                for i_p in self.ref_paths_agent_related.nearing_points_left_boundary[
                    env_index,
                    agent_i,
                ]:
                    circle = rendering.make_circle(radius=0.2, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)
                geom = rendering.PolyLine(
                    v=self.ref_paths_agent_related.nearing_points_right_boundary[
                        env_index,
                        agent_i,
                    ],
                    close=False,
                )
                xform = rendering.Transform()
                geom.add_attr(xform)
                geom.set_linewidth(2)
                geom.set_color(*self.world.agents[agent_i].color)
                geoms.append(geom)
                for i_p in self.ref_paths_agent_related.nearing_points_right_boundary[
                    env_index,
                    agent_i,
                ]:
                    circle = rendering.make_circle(radius=0.2, filled=True)
                    xform = rendering.Transform()
                    circle.add_attr(xform)
                    xform.set_translation(i_p[0], i_p[1])
                    circle.set_color(*self.world.agents[agent_i].color)
                    geoms.append(circle)
        return geoms

    def extra_render_extend_road(self, env_index: int=0):
        """
        Render the extended road area with gray fill.
        """
        left_pts1 = self.road.get_pts(torch.tensor(0, device=self.device), env_index, 'left')
        left_pts2 = self.road.get_pts(
            torch.tensor(self.rod_len + 1, device=self.device),
            env_index,
            'left',
        )
        right_pts1 = self.road.get_pts(torch.tensor(0, device=self.device), env_index, 'right')
        right_pts2 = self.road.get_pts(
            torch.tensor(self.rod_len + 1, device=self.device),
            env_index,
            'right',
        )
        extend_road_pts = [left_pts1, left_pts2, right_pts2, right_pts1]
        extend_road_polygon1 = rendering.make_polygon(extend_road_pts, draw_border=False)
        extend_road_polygon1.set_color(0.7, 0.7, 0.7, alpha=1.0)
        s_max = self.road.get_s_max()[env_index]
        left_pts3 = self.road.get_pts(s_max - self.rod_len - 1, env_index, 'left')
        left_pts4 = self.road.get_pts(s_max, env_index, 'left')
        right_pts3 = self.road.get_pts(s_max - self.rod_len - 1, env_index, 'right')
        right_pts4 = self.road.get_pts(s_max, env_index, 'right')
        extend_road_pts = [left_pts3, left_pts4, right_pts4, right_pts3]
        extend_road_polygon2 = rendering.make_polygon(extend_road_pts, draw_border=False)
        extend_road_polygon2.set_color(0.7, 0.7, 0.7, alpha=1.0)
        return [extend_road_polygon1, extend_road_polygon2]

    def extra_render_map(self, env_index: int=0):
        """
        Render the road map:
        1) road center line (thin black line)
        2) left and right road boundaries (black lines)
        3) polygon area enclosed by the boundaries (gray fill)
        """
        geoms = []
        try:
            scenario = self.road.get_scenario_by_env_index(env_index)
        except:
            return geoms
        lanelets = scenario.lanelet_network.lanelets
        for lanelet in lanelets:
            left_vertices = lanelet.left_vertices
            right_vertices = lanelet.right_vertices
            if left_vertices is not None and right_vertices is not None:
                SEGMENT_VERTEX_COUNT = 3
                n_left = len(left_vertices)
                n_right = len(right_vertices)
                n_vertices = min(n_left, n_right)
                for i in range(0, n_vertices - 1, SEGMENT_VERTEX_COUNT):
                    end_idx = min(i + SEGMENT_VERTEX_COUNT, n_vertices - 1)
                    segment_pts = []
                    for j in range(i, end_idx + 1):
                        (x, y) = left_vertices[j]
                        segment_pts.append((float(x), float(y)))
                    for j in range(end_idx, i - 1, -1):
                        (x, y) = right_vertices[j]
                        segment_pts.append((float(x), float(y)))
                    road_polygon = rendering.make_polygon(segment_pts, draw_border=False)
                    road_polygon.set_color(0.7, 0.7, 0.7, alpha=1.0)
                    geoms.append(road_polygon)
            center_vertices = lanelet.center_vertices
            if center_vertices is not None:
                center_line = rendering.PolyLine(
                    v=[(float(x), float(y)) for (x, y) in center_vertices],
                    close=False,
                )
                center_line.set_color(*Color.BLACK.value, alpha=1.0)
                center_line.set_linewidth(1.0)
                center_line.add_attr(rendering.LineStyle(255))
                geoms.append(center_line)
            if left_vertices is not None:
                left_line = rendering.PolyLine(
                    v=[(float(x), float(y)) for (x, y) in left_vertices],
                    close=False,
                )
                left_line.set_color(*Color.BLACK.value, alpha=1.0)
                left_line.set_linewidth(2.0)
                geoms.append(left_line)
            if right_vertices is not None:
                right_line = rendering.PolyLine(
                    v=[(float(x), float(y)) for (x, y) in right_vertices],
                    close=False,
                )
                right_line.set_color(*Color.BLACK.value, alpha=1.0)
                right_line.set_linewidth(2.0)
                geoms.append(right_line)
        return geoms
if __name__ == '__main__':
    render_interactively(
        __file__,
        control_two_agents=True,
        display_info=False,
        seed=None,
        agent_index_focus=AGENT_INDEX_FOCUS,
    )
