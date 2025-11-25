# oversize_transport.py
# VMAS scenario skeleton for "oversize cargo with front/rear tractors + follower vehicles"
# 关键特性：
# - 前/后牵引车：脚本体，不在 policy_agents；受"端点贴合道路 + 固定弦长"约束
# - 随动车辆：policy agents；支持 dock/free 状态机；docked 时与货物锚点刚性/软性约束
# - 并行批量计算（num_envs = batch_dim），核心几何/求解均为向量化张量运算

import time
from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor

from vmas import render_interactively
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import World, Agent, Sphere, Box  # 按你的工程结构调整
from vmas.simulator.utils import Color
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle   # 视你的动力学模块而定
from vmas.simulator import rendering
from vmas.simulator.utils import Color, ScenarioUtils

from vmas.scenarios.road_traffic import get_perpendicular_distances,get_distances_between_agents,get_rectangle_vertices,\
    transform_from_global_to_local_coordinate,interX,exponential_decreasing_fcn,\
    Rewards,Penalties,Collisions,Distances,Constants,CircularBuffer,Normalizers,Timer,Thresholds,StateBuffer

X, Y = 0, 1  # 方便阅读\

def get_short_term_reference_path_simple(
    polyline: torch.Tensor,
    index_closest_point: torch.Tensor,
    n_points_to_return: int,
    device=None,
    sample_interval: int = 2,
    n_points_shift: int = 1,
):
    """
    Args:
        polyline:                   [batch_size, num_points, 2] or [num_points, 2]. In the case of the latter, batch_dim is deemed as 1.
        index_closest_point:        [batch_size, 1] or [1] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_to_return:         [1] or []. In the case of the latter, batch_dim is deemed as 1.
        sample_interval:            Sample interval to match specific purposes;
                                    set to 2 when using this function to get the short-term reference path;
                                    set to 1 when using this function to get the nearing boundary points.
        n_points_shift:             Number of points to be shifted to match specific purposes;
                                    set to 1 when using this function to get the short-term reference path to "force" the first point of the short-term reference path being in front of the agent;
                                    set to -2 when using this function to get the nearing boundary points to consider the points behind the agent.
    """
    if polyline.ndim == 2:
        polyline = polyline.unsqueeze(0)
    if index_closest_point.ndim == 1:
        index_closest_point = index_closest_point.unsqueeze(1)
    elif index_closest_point.ndim == 0:
        index_closest_point = index_closest_point.unsqueeze(0).unsqueeze(1)
    if device is None:
        device = torch.device("cpu")
    batch_size = index_closest_point.shape[0]
    future_points_idx = (
        torch.arange(n_points_to_return, device=device) * sample_interval
        + index_closest_point
        + n_points_shift
    )
    short_term_path = polyline[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(
            1
        ),  # For broadcasting
        future_points_idx,
    ]
    return short_term_path, future_points_idx

class Observations:
    def __init__(
        self,
        is_partial=None,
        n_nearing_agents=None,
        nearing_agents_indices=None,
        noise_level=None,
        n_stored_steps=None,
        n_observed_steps=None,
        past_pos: CircularBuffer = None,
        past_rot: CircularBuffer = None,
        past_vertices: CircularBuffer = None,
        past_vel: CircularBuffer = None,
        past_short_term_ref_points: CircularBuffer = None,
        past_action_vel: CircularBuffer = None,
        past_action_steering: CircularBuffer = None,
        past_distance_to_ref_path: CircularBuffer = None,
        past_distance_to_boundaries: CircularBuffer = None,
        past_distance_to_left_boundary: CircularBuffer = None,
        past_distance_to_right_boundary: CircularBuffer = None,
        past_distance_to_agents: CircularBuffer = None,
        past_left_boundary: CircularBuffer = None,
        past_right_boundary: CircularBuffer = None,
    ):
        self.is_partial = is_partial  # Local observation
        self.n_nearing_agents = n_nearing_agents
        self.nearing_agents_indices = nearing_agents_indices
        self.noise_level = noise_level  # Whether to add noise to observations
        self.n_stored_steps = n_stored_steps  # Number of past steps to store
        self.n_observed_steps = n_observed_steps  # Number of past steps to observe

        self.past_pos = past_pos  # Past positions
        self.past_rot = past_rot  # Past rotations
        self.past_vertices = past_vertices  # Past vertices
        self.past_vel = past_vel  # Past velocites

        self.past_short_term_ref_points = (
            past_short_term_ref_points  # Past short-term reference points
        )
        self.past_left_boundary = past_left_boundary  # Past left lanelet boundary
        self.past_right_boundary = past_right_boundary  # Past right lanelet boundary

        self.past_action_vel = past_action_vel  # Past velocity action
        self.past_action_steering = past_action_steering  # Past steering action
        self.past_distance_to_ref_path = (
            past_distance_to_ref_path  # Past distance to refrence path
        )
        self.past_distance_to_boundaries = (
            past_distance_to_boundaries  # Past distance to lanelet boundaries
        )
        self.past_distance_to_left_boundary = (
            past_distance_to_left_boundary  # Past distance to left lanelet boundary
        )
        self.past_distance_to_right_boundary = (
            past_distance_to_right_boundary  # Past distance to right lanelet boundary
        )
        self.past_distance_to_agents = (
            past_distance_to_agents  # Past mutual distance between agents
        )

class ReferencePathsAgentRelated:
    def __init__(
        self,
        long_term: torch.Tensor = None,
        left_boundary: torch.Tensor = None,
        right_boundary: torch.Tensor = None,
        nearing_points_left_boundary: torch.Tensor = None,
        nearing_points_right_boundary: torch.Tensor = None,
        short_term: torch.Tensor = None,
        short_term_indices: torch.Tensor = None,
    ):
        self.long_term = long_term  # Actual long-term reference paths of agents
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.short_term = short_term  # Short-term reference path
        self.short_term_indices = short_term_indices  # Indices that indicate which part of the long-term reference path is used to build the short-term reference path
        self.nearing_points_left_boundary = nearing_points_left_boundary  # Nearing left boundary
        self.nearing_points_right_boundary = nearing_points_right_boundary  # Nearing right boundary
from enum import Enum
class TaskClass(Enum):
    TRACKING = 0 # only reward tracking 
    PLATOON = 1 # reward tracking and platooning
    AVOIDING = 2 # reward tracking and avoiding collision
class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.device = device
        self.batch_dim = batch_dim
        self.init_params(batch_dim, device, **kwargs)
        world = self.init_world(batch_dim, device)
        self.init_agents(world, batch_dim, device)
        return world

    # ========== 1) 读取参数 & 分配并行批缓存 ==========
    def init_params(self, batch_dim: int, device: torch.device, **kwargs):
        # world params
        self.device = device
        self.batch_dim = batch_dim
        self.task_class=kwargs.pop("task_class", TaskClass.TRACKING)

        self.dt = float(kwargs.get("dt", 0.05))
        self.n_agents=kwargs.pop("n_agents", 5)
        self.n_followers = self.n_agents - 2
        self.n_nearing_agents_observed=kwargs.pop("n_nearing_agents_observed", 2)
        if self.n_nearing_agents_observed >= self.n_agents:
            raise ValueError("n_nearing_agents_observed must be less than n_agents")

        self.is_real_time_rendering=kwargs.pop("is_real_time_rendering", False)
        self.n_points_short_term=kwargs.pop("n_points_short_term", 3)
        self.n_points_nearing_boundary=kwargs.pop("n_points_nearing_boundary", 5)
        self.is_ego_view=kwargs.pop("is_ego_view", True)
        self.is_apply_mask=kwargs.pop("is_apply_mask", True)
        self.is_observe_vertices=kwargs.pop("is_observe_vertices", True)
        self.is_observe_distance_to_agents=kwargs.pop(
            "is_observe_distance_to_agents", True
        )
        self.is_observe_distance_to_boundaries=kwargs.pop(
            "is_observe_distance_to_boundaries", True
        )
        self.is_observe_distance_to_center_line=kwargs.pop(
            "is_observe_distance_to_center_line", True
        )
        self.is_add_noise=kwargs.pop("is_add_noise", True)
        self.is_observe_ref_path_other_agents=kwargs.pop(
            "is_observe_ref_path_other_agents", False
        )
        is_partial_observation=kwargs.pop("is_partial_observation", True)
        
        # Visualization
        self.visualize_semidims=False
        self.viewer_zoom = float(kwargs.get("viewer_zoom", 1.0))
        self.world_x_dim = kwargs.pop(
            "world_x_dim", 50
        )  # The x-dimension of the world in [m]
        self.world_y_dim = kwargs.pop(
            "world_y_dim", 40
        )  # The y-dimension of the world in [m]
        self.resolution_factor = kwargs.pop("resolution_factor", 20)  # Default 200
        self.render_origin = kwargs.pop(
            "render_origin", [self.world_x_dim / 2, self.world_y_dim / 2*3]
        )
        self.viewer_size = kwargs.pop(
            "viewer_size",
            (
                int(self.world_x_dim * self.resolution_factor),
                int(self.world_y_dim * self.resolution_factor),
            ),
        )
        # agent params
        self.v_front_ref = torch.full((self.batch_dim,), kwargs.get("v_front_ref", 7.0), device=device)  # 前端参考速度

        self.rod_len     = float(kwargs.get("rod_len", 40.0))   # 货物长度 L
        self.n_latch = int(kwargs.get("n_latch", 5))
        self.cargo_half_width = float(kwargs.get("cargo_half_width", 2.5))
        self.max_speed = float(kwargs.get("max_speed", 7.0))
        self.max_steering_angle = kwargs.pop(
            "max_steering_angle",
            torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)),
        )
        self.agent_width = float(kwargs.get("agent_width", 3.0))
        self.l_f = float(kwargs.get("l_f", 2.7))
        self.l_r = float(kwargs.get("l_r", 2.8))
        self.agent_length = self.l_f + self.l_r
        
        noise_level = kwargs.pop(
            "noise_level", 0.2 * self.agent_width
        )  # Noise will be generated by the standary normal distribution. This parameter controls the noise level
        n_stored_steps = kwargs.pop(
            "n_stored_steps",
            5,  # The number of steps to store (include the current step). At least one
        )
        n_observed_steps = kwargs.pop(
            "n_observed_steps", 1
        )  # The number of steps to observe (include the current step). At least one, and at most `n_stored_steps`

        #map params
        straight_length = 50.0
        radius = 25.0
        pts_gap=1.0
        self.lane_width  = float(kwargs.get("lane_width", 10.0))  # 道路宽度
        
        # 生成道路
        B = batch_dim
        n_straight = int(straight_length // pts_gap)
        n_circle = int((3.14*radius) // pts_gap)
        points = []
        # 第一段直线 (0,0) -> (straight_length,0)
        x1 = torch.linspace(1.0, straight_length-1.0, n_straight, device=device)
        y1 = torch.zeros(n_straight, device=device)
        points.append(torch.stack([x1, y1], dim=-1))
        
        # 第一个半圆 (straight_length,0) -> (straight_length,2*radius)
        theta1 = torch.linspace(0.0, torch.pi, n_circle, device=device)
        x2 = straight_length + radius * torch.sin(theta1)
        y2 = radius - radius * torch.cos(theta1)
        points.append(torch.stack([x2, y2], dim=-1))
        
        # 第二段直线 (straight_length,2*radius) -> (0,2*radius)
        x3 = torch.linspace(straight_length-1.0, 1.0, n_straight, device=device)
        y3 = torch.full((n_straight,), 2*radius, device=device)
        points.append(torch.stack([x3, y3], dim=-1))
        
        # 第二个半圆 (0,2*radius) -> (0,0)
        theta2 = torch.linspace(torch.pi, 2*torch.pi, n_circle, device=device)
        x4 = radius * torch.sin(theta2)
        y4 = 3*radius + radius * torch.cos(theta2)
        points.append(torch.stack([x4, y4], dim=-1))
        
        # 第三段直线 (straight_length,2*radius) -> (0,2*radius)
        x5 = torch.linspace(1.0, straight_length-1.0, n_straight, device=device)
        y5 = torch.full((n_straight,), 4*radius, device=device)
        points.append(torch.stack([x5, y5], dim=-1))
        
        # 合并所有点
        road_pts = torch.cat(points, dim=0)
        
        # 扩展到batch维度
        self.road_pts = road_pts.expand(B, -1, 2)
        # 计算累积弧长
        seg = self.road_pts[:, 1:, :] - self.road_pts[:, :-1, :]
        seg_len = torch.linalg.norm(seg, dim=-1)
        zero = torch.zeros(B, 1, device=device)
        self.road_cum_s = torch.cat([zero, torch.cumsum(seg_len, dim=-1)], dim=-1)
        self.s_start = self.road_cum_s[:, 0]
        
        # 3. 生成左右边界点
        tangents = self.road_pts[:, 1:, :] - self.road_pts[:, :-1, :]  # [B, N-1, 2]
        norm_tangents = torch.linalg.norm(tangents, dim=-1, keepdim=True) + 1e-8  # [B, N-1, 1]
        unit_tangents = tangents / norm_tangents  # [B, N-1, 2]
        normals = torch.stack([-unit_tangents[..., 1], unit_tangents[..., 0]], dim=-1)  # [B, N-1, 2]
        
        point_normals = torch.zeros_like(self.road_pts)  # [B, N, 2]
        point_normals[:, 0, :] = normals[:, 0, :]
        mid_normals = (normals[:, :-1, :] + normals[:, 1:, :]) / 2  # [B, N-2, 2]
        point_normals[:, 1:-1, :] = mid_normals
        point_normals[:, -1, :] = normals[:, -1, :]
        point_normals = point_normals / torch.linalg.norm(point_normals, dim=-1, keepdim=True) + 1e-8
        
        self.road_left_pts = self.road_pts + point_normals * self.lane_width/2  # [B, N, 2]
        self.road_right_pts = self.road_pts - point_normals * self.lane_width/2  # [B, N, 2]

        self.ref_paths_agent_related = ReferencePathsAgentRelated(
            long_term=self.road_pts.unsqueeze(1).expand(-1,self.n_agents,-1,-1),
            left_boundary=self.road_left_pts.unsqueeze(1).expand(-1,self.n_agents,-1,-1),
            right_boundary=self.road_right_pts.unsqueeze(1).expand(-1,self.n_agents,-1,-1),
            
            short_term=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_short_term, 2),
                device=device,
                dtype=torch.float32,
            ),  # Short-term reference path
            short_term_indices=torch.zeros(
                (batch_dim, self.n_agents, self.n_points_short_term),
                device=device,
                dtype=torch.int32,
            ),
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
        )
        # ---- 前/后端的弧长（按路起点放置）----
        # 初始：前端在 s_start + rod_len，后端在 s_start（如果路够长；否则夹取）
        s0 = self.s_start
        s1 = torch.clamp(s0 + self.rod_len, max=self.road_cum_s[:, -1] - 1e-6)
        self.s_front = s1.clone()   # [B]
        self.s_rear  = s0.clone()   # [B]

        # ---- 锚点（沿杆的比例 alpha）----
        self.latch_alpha = torch.linspace(0.0, 1.0, self.n_latch, device=device)  # [n_latch]
        self.latch_pos_world   = torch.zeros(B, self.n_latch, 2, device=device)   # [B,nL,2]
        self.latch_theta_world = torch.zeros(B, self.n_latch,   device=device)    # [B,nL]

        # ---- 随动车辆的 dock 状态/绑定锚点 ----
        F = self.n_followers
        self.dock_state      = torch.zeros(B, F, dtype=torch.bool, device=device) # 全部 free
        self.bound_latch_id  = torch.full((B, F), -1, dtype=torch.long, device=device)

        # 目标位姿缓存（post_step 投影用）
        self.target_pos   = torch.zeros(B, F, 2, device=device)
        self.target_theta = torch.zeros(B, F,   device=device)

        # 计时器（奖励/日志可用）
        self.dock_timer = torch.zeros(B, F, device=device)
        
        # 初始化 infeasible_mask
        self.infeasible_mask = torch.zeros(B, dtype=torch.bool, device=device)

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
        self.constants = Constants(
            env_idx_broadcasting=torch.arange(
                batch_dim, device=device, dtype=torch.int32
            ).unsqueeze(-1),
            empty_action_vel=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            empty_action_steering=torch.zeros(
                (batch_dim, self.n_agents), device=device, dtype=torch.float32
            ),
            mask_pos=torch.tensor(1, device=device, dtype=torch.float32),
            mask_zero=torch.tensor(0, device=device, dtype=torch.float32),
            mask_one=torch.tensor(1, device=device, dtype=torch.float32),
            reset_agent_min_distance=torch.tensor(
                (self.l_f + self.l_r) ** 2 + self.agent_width**2,
                device=device,
                dtype=torch.float32,
            ).sqrt()
            * 1.2,
        )

        self.normalizers = Normalizers(
            pos=torch.tensor(
                [self.agent_length * 10, self.agent_length * 10],
                device=device,
                dtype=torch.float32,
            ),
            pos_world=torch.tensor(
                [self.world_x_dim, self.world_y_dim], device=device, dtype=torch.float32
            ),
            v=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            rot=torch.tensor(2 * torch.pi, device=device, dtype=torch.float32),
            action_steering=self.max_steering_angle,
            action_vel=torch.tensor(self.max_speed, device=device, dtype=torch.float32),
            distance_lanelet=torch.tensor(
                self.lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_ref=torch.tensor(
                self.lane_width * 3, device=device, dtype=torch.float32
            ),
            distance_agent=torch.tensor(
                self.agent_length * 10, device=device, dtype=torch.float32
            ),
        )

        self.observations = Observations(
            is_partial=torch.tensor(
                is_partial_observation, device=device, dtype=torch.bool
            ),
            n_nearing_agents=torch.tensor(
                self.n_agents,
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
            past_short_term_ref_points = CircularBuffer(
                torch.zeros(
                    (
                        n_stored_steps,
                        batch_dim,
                        self.n_agents,
                        self.n_agents,
                        self.n_points_short_term,
                        2,
                    ),
                    device=device,
                    dtype=torch.float32,
                )
            ),
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
            past_action_vel = CircularBuffer(
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

        self.distances = Distances(
            agents=torch.zeros(
                batch_dim, self.n_agents, self.n_agents, dtype=torch.float32
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

        # Penalty
        threshold_deviate_from_ref_path = kwargs.pop(
            "threshold_deviate_from_ref_path", (self.lane_width - self.agent_width) / 2
        )  # Use for penalizing of deviating from reference path

        threshold_reach_goal = kwargs.pop(
            "threshold_reach_goal", self.agent_width / 2
        )  # Threshold less than which agents are considered at their goal positions

        threshold_change_steering = kwargs.pop(
            "threshold_change_steering", 10
        )  # Threshold above which agents will be penalized for changing steering too quick [degree]

        threshold_near_boundary_high = kwargs.pop(
            "threshold_near_boundary_high", (self.lane_width - self.agent_width) / 2 * 0.9
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to lanelet boundaries
        threshold_near_boundary_low = kwargs.pop(
            "threshold_near_boundary_low", 0
        )  # Threshold above which agents will be penalized for being too close to lanelet boundaries

        threshold_near_other_agents_c2c_high = kwargs.pop(
            "threshold_near_other_agents_c2c_high", self.agent_length + self.agent_width
        )  # Threshold beneath which agents will started be
        # Penalized for being too close to other agents (for center-to-center distance)
        threshold_near_other_agents_c2c_low = kwargs.pop(
            "threshold_near_other_agents_c2c_low",
            (self.agent_length + self.agent_width) / 2,
        )  # Threshold above which agents will be penalized (for center-to-center distance,
        # If a c2c distance is less than the half of the agent width, they are colliding, which will be penalized by another penalty)

        threshold_no_reward_if_too_close_to_boundaries = kwargs.pop(
            "threshold_no_reward_if_too_close_to_boundaries", self.agent_width / 10
        )
        threshold_no_reward_if_too_close_to_other_agents = kwargs.pop(
            "threshold_no_reward_if_too_close_to_other_agents", self.agent_width / 6
        )
        self.thresholds = Thresholds(
            reach_goal=torch.tensor(
                threshold_reach_goal, device=device, dtype=torch.float32
            ),
            deviate_from_ref_path=torch.tensor(
                threshold_deviate_from_ref_path, device=device, dtype=torch.float32
            ),
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
            no_reward_if_too_close_to_boundaries=torch.tensor(
                threshold_no_reward_if_too_close_to_boundaries,
                device=device,
                dtype=torch.float32,
            ),
            no_reward_if_too_close_to_other_agents=torch.tensor(
                threshold_no_reward_if_too_close_to_other_agents,
                device=device,
                dtype=torch.float32,
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
            )
        )
        # Train on the entire map
        max_ref_path_points = self.road_pts.shape[0]
        # Initialize agent-specific reference paths, which will be determined in `reset_world_at` function
        
        # The shape of each agent is considered a rectangle with 4 vertices.
        # The first vertex is repeated at the end to close the shape.
        self.vertices = torch.zeros(
            (batch_dim, self.n_agents, 5, 2), device=device, dtype=torch.float32
        )

        weighting_ref_directions = torch.linspace(
            1,
            0.2,
            steps=self.n_points_short_term,
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
            kwargs.pop("reward_vel", 5) / r_p_normalizer
        )  # Reward for moving in high velocities.
        reward_reach_goal = (
            kwargs.pop("reward_reach_goal", 0) / r_p_normalizer
        )  # Goal-reaching reward

        self.rewards = Rewards(
            progress=torch.tensor(reward_progress, device=device, dtype=torch.float32),
            weighting_ref_directions=weighting_ref_directions,  # Progress in the weighted directions (directions indicating by
            # closer short-term reference points have higher weights)
            higth_v=torch.tensor(reward_vel, device=device, dtype=torch.float32),
            reach_goal=torch.tensor(
                reward_reach_goal, device=device, dtype=torch.float32
            ),
        )
        self.rew = torch.zeros(batch_dim, device=device, dtype=torch.float32)

        self.penalties = Penalties(
            deviate_from_ref_path=torch.tensor(
                -2 / 100, device=device, dtype=torch.float32
            ),
            weighting_deviate_from_ref_path=self.lane_width / 2,
            near_boundary=torch.tensor(-20 / 100, device=device, dtype=torch.float32),
            near_other_agents=torch.tensor(
                -20 / 100, device=device, dtype=torch.float32
            ),
            collide_with_agents=torch.tensor(
                -100 / 100, device=device, dtype=torch.float32
            ),
            collide_with_boundaries=torch.tensor(
                -100 / 100, device=device, dtype=torch.float32
            ),
            change_steering=torch.tensor(-2 / 100, device=device, dtype=torch.float32),
            time=torch.tensor(5 / 100, device=device, dtype=torch.float32),
        )

        ScenarioUtils.check_kwargs_consumed(kwargs)
        self.n_steps_before_recording=kwargs.pop("n_steps_before_recording", 10)

        self.state_buffer = StateBuffer(
            buffer=torch.zeros(
                (self.n_steps_before_recording, batch_dim, self.n_agents, 5),
                device=device,
                dtype=torch.float32,
            )  # [pos_x, pos_y, rot, vel_x, vel_y],
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

        # 牵引车（脚本控制，不进 policy_agents）
        # 设置合适的形状和动作范围
        

        for i in range(self.n_followers):
            a = Agent(
                    name=f"follower_{i}", 
                    shape=Box(length=self.l_f + self.l_r, width=self.agent_width),
                    color=tuple(
                        torch.rand(3, device=world.device, dtype=torch.float32).tolist()
                    ),
                    collide=False,
                    render_action=False,
                    u_range=[
                        self.max_speed,
                        self.max_steering_angle,
                    ],  # Control command serves as velocity command
                    u_multiplier=[1, 1],
                    max_speed=self.max_speed,
                    dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                        world,
                        width=self.agent_width,
                        l_f=self.l_f,
                        l_r=self.l_r,
                        max_steering_angle=self.max_steering_angle,
                        integration="rk4",  # one of {"euler", "rk4"}
                    ),
                )
            world.add_agent(a)
            self.followers.append(a)
        self.tractor_front = Agent(
            name="tractor_front", 
            shape=Box(length=self.l_f + self.l_r, width=self.agent_width),
            color=Color.RED,
            collide=False,
            render_action=False,
            u_range=[
                self.max_speed,
                self.max_steering_angle,
            ],  # Control command serves as velocity command
            u_multiplier=[1, 1],
            max_speed=self.max_speed,
            dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                world,
                width=self.agent_width,
                l_f=self.l_f,
                l_r=self.l_r,
                max_steering_angle=self.max_steering_angle,
                integration="rk4",  # one of {"euler", "rk4"}
            ),
        )
        self.tractor_rear  = Agent(
            name="tractor_rear",  
            shape=Box(length=self.l_f + self.l_r, width=self.agent_width),
            color=Color.BLUE,
            collide=False,
            render_action=False,
            u_range=[
                self.max_speed,
                self.max_steering_angle,
            ],  # Control command serves as velocity command
            u_multiplier=[1, 1],
            max_speed=self.max_speed,
            dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                world,
                width=self.agent_width,
                l_f=self.l_f,
                l_r=self.l_r,
                max_steering_angle=self.max_steering_angle,
                integration="rk4",  # one of {"euler", "rk4"}
            ),
        )
        world.add_agent(self.tractor_front)
        world.add_agent(self.tractor_rear)
    def reset_world_at(self, env_index: Optional[int] = None, agent_index: Optional[int] = None):
        """
        This function resets the world at the specified env_index and the specified agent_index.
        If env_index is given as None, the majority part of computation will be done in a vectorized manner.

        Args:
        :param env_index: index of the environment to reset. If None a vectorized reset should be performed
        :param agent_index: index of the agent to reset. If None all agents in the specified environment will be reset.
        """
        B = self.batch_dim
        device = self.device

        
        if env_index is None:
            idx_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            self._check_batch_index(env_index)  # 如果你沿用 VMAS 的检查
            idx_mask = torch.zeros(B, dtype=torch.bool, device=device)
            idx_mask[env_index] = True

        # ---- 放置前/后端弧长 ----
        # 让 rear 在 s_start，front = s_start + L（夹取到道路范围）
        s0 = self.s_start.clone()
        s1 = torch.clamp(s0 + self.rod_len, max=self.road_cum_s[:, -1] - 1e-6)
        self.s_front = torch.where(idx_mask, s1, self.s_front)
        self.s_rear  = torch.where(idx_mask, s0, self.s_rear)

        # ---- 端点坐标与朝向（用于设置牵引车初始位姿）----
        p_front = self.road_C(self.s_front)    # [B,2]
        p_rear  = self.road_C(self.s_rear)     # [B,2]
        # 杆方向
        rod_vec = (p_front - p_rear)           # [B,2]
        rod_theta = torch.atan2(rod_vec[:, 1], rod_vec[:, 0])  # [B]

        # 设置牵引车初始位姿（只对 mask 的样本赋值）
        def _set_pose(agent: Agent, pos: Tensor, theta: Tensor):
            if hasattr(agent.state, "pos"):
                agent.state.pos[idx_mask] = pos[idx_mask]
            if hasattr(agent.state, "rot"):
                # Ensure theta has the right shape for assignment
                theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
                agent.state.rot[idx_mask] = theta_reshaped[idx_mask]
            elif hasattr(agent.state, "angle"):
                # Ensure theta has the right shape for assignment
                theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
                agent.state.angle[idx_mask] = theta_reshaped[idx_mask]
            if hasattr(agent.state, "vel"):
                agent.state.vel[idx_mask] = 0.0
        # 计算道路切线方向而不是使用货物方向
        front_theta = self.road_tangent(self.s_front)
        rear_theta = self.road_tangent(self.s_rear)
        
        # 设置牵引车初始位姿
        _set_pose(self.tractor_front, p_front, front_theta)
        _set_pose(self.tractor_rear, p_rear, rear_theta)

        # ---- 随动：初始 free & 随机放置在杆附近（不 dock）----
        self.dock_state[idx_mask, :]     = False
        self.bound_latch_id[idx_mask, :] = -1   
        self.dock_timer[idx_mask, :]     = 0.0

        # 在 rear→front 方向均匀分布随动车辆
        alpha = torch.linspace(0.0, 1.0, self.n_agents, device=device)[1:-1]  # 首尾去掉(牵引车位置)
        alpha = alpha.expand(B, -1)  # [B,F]
        base = p_rear[:, None, :] + alpha[..., None] * rod_vec[:, None, :]  # [B,F,2]
        
        # 设置随动初始位姿(无横向偏移)
        pos_f_init = base

        for i, ag in enumerate(self.followers):
            if hasattr(ag.state, "pos"):
                ag.state.pos[idx_mask] = pos_f_init[idx_mask, i, :]
            if hasattr(ag.state, "rot"):
                # Ensure rod_theta has the right shape for assignment
                theta_reshaped = rod_theta.unsqueeze(-1) if rod_theta.dim() == 1 else rod_theta
                ag.state.rot[idx_mask] = theta_reshaped[idx_mask]  # 朝向与杆一致先
            elif hasattr(ag.state, "angle"):
                # Ensure rod_theta has the right shape for assignment
                theta_reshaped = rod_theta.unsqueeze(-1) if rod_theta.dim() == 1 else rod_theta
                ag.state.angle[idx_mask] = theta_reshaped[idx_mask]
            if hasattr(ag.state, "vel"):
                ag.state.vel[idx_mask] = 0.0

        # ---- 预计算锚点世界位姿（沿杆等间距）----
        # latch_pos = p_rear + alpha * (p_front - p_rear)
        self.latch_pos_world = p_rear[:, None, :] + self.latch_alpha[None, :, None] * rod_vec[:, None, :]
        self.latch_theta_world = rod_theta[:, None].expand(-1, self.n_latch)

        # 给一个初始的 target（free 也写当前位姿，post_step 会用 mask 过滤）
        self.compute_latch_targets()

        agents = self.world.agents

        is_reset_single_agent = agent_index is not None

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

            for i_agent in (
                range(self.n_agents)
                if not is_reset_single_agent
                else agent_index.unsqueeze(0)
            ):

                self.reset_init_distances_and_short_term_ref_path(
                    env_j, i_agent, agents
                )

            # Compute mutual distances between agents
            mutual_distances = get_distances_between_agents(
                self=self, is_set_diagonal=True
            )
            # Reset mutual distances of all envs
            self.distances.agents[env_j, :, :] = mutual_distances[env_j, :, :]

            # Reset the collision matrix
            self.collisions.with_agents[env_j, :, :] = False
            self.collisions.with_lanelets[env_j, :] = False

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

    def reset_init_distances_and_short_term_ref_path(self, env_j, i_agent, agents):
        """
        This function calculates the distances from the agent's center of gravity (CG) to its reference path and boundaries,
        and computes the positions of the four vertices of the agent. It also determines the short-term reference paths
        for the agent based on the long-term reference paths and the agent's current position.
        """
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
        # Calculate the positions of the four vertices of the agents
        self.vertices[env_j, i_agent] = get_rectangle_vertices(
            center=agents[i_agent].state.pos[env_j, :],
            yaw=agents[i_agent].state.rot[env_j, :],
            width=agents[i_agent].shape.width,
            length=agents[i_agent].shape.length,
            is_close_shape=True,
        )
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

        # Get the short-term reference paths
        (
            self.ref_paths_agent_related.short_term[env_j, i_agent],
            _,
        ) = get_short_term_reference_path_simple(
            polyline=self.ref_paths_agent_related.long_term[env_j, i_agent],
            index_closest_point=self.distances.closest_point_on_ref_path[
                env_j, i_agent
            ],
            n_points_to_return=self.n_points_short_term,
            device=self.world.device,
            sample_interval=1,
            n_points_shift=1,
        )

        if not self.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
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
                sample_interval=1,
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
                sample_interval=1,
                n_points_shift=1,
            )
    # =========================
    # 核心动作处理/几何解算
    # =========================
    def process_action(self, agent):
        """
        调用时序：在 world.step() 之前，且在 env 分发完 RL 动作之后
        作用：docked 时覆盖随动车辆的物理动作；free 时保持 RL 动作
        """
        # 只处理随动车辆
        if agent not in self.followers:
            return

        i = self.followers.index(agent)            # 随动车辆在列表中的索引
        docked_mask = self.dock_state[:, i]        # [B] bool

        # ---- 硬约束：docked 时忽略物理控制（例如 [速度, 转向] 两维清零）----
        # 动作张量形状通常是 [B, action_dim]；只改物理通道，保留你设计的 attach/undock 位
        if docked_mask.any() and hasattr(agent, 'action') and agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
            # 把该 agent 在 batch 中 d==True 的样本的物理控制清零
            u = agent.action.u                      # [B, action_dim]
            # 假设前两维为物理控制（按你的动作定义改）
            if u.shape[-1] >= 2:
                u[docked_mask, :2] =  torch.zeros_like(u[docked_mask, :2])
                agent.action.u = u

    def pre_step(self):
        """
        每次 world.step() 之前：
        1) 推进前端弧长 s_front
        2) 固定弦长解 Δs -> s_rear
        3) 计算端点坐标与杆朝向
        4) 更新所有锚点的世界位姿（供 docked 目标用）
        5) 计算随动车辆的 target_pos / target_theta（按绑定锚点）
        """

        # ---- 1) 推进前端弧长并夹取到道路范围 ----
        s_front = self.s_front + self.v_front_ref * self.dt                   # [B]
        # 不要超过道路最大 s；留一点 eps 免得插值越界
        s_max = self.road_cum_s[:, -1] - 1e-6
        s_min = self.s_start
        s_front = torch.clamp(s_front, min=s_min, max=s_max)

        # ---- 2) 固定弦长解 Δs -> s_rear ----
        delta_s, infeasible = self.solve_delta_s(s_front, torch.full_like(s_front, self.rod_len))
        s_rear = s_front - delta_s
        s_rear = torch.clamp(s_rear, min=s_min, max=s_max)

        # 缓存回成员（别忘了）
        self.s_front = s_front
        self.s_rear  = s_rear
        self.infeasible_mask = infeasible    # 你后面可据此降速/强制 undock，加惩罚等

        # ---- 3) 端点坐标与杆朝向 ----
        p_front = self.road_C(s_front)                                         # [B,2]
        p_rear  = self.road_C(s_rear)                                          # [B,2]
        rod_vec = p_front - p_rear                                             # [B,2]
        theta_rod = torch.atan2(rod_vec[:, 1], rod_vec[:, 0])                  # [B]

        # ---- 4) 更新锚点世界位姿（等间距示例；若你有自定义分布，就替换这里）----
        # latch at: p = p_rear + α * (p_front - p_rear)
        self.latch_pos_world   = p_rear[:, None, :] + self.latch_alpha[None, :, None] * rod_vec[:, None, :]
        self.latch_theta_world = theta_rod[:, None].expand(-1, self.n_latch)   # 全部用杆朝向

        # （可选）更新牵引车脚本动作的目标，这里只算，不写 action；你可在 env_process_action 里写入

        # ---- 5) 计算随动车辆目标（按绑定锚点），shape 必须是 [B, nF, 2] / [B, nF] ----
        self.compute_latch_targets()  # 这个函数我们之前已经给了实现


    def post_step(self):
        """
        每次 world.step() 之后：
        - 对 docked 的随动车辆做硬投影（对齐到各自锚点）
        - 统计 dock 计时等
        """
        B = self.batch_dim

        for i, agent in enumerate(self.followers):
            docked_mask = self.dock_state[:, i]      # [B] bool
            if not docked_mask.any():
                continue

            # 注意：这里的 target_* 是 [B, n_followers, ...]
            pos_i   = self.target_pos[:, i, :]       # [B, 2]
            theta_i = self.target_theta[:, i]        # [B]

            self._project_agent_to_pose(
                agent,
                pos_i[docked_mask, :],               # [M,2]
                theta_i[docked_mask],               # [M]
                mask=docked_mask,                   # [B]
            )
        # 更新牵引车位置以跟随货物（修改后的版本）
        p_front = self.road_C(self.s_front)    # [B,2]
        p_rear  = self.road_C(self.s_rear)     # [B,2]
        # 计算道路切线方向而不是使用货物方向
        front_theta = self.road_tangent(self.s_front)
        rear_theta = self.road_tangent(self.s_rear)
        rod_theta = torch.atan2(p_front[:,1]-p_rear[:,1], p_front[:,0]-p_rear[:,0])  # [B]
        
        # 分别为前后牵引车设置各自的道路切线方向
        for i, agent in enumerate([self.tractor_front, self.tractor_rear]):
            if hasattr(agent.state, "pos"):
                # 使用杆端点位置
                pos = p_front if i == 0 else p_rear
                agent.state.pos = pos
            if hasattr(agent.state, "rot"):
                # 使用道路切线方向
                theta = front_theta if i == 0 else rear_theta
                agent.state.rot = theta.unsqueeze(-1)
            elif hasattr(agent.state, "angle"):
                # 使用道路切线方向
                theta = front_theta if i == 0 else rear_theta
                agent.state.angle = theta
        

        # 计时器（可用于奖励）：每步给 docked 的样本累加 dt
        self.dock_timer += self.dock_state.to(self.dock_timer.dtype) * self.dt
    # 添加一个方法来计算道路切线方向
    def road_tangent(self, s: Tensor) -> Tensor:
        """计算道路上弧长s处的切线方向角"""
        # 使用小扰动法计算切线方向
        epsilon = 1e-3  # 小扰动值
        s_plus = torch.clamp(s + epsilon, max=self.road_cum_s[:, -1] - 1e-6)
        pos_plus = self.road_C(s_plus)  # [B,2]
        pos = self.road_C(s)  # [B,2]
        tangent_vec = pos_plus - pos
        # 计算切线方向角（弧度）
        tangent_theta = torch.atan2(tangent_vec[:, 1], tangent_vec[:, 0])
        return tangent_theta

    def compute_latch_targets(self):
        """
        写出:
        self.target_pos   : [B, n_followers, 2]
        self.target_theta : [B, n_followers]
        规则:
        - 若某随动处于 docked 且有有效锚点 id(>=0)，目标=锚点位姿
        - 否则(自由态或无效 id)，目标=当前车位姿(这样 post_step 的硬投影对自由态不产生影响)
        """
        device = self.device
        B = self.batch_dim
        nF = len(self.followers)

        # 取出所有随动的当前位姿，拼成张量，方便向量化
        # pos_f: [B, nF, 2], theta_f: [B, nF]
        pos_f_list = []
        theta_f_list = []
        for ag in self.followers:
            pos_f_list.append(ag.state.pos)         # [B,2]
            # 兼容不同版本的朝向字段: rot / angle
            if hasattr(ag.state, "rot"):
                theta_f_list.append(ag.state.rot.squeeze(-1))   # [B] - remove last dimension if it's [B,1]
            elif hasattr(ag.state, "angle"):
                theta_f_list.append(ag.state.angle.squeeze(-1)) # [B] - remove last dimension if it's [B,1]
            else:
                raise AttributeError("Agent.state has no 'rot' or 'angle'")
        pos_f   = torch.stack(pos_f_list, dim=1).to(device)   # [B,nF,2]
        theta_f = torch.stack(theta_f_list, dim=1).to(device) # [B,nF]

        # 目标张量初始化为"当前位姿"（自由态默认不动）
        target_pos   = pos_f.clone()
        target_theta = theta_f.clone()

        # 锚点表
        latch_pos   = self.latch_pos_world.to(device)     # [B, nL, 2]
        latch_theta = self.latch_theta_world.to(device)   # [B, nL]
        nL = latch_pos.shape[1]

        # 有效(docked 且 id>=0)的掩码
        dock = self.dock_state.to(device)                 # [B, nF]
        ids  = self.bound_latch_id.to(device).long()      # [B, nF]
        valid = dock & (ids >= 0)                         # [B, nF]

        if valid.any():
            # 为 gather 构造索引
            # pos: [B,nF,2] <- latch_pos.gather(dim=1, idx=[B,nF,2])
            idx_pos = ids.clamp(min=0, max=nL-1)[..., None].expand(-1, -1, 2)  # [B,nF,2]
            sel_pos = torch.gather(latch_pos, dim=1, index=idx_pos)            # [B,nF,2]

            # theta: [B,nF] <- take_along_dim(latch_theta, idx=[B,nF])
            sel_theta = torch.take_along_dim(latch_theta, ids.clamp(min=0, max=nL-1), dim=1)  # [B,nF]

            # 仅对 valid 的位置/朝向进行覆盖
            valid_exp = valid.unsqueeze(-1)                                    # [B,nF,1]
            target_pos   = torch.where(valid_exp, sel_pos,   target_pos)
            target_theta = torch.where(valid,     sel_theta, target_theta)

        # 缓存输出，供 post_step 使用
        self.target_pos   = target_pos   # [B, nF, 2]
        self.target_theta = target_theta # [B, nF]


    def _project_agent_to_pose(self, agent, pos_batch: torch.Tensor, theta_batch: torch.Tensor, mask: torch.Tensor):
        """
        把该 agent 在 mask==True 的那些 batch 样本的状态(位置/朝向/速度)硬对齐到给定目标:
        - agent.state.pos[mask] = pos_batch
        - agent.state.rot/angle[mask] = theta_batch
        - 速度: 为了稳定，清零(或按需设置为沿切向的目标速度)
        形状:
        - agent.state.pos : [B,2]
        - pos_batch       : [sum(mask), 2]
        - theta_batch     : [sum(mask)]
        - mask            : [B] (bool)
        """
        if not mask.any():
            return

        # 取 mask 索引
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # [M], M = sum(mask)

        # --- 位置 ---
        agent.state.pos[idx] = pos_batch.to(agent.state.pos.dtype)

        # --- 朝向 ---
        if hasattr(agent.state, "rot"):
            # Ensure theta_batch has the right shape for assignment
            theta_reshaped = theta_batch.unsqueeze(-1) if theta_batch.dim() == 1 else theta_batch
            agent.state.rot[idx] = theta_reshaped.to(agent.state.rot.dtype)
        elif hasattr(agent.state, "angle"):
            # Ensure theta_batch has the right shape for assignment
            theta_reshaped = theta_batch.unsqueeze(-1) if theta_batch.dim() == 1 else theta_batch
            agent.state.angle[idx] = theta_reshaped.to(agent.state.angle.dtype)
        else:
            raise AttributeError("Agent.state has no 'rot' or 'angle' to set orientation")

        # --- 线速度/角速度（硬约束：清零最稳；如需更物理，可填入锚点切向速度） ---
        if hasattr(agent.state, "vel"):
            # vel: [B,2]
            agent.state.vel[idx] = 0.0
        if hasattr(agent.state, "omega"):
            # 有的实现有角速度标量
            agent.state.omega[idx] = 0.0

    def _check_batch_index(self, env_index: int):
        """检查批次索引是否有效"""
        if env_index < 0 or env_index >= self.batch_dim:
            raise ValueError(f"Invalid env_index {env_index}, must be in [0, {self.batch_dim})")

    def get_scenario_info(self):
        """获取场景信息，用于调试和验证"""
        return {
            "batch_dim": self.batch_dim,
            "n_agents": self.n_agents,
            "n_followers": self.n_followers,
            "rod_len": self.rod_len,
            "dt": self.dt,
            "device": str(self.device),
            "road_points": self.road_pts.shape,
            "n_latch": self.n_latch
        }
    
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

            if self.is_ego_view:
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
                    (self.observations.past_short_term_ref_points.get_latest())
                )  # Reference paths of other agents relative to agent i
                l_b_i_others = torch.zeros_like(
                    (self.observations.past_left_boundary.get_latest())
                )  # Left boundaries of other agents relative to agent i
                r_b_i_others = torch.zeros_like(
                    (self.observations.past_right_boundary.get_latest())
                )  # Right boundaries of other agents relative to agent i
                ver_i_others = torch.zeros_like(
                    (self.observations.past_vertices.get_latest())
                )  # Vertices of other agents relative to agent i

                for a_i in range(self.n_agents):
                    pos_i = self.world.agents[a_i].state.pos
                    rot_i = self.world.agents[a_i].state.rot

                    # Store new observation - position
                    pos_i_others[:, a_i] = transform_from_global_to_local_coordinate(
                        pos_i=pos_i,
                        pos_j=positions_global,
                        rot_i=rot_i,
                    )

                    # Store new observation - rotation
                    rot_i_others[:, a_i] = rotations_global - rot_i

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
                            :, a_i, a_j
                        ] = transform_from_global_to_local_coordinate(
                            pos_i=pos_i,
                            pos_j=self.ref_paths_agent_related.short_term[:, a_j],
                            rot_i=rot_i,
                        )

                        # Store new observation - left boundary
                        if not self.is_observe_distance_to_boundaries:
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

                # Add new observations & normalize
                self.observations.past_pos.add(
                    pos_i_others
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_rot.add(rot_i_others / self.normalizers.rot)
                self.observations.past_vel.add(vel_i_others / self.normalizers.v)
                self.observations.past_short_term_ref_points.add(
                    ref_i_others
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_left_boundary.add(
                    l_b_i_others
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_right_boundary.add(
                    r_b_i_others
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_vertices.add(
                    ver_i_others
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )

            else:  # Global coordinate system
                # Store new observations
                self.observations.past_pos.add(
                    positions_global
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_vel.add(
                    torch.stack([a.state.vel for a in self.world.agents], dim=1)
                    / self.normalizers.v
                )
                self.observations.past_rot.add(
                    rotations_global[:] / self.normalizers.rot
                )
                self.observations.past_vertices.add(
                    self.vertices[:, :, 0:4, :]
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_short_term_ref_points.add(
                    self.ref_paths_agent_related.short_term[:]
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_left_boundary.add(
                    self.ref_paths_agent_related.nearing_points_left_boundary
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )
                self.observations.past_right_boundary.add(
                    self.ref_paths_agent_related.nearing_points_right_boundary
                    / (
                        self.normalizers.pos
                        if self.is_ego_view
                        else self.normalizers.pos_world
                    )
                )

            # Add new observation - actions & normalize
            if agent.action.u is None:
                self.observations.past_action_vel.add(self.constants.empty_action_vel)
                self.observations.past_action_steering.add(
                    self.constants.empty_action_steering
                )
            else:
                self.observations.past_action_vel.add(
                    torch.stack([a.action.u[:, 0] for a in self.world.agents], dim=1)
                    / self.normalizers.action_vel
                )
                self.observations.past_action_steering.add(
                    torch.stack([a.action.u[:, 1] for a in self.world.agents], dim=1)
                    / self.normalizers.action_steering
                )

    def observe_self(self, agent_index):
        """Observe the given agent itself."""
        indexing_tuple_3 = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index,) if self.is_ego_view else ())
        )
        indexing_tuple_vel = (
            (self.constants.env_idx_broadcasting,)
            + (agent_index,)
            + ((agent_index, 0) if self.is_ego_view else ())
        )  # In local coordinate system, only the first component is interesting, as the second is always 0

        # Self-observations
        obs_self = [
            None
            if self.is_ego_view
            else self.observations.past_pos.get_latest()[indexing_tuple_3].reshape(
                self.world.batch_dim, -1
            ),  # [own] position,
            None
            if self.is_ego_view
            else self.observations.past_rot.get_latest()[indexing_tuple_3].reshape(
                self.world.batch_dim, -1
            ),  # [own] rotation,
            self.observations.past_vel.get_latest()[indexing_tuple_vel].reshape(
                self.world.batch_dim, -1
            ),  # [own] velocity
            self.observations.past_short_term_ref_points.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.world.batch_dim, -1
            ),  # [own] short-term reference path
            self.observations.past_distance_to_ref_path.get_latest()[
                :, agent_index
            ].reshape(self.world.batch_dim, -1)
            if self.is_observe_distance_to_center_line
            else None,  # [own] distances to reference paths
            self.observations.past_distance_to_left_boundary.get_latest()[
                :, agent_index
            ].reshape(self.world.batch_dim, -1)
            if self.is_observe_distance_to_boundaries
            else self.observations.past_left_boundary.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.world.batch_dim, -1
            ),  # [own] left boundaries
            self.observations.past_distance_to_right_boundary.get_latest()[
                :, agent_index
            ].reshape(self.world.batch_dim, -1)
            if self.is_observe_distance_to_boundaries
            else self.observations.past_right_boundary.get_latest()[
                indexing_tuple_3
            ].reshape(
                self.world.batch_dim, -1
            ),  # [own] right boundaries
        ]
        return obs_self
    
    def observe_other_agents(self, agent_index):
        """Observe surrounding agents."""
        if self.observations.is_partial:
            # Each agent observes only a fixed number of nearest agents
            nearing_agents_distances, nearing_agents_indices = torch.topk(
                self.distances.agents[:, agent_index],
                k=self.observations.n_nearing_agents,
                largest=False,
            )

            if self.is_apply_mask:
                # Nearing agents that are distant will be masked
                mask_nearing_agents_too_far = (
                    nearing_agents_distances >= self.thresholds.distance_mask_agents
                )
            else:
                # Otherwise no agents will be masked
                mask_nearing_agents_too_far = torch.zeros(
                    (self.world.batch_dim, self.n_nearing_agents_observed),
                    device=self.world.device,
                    dtype=torch.bool,
                )

            indexing_tuple_1 = (
                (self.constants.env_idx_broadcasting,)
                + ((agent_index,) if self.is_ego_view else ())
                + (nearing_agents_indices,)
            )

            # Positions of nearing agents
            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 2]
            obs_pos_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Position mask

            # Rotations of nearing agents
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_rot_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_zero  # Rotation mask

            # Velocities of nearing agents
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents]
            obs_vel_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_zero  # Velocity mask

            # Reference paths of nearing agents
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    indexing_tuple_1
                ]
            )  # [batch_size, n_nearing_agents, n_points_short_term, 2]
            obs_ref_path_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Reference-path mask

            # vertices of nearing agents
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                indexing_tuple_1
            ]  # [batch_size, n_nearing_agents, 4, 2]
            obs_vertices_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Reference-path mask

            # Distances to nearing agents
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[
                    self.constants.env_idx_broadcasting,
                    agent_index,
                    nearing_agents_indices,
                ]
            )  # [batch_size, n_nearing_agents]
            obs_distance_other_agents[
                mask_nearing_agents_too_far
            ] = self.constants.mask_one  # Distance mask

        else:
            obs_pos_other_agents = self.observations.past_pos.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, 2]
            obs_rot_other_agents = self.observations.past_rot.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, (n_agents)]
            obs_vel_other_agents = self.observations.past_vel.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, 2]
            obs_ref_path_other_agents = (
                self.observations.past_short_term_ref_points.get_latest()[
                    :, agent_index
                ]
            )  # [batch_size, n_agents, n_points_short_term, 2]
            obs_vertices_other_agents = self.observations.past_vertices.get_latest()[
                :, agent_index
            ]  # [batch_size, n_agents, 4, 2]
            obs_distance_other_agents = (
                self.observations.past_distance_to_agents.get_latest()[:, agent_index]
            )  # [batch_size, n_agents]
            obs_distance_other_agents[
                :, agent_index
            ] = 0  # Reset self-self distance to zero

        # Flatten the last dimensions to combine all features into a single dimension
        obs_pos_other_agents_flat = obs_pos_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_rot_other_agents_flat = obs_rot_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vel_other_agents_flat = obs_vel_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_ref_path_other_agents_flat = obs_ref_path_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_vertices_other_agents_flat = obs_vertices_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )
        obs_distance_other_agents_flat = obs_distance_other_agents.reshape(
            self.world.batch_dim, self.observations.n_nearing_agents, -1
        )

        # Observation of other agents
        obs_others_list = [
            obs_vertices_other_agents_flat
            if self.is_observe_vertices
            else torch.cat(  # [other] vertices
                [
                    obs_pos_other_agents_flat,  # [others] positions
                    obs_rot_other_agents_flat,  # [others] rotations
                ],
                dim=-1,
            ),
            obs_vel_other_agents_flat,  # [others] velocities
            obs_distance_other_agents_flat
            if self.is_observe_distance_to_agents
            else None,  # [others] mutual distances
            obs_ref_path_other_agents_flat
            if self.is_observe_ref_path_other_agents
            else None,  # [others] reference paths
        ]
        obs_others_list = [
            o for o in obs_others_list if o is not None
        ]  # Filter out None values
        obs_other_agents = torch.cat(obs_others_list, dim=-1).reshape(
            self.world.batch_dim, -1
        )  # [batch_size, -1]

        return obs_other_agents
    def observation(self, agent: Agent):
        """
        Generate an observation for the given agent in all envs.

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            The observation for the given agent in all envs, which consists of the observation of this agent itself and possibly the observation of its surrounding agents.
                The observation of this agent itself includes
                    position (in case of using bird view),
                    rotation (in case of using bird view),
                    velocity,
                    short-term reference path,
                    distance to its reference path (optional), and
                    lane boundaries (or distances to them).
                The observation of its surrounding agents includes their
                    vertices (or positions and rotations),
                    velocities,
                    distances to them (optional), and
                    reference paths (optional).
        """
        agent_index = self.world.agents.index(agent)

        self.update_observation_and_normalize(agent, agent_index)

        # Observation of other agents
        obs_other_agents = self.observe_other_agents(agent_index)

        obs_self = self.observe_self(agent_index)

        obs_self.append(obs_other_agents)  # Append the observations of other agents

        obs_all = [o for o in obs_self if o is not None]  # Filter out None values

        obs = torch.hstack(obs_all)  # Convert from list to tensor

        if self.is_add_noise:
            # Add sensor noise if required
            return obs + (
                self.observations.noise_level
                * torch.rand_like(obs, device=self.world.device, dtype=torch.float32)
            )
        else:
            # Return without sensor noise
            return obs

    def reward(self, agent: Agent):
        """
        Issue rewards for the given agent in all envs.
            Positive Rewards:
                Moving forward (become negative if the projection of the moving direction to its reference path is negative)
                Moving forward with high speed (become negative if the projection of the moving direction to its reference path is negative)
                Reaching goal (optional)

            Negative Rewards (penalties):
                Too close to lane boundaries
                Too close to other agents
                Deviating from reference paths
                Changing steering too quick
                Colliding with other agents
                Colliding with lane boundaries

        Args:
            agent: The agent for which the observation is to be generated.

        Returns:
            A tensor with shape [batch_dim].
        """
        # Initialize
        self.rew[:] = 0
        # Get the index of the current agent
        agent_index = self.world.agents.index(agent)
        # we exclude the front vehicle and end vehicle
        if agent_index>=self.n_followers:
            return self.rew
        # [update] mutual distances between agents, vertices of each agent, and collision matrices
        self.update_state_before_rewarding(agent, agent_index)

        # [reward] forward movement
        latest_state = self.state_buffer.get_latest(n=1)
        move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(
            1
        )  # Vector of the current movement

        ref_points_vecs = self.ref_paths_agent_related.short_term[
            :, agent_index
        ] - latest_state[:, agent_index, 0:2].unsqueeze(
            1
        )  # Vectors from the previous position to the points on the short-term reference path
        move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
        move_projected_weighted = torch.matmul(
            move_projected, self.rewards.weighting_ref_directions
        )  # Put more weights on nearing reference points

        reward_movement = (
            move_projected_weighted
            / (agent.max_speed * self.world.dt)
            * self.rewards.progress
        )
        self.rew += reward_movement  # Relative to the maximum possible movement

        # [reward] high velocity
        v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(
            -1
        )
        factor_moving_direction = torch.where(
            v_proj > 0, 1, 2
        )  # Get penalty if move in negative direction

        reward_vel = (
            factor_moving_direction * v_proj / agent.max_speed * self.rewards.higth_v
        )
        self.rew += reward_vel

        # [reward] reach goal
        # reward_goal = (
        #     self.collisions.with_exit_segments[:, agent_index] * self.rewards.reach_goal
        # )
        # self.rew += reward_goal

        # [penalty] close to lanelet boundaries
        penalty_close_to_lanelets = (
            exponential_decreasing_fcn(
                x=self.distances.boundaries[:, agent_index],
                x0=self.thresholds.near_boundary_low,
                x1=self.thresholds.near_boundary_high,
            )
            * self.penalties.near_boundary
        )
        self.rew += penalty_close_to_lanelets

        # [penalty] close to other agents
        mutual_distance_exp_fcn = exponential_decreasing_fcn(
            x=self.distances.agents[:, agent_index, :],
            x0=self.thresholds.near_other_agents_low,
            x1=self.thresholds.near_other_agents_high,
        )
        penalty_close_to_agents = (
            torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
        )
        self.rew += penalty_close_to_agents

        # [penalty] deviating from reference path
        self.rew += (
            self.distances.ref_paths[:, agent_index]
            / self.penalties.weighting_deviate_from_ref_path
            * self.penalties.deviate_from_ref_path
        )

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
        steering_change_reward_factor = steering_change / (
            2 * agent.u_range[1] - 2 * self.thresholds.change_steering
        )
        penalty_change_steering = (
            steering_change_reward_factor * self.penalties.change_steering
        )
        self.rew += penalty_change_steering

        # [penalty] colliding with other agents
        is_collide_with_agents = self.collisions.with_agents[:, agent_index]
        penalty_collide_other_agents = (
            is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
        )
        self.rew += penalty_collide_other_agents

        # [penalty] colliding with lanelet boundaries
        is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
        penalty_collide_lanelet = (
            is_collide_with_lanelets * self.penalties.collide_with_boundaries
        )
        self.rew += penalty_collide_lanelet

        # [penalty/reward] time
        # Get time reward if moving in positive direction; otherwise get time penalty
        time_reward = (
            torch.where(v_proj > 0, 1, -1)
            * agent.state.vel.norm(dim=-1)
            / agent.max_speed
            * self.penalties.time
        )
        self.rew += time_reward

        # [update] previous positions and short-term reference paths
        self.update_state_after_rewarding(agent_index)

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

            # Update distances between agents
            self.distances.agents = get_distances_between_agents(
                self=self, is_set_diagonal=True
            )
            self.collisions.with_agents[:] = False  # Reset
            self.collisions.with_lanelets[:] = False  # Reset

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

                # Check for collisions between agents and lanelet boundaries
                collision_with_left_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.left_boundary[:, a_i],
                    is_return_points=False,
                )  # [batch_dim]
                collision_with_right_boundary = interX(
                    L1=self.vertices[:, a_i],
                    L2=self.ref_paths_agent_related.right_boundary[:, a_i],
                    is_return_points=False,
                )  # [batch_dim]
                self.collisions.with_lanelets[
                    (collision_with_left_boundary | collision_with_right_boundary), a_i
                ] = True


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

        (
            self.ref_paths_agent_related.short_term[:, agent_index],
            _,
        ) = get_short_term_reference_path_simple(
            polyline=self.ref_paths_agent_related.long_term[:, agent_index],
            index_closest_point=self.distances.closest_point_on_ref_path[
                :, agent_index
            ],
            n_points_to_return=self.n_points_short_term,
            device=self.world.device,
            sample_interval=1,
        )

        if not self.is_observe_distance_to_boundaries:
            # Get nearing points on boundaries
            (
                self.ref_paths_agent_related.nearing_points_left_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.left_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_left_b[
                    :, agent_index
                ],
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=1,
                n_points_shift=-2,
            )
            (
                self.ref_paths_agent_related.nearing_points_right_boundary[
                    :, agent_index
                ],
                _,
            ) = get_short_term_reference_path_simple(
                polyline=self.ref_paths_agent_related.right_boundary[:, agent_index],
                index_closest_point=self.distances.closest_point_on_right_b[
                    :, agent_index
                ],
                n_points_to_return=self.n_points_nearing_boundary,
                device=self.world.device,
                sample_interval=1,
                n_points_shift=-2,
            )
    
    def done(self):
        """
        This function computes the done flag for each env in a vectorized way.
        """
        is_collision_with_agents = self.collisions.with_agents.view(
            self.world.batch_dim, -1
        ).any(
            dim=-1
        )  # [batch_dim]
        is_collision_with_lanelets = self.collisions.with_lanelets.any(dim=-1)
        is_done = is_collision_with_agents | is_collision_with_lanelets

        return is_done

    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        返回：字典，值均为 [B, ...] 张量，用于日志与评估
        建议包含：docked 标志、dock_timer、最近锚点距离/越界、最小边界距离、与他车/货物最小间距、是否不可达降级等
        """
        B = self.batch_dim
        device = self.device

        if agent in self.followers:
            i = self.followers.index(agent)
            return {
                "docked": self.dock_state[:, i].float(),     # [B]
                "dock_timer": self.dock_timer[:, i],         # [B]
            }
        else:
            # 牵引车给空或基础指标
            return {
                "docked": torch.zeros(B, device=device),
                "dock_timer": torch.zeros(B, device=device),
            }

    def extra_render(self, env_index: int = 0):
        """
        画三类要素：
        1) 路网中心线（road_pts）
        2) 超大件：用 p_rear ↔ p_front 生成加厚矩形；并在矩形上画出各个铰接点（latch）
        - 绿色：被某随动车辆占用（docked & bound 到该点）
        - 灰色：空闲
        3) 随动车辆中心黑点；若 docked，则画一条连线到其绑定的铰接点
        """
        from vmas.simulator import rendering

        if self.is_real_time_rendering:
            if self.timer.step[0] == 0:
                pause_duration = 0  # Not sure how long should the simulation be paused at time step 0, so rather 0
            else:
                pause_duration = self.world.dt - (time.time() - self.timer.render_begin)
            if pause_duration > 0:
                time.sleep(pause_duration)

            self.timer.render_begin = time.time()  # Update
        geoms = []
        # ---------- 1) 路网中心线 ----------
        if hasattr(self, "road_pts"):
            pts = self.road_pts[env_index]  # [N,2]
            # rendering.PolyLine 接受 list[(x,y)]，转成 python list 更稳
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.GRAY.value, alpha=0.7)
            geoms.append(geom)
        
        # ---------- 2) 道路左边界（黑实线） ----------
        if hasattr(self, "road_left_pts"):
            left_pts = self.road_left_pts[env_index]  # [N,2]
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in left_pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(2.0)  # 设置左边界线宽度
            geoms.append(geom)
        
        # ---------- 3) 道路右边界（黑实线） ----------
        if hasattr(self, "road_right_pts"):
            right_pts = self.road_right_pts[env_index]  # [N,2]
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in right_pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(2.0)  # 设置右边界线宽度
            geoms.append(geom)


        # ---------- 2) 超大件（杆 + 锚点） ----------
        # 端点坐标
        p_front = self.road_C(self.s_front)[env_index]  # [2]
        p_rear  = self.road_C(self.s_rear)[env_index]   # [2]
        pf = p_front.detach().cpu()
        pr = p_rear.detach().cpu()
        rod = pf - pr
        rod_len = torch.linalg.norm(rod).item() + 1e-9
        t_hat = (rod / rod_len)            # 切向
        n_hat = torch.tensor([-t_hat[1], t_hat[0]])  # 法向（左法向）

        # 超大件半宽（可选参数）
        cargo_half_w = self.cargo_half_width

        # 四个角点：rear±n*w, front±n*w
        rear_left   = (pr + n_hat * cargo_half_w).tolist()
        rear_right  = (pr - n_hat * cargo_half_w).tolist()
        front_left  = (pf + n_hat * cargo_half_w).tolist()
        front_right = (pf - n_hat * cargo_half_w).tolist()

        # 用 PolyLine(close=True) 画矩形外框
        cargo_outline = rendering.PolyLine(
            v=[tuple(rear_left), tuple(front_left), tuple(front_right), tuple(rear_right)],
            close=True
        )
        cargo_outline.set_color(*Color.BLACK.value, alpha=0.9)
        geoms.append(cargo_outline)

        # # 中心线（细线，帮助看朝向）
        # cargo_axis = rendering.PolyLine(v=[tuple(pr.tolist()), tuple(pf.tolist())], close=False)
        # cargo_axis.set_color(*Color.BLACK.value, alpha=0.8)
        # geoms.append(cargo_axis)

        # 锚点：self.latch_pos_world[B, nL, 2]
        if hasattr(self, "latch_pos_world") and hasattr(self, "n_latch"):
            latch_xy = self.latch_pos_world[env_index]  # [nL,2]
            nL = int(self.n_latch)

            # 计算占用：该 env 下，是否有随动 docked 且绑定此 latch_id
            occ = torch.zeros(nL, dtype=torch.bool, device=latch_xy.device)
            if hasattr(self, "bound_latch_id") and hasattr(self, "dock_state"):
                ids = self.bound_latch_id[env_index]    # [F]
                dkd = self.dock_state[env_index]        # [F] bool
                for j in range(nL):
                    occ[j] = torch.any((ids == j) & dkd)

            for j in range(nL):
                x, y = latch_xy[j].detach().cpu().tolist()
                circle = rendering.make_circle(radius=0.15, filled=True)
                xform = rendering.Transform()
                circle.add_attr(xform)
                xform.set_translation(float(x), float(y))
                if bool(occ[j].item()):
                    circle.set_color(*(0.2, 0.8, 0.2), alpha=1.0)  # 绿色：占用
                else:
                    circle.set_color(*(0.5, 0.5, 0.5), alpha=0.9)  # 灰色：空闲
                geoms.append(circle)

        # ---------- 3) 随动车辆：中心黑点 + docked 连线 ----------
        if hasattr(self, "followers"):
            for i, ag in enumerate(self.followers):
                pos = ag.state.pos[env_index].detach().cpu().tolist()
                # 中心黑点
                dot = rendering.make_circle(radius=0.12, filled=True)
                xf = rendering.Transform()
                dot.add_attr(xf)
                xf.set_translation(float(pos[0]), float(pos[1]))
                dot.set_color(*Color.BLACK.value)  # 黑点
                geoms.append(dot)

                # 若 docked，画到其锚点的连线
                if hasattr(self, "dock_state") and hasattr(self, "bound_latch_id") and hasattr(self, "latch_pos_world"):
                    if bool(self.dock_state[env_index, i].item()):
                        lid = int(self.bound_latch_id[env_index, i].item())
                        if lid >= 0 and lid < self.n_latch:
                            latch_p = self.latch_pos_world[env_index, lid].detach().cpu().tolist()
                            l = rendering.PolyLine(v=[tuple(pos), tuple(latch_p)], close=False)
                            l.set_color(*(0.1, 0.6, 0.1), alpha=0.9)
                            geoms.append(l)

        # （可选）标注：端点索引、Δs 不可达标记等 —— 未来再加
        return geoms

    # =========================
    # —— 辅助函数（全部张量化）——
    # =========================
    # 提示：以下函数仅列出职责与 I/O；实现时保持纯张量运算 + 掩码/where 分支，避免 Python for

    # ---- 道路几何 ----
    def road_C(self, s: Tensor) -> Tensor:
        """
        输入:
            s: [B] 或 [B,K] 的弧长参数，单位需与 self.road_cum_s 一致
        需要的成员:
            self.road_cum_s: [B, N] 各环境折线节点的累积弧长（严格非降）
            self.road_pts:   [B, N, 2] 各环境折线节点坐标
            (可选) self.road_valid_n: [B] 每个环境有效节点数；若不存在视为全满 N
        输出:
            p: [B,2] 或 [B,K,2] 上对应的坐标（分段线性插值）
        说明:
            - 使用 torch.searchsorted 在最后维度上找段索引（向量化、支持 GPU）
            - 自动夹取到合法弧长范围 [s_min, s_max]
        """
        cum_s = self.road_cum_s              # [B, N]
        pts   = self.road_pts                # [B, N, 2]
        B, N  = cum_s.shape
        eps   = 1e-8

        # 兼容 [B] 或 [B,K]
        s_in = s
        if s.dim() == 1:
            s = s[:, None]                   # -> [B,1]
            squeeze_back = True
        else:
            squeeze_back = False

        # 每环境有效范围
        if hasattr(self, "road_valid_n") and self.road_valid_n is not None:
            # 最小/最大弧长（按有效节点）
            idx_last = torch.clamp(self.road_valid_n - 1, min=1)              # [B]
            s_min = cum_s[:, 0]
            s_max = cum_s.gather(-1, idx_last[:, None]).squeeze(-1)           # [B]
        else:
            s_min = cum_s[:, 0]
            s_max = cum_s[:, -1]

        # 夹取 s 到合法范围（广播到 [B,K]）
        s = torch.maximum(s, s_min[:, None])
        s = torch.minimum(s, s_max[:, None] - eps)

        # searchsorted: 在最后一维上搜索；返回右边界索引（段右端点）
        # idx_right ∈ [1, N-1]，我们使用左端点 idx0 = idx_right - 1
        idx_right = torch.searchsorted(cum_s, s, right=False)                 # [B,K]
        idx0 = torch.clamp(idx_right - 1, min=0, max=N-2)                     # [B,K]
        idx1 = idx0 + 1                                                       # [B,K]

        # 取段端点的 s 值
        s0 = torch.take_along_dim(cum_s, idx0, dim=-1)                        # [B,K]
        s1 = torch.take_along_dim(cum_s, idx1, dim=-1)                        # [B,K]
        denom = (s1 - s0).clamp_min(eps)
        t = (s - s0) / denom                                                  # [B,K] in [0,1]

        # 取段端点坐标并线性插值
        # 扩展索引用于 [B, N, 2] 按 dim=-2 抓取
        gather_idx0 = idx0[..., None].expand(-1, -1, 2)                       # [B,K,2]
        gather_idx1 = idx1[..., None].expand(-1, -1, 2)                       # [B,K,2]
        p0 = torch.take_along_dim(pts, gather_idx0, dim=-2)                   # [B,K,2]
        p1 = torch.take_along_dim(pts, gather_idx1, dim=-2)                   # [B,K,2]
        p  = p0 + t[..., None] * (p1 - p0)                                    # [B,K,2]

        if squeeze_back:
            p = p[:, 0, :]                                                    # [B,2]
        return p

    # ===== 固定弦长：批量求 Δs（向量化二分） =====
    def solve_delta_s(self, s_front: Tensor, L: Tensor, *, max_iter: int = 20) -> Tuple[Tensor, Tensor]:
        """
        输入:
            s_front: [B] 前端点弧长
            L:       [B] 固定弦长（货物长度）
        需要的成员:
            self.s_start:   [B] 每环境的起点弧长（用于下界）
            self.road_cum_s / self.road_pts: 供 road_C 调用
        输出:
            delta_s:        [B] 解出的 Δs (>=0)
            infeasible:     [B] bool 掩码；若在给定上界下最大弦长仍 < L，则置 True（需降速/脱开）
        数值:
            - 固定迭代步数、纯张量 where 更新，无 Python 分支
            - 单调性假设: chord(Δs) 随 Δs 近似单调增（对折线/样条在小步足够）
        """
        B = s_front.shape[0]
        eps = 1e-8

        # 下/上界
        lo = torch.zeros_like(s_front)                                         # [B]
        # 允许跨多段: hi = s_front - s_start（若 <0 则 0）
        hi = torch.clamp(s_front - self.s_start, min=0.0)                      # [B]

        # 预计算前端点坐标
        p_f = self.road_C(s_front)                                            # [B,2]

        # 不可解判断：最大 Δs=hi 时的弦长仍 < L
        p_r_hi = self.road_C(s_front - hi)                                    # [B,2]
        chord_max = torch.linalg.norm(p_f - p_r_hi, dim=-1)                   # [B]
        infeasible = (chord_max + 1e-6) < L                                   # [B] 需要外部策略处理

        # 向量化二分
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)                                             # [B]
            p_r = self.road_C(s_front - mid)                                  # [B,2]
            chord = torch.linalg.norm(p_f - p_r, dim=-1)                      # [B]
            go_left = chord > L                                               # [B]
            hi = torch.where(go_left, mid, hi)
            lo = torch.where(go_left, lo, mid)

        delta_s = 0.5 * (lo + hi)                                             # [B]
        # 对不可解样本，保持 delta_s 为 0（或可改为 hi），由上层降速/脱开处理
        delta_s = torch.where(infeasible, torch.zeros_like(delta_s), delta_s)
        return delta_s, infeasible

    def _check_batch_index(self, env_index: int):
        """检查批次索引是否有效"""
        if env_index < 0 or env_index >= self.batch_dim:
            raise ValueError(f"Invalid env_index {env_index}, must be in [0, {self.batch_dim})")
if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )

'''
from typing import Dict, List, Tuple, Optional
import torch
from torch import Tensor
from vmas import render_interactively

from vmas.simulator.scenario import BaseScenario
from vmas.simulator.core import World, Agent, Sphere, Box  # 按你的工程结构调整
from vmas.simulator.utils import Color
from vmas.simulator.dynamics.kinematic_bicycle import KinematicBicycle   # 视你的动力学模块而定
from vmas.simulator import rendering
X, Y = 0, 1  # 方便阅读

class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        self.device = device
        self.batch_dim = batch_dim
        self.init_params(batch_dim, device, **kwargs)
        world = self.init_world(batch_dim, device)
        self.init_agents(world, batch_dim, device)
        return world

    # ========== 1) 读取参数 & 分配并行批缓存 ==========
    def init_params(self, batch_dim: int, device: torch.device, **kwargs):
        # world params
        self.device = device
        self.batch_dim = batch_dim
        self.dt = float(kwargs.get("dt", 0.05))

        # Visualization
        self.visualize_semidims=False
        self.viewer_zoom = float(kwargs.get("viewer_zoom", 1.0))
        self.world_x_dim = kwargs.pop(
            "world_x_dim", 50
        )  # The x-dimension of the world in [m]
        self.world_y_dim = kwargs.pop(
            "world_y_dim", 40
        )  # The y-dimension of the world in [m]
        self.resolution_factor = kwargs.pop("resolution_factor", 20)  # Default 200
        self.render_origin = kwargs.pop(
            "render_origin", [self.world_x_dim / 2, self.world_y_dim / 2*3]
        )
        self.viewer_size = kwargs.pop(
            "viewer_size",
            (
                int(self.world_x_dim * self.resolution_factor),
                int(self.world_y_dim * self.resolution_factor),
            ),
        )
        # agent params
        # 生成道路
        B = batch_dim
        self.v_front_ref = torch.full((B,), kwargs.get("v_front_ref", 6.0), device=device)  # 前端参考速度
        self.n_followers = int(kwargs.get("n_followers", 3))

        self.rod_len     = float(kwargs.get("rod_len", 40.0))   # 货物长度 L
        self.n_latch = int(kwargs.get("n_latch", 5))
        self.cargo_half_width = float(kwargs.get("cargo_half_width", 2.5))
        self.max_speed = float(kwargs.get("max_speed", 7.0))
        self.max_steering_angle = kwargs.pop(
            "max_steering_angle",
            torch.deg2rad(torch.tensor(35, device=device, dtype=torch.float32)),
        )
        self.agent_width = float(kwargs.get("agent_width", 3.0))
        self.l_f = float(kwargs.get("l_f", 2.7))
        self.l_r = float(kwargs.get("l_r", 2.8))

        #map params
        straight_length = 50.0
        radius = 30.0
        pts_gap=1.0
        self.road_width  = float(kwargs.get("road_width", 10.0))  # 道路宽度
        self.road_half_width = self.road_width / 2.0
        
        # 生成道路
        B = batch_dim
        n_straight = int(straight_length // pts_gap)
        n_circle = int((3.14*radius) // pts_gap)
        points = []
        # 第一段直线 (0,0) -> (straight_length,0)
        x1 = torch.linspace(1.0, straight_length-1.0, n_straight, device=device)
        y1 = torch.zeros(n_straight, device=device)
        points.append(torch.stack([x1, y1], dim=-1))
        
        # 第一个半圆 (straight_length,0) -> (straight_length,2*radius)
        theta1 = torch.linspace(0.0, torch.pi, n_circle, device=device)
        x2 = straight_length + radius * torch.sin(theta1)
        y2 = radius - radius * torch.cos(theta1)
        points.append(torch.stack([x2, y2], dim=-1))
        
        # 第二段直线 (straight_length,2*radius) -> (0,2*radius)
        x3 = torch.linspace(straight_length-1.0, 1.0, n_straight, device=device)
        y3 = torch.full((n_straight,), 2*radius, device=device)
        points.append(torch.stack([x3, y3], dim=-1))
        
        # 第二个半圆 (0,2*radius) -> (0,0)
        theta2 = torch.linspace(torch.pi, 2*torch.pi, n_circle, device=device)
        x4 = radius * torch.sin(theta2)
        y4 = 3*radius + radius * torch.cos(theta2)
        points.append(torch.stack([x4, y4], dim=-1))
        
        # 第三段直线 (straight_length,2*radius) -> (0,2*radius)
        x5 = torch.linspace(1.0, straight_length-1.0, n_straight, device=device)
        y5 = torch.full((n_straight,), 4*radius, device=device)
        points.append(torch.stack([x5, y5], dim=-1))
        
        # 合并所有点
        road_pts = torch.cat(points, dim=0)
        
        # 扩展到batch维度
        self.road_pts = road_pts.expand(B, -1, 2)
        # 计算累积弧长
        seg = self.road_pts[:, 1:, :] - self.road_pts[:, :-1, :]
        seg_len = torch.linalg.norm(seg, dim=-1)
        zero = torch.zeros(B, 1, device=device)
        self.road_cum_s = torch.cat([zero, torch.cumsum(seg_len, dim=-1)], dim=-1)
        self.s_start = self.road_cum_s[:, 0]
        
        # 3. 生成左右边界点
        tangents = self.road_pts[:, 1:, :] - self.road_pts[:, :-1, :]  # [B, N-1, 2]
        norm_tangents = torch.linalg.norm(tangents, dim=-1, keepdim=True) + 1e-8  # [B, N-1, 1]
        unit_tangents = tangents / norm_tangents  # [B, N-1, 2]
        normals = torch.stack([-unit_tangents[..., 1], unit_tangents[..., 0]], dim=-1)  # [B, N-1, 2]
        
        point_normals = torch.zeros_like(self.road_pts)  # [B, N, 2]
        point_normals[:, 0, :] = normals[:, 0, :]
        mid_normals = (normals[:, :-1, :] + normals[:, 1:, :]) / 2  # [B, N-2, 2]
        point_normals[:, 1:-1, :] = mid_normals
        point_normals[:, -1, :] = normals[:, -1, :]
        point_normals = point_normals / torch.linalg.norm(point_normals, dim=-1, keepdim=True) + 1e-8
        
        self.road_left_pts = self.road_pts + point_normals * self.road_half_width  # [B, N, 2]
        self.road_right_pts = self.road_pts - point_normals * self.road_half_width  # [B, N, 2]

        # ---- 前/后端的弧长（按路起点放置）----
        # 初始：前端在 s_start + rod_len，后端在 s_start（如果路够长；否则夹取）
        s0 = self.s_start
        s1 = torch.clamp(s0 + self.rod_len, max=self.road_cum_s[:, -1] - 1e-6)
        self.s_front = s1.clone()   # [B]
        self.s_rear  = s0.clone()   # [B]

        # ---- 锚点（沿杆的比例 alpha）----
        self.latch_alpha = torch.linspace(0.0, 1.0, self.n_latch, device=device)  # [n_latch]
        self.latch_pos_world   = torch.zeros(B, self.n_latch, 2, device=device)   # [B,nL,2]
        self.latch_theta_world = torch.zeros(B, self.n_latch,   device=device)    # [B,nL]

        # ---- 随动车辆的 dock 状态/绑定锚点 ----
        F = self.n_followers
        self.dock_state      = torch.zeros(B, F, dtype=torch.bool, device=device) # 全部 free
        self.bound_latch_id  = torch.full((B, F), -1, dtype=torch.long, device=device)

        # 目标位姿缓存（post_step 投影用）
        self.target_pos   = torch.zeros(B, F, 2, device=device)
        self.target_theta = torch.zeros(B, F,   device=device)

        # 计时器（奖励/日志可用）
        self.dock_timer = torch.zeros(B, F, device=device)
        
        # 初始化 infeasible_mask
        self.infeasible_mask = torch.zeros(B, dtype=torch.bool, device=device)
        

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

        # 牵引车（脚本控制，不进 policy_agents）
        # 设置合适的形状和动作范围
        self.tractor_front = Agent(
            name="tractor_front", 
            shape=Box(length=self.l_f + self.l_r, width=self.agent_width),
            color=Color.RED,
            collide=False,
            render_action=False,
            u_range=[
                self.max_speed,
                self.max_steering_angle,
            ],  # Control command serves as velocity command
            u_multiplier=[1, 1],
            max_speed=self.max_speed,
            dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                world,
                width=self.agent_width,
                l_f=self.l_f,
                l_r=self.l_r,
                max_steering_angle=self.max_steering_angle,
                integration="rk4",  # one of {"euler", "rk4"}
            ),
        )
        self.tractor_rear  = Agent(
            name="tractor_rear",  
            shape=Box(length=self.l_f + self.l_r, width=self.agent_width),
            color=Color.BLUE,
            collide=False,
            render_action=False,
            u_range=[
                self.max_speed,
                self.max_steering_angle,
            ],  # Control command serves as velocity command
            u_multiplier=[1, 1],
            max_speed=self.max_speed,
            dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                world,
                width=self.agent_width,
                l_f=self.l_f,
                l_r=self.l_r,
                max_steering_angle=self.max_steering_angle,
                integration="rk4",  # one of {"euler", "rk4"}
            ),
        )
        world.add_agent(self.tractor_front)
        world.add_agent(self.tractor_rear)

        for i in range(self.n_followers):
            a = Agent(
                    name=f"follower_{i}", 
                    shape=Box(length=self.l_f + self.l_r, width=self.agent_width),
                    color=tuple(
                        torch.rand(3, device=world.device, dtype=torch.float32).tolist()
                    ),
                    collide=False,
                    render_action=False,
                    u_range=[
                        self.max_speed,
                        self.max_steering_angle,
                    ],  # Control command serves as velocity command
                    u_multiplier=[1, 1],
                    max_speed=self.max_speed,
                    dynamics=KinematicBicycle(  # Use the kinematic bicycle model for each agent
                        world,
                        width=self.agent_width,
                        l_f=self.l_f,
                        l_r=self.l_r,
                        max_steering_angle=self.max_steering_angle,
                        integration="rk4",  # one of {"euler", "rk4"}
                    ),
                )
            world.add_agent(a)
            self.followers.append(a)

    # ========== 4) 重置 ==========
    def reset_world_at(self, env_index: Optional[int] = None, agent_index: Optional[int] = None):
        """
        - env_index=None：重置全部并行环境
        - 否则只重置指定环境
        - 初始化：s_front/s_rear → 端点坐标 → 牵引车位姿 → 随动初始位姿（随机放在杆附近但不 dock）
        - 预计算一遍锚点位姿缓存
        """
        B = self.batch_dim
        device = self.device

        if env_index is None:
            idx_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            self._check_batch_index(env_index)  # 如果你沿用 VMAS 的检查
            idx_mask = torch.zeros(B, dtype=torch.bool, device=device)
            idx_mask[env_index] = True

        # ---- 放置前/后端弧长 ----
        # 让 rear 在 s_start，front = s_start + L（夹取到道路范围）
        s0 = self.s_start.clone()
        s1 = torch.clamp(s0 + self.rod_len, max=self.road_cum_s[:, -1] - 1e-6)
        self.s_front = torch.where(idx_mask, s1, self.s_front)
        self.s_rear  = torch.where(idx_mask, s0, self.s_rear)

        # ---- 端点坐标与朝向（用于设置牵引车初始位姿）----
        p_front = self.road_C(self.s_front)    # [B,2]
        p_rear  = self.road_C(self.s_rear)     # [B,2]
        # 杆方向
        rod_vec = (p_front - p_rear)           # [B,2]
        rod_theta = torch.atan2(rod_vec[:, 1], rod_vec[:, 0])  # [B]

        # 设置牵引车初始位姿（只对 mask 的样本赋值）
        def _set_pose(agent: Agent, pos: Tensor, theta: Tensor):
            if hasattr(agent.state, "pos"):
                agent.state.pos[idx_mask] = pos[idx_mask]
            if hasattr(agent.state, "rot"):
                # Ensure theta has the right shape for assignment
                theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
                agent.state.rot[idx_mask] = theta_reshaped[idx_mask]
            elif hasattr(agent.state, "angle"):
                # Ensure theta has the right shape for assignment
                theta_reshaped = theta.unsqueeze(-1) if theta.dim() == 1 else theta
                agent.state.angle[idx_mask] = theta_reshaped[idx_mask]
            if hasattr(agent.state, "vel"):
                agent.state.vel[idx_mask] = 0.0
        # 计算道路切线方向而不是使用货物方向
        front_theta = self.road_tangent(self.s_front)
        rear_theta = self.road_tangent(self.s_rear)
        
        # 设置牵引车初始位姿
        _set_pose(self.tractor_front, p_front, front_theta)
        _set_pose(self.tractor_rear, p_rear, rear_theta)

        # ---- 随动：初始 free & 随机放置在杆附近（不 dock）----
        self.dock_state[idx_mask, :]     = False
        self.bound_latch_id[idx_mask, :] = -1
        self.dock_timer[idx_mask, :]     = 0.0

        # 在 rear→front 方向均匀分布随动车辆
        alpha = torch.linspace(0.0, 1.0, self.n_followers + 2, device=device)[1:-1]  # 首尾去掉(牵引车位置)
        alpha = alpha.expand(B, -1)  # [B,F]
        base = p_rear[:, None, :] + alpha[..., None] * rod_vec[:, None, :]  # [B,F,2]
        
        # 设置随动初始位姿(无横向偏移)
        pos_f_init = base

        for i, ag in enumerate(self.followers):
            if hasattr(ag.state, "pos"):
                ag.state.pos[idx_mask] = pos_f_init[idx_mask, i, :]
            if hasattr(ag.state, "rot"):
                # Ensure rod_theta has the right shape for assignment
                theta_reshaped = rod_theta.unsqueeze(-1) if rod_theta.dim() == 1 else rod_theta
                ag.state.rot[idx_mask] = theta_reshaped[idx_mask]  # 朝向与杆一致先
            elif hasattr(ag.state, "angle"):
                # Ensure rod_theta has the right shape for assignment
                theta_reshaped = rod_theta.unsqueeze(-1) if rod_theta.dim() == 1 else rod_theta
                ag.state.angle[idx_mask] = theta_reshaped[idx_mask]
            if hasattr(ag.state, "vel"):
                ag.state.vel[idx_mask] = 0.0

        # ---- 预计算锚点世界位姿（沿杆等间距）----
        # latch_pos = p_rear + alpha * (p_front - p_rear)
        self.latch_pos_world = p_rear[:, None, :] + self.latch_alpha[None, :, None] * rod_vec[:, None, :]
        self.latch_theta_world = rod_theta[:, None].expand(-1, self.n_latch)

        # 给一个初始的 target（free 也写当前位姿，post_step 会用 mask 过滤）
        self.compute_latch_targets()

    # =========================
    # 核心动作处理/几何解算
    # =========================
    def process_action(self, agent):
        """
        调用时序：在 world.step() 之前，且在 env 分发完 RL 动作之后
        作用：docked 时覆盖随动车辆的物理动作；free 时保持 RL 动作
        """
        # 只处理随动车辆
        if agent not in self.followers:
            return

        i = self.followers.index(agent)            # 随动车辆在列表中的索引
        docked_mask = self.dock_state[:, i]        # [B] bool

        # ---- 硬约束：docked 时忽略物理控制（例如 [速度, 转向] 两维清零）----
        # 动作张量形状通常是 [B, action_dim]；只改物理通道，保留你设计的 attach/undock 位
        if docked_mask.any() and hasattr(agent, 'action') and agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
            # 把该 agent 在 batch 中 d==True 的样本的物理控制清零
            u = agent.action.u                      # [B, action_dim]
            # 假设前两维为物理控制（按你的动作定义改）
            if u.shape[-1] >= 2:
                u[docked_mask, :2] =  torch.zeros_like(u[docked_mask, :2])
                agent.action.u = u

    def pre_step(self):
        """
        每次 world.step() 之前：
        1) 推进前端弧长 s_front
        2) 固定弦长解 Δs -> s_rear
        3) 计算端点坐标与杆朝向
        4) 更新所有锚点的世界位姿（供 docked 目标用）
        5) 计算随动车辆的 target_pos / target_theta（按绑定锚点）
        """

        # ---- 1) 推进前端弧长并夹取到道路范围 ----
        s_front = self.s_front + self.v_front_ref * self.dt                   # [B]
        # 不要超过道路最大 s；留一点 eps 免得插值越界
        s_max = self.road_cum_s[:, -1] - 1e-6
        s_min = self.s_start
        s_front = torch.clamp(s_front, min=s_min, max=s_max)

        # ---- 2) 固定弦长解 Δs -> s_rear ----
        delta_s, infeasible = self.solve_delta_s(s_front, torch.full_like(s_front, self.rod_len))
        s_rear = s_front - delta_s
        s_rear = torch.clamp(s_rear, min=s_min, max=s_max)

        # 缓存回成员（别忘了）
        self.s_front = s_front
        self.s_rear  = s_rear
        self.infeasible_mask = infeasible    # 你后面可据此降速/强制 undock，加惩罚等

        # ---- 3) 端点坐标与杆朝向 ----
        p_front = self.road_C(s_front)                                         # [B,2]
        p_rear  = self.road_C(s_rear)                                          # [B,2]
        rod_vec = p_front - p_rear                                             # [B,2]
        theta_rod = torch.atan2(rod_vec[:, 1], rod_vec[:, 0])                  # [B]

        # ---- 4) 更新锚点世界位姿（等间距示例；若你有自定义分布，就替换这里）----
        # latch at: p = p_rear + α * (p_front - p_rear)
        self.latch_pos_world   = p_rear[:, None, :] + self.latch_alpha[None, :, None] * rod_vec[:, None, :]
        self.latch_theta_world = theta_rod[:, None].expand(-1, self.n_latch)   # 全部用杆朝向

        # （可选）更新牵引车脚本动作的目标，这里只算，不写 action；你可在 env_process_action 里写入

        # ---- 5) 计算随动车辆目标（按绑定锚点），shape 必须是 [B, nF, 2] / [B, nF] ----
        self.compute_latch_targets()  # 这个函数我们之前已经给了实现


    def post_step(self):
        """
        每次 world.step() 之后：
        - 对 docked 的随动车辆做硬投影（对齐到各自锚点）
        - 统计 dock 计时等
        """
        B = self.batch_dim

        for i, agent in enumerate(self.followers):
            docked_mask = self.dock_state[:, i]      # [B] bool
            if not docked_mask.any():
                continue

            # 注意：这里的 target_* 是 [B, n_followers, ...]
            pos_i   = self.target_pos[:, i, :]       # [B, 2]
            theta_i = self.target_theta[:, i]        # [B]

            self._project_agent_to_pose(
                agent,
                pos_i[docked_mask, :],               # [M,2]
                theta_i[docked_mask],               # [M]
                mask=docked_mask,                   # [B]
            )
        # 更新牵引车位置以跟随货物（修改后的版本）
        p_front = self.road_C(self.s_front)    # [B,2]
        p_rear  = self.road_C(self.s_rear)     # [B,2]
        # 计算道路切线方向而不是使用货物方向
        front_theta = self.road_tangent(self.s_front)
        rear_theta = self.road_tangent(self.s_rear)
        rod_theta = torch.atan2(p_front[:,1]-p_rear[:,1], p_front[:,0]-p_rear[:,0])  # [B]
        
        # 分别为前后牵引车设置各自的道路切线方向
        for i, agent in enumerate([self.tractor_front, self.tractor_rear]):
            if hasattr(agent.state, "pos"):
                # 使用杆端点位置
                pos = p_front if i == 0 else p_rear
                agent.state.pos = pos
            if hasattr(agent.state, "rot"):
                # 使用道路切线方向
                theta = front_theta if i == 0 else rear_theta
                agent.state.rot = theta.unsqueeze(-1)
            elif hasattr(agent.state, "angle"):
                # 使用道路切线方向
                theta = front_theta if i == 0 else rear_theta
                agent.state.angle = theta
        

        # 计时器（可用于奖励）：每步给 docked 的样本累加 dt
        self.dock_timer += self.dock_state.to(self.dock_timer.dtype) * self.dt
    # 添加一个方法来计算道路切线方向
    def road_tangent(self, s: Tensor) -> Tensor:
        """计算道路上弧长s处的切线方向角"""
        # 使用小扰动法计算切线方向
        epsilon = 1e-3  # 小扰动值
        s_plus = torch.clamp(s + epsilon, max=self.road_cum_s[:, -1] - 1e-6)
        pos_plus = self.road_C(s_plus)  # [B,2]
        pos = self.road_C(s)  # [B,2]
        tangent_vec = pos_plus - pos
        # 计算切线方向角（弧度）
        tangent_theta = torch.atan2(tangent_vec[:, 1], tangent_vec[:, 0])
        return tangent_theta

    def compute_latch_targets(self):
        """
        写出:
        self.target_pos   : [B, n_followers, 2]
        self.target_theta : [B, n_followers]
        规则:
        - 若某随动处于 docked 且有有效锚点 id(>=0)，目标=锚点位姿
        - 否则(自由态或无效 id)，目标=当前车位姿(这样 post_step 的硬投影对自由态不产生影响)
        """
        device = self.device
        B = self.batch_dim
        nF = len(self.followers)

        # 取出所有随动的当前位姿，拼成张量，方便向量化
        # pos_f: [B, nF, 2], theta_f: [B, nF]
        pos_f_list = []
        theta_f_list = []
        for ag in self.followers:
            pos_f_list.append(ag.state.pos)         # [B,2]
            # 兼容不同版本的朝向字段: rot / angle
            if hasattr(ag.state, "rot"):
                theta_f_list.append(ag.state.rot.squeeze(-1))   # [B] - remove last dimension if it's [B,1]
            elif hasattr(ag.state, "angle"):
                theta_f_list.append(ag.state.angle.squeeze(-1)) # [B] - remove last dimension if it's [B,1]
            else:
                raise AttributeError("Agent.state has no 'rot' or 'angle'")
        pos_f   = torch.stack(pos_f_list, dim=1).to(device)   # [B,nF,2]
        theta_f = torch.stack(theta_f_list, dim=1).to(device) # [B,nF]

        # 目标张量初始化为"当前位姿"（自由态默认不动）
        target_pos   = pos_f.clone()
        target_theta = theta_f.clone()

        # 锚点表
        latch_pos   = self.latch_pos_world.to(device)     # [B, nL, 2]
        latch_theta = self.latch_theta_world.to(device)   # [B, nL]
        nL = latch_pos.shape[1]

        # 有效(docked 且 id>=0)的掩码
        dock = self.dock_state.to(device)                 # [B, nF]
        ids  = self.bound_latch_id.to(device).long()      # [B, nF]
        valid = dock & (ids >= 0)                         # [B, nF]

        if valid.any():
            # 为 gather 构造索引
            # pos: [B,nF,2] <- latch_pos.gather(dim=1, idx=[B,nF,2])
            idx_pos = ids.clamp(min=0, max=nL-1)[..., None].expand(-1, -1, 2)  # [B,nF,2]
            sel_pos = torch.gather(latch_pos, dim=1, index=idx_pos)            # [B,nF,2]

            # theta: [B,nF] <- take_along_dim(latch_theta, idx=[B,nF])
            sel_theta = torch.take_along_dim(latch_theta, ids.clamp(min=0, max=nL-1), dim=1)  # [B,nF]

            # 仅对 valid 的位置/朝向进行覆盖
            valid_exp = valid.unsqueeze(-1)                                    # [B,nF,1]
            target_pos   = torch.where(valid_exp, sel_pos,   target_pos)
            target_theta = torch.where(valid,     sel_theta, target_theta)

        # 缓存输出，供 post_step 使用
        self.target_pos   = target_pos   # [B, nF, 2]
        self.target_theta = target_theta # [B, nF]


    def _project_agent_to_pose(self, agent, pos_batch: torch.Tensor, theta_batch: torch.Tensor, mask: torch.Tensor):
        """
        把该 agent 在 mask==True 的那些 batch 样本的状态(位置/朝向/速度)硬对齐到给定目标:
        - agent.state.pos[mask] = pos_batch
        - agent.state.rot/angle[mask] = theta_batch
        - 速度: 为了稳定，清零(或按需设置为沿切向的目标速度)
        形状:
        - agent.state.pos : [B,2]
        - pos_batch       : [sum(mask), 2]
        - theta_batch     : [sum(mask)]
        - mask            : [B] (bool)
        """
        if not mask.any():
            return

        # 取 mask 索引
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)  # [M], M = sum(mask)

        # --- 位置 ---
        agent.state.pos[idx] = pos_batch.to(agent.state.pos.dtype)

        # --- 朝向 ---
        if hasattr(agent.state, "rot"):
            # Ensure theta_batch has the right shape for assignment
            theta_reshaped = theta_batch.unsqueeze(-1) if theta_batch.dim() == 1 else theta_batch
            agent.state.rot[idx] = theta_reshaped.to(agent.state.rot.dtype)
        elif hasattr(agent.state, "angle"):
            # Ensure theta_batch has the right shape for assignment
            theta_reshaped = theta_batch.unsqueeze(-1) if theta_batch.dim() == 1 else theta_batch
            agent.state.angle[idx] = theta_reshaped.to(agent.state.angle.dtype)
        else:
            raise AttributeError("Agent.state has no 'rot' or 'angle' to set orientation")

        # --- 线速度/角速度（硬约束：清零最稳；如需更物理，可填入锚点切向速度） ---
        if hasattr(agent.state, "vel"):
            # vel: [B,2]
            agent.state.vel[idx] = 0.0
        if hasattr(agent.state, "omega"):
            # 有的实现有角速度标量
            agent.state.omega[idx] = 0.0

    def _check_batch_index(self, env_index: int):
        """检查批次索引是否有效"""
        if env_index < 0 or env_index >= self.batch_dim:
            raise ValueError(f"Invalid env_index {env_index}, must be in [0, {self.batch_dim})")

    def get_scenario_info(self):
        """获取场景信息，用于调试和验证"""
        return {
            "batch_dim": self.batch_dim,
            "n_followers": self.n_followers,
            "rod_len": self.rod_len,
            "dt": self.dt,
            "device": str(self.device),
            "road_points": self.road_pts.shape,
            "n_latch": self.n_latch
        }

    # =========================
    # 观测 / 奖励 / 终止 / 信息
    # =========================
    def observation(self, agent):
        B = self.batch_dim
        device = self.device

        # 自车基础量
        pos = agent.state.pos                       # [B,2]
        if hasattr(agent.state, "rot"):
            theta = agent.state.rot.squeeze(-1)    # [B] - remove last dimension if it's [B,1]
        elif hasattr(agent.state, "angle"):
            theta = agent.state.angle.squeeze(-1)  # [B] - remove last dimension if it's [B,1]
        else:
            theta = torch.zeros(B, device=device)

        # 是否是随动车辆（用于取 dock_state）
        if agent in self.followers:
            i = self.followers.index(agent)
            docked = self.dock_state[:, i].float().unsqueeze(-1)   # [B,1]
        else:
            docked = torch.zeros(B, 1, device=device)

        # 简单观测： [x, y, cosθ, sinθ, docked]
        obs = torch.cat([pos,
                        torch.cos(theta).unsqueeze(-1),
                        torch.sin(theta).unsqueeze(-1),
                        docked], dim=-1)           # [B,5]
        return obs

    def reward(self, agent):
        B = self.batch_dim
        device = self.device
        rew = torch.zeros(B, device=device)

        # 只给随动车辆奖励
        if agent in self.followers:
            i = self.followers.index(agent)
            # 每步保持 docked 给正奖励
            alpha = 0.01
            rew = rew + alpha * self.dock_state[:, i].float()

            # （可选 placeholder）若有 infeasible 批次、强制 undock 可给轻惩罚
            # rew = rew - 0.01 * self.infeasible_mask.float()

        return rew


    def done(self) -> Tensor:
        """
        返回：[B] bool 张量
        典型：
        - 碰撞/严重越界/到达终点 → True
        - 也可训练时持续运行（始终 False），只在违规个体 reset_world_at(agent_index=...)；
          测试时再按整环境规则 done
        """
        return torch.zeros(self.batch_dim, dtype=torch.bool, device=self.device)


    def info(self, agent: Agent) -> Dict[str, Tensor]:
        """
        返回：字典，值均为 [B, ...] 张量，用于日志与评估
        建议包含：docked 标志、dock_timer、最近锚点距离/越界、最小边界距离、与他车/货物最小间距、是否不可达降级等
        """
        B = self.batch_dim
        device = self.device

        if agent in self.followers:
            i = self.followers.index(agent)
            return {
                "docked": self.dock_state[:, i].float(),     # [B]
                "dock_timer": self.dock_timer[:, i],         # [B]
            }
        else:
            # 牵引车给空或基础指标
            return {
                "docked": torch.zeros(B, device=device),
                "dock_timer": torch.zeros(B, device=device),
            }

    def extra_render(self, env_index: int = 0):
        """
        画三类要素：
        1) 路网中心线（road_pts）
        2) 超大件：用 p_rear ↔ p_front 生成加厚矩形；并在矩形上画出各个铰接点（latch）
        - 绿色：被某随动车辆占用（docked & bound 到该点）
        - 灰色：空闲
        3) 随动车辆中心黑点；若 docked，则画一条连线到其绑定的铰接点
        """
        
        geoms = []
        # ---------- 1) 路网中心线 ----------
        if hasattr(self, "road_pts"):
            pts = self.road_pts[env_index]  # [N,2]
            # rendering.PolyLine 接受 list[(x,y)]，转成 python list 更稳
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.GRAY.value, alpha=0.7)
            geoms.append(geom)
        
        # ---------- 2) 道路左边界（黑实线） ----------
        if hasattr(self, "road_left_pts"):
            left_pts = self.road_left_pts[env_index]  # [N,2]
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in left_pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(2.0)  # 设置左边界线宽度
            geoms.append(geom)
        
        # ---------- 3) 道路右边界（黑实线） ----------
        if hasattr(self, "road_right_pts"):
            right_pts = self.road_right_pts[env_index]  # [N,2]
            geom = rendering.PolyLine(v=[(float(x), float(y)) for x, y in right_pts.detach().cpu().tolist()],
                                    close=False)
            geom.set_color(*Color.BLACK.value, alpha=1.0)
            geom.set_linewidth(2.0)  # 设置右边界线宽度
            geoms.append(geom)


        # ---------- 2) 超大件（杆 + 锚点） ----------
        # 端点坐标
        p_front = self.road_C(self.s_front)[env_index]  # [2]
        p_rear  = self.road_C(self.s_rear)[env_index]   # [2]
        pf = p_front.detach().cpu()
        pr = p_rear.detach().cpu()
        rod = pf - pr
        rod_len = torch.linalg.norm(rod).item() + 1e-9
        t_hat = (rod / rod_len)            # 切向
        n_hat = torch.tensor([-t_hat[1], t_hat[0]])  # 法向（左法向）

        # 超大件半宽（可选参数）
        cargo_half_w = self.cargo_half_width

        # 四个角点：rear±n*w, front±n*w
        rear_left   = (pr + n_hat * cargo_half_w).tolist()
        rear_right  = (pr - n_hat * cargo_half_w).tolist()
        front_left  = (pf + n_hat * cargo_half_w).tolist()
        front_right = (pf - n_hat * cargo_half_w).tolist()

        # 用 PolyLine(close=True) 画矩形外框
        cargo_outline = rendering.PolyLine(
            v=[tuple(rear_left), tuple(front_left), tuple(front_right), tuple(rear_right)],
            close=True
        )
        cargo_outline.set_color(*Color.BLACK.value, alpha=0.9)
        geoms.append(cargo_outline)

        # # 中心线（细线，帮助看朝向）
        # cargo_axis = rendering.PolyLine(v=[tuple(pr.tolist()), tuple(pf.tolist())], close=False)
        # cargo_axis.set_color(*Color.BLACK.value, alpha=0.8)
        # geoms.append(cargo_axis)

        # 锚点：self.latch_pos_world[B, nL, 2]
        if hasattr(self, "latch_pos_world") and hasattr(self, "n_latch"):
            latch_xy = self.latch_pos_world[env_index]  # [nL,2]
            nL = int(self.n_latch)

            # 计算占用：该 env 下，是否有随动 docked 且绑定此 latch_id
            occ = torch.zeros(nL, dtype=torch.bool, device=latch_xy.device)
            if hasattr(self, "bound_latch_id") and hasattr(self, "dock_state"):
                ids = self.bound_latch_id[env_index]    # [F]
                dkd = self.dock_state[env_index]        # [F] bool
                for j in range(nL):
                    occ[j] = torch.any((ids == j) & dkd)

            for j in range(nL):
                x, y = latch_xy[j].detach().cpu().tolist()
                circle = rendering.make_circle(radius=0.15, filled=True)
                xform = rendering.Transform()
                circle.add_attr(xform)
                xform.set_translation(float(x), float(y))
                if bool(occ[j].item()):
                    circle.set_color(*(0.2, 0.8, 0.2), alpha=1.0)  # 绿色：占用
                else:
                    circle.set_color(*(0.5, 0.5, 0.5), alpha=0.9)  # 灰色：空闲
                geoms.append(circle)

        # ---------- 3) 随动车辆：中心黑点 + docked 连线 ----------
        if hasattr(self, "followers"):
            for i, ag in enumerate(self.followers):
                pos = ag.state.pos[env_index].detach().cpu().tolist()
                # 中心黑点
                dot = rendering.make_circle(radius=0.12, filled=True)
                xf = rendering.Transform()
                dot.add_attr(xf)
                xf.set_translation(float(pos[0]), float(pos[1]))
                dot.set_color(*Color.BLACK.value)  # 黑点
                geoms.append(dot)

                # 若 docked，画到其锚点的连线
                if hasattr(self, "dock_state") and hasattr(self, "bound_latch_id") and hasattr(self, "latch_pos_world"):
                    if bool(self.dock_state[env_index, i].item()):
                        lid = int(self.bound_latch_id[env_index, i].item())
                        if lid >= 0 and lid < self.n_latch:
                            latch_p = self.latch_pos_world[env_index, lid].detach().cpu().tolist()
                            l = rendering.PolyLine(v=[tuple(pos), tuple(latch_p)], close=False)
                            l.set_color(*(0.1, 0.6, 0.1), alpha=0.9)
                            geoms.append(l)

        # （可选）标注：端点索引、Δs 不可达标记等 —— 未来再加
        return geoms

    # =========================
    # —— 辅助函数（全部张量化）——
    # =========================
    # 提示：以下函数仅列出职责与 I/O；实现时保持纯张量运算 + 掩码/where 分支，避免 Python for

    # ---- 道路几何 ----
    def road_C(self, s: Tensor) -> Tensor:
        """
        输入:
            s: [B] 或 [B,K] 的弧长参数，单位需与 self.road_cum_s 一致
        需要的成员:
            self.road_cum_s: [B, N] 各环境折线节点的累积弧长（严格非降）
            self.road_pts:   [B, N, 2] 各环境折线节点坐标
            (可选) self.road_valid_n: [B] 每个环境有效节点数；若不存在视为全满 N
        输出:
            p: [B,2] 或 [B,K,2] 上对应的坐标（分段线性插值）
        说明:
            - 使用 torch.searchsorted 在最后维度上找段索引（向量化、支持 GPU）
            - 自动夹取到合法弧长范围 [s_min, s_max]
        """
        cum_s = self.road_cum_s              # [B, N]
        pts   = self.road_pts                # [B, N, 2]
        B, N  = cum_s.shape
        eps   = 1e-8

        # 兼容 [B] 或 [B,K]
        s_in = s
        if s.dim() == 1:
            s = s[:, None]                   # -> [B,1]
            squeeze_back = True
        else:
            squeeze_back = False

        # 每环境有效范围
        if hasattr(self, "road_valid_n") and self.road_valid_n is not None:
            # 最小/最大弧长（按有效节点）
            idx_last = torch.clamp(self.road_valid_n - 1, min=1)              # [B]
            s_min = cum_s[:, 0]
            s_max = cum_s.gather(-1, idx_last[:, None]).squeeze(-1)           # [B]
        else:
            s_min = cum_s[:, 0]
            s_max = cum_s[:, -1]

        # 夹取 s 到合法范围（广播到 [B,K]）
        s = torch.maximum(s, s_min[:, None])
        s = torch.minimum(s, s_max[:, None] - eps)

        # searchsorted: 在最后一维上搜索；返回右边界索引（段右端点）
        # idx_right ∈ [1, N-1]，我们使用左端点 idx0 = idx_right - 1
        idx_right = torch.searchsorted(cum_s, s, right=False)                 # [B,K]
        idx0 = torch.clamp(idx_right - 1, min=0, max=N-2)                     # [B,K]
        idx1 = idx0 + 1                                                       # [B,K]

        # 取段端点的 s 值
        s0 = torch.take_along_dim(cum_s, idx0, dim=-1)                        # [B,K]
        s1 = torch.take_along_dim(cum_s, idx1, dim=-1)                        # [B,K]
        denom = (s1 - s0).clamp_min(eps)
        t = (s - s0) / denom                                                  # [B,K] in [0,1]

        # 取段端点坐标并线性插值
        # 扩展索引用于 [B, N, 2] 按 dim=-2 抓取
        gather_idx0 = idx0[..., None].expand(-1, -1, 2)                       # [B,K,2]
        gather_idx1 = idx1[..., None].expand(-1, -1, 2)                       # [B,K,2]
        p0 = torch.take_along_dim(pts, gather_idx0, dim=-2)                   # [B,K,2]
        p1 = torch.take_along_dim(pts, gather_idx1, dim=-2)                   # [B,K,2]
        p  = p0 + t[..., None] * (p1 - p0)                                    # [B,K,2]

        if squeeze_back:
            p = p[:, 0, :]                                                    # [B,2]
        return p

    # ===== 固定弦长：批量求 Δs（向量化二分） =====
    def solve_delta_s(self, s_front: Tensor, L: Tensor, *, max_iter: int = 20) -> Tuple[Tensor, Tensor]:
        """
        输入:
            s_front: [B] 前端点弧长
            L:       [B] 固定弦长（货物长度）
        需要的成员:
            self.s_start:   [B] 每环境的起点弧长（用于下界）
            self.road_cum_s / self.road_pts: 供 road_C 调用
        输出:
            delta_s:        [B] 解出的 Δs (>=0)
            infeasible:     [B] bool 掩码；若在给定上界下最大弦长仍 < L，则置 True（需降速/脱开）
        数值:
            - 固定迭代步数、纯张量 where 更新，无 Python 分支
            - 单调性假设: chord(Δs) 随 Δs 近似单调增（对折线/样条在小步足够）
        """
        B = s_front.shape[0]
        eps = 1e-8

        # 下/上界
        lo = torch.zeros_like(s_front)                                         # [B]
        # 允许跨多段: hi = s_front - s_start（若 <0 则 0）
        hi = torch.clamp(s_front - self.s_start, min=0.0)                      # [B]

        # 预计算前端点坐标
        p_f = self.road_C(s_front)                                            # [B,2]

        # 不可解判断：最大 Δs=hi 时的弦长仍 < L
        p_r_hi = self.road_C(s_front - hi)                                    # [B,2]
        chord_max = torch.linalg.norm(p_f - p_r_hi, dim=-1)                   # [B]
        infeasible = (chord_max + 1e-6) < L                                   # [B] 需要外部策略处理

        # 向量化二分
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)                                             # [B]
            p_r = self.road_C(s_front - mid)                                  # [B,2]
            chord = torch.linalg.norm(p_f - p_r, dim=-1)                      # [B]
            go_left = chord > L                                               # [B]
            hi = torch.where(go_left, mid, hi)
            lo = torch.where(go_left, lo, mid)

        delta_s = 0.5 * (lo + hi)                                             # [B]
        # 对不可解样本，保持 delta_s 为 0（或可改为 hi），由上层降速/脱开处理
        delta_s = torch.where(infeasible, torch.zeros_like(delta_s), delta_s)
        return delta_s, infeasible

    def _check_batch_index(self, env_index: int):
        """检查批次索引是否有效"""
        if env_index < 0 or env_index >= self.batch_dim:
            raise ValueError(f"Invalid env_index {env_index}, must be in [0, {self.batch_dim})")
if __name__ == "__main__":
    render_interactively(
        __file__,
        control_two_agents=True,
    )

'''