from vmas.scenarios.road_traffic import CircularBuffer, get_perpendicular_distances
import torch
from torch import Tensor
from vmas.scenarios.occt_map import OcctMap,OcctCRMap,MapBase
from vmas.simulator.core import World, Agent, Sphere, Box
import torch.nn.functional as F
class OcctConstants:
    # Predefined constants that may be used during simulations
    def __init__(
        self,
        env_idx_broadcasting: Tensor = None,
        empty_action_acc: Tensor = None,
        empty_action_steering: Tensor = None,
        mask_pos: Tensor = None,
        mask_vel: Tensor = None,
        mask_rot: Tensor = None,
        mask_zero: Tensor = None,
        mask_one: Tensor = None,
        reset_agent_min_distance: Tensor = None,
    ):
        self.env_idx_broadcasting = env_idx_broadcasting
        self.empty_action_acc = empty_action_acc
        self.empty_action_steering = empty_action_steering
        self.mask_pos = mask_pos
        self.mask_zero = mask_zero
        self.mask_one = mask_one
        self.mask_vel = mask_vel
        self.mask_rot = mask_rot
        self.reset_agent_min_distance = reset_agent_min_distance  # The minimum distance between agents when being reset

class OcctNormalizers:
    """Normalizers for positions, velocities, rotations, etc."""

    def __init__(
        self,
        pos=None,
        error_pos=None,
        pos_world=None,
        v=None,
        error_v=None,
        rot=None,
        action_steering=None,
        action_vel=None,
        action_steering_rate=None,
        action_acc=None,
        distance_lanelet=None,
        distance_agent=None,
        distance_ref=None,
        hinge_step=None,
    ):
        self.pos = pos
        self.error_pos = error_pos
        self.pos_world = pos_world
        self.v = v
        self.error_v = error_v
        self.rot = rot
        self.action_steering = action_steering
        self.action_vel = action_vel
        self.action_steering_rate = action_steering_rate
        self.action_acc = action_acc
        self.distance_lanelet = distance_lanelet
        self.distance_agent = distance_agent
        self.distance_ref = distance_ref
        self.hinge_step = hinge_step
class OcctRewards:
    """Reward weights used by the current OCCT platoon reward pipeline."""
    def __init__(
        self,
        reward_progress=None,
        weighting_ref_directions=None,
        reward_vel=None,
        reward_goal=None,
        reward_platoon_heading=None,
        reward_platoon_space=None,
        reward_hinge_space=None,
        reward_platoon_vel=None,
        reward_hinge_vel=None,
        reward_platoon_ref=None,
        reward_hinge_ref=None,
        reward_hinge=None,
    ):
        self.reward_progress = reward_progress
        self.weighting_ref_directions = weighting_ref_directions
        self.reward_vel = reward_vel
        self.reward_goal = reward_goal
        self.reward_platoon_heading = reward_platoon_heading
        self.reward_platoon_space = reward_platoon_space
        self.reward_hinge_space = reward_hinge_space
        self.reward_platoon_vel = reward_platoon_vel
        self.reward_hinge_vel = reward_hinge_vel
        self.reward_platoon_ref = reward_platoon_ref
        self.reward_hinge_ref = reward_hinge_ref
        self.reward_hinge = reward_hinge

class OcctPenalties:
    """Penalties for collisions, being too close to other agents or lane boundaries, etc."""

    def __init__(
        self,
        near_boundary=None,
        near_other_agents=None,
        collide_with_agents=None,
        collide_with_boundaries=None,
        backward=None,
        change_steering=None,
        change_acc=None,
        hinge_time_cost=None,
    ):
        self.near_boundary = (
            near_boundary  # Penalty for being too close to lanelet boundaries
        )
        self.near_other_agents = (
            near_other_agents  # Penalty for being too close to other agents
        )
        self.collide_with_agents = (
            collide_with_agents  # Penalty for colliding with other agents
        )
        self.collide_with_boundaries = (
            collide_with_boundaries  # Penalty for colliding with lanelet boundaries
        )
        self.backward = backward  # Penalty for leaving the world
        self.change_steering = (
            change_steering  # Penalty for changing steering direction
        )
        self.change_acc = change_acc  # Penalty for changing acceleration direction
        self.hinge_time_cost = hinge_time_cost

class OcctThresholds:
    """Different thresholds, such as starting from which distance agents are deemed being too close to other agents."""

    def __init__(
        self,
        near_boundary_low=None,
        near_boundary_high=None,
        near_other_agents_low=None,
        near_other_agents_high=None,
        change_steering=None,
        change_acc=None,
        distance_mask_agents=None,
    ):
        self.near_boundary_low = near_boundary_low
        self.near_boundary_high = near_boundary_high
        self.near_other_agents_low = near_other_agents_low
        self.near_other_agents_high = near_other_agents_high
        self.change_steering = change_steering  # Threshold above which agents will be penalized for changing steering too quick [degree]
        self.change_acc = change_acc  # Threshold above which agents will be penalized for changing acceleration too quick [m/s^2]
        self.distance_mask_agents = (
            distance_mask_agents  # Threshold above which nearing agents will be masked
        )

class OcctObservations:
    def __init__(
        self,
        is_partial=None,
        n_nearing_agents=None,
        nearing_agents_indices=None,
        noise_level=None,
        n_stored_steps=None,
        n_observed_steps=None,
        error_vel=None,
        platoon_error_vel=None,
        hinge_error_vel=None,
        past_platoon_error_vel: CircularBuffer = None,
        past_hinge_error_vel: CircularBuffer = None,
        self_platoon_error_space: CircularBuffer = None,
        agent_hinge_status: CircularBuffer = None,
        agent_s=None,
        past_pos: CircularBuffer = None,
        past_rot: CircularBuffer = None,
        past_vertices: CircularBuffer = None,
        past_vel: CircularBuffer = None,
        past_steering: CircularBuffer = None,
        past_relative_ref_info: CircularBuffer = None,
        past_relative_hinge_info: CircularBuffer = None,
        past_action_acc: CircularBuffer = None,
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
        self.error_vel = error_vel  # Front/rear relative longitudinal velocity in ego-local frame
        self.platoon_error_vel = platoon_error_vel  # Platoon-mode front/rear speed errors
        self.hinge_error_vel = hinge_error_vel  # Hinge-mode target speed errors
        self.past_platoon_error_vel = past_platoon_error_vel
        self.past_hinge_error_vel = past_hinge_error_vel
        self.self_platoon_error_space = self_platoon_error_space  # Platoon gap error relative to reference gap (unnormalized)
        self.agent_hinge_status = agent_hinge_status  # 车辆铰接状态（0：未铰接，1：铰接），铰接后车辆被动行驶且奖励屏蔽
        self.agent_s = agent_s  # Arc length position
        
        self.past_pos = past_pos  # Past positions
        self.past_rot = past_rot  # Past rotations
        self.past_vertices = past_vertices  # Past vertices
        self.past_vel = past_vel  # Past velocites
        self.past_steering = past_steering  # Past steering actions

        self.past_relative_ref_info = (
            past_relative_ref_info  # Past short-term reference points
        )
        self.past_relative_hinge_info = (
            past_relative_hinge_info  # Past short-term hinge points
        )
        self.past_left_boundary = past_left_boundary  # Past left lanelet boundary
        self.past_right_boundary = past_right_boundary  # Past right lanelet boundary

        self.past_action_acc = past_action_acc  # Past velocity action
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

class OcctDistances:
    def __init__(
        self,
        agents=None,
        agents_frenet=None,
        left_boundaries=None,
        right_boundaries=None,
        boundaries=None,
        ref_paths=None,
        lookahead_pts=None,
        closest_point_on_ref_path=None,
        closest_point_on_left_b=None,
        closest_point_on_right_b=None,
        goal=None,
        obstacles=None,
    ):
        self.agents = agents  # Distances between agents
        self.agents_frenet = agents_frenet  # Frenet distances between agents
        self.left_boundaries = left_boundaries  # Distances between agents and the left boundaries of their current lanelets (for each vertex of each agent)
        self.right_boundaries = right_boundaries  # Distances between agents and the right boundaries of their current lanelets (for each vertex of each agent)
        self.boundaries = boundaries  # The minimum distances between agents and the boundaries of their current lanelets
        self.ref_paths = ref_paths  # Distances between agents and the center line of their current lanelets
        self.lookahead_pts = lookahead_pts  # Distances between agents and the lookahead points
        self.closest_point_on_ref_path = (
            closest_point_on_ref_path  # Index of the closest point on reference path
        )
        self.closest_point_on_left_b = (
            closest_point_on_left_b  # Index of the closest point on left boundary
        )
        self.closest_point_on_right_b = (
            closest_point_on_right_b  # Index of the closest point on right boundary
        )
        self.goal = goal  # Distances to goal positions
        self.obstacles = obstacles  # Distances to obstacles

class OcctReferencePathsAgentRelated:
    def __init__(
        self,
        long_term: Tensor = None,
        left_boundary: Tensor = None,
        right_boundary: Tensor = None,
        nearing_points_left_boundary: Tensor = None,
        nearing_points_right_boundary: Tensor = None,
        short_term: Tensor = None,
        hinge_short_term: Tensor = None,
        short_term_indices: Tensor = None,
        agent_hinge_status: CircularBuffer = None,
        hinge_status: Tensor = None,
        hinge_heading_vel_angle_diff_deg: Tensor = None,
        agent_heading_hinge_heading_angle_diff_deg: Tensor = None,
        exit: Tensor = None,
    ):
        self.long_term = long_term  # Actual long-term reference paths of agents
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.short_term = short_term  # Short-term reference path
        self.hinge_short_term = hinge_short_term  # Short-term reference path for the hinge
        self.short_term_indices = short_term_indices  # Indices that indicate which part of the long-term reference path is used to build the short-term reference path
        self.nearing_points_left_boundary = nearing_points_left_boundary  # Nearing left boundary
        self.nearing_points_right_boundary = nearing_points_right_boundary  # Nearing right boundary
        self.agent_hinge_status = agent_hinge_status  # Hinge status for each agent
        self.hinge_status = hinge_status  # Each Hinge status
        self.hinge_heading_vel_angle_diff_deg = hinge_heading_vel_angle_diff_deg
        self.agent_heading_hinge_heading_angle_diff_deg = (
            agent_heading_hinge_heading_angle_diff_deg
        )
        self.exit = exit  # Exit segment
        
def check_validity(obj):
    for attr_name, attr_value in obj.__dict__.items():
        if isinstance(attr_value, Tensor) and torch.isnan(attr_value).any():
            nan_indices = torch.nonzero(torch.isnan(attr_value), as_tuple=False)
            raise ValueError(f"NaN found in self.{attr_name}, index:{nan_indices}")
def check_boolean_block(tensor):
        """
        检查B×M布尔张量的每个样本是否为连续块（无0101交替），并判断顺序
        
        参数：
            tensor: torch.BoolTensor，形状为 [B, M]
        
        返回：
            is_block: torch.BoolTensor，形状 [B]，True表示该样本是连续块
            block_order: torch.LongTensor，形状 [B]，
                        -1=纯0，0=先0后1，1=先1后0，2=纯1
        """
        # 1. 转换为float张量（方便计算差值），形状 [B, M]
        device = tensor.device
        t_float = tensor.float()
        
        # 2. 计算相邻元素的差值（M-1个），形状 [B, M-1]
        diff = t_float[:, 1:] - t_float[:, :-1]
        
        # 3. 统计每个样本的切换次数（差值非0的数量）
        switch_count = torch.abs(diff).sum(dim=1).long()  # 形状 [B]
        
        # 4. 判断是否为连续块（切换次数≤1）
        is_block = switch_count <= 1  # 形状 [B]
        
        # 5. 初始化顺序标记：-1=纯0，0=先0后1，1=先1后0，2=纯1
        block_order = torch.full((tensor.shape[0],), -1, dtype=torch.long, device=device)
        
        # 6. 分别处理不同情况
        # 情况1：纯1（所有元素为1）
        all_ones = torch.all(tensor, dim=1)  # 形状 [B]
        block_order[all_ones] = 2
        
        # 情况2：纯0（所有元素为0）
        all_zeros = torch.all(~tensor, dim=1)  # 形状 [B]
        block_order[all_zeros] = -1
        
        # 情况3：有切换（切换次数=1），判断是先0后1还是先1后0
        has_switch = switch_count == 1  # 形状 [B]
        if torch.any(has_switch):
            # 找到每个样本的切换位置（第一个差值非0的索引）
            switch_mask = (diff != 0)  # 形状 [B, M-1]
            # 找到第一个切换位置（若没有则为M-1）
            switch_pos = torch.argmax(switch_mask.int(), dim=1)  # 形状 [B]
            # 切换位置的前一个值：0→先0后1，1→先1后0
            pre_switch_val = t_float[has_switch, switch_pos[has_switch]]
            # 赋值：前值为0 → 先0后1（0）；前值为1 → 先1后0（1）
            block_order[has_switch] = (pre_switch_val == 1).long()
        
        return is_block, block_order

def get_short_term_reference_path_simple(
    polyline: Tensor,
    index_closest_point: Tensor,
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
    # prevent index out of range as well as polyline NaN within batch
    has_nan = torch.isnan(polyline).any(dim=-1)  # [batch_size, num_points]
    # 对于每个样本，找到最后一个非NaN点的索引
    valid_indices = torch.arange(polyline.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
    valid_points_mask = ~has_nan
    assert not (valid_points_mask.sum(dim=-1)==0).any(), "Some samples have no valid points!"
    max_valid_indices = (valid_indices * valid_points_mask).argmax(dim=1)
    max_valid_indices_expanded = max_valid_indices.unsqueeze(1).expand(-1, n_points_to_return)
    min_valid_indices_expanded = torch.zeros_like(max_valid_indices_expanded)
    future_points_idx = torch.clamp(future_points_idx, \
                                    min_valid_indices_expanded, max_valid_indices_expanded)
    short_term_path = polyline[
        torch.arange(batch_size, device=device, dtype=torch.int).unsqueeze(
            1
        ),  # For broadcasting
        future_points_idx,
    ]
    has_nan = torch.isnan(short_term_path).any(dim=(1, 2))
    nan_sample_indices = torch.nonzero(has_nan).squeeze(dim=1)
    if len(nan_sample_indices):
        print(f"包含NaN的样本索引：{nan_sample_indices}")
        print(f"共有 {len(nan_sample_indices)} 个样本包含NaN")
        assert True, "Some samples have NaN points in the short-term reference path!"
    return short_term_path, future_points_idx


def get_short_term_reference_path_by_s(
    occt_map: MapBase,
    agent_s : Tensor, 
    n_points_to_return: int,
    device=None,
    sample_interval: int = 2,
    return_ref_v: bool = False,
    env_j: int = None,
    line: str = "center",
):
    """
    Args:
        occt_map:                   OcctMap or OcctCRMap.
        agent_s:                    [batch_size, 1] or [1] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_to_return:         [1] or []. In the case of the latter, batch_dim is deemed as 1.
        sample_interval:            Sample interval to match specific purposes;
                                    set to 2 when using this function to get the short-term reference path;
                                    set to 1 when using this function to get the nearing boundary points."""
    if device is None:
        device = torch.device("cpu")
    B=agent_s.shape[0] if agent_s.dim() else 1
    short_term_path=torch.zeros((B, n_points_to_return , 3 if return_ref_v else 2), device=device, dtype=torch.float32)
    agent_s_query=torch.stack([agent_s+i*sample_interval for i in range(n_points_to_return)],dim=-1)
    ref_pts = occt_map.get_pts(agent_s_query, env_j, line)
    short_term_path[:, :, :2] = ref_pts
    if return_ref_v:
        ref_v = occt_map.get_ref_v(agent_s_query, env_j).squeeze(dim=-1)
        short_term_path[:, :, 2] = ref_v
    
    has_nan = torch.isnan(short_term_path).any(dim=(1, 2))
    nan_sample_indices = torch.nonzero(has_nan).squeeze(dim=1)
    # if len(nan_sample_indices):
    #     print(f"包含NaN的样本索引：{nan_sample_indices}")
    #     print(f"共有 {len(nan_sample_indices)} 个样本包含NaN")
    #     assert True, "Some samples have NaN points in the short-term reference path!"
    if agent_s.dim()==0:
        return short_term_path[0,...]
    return short_term_path
def check_hinge_points_in_boundary(
    ref_left_boundary: torch.Tensor,  # [B, 24, 2] 左边界点集
    ref_right_boundary: torch.Tensor, # [B, 24, 2] 右边界点集
    hinge_short_term: torch.Tensor,   # [B, 4, 4, 2] 预瞄点 (x,y)
    K: float = 1.0,                   # 与边界的最小距离（米）
    return_boundary_margin: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    并行判断预瞄点是否在左右边界内且与边界至少距离K米
    核心修复：仅针对「最近的边界线段」判断位置合法性，适配弯道边界
    return_boundary_margin=True 时，boundary_margin 返回 signed feasibility margin:
    大于0表示在可行范围内，等于0表示恰好压在可行边界上，小于0表示越界或侵入K缓冲区
    返回形状：[B, n_agent, n_pts, 1] 的布尔张量（True=满足条件，False=不满足）
    """
    # ========== 1. 预处理维度，适配广播 ==========
    B = ref_left_boundary.shape[0]
    n_agent = hinge_short_term.shape[1]  # 4
    n_pts = hinge_short_term.shape[2]    # 4
    
    # 提取预瞄点 (x,y)：[B, 4, 4, 2]
    hinge_pts = hinge_short_term[..., :2].contiguous()  # 确保内存连续
    
    # ---------------- 左边界预处理 ----------------
    left_seg_start = ref_left_boundary[:, :-1, :]  # [B,23,2] 23条线段的起点
    left_seg_end = ref_left_boundary[:, 1:, :]    # [B,23,2] 23条线段的终点
    left_seg_vec = left_seg_end - left_seg_start  # [B,23,2] 线段向量
    # 扩维适配广播：[B,23,2] → [B,1,1,23,2]
    left_seg_start_exp = left_seg_start.unsqueeze(1).unsqueeze(1)
    left_seg_end_exp = left_seg_end.unsqueeze(1).unsqueeze(1)
    left_seg_vec_exp = left_seg_vec.unsqueeze(1).unsqueeze(1)
    
    # ---------------- 右边界预处理 ----------------
    right_seg_start = ref_right_boundary[:, :-1, :]  # [B,23,2]
    right_seg_end = ref_right_boundary[:, 1:, :]    # [B,23,2]
    right_seg_vec = right_seg_end - right_seg_start  # [B,23,2]
    # 扩维适配广播：[B,23,2] → [B,1,1,23,2]
    right_seg_start_exp = right_seg_start.unsqueeze(1).unsqueeze(1)
    right_seg_vec_exp = right_seg_vec.unsqueeze(1).unsqueeze(1)
    
    # ---------------- 预瞄点扩维 ----------------
    hinge_pts_exp = hinge_pts.unsqueeze(3)  # [B,4,4,1,2] → 适配23条线段的广播

    # ========== 2. 通用化计算：最近线段+位置判断+距离 ==========
    def get_nearest_seg_info(pt_exp, seg_start_exp, seg_vec_exp, is_left_bound: bool):
        """
        计算点到边界的「最近线段」的距离 + 位置合法性
        步骤：1. 找最近线段 → 2. 仅对最近线段判断位置 → 3. 计算垂直距离
        返回：min_dist [B,4,4], in_bound [B,4,4]
        """
        # ------------ 步骤1：计算点到每条线段的「原始距离」（无限制，找最近线段） ------------
        pt_to_start = pt_exp - seg_start_exp  # [B,4,4,23,2]
        # 投影比例（不限制0~1）
        seg_len_sq = torch.sum(seg_vec_exp**2, dim=-1, keepdim=False)  # [B,1,1,23]
        seg_len_sq = torch.clamp(seg_len_sq, min=1e-8)
        proj = torch.sum(pt_to_start * seg_vec_exp, dim=-1) / seg_len_sq  # [B,4,4,23]
        # 线段上的最近点（投影超出线段则取端点）
        proj_clamped = torch.clamp(proj, 0.0, 1.0).unsqueeze(-1)  # [B,4,4,23,1]
        closest_pt = seg_start_exp + proj_clamped * seg_vec_exp  # [B,4,4,23,2]
        # 点到每条线段的原始距离（用于找最近线段）
        seg_dist = torch.norm(pt_exp - closest_pt, dim=-1)  # [B,4,4,23]
        # 找到最近线段的索引：[B,4,4]
        nearest_seg_idx = torch.argmin(seg_dist, dim=-1)  # 每个点对应1条最近线段
        
        # ------------ 步骤2：仅对「最近线段」判断位置合法性 ------------
        # 生成最近线段的掩码：[B,4,4,23] → 仅最近线段为True，其余为False
        B_idx = torch.arange(B).view(B,1,1).expand(-1, n_agent, n_pts)
        agent_idx = torch.arange(n_agent).view(1,n_agent,1).expand(B, -1, n_pts)
        pt_idx = torch.arange(n_pts).view(1,1,n_pts).expand(B, n_agent, -1)
        nearest_mask = torch.zeros_like(seg_dist, dtype=torch.bool)  # [B,4,4,23]
        nearest_mask[B_idx, agent_idx, pt_idx, nearest_seg_idx] = True  # 仅最近线段为True
        
        # 计算叉乘（位置判断）：仅保留最近线段的叉乘结果
        cross = pt_to_start[..., 0] * seg_vec_exp[..., 1] - pt_to_start[..., 1] * seg_vec_exp[..., 0]  # [B,4,4,23]
        nearest_cross = torch.masked_select(cross, nearest_mask).reshape(B, n_agent, n_pts)  # [B,4,4]
        
        # 位置合法性判断（仅针对最近线段）
        if is_left_bound:
            # 左边界：点应在最近线段的右侧 → cross > 0（适配竖直/弯道线段）
            pos_valid = nearest_cross > 0
        else:
            # 右边界：点应在最近线段的左侧 → cross < 0
            pos_valid = nearest_cross < 0
        
        # ------------ 步骤3：计算到最近线段的垂直距离 ------------
        # 提取最近线段的距离：[B,4,4]
        min_dist = torch.gather(seg_dist, dim=-1, index=nearest_seg_idx.unsqueeze(-1)).squeeze(-1)
        
        # 位置合法 = 最近线段的位置正确
        in_bound = pos_valid  # [B,4,4]
        
        return min_dist, in_bound

    # ========== 3. 计算左/右边界的最近线段信息 ==========
    # 左边界：最近线段+位置+距离
    left_min_dist, left_in_bound = get_nearest_seg_info(
        hinge_pts_exp, left_seg_start_exp, left_seg_vec_exp, is_left_bound=True
    )
    # 右边界：最近线段+位置+距离
    right_min_dist, right_in_bound = get_nearest_seg_info(
        hinge_pts_exp, right_seg_start_exp, right_seg_vec_exp, is_left_bound=False
    )

    # ========== 4. 最终条件判断 ==========
    # 条件1：在左右边界内（仅需最近线段位置合法）
    cond_in_bound = left_in_bound & right_in_bound  # [B,4,4]
    # 条件2：与左边界距离 ≥ K
    cond_left_dist = (left_min_dist >= K)  # [B,4,4]
    # 条件3：与右边界距离 ≥ K
    cond_right_dist = (right_min_dist >= K)  # [B,4,4]
    
    # 最终条件：同时满足所有条件
    final_cond = cond_in_bound & cond_left_dist & cond_right_dist  # [B,4,4]
    left_signed_margin = torch.where(
        left_in_bound,
        left_min_dist - K,
        -(left_min_dist + K),
    )
    right_signed_margin = torch.where(
        right_in_bound,
        right_min_dist - K,
        -(right_min_dist + K),
    )
    boundary_margin = torch.minimum(left_signed_margin, right_signed_margin).unsqueeze(-1)
    # 调整形状为 [B,4,4,1]
    final_cond = final_cond.unsqueeze(-1)

    if return_boundary_margin:
        return final_cond.to(dtype=torch.bool), boundary_margin.to(dtype=torch.float32)
    return final_cond.to(dtype=torch.bool)
def get_short_term_hinge_path_by_s(
    occt_map: OcctCRMap,
    agents : Agent, 
    agent_s: Tensor,
    n_points_to_return: int,
    tractor_slice: list,
    device=None,
    sample_dt: int = 1,
    sample_ds: int = None,
    env_j: int = None,
    hinge_relative_pos: Tensor = None,
):
    if device is None:
        device = torch.device("cpu")
    
    # --- DEBUG START ---
    # print(f"\n[DEBUG] === Entering get_short_term_hinge_path_by_s ===")
    # print(f"[DEBUG] agent_s shape: {agent_s.shape}")
    # print(f"[DEBUG] n_points_to_return: {n_points_to_return}")
    # --- DEBUG END ---

    HINGE_FIRST_INDEX = tractor_slice[0]
    HINGE_LAST_INDEX = tractor_slice[-1]
    first_agent_s = agent_s[env_j, HINGE_FIRST_INDEX]
    last_agent_s = agent_s[env_j, HINGE_LAST_INDEX]
    B = agent_s.shape[0] if agent_s.dim() else 1
    hinge_pts_num = len(agents) if hinge_relative_pos is None else hinge_relative_pos.shape[0]
    hinge_short_term = torch.zeros((B, hinge_pts_num, n_points_to_return, 5), device=device, dtype=torch.float32)
    # print(f"[DEBUG] Initialized hinge_short_term shape: {hinge_short_term.shape}")

    # 提取速度
    first_agent_vel_vec = agents[HINGE_FIRST_INDEX].state.vel  # [B?, 2]
    last_agent_vel_vec = agents[HINGE_LAST_INDEX].state.vel    # [B?, 2]
    # print(f"[DEBUG] first_agent_vel_vec shape: {first_agent_vel_vec.shape}")

    first_vel_expanded = first_agent_vel_vec.unsqueeze(1).expand(-1, n_points_to_return, -1)
    last_vel_expanded = last_agent_vel_vec.unsqueeze(1).expand(-1, n_points_to_return, -1)
    # print(f"[DEBUG] first_vel_expanded shape: {first_vel_expanded.shape}")

    if sample_ds is None:
        first_agent_vel_mag = 2 * torch.ones_like(torch.linalg.norm(first_agent_vel_vec, dim=-1))
        last_agent_vel_mag = 2 * torch.ones_like(torch.linalg.norm(last_agent_vel_vec, dim=-1))
        first_agent_s_query = torch.stack([first_agent_s + i * sample_dt * first_agent_vel_mag for i in range(n_points_to_return)], dim=-1)
        last_agent_s_query = torch.stack([last_agent_s + i * sample_dt * last_agent_vel_mag for i in range(n_points_to_return)], dim=-1)
    else:
        tmp = torch.ones_like(torch.linalg.norm(first_agent_vel_vec, dim=-1))
        first_agent_s_query = torch.stack([first_agent_s + i * sample_ds * tmp for i in range(n_points_to_return)], dim=-1)
        last_agent_s_query = torch.stack([last_agent_s + i * sample_ds * tmp for i in range(n_points_to_return)], dim=-1)
    
    first_last_pred_traj = occt_map.get_pts(torch.cat([first_agent_s_query, last_agent_s_query], dim=1), env_j)
    first_pred_traj = first_last_pred_traj[:, :n_points_to_return] 
    last_pred_traj = first_last_pred_traj[:, n_points_to_return:] 
    # print(f"[DEBUG] first_pred_traj shape: {first_pred_traj.shape}")

    if hinge_relative_pos is None:
        for i in range(len(agents)):
            ratio = i / (len(agents) - 1) if len(agents) > 1 else 0.0
            hinge_short_term[..., i, :, :2] = first_pred_traj + ratio * (last_pred_traj - first_pred_traj)
            hinge_short_term[..., i, :, 2:4] = first_vel_expanded + ratio * (last_vel_expanded - first_vel_expanded)
    else:
        rod_vector = last_pred_traj - first_pred_traj 
        rod_length = torch.norm(rod_vector, dim=-1, keepdim=True) 
        rod_length_safe = torch.where(rod_length > 1e-6, rod_length, torch.ones_like(rod_length))
        rod_dir = rod_vector / rod_length_safe 
        rod_perp = torch.stack([rod_dir[..., 1], -rod_dir[..., 0]], dim=-1) 

        rel = hinge_relative_pos.unsqueeze(0).unsqueeze(2)  # [1, N_hinge, 1, 2]
        rod_dir_exp = rod_dir.unsqueeze(1)  # [B, 1, n_points, 2]
        rod_perp_exp = rod_perp.unsqueeze(1) 
        
        offset = rel[..., 0:1] * rod_perp_exp + rel[..., 1:2] * rod_dir_exp
        # print(f"[DEBUG] offset shape: {offset.shape}")

        r_ap = offset 
        vel_diff = last_vel_expanded - first_vel_expanded 
        vel_diff_dot_perp = torch.sum(vel_diff * rod_perp, dim=-1, keepdim=True)
        vel_diff_dot_perp_exp = vel_diff_dot_perp.unsqueeze(1) 
        
        omega = vel_diff_dot_perp_exp / rod_length_safe.unsqueeze(1)
        # print(f"[DEBUG] omega shape: {omega.shape}")

        # 计算转动速度
        rot_vel_raw = torch.stack([
            -omega * r_ap[..., 1:2], 
            omega * r_ap[..., 0:1]    
        ], dim=-1)
        # print(f"[DEBUG] rot_vel_raw (before squeeze) shape: {rot_vel_raw.shape}")
        
        rot_vel = rot_vel_raw.squeeze(-2) 
        # print(f"[DEBUG] rot_vel (after squeeze) shape: {rot_vel.shape}")

        first_vel_exp = first_vel_expanded.unsqueeze(1) 
        # print(f"[DEBUG] first_vel_exp shape: {first_vel_exp.shape}")
        
        n = hinge_relative_pos.size(0)
        target_slice_shape = hinge_short_term[:, :n, :, 2:4].shape
        # print(f"[DEBUG] Target slice shape: {target_slice_shape}")
        
        hinge_short_term[:, :n, :, 2:4] = first_vel_exp + rot_vel 
        first_pos_exp = first_pred_traj.unsqueeze(1)
        hinge_short_term[:, :n, :, 0:2] = first_pos_exp + offset
    # print(f"[DEBUG] Proceeding to ready bit calculation...")
    
    # 1. 基础 S 坐标处理
    end = first_agent_s_query[:, -1:].float()    # [B, 1]
    start = last_agent_s_query[:, 0:1].float()   # [B, 1]
    
    # size 计算（防止空插值）
    size_val = torch.max(end - start).item()
    size = max(int(round(size_val)), 1)
    
    start_end = torch.cat([start, end], dim=1).unsqueeze(1) # [B, 1, 2]
    interpolated = torch.nn.functional.interpolate(
        start_end, 
        size=size, 
        mode='linear', 
        align_corners=True
    ).squeeze(1) # [B, steps]
    
    ref_left_boundary = occt_map.get_pts(interpolated, env_j, line="left")
    ref_right_boundary = occt_map.get_pts(interpolated, env_j, line="right")
    
    _, boundary_margin = check_hinge_points_in_boundary(
        ref_left_boundary=ref_left_boundary,
        ref_right_boundary=ref_right_boundary,
        hinge_short_term=hinge_short_term,
        K=0,
        return_boundary_margin=True,
    )
    # is_after_corner = (torch.atleast_1d(first_agent_s) > corner_s).view(B, 1, 1, 1)
    # is_after_corner = is_after_corner.expand(-1, hinge_pts_num, n_points_to_return, 1)
    # is_in_straight = (torch.atleast_1d(first_agent_s) < (corner_s-25)).view(B, 1, 1, 1)
    # is_in_straight = is_in_straight.expand(-1, hinge_pts_num, n_points_to_return, 1)
    # hinge_ready_mask = is_in_boundary # & is_after_corner
    hinge_short_term[..., 4:5] = boundary_margin
    return hinge_short_term

def get_short_term_hinge_path_by_s_backup(
    occt_map: OcctCRMap,
    agents : Agent, 
    agent_s: Tensor,
    n_points_to_return: int,
    tractor_slice: list,
    device=None,
    sample_dt: int = 1,
    env_j: int = None,
):
    """
    Args:
        occt_map:                   OcctMap or OcctCRMap.
        agent_s:                    [batch_size, 1] or [1] or []. In the case of the latter, batch_dim is deemed as 1.
        n_points_to_return:         [1] or []. In the case of the latter, batch_dim is deemed as 1.
        sample_interval:            Sample interval to match specific purposes;
                                    set to 2 when using this function to get the short-term reference path;
                                    set to 1 when using this function to get the nearing boundary points."""
    if device is None:
        device = torch.device("cpu")
    HINGE_FIRST_INDEX=tractor_slice[0]
    HINGE_LAST_INDEX=tractor_slice[-1]
    first_agent_s = agent_s[env_j, HINGE_FIRST_INDEX]
    last_agent_s = agent_s[env_j, HINGE_LAST_INDEX]
    B=agent_s.shape[0] if agent_s.dim() else 1
    hinge_short_term=torch.zeros((B, len(agents), n_points_to_return , 3), device=device, dtype=torch.float32)
    first_agent_vel = torch.linalg.norm(agents[HINGE_FIRST_INDEX].state.vel, dim=-1)
    last_agent_vel = torch.linalg.norm(agents[HINGE_LAST_INDEX].state.vel, dim=-1)
    first_agent_s_query = torch.stack([first_agent_s+i*sample_dt*first_agent_vel for i in range(n_points_to_return)],dim=-1)
    last_agent_s_query = torch.stack([last_agent_s+i*sample_dt*last_agent_vel for i in range(n_points_to_return)],dim=-1)
    first_last_pred_traj = occt_map.get_pts(torch.cat([first_agent_s_query, last_agent_s_query], dim=1), env_j)
    first_pred_traj = first_last_pred_traj[:,:n_points_to_return] #[B, n_points_to_return, 2]
    last_pred_traj = first_last_pred_traj[:,n_points_to_return:] #[B, n_points_to_return, 2]
    for i in range(len(agents)):
        hinge_short_term[...,i,:,:2] = first_pred_traj + (i/(len(agents)-1))*(last_pred_traj - first_pred_traj)
    hinges_status = occt_map.get_hinge_status(first_agent_s_query, env_j).transpose(-2,-1) # [batch_size, 2] or [2]
    hinge_short_term[...,-1] = hinges_status
    return hinge_short_term

# def get_short_term_reference_path_by_s_all_agents_backup(
#     occt_map: OcctCRMap,
#     agent_s : Tensor, 
#     n_points_to_return: int,
#     device=None,
#     sample_dt: int = 2,
#     env_j: int = None,
#     line: str = "center",
# ):
#     """
#     Args:
#         occt_map:                   OcctMap or OcctCRMap.
#         agent_s:                    [batch_size, n_agents]. In the case of the latter, batch_dim is deemed as 1.
#         n_points_to_return:         [1] or []. In the case of the latter, batch_dim is deemed as 1.
#         sample_dt:                  Sample dt to match specific purposes"""
#     # warning: not compatible with env_j=int
#     if device is None:
#         device = torch.device("cpu")
#     assert agent_s.dim()==2, "agent_s must be [batch_size, n_agents]!"
#     B=agent_s.shape[0]
#     short_term_path = torch.zeros((B, agent_s.shape[1], n_points_to_return, 3), device=device, dtype=torch.float32)
#     if line=="center":
#         agent_s_query = torch.zeros((B, agent_s.shape[1], n_points_to_return), device=device, dtype=torch.float32)
#         agent_s_query[..., 0] = agent_s
#         short_term_path[..., 0, 2] = occt_map.get_ref_v(agent_s_query[:, :, 0], env_j).squeeze(dim=-1).reshape(B, agent_s.shape[1])
#         for i in range(1, n_points_to_return):
#             ref_v = occt_map.get_ref_v(agent_s_query[:, :, i-1], env_j).squeeze(dim=-1).reshape(B, agent_s.shape[1])
#             agent_s_query[..., i] = agent_s_query[:, :, i-1] + sample_dt * ref_v
#             short_term_path[..., i, 2] = ref_v
#     else:
#         agent_s_query=torch.hstack([agent_s+i*sample_dt*4 for i in range(n_points_to_return)])
#         ref_v = occt_map.get_ref_v(agent_s_query, env_j).squeeze(dim=-1).reshape(B, agent_s.shape[1], n_points_to_returny)
#         short_term_path[..., 2] = ref_v
#     agent_s_query = agent_s_query.reshape(B, -1)
#     ref_pts = occt_map.get_pts(agent_s_query, env_j, line).reshape(B, agent_s.shape[1], n_points_to_return, 2)
#     short_term_path[...,:2] = ref_pts
#     return short_term_path

def calibrate_agent_s_by_road_pts(
    agent_pos: torch.Tensor,
    ref_agent_s: torch.Tensor,
    road_get_pts_func,
    interval: float = 0.25,
    precision: float = 0.05,
    forward_search: bool = True,  # 重命名+默认True：默认向前搜索
    device: torch.device = None
) -> torch.Tensor:
    """
    基于道路中心线物理点校准agent_s（纯batch操作，兼容任意B值）
    核心：全程保留batch维度，默认向前搜索（适配车辆正常向前行驶场景）
    
    Args:
        agent_pos: 车辆当前位置，shape=[B, F, 2]（必须是3维，B≥1）
        ref_agent_s: 上一时刻的参考agent_s，shape=[B, F]（必须是2维，B≥1）
        road_get_pts_func: 道路获取点的函数，输入[s_list]返回[pts_list]，输入shape=[B, N]，输出shape=[B, N, 2]
        interval: 搜索范围（以ref_agent_s为中心的覆盖距离），单位m，默认0.5m
        precision: 搜索点的步长（精度），单位m，默认0.05m
        forward_search: 是否仅向前搜索（True=仅ref_agent_s ~ ref_agent_s+interval，False=双向搜索），默认True
        device: 计算设备（CPU/GPU），默认自动匹配agent_pos的设备
    
    Returns:
        new_agent_s: 校准后的agent_s，shape=[B, F]（与输入维度完全一致）
    """
    # 0. 严格校验输入维度（避免维度错误）
    assert len(agent_pos.shape) == 3 and agent_pos.shape[-1] == 2, \
        f"agent_pos必须是[B, F, 2]维度，当前为{agent_pos.shape}"
    assert len(ref_agent_s.shape) == 2, \
        f"ref_agent_s必须是[B, F]维度，当前为{ref_agent_s.shape}"
    assert agent_pos.shape[0] == ref_agent_s.shape[0] and agent_pos.shape[1] == ref_agent_s.shape[1], \
        f"agent_pos和ref_agent_s的B/F维度不匹配：agent_pos={agent_pos.shape}, ref_agent_s={ref_agent_s.shape}"
    
    # 设备适配
    if device is None:
        device = agent_pos.device
    agent_pos = agent_pos.to(device)
    ref_agent_s = ref_agent_s.to(device)
    
    # 1. 提取基础维度（全程保留B/F）
    B, F, _ = agent_pos.shape
    # 计算搜索点数（纯数值计算，不涉及tensor维度）
    if forward_search:
        # 仅向前搜索：ref_agent_s ~ ref_agent_s+interval（适配车辆向前行驶）
        n_nearby = int(interval / precision) + 1
        steps = torch.linspace(0, interval, n_nearby, device=device).reshape(1, 1, -1)
    else:
        # 双向搜索：ref_agent_s-interval ~ ref_agent_s+interval
        n_nearby = int((2 * interval) / precision) + 1
        steps = torch.linspace(-interval, interval, n_nearby, device=device).reshape(1, 1, -1)
    
    # 2. 生成搜索用的agent_s序列（纯batch操作）
    # [B,F,1] + [1,1,n_nearby] → [B,F,n_nearby]（保留B/F维度）
    nearby_agent_s = ref_agent_s.unsqueeze(-1) + steps
    
    # 3. 展平用于road_get_pts_func（保留B维度）
    nearby_agent_s_flat = nearby_agent_s.reshape(B, -1)  # [B, F*n_nearby]
    
    # 4. 获取搜索s对应的道路中心线点（保留B维度）
    nearby_pts = road_get_pts_func(nearby_agent_s_flat)  # [B, F*n_nearby, 2]
    nearby_pts = nearby_pts.reshape(B, F, n_nearby, 2).to(device)  # [B, F, n_nearby, 2]
    
    # 5. 计算车辆到每个搜索点的距离（纯batch操作）
    agent_pos_expand = agent_pos.unsqueeze(2)  # [B, F, 1, 2]
    dists = torch.norm(nearby_pts - agent_pos_expand, dim=-1)  # [B, F, n_nearby]
    
    # 6. 找到最近点索引（保留B/F维度）
    min_indices = torch.argmin(dists, dim=-1)  # [B, F]
    
    # 7. 根据索引获取校准后的agent_s（纯batch gather操作）
    new_agent_s = torch.gather(
        nearby_agent_s,
        dim=2,
        index=min_indices.unsqueeze(-1)  # [B, F, 1]
    ).squeeze(-1)  # [B, F]（仅压缩最后一维，B/F保留）
    
    # 最终返回维度严格为[B, F]，与输入完全一致
    return new_agent_s

def is_point_left_of_polyline(point, polyline):
        """
        判断点是否在边界左侧
        
        参数:
            point: [B, 2] 形状的张量，表示点的坐标
            polyline: [B, N, 2] 形状的张量，表示折线的坐标点
        
        返回:
            is_left: [B] 形状的布尔张量，表示点是否在边界左侧
        """
        # 获取边界线段的起点和终点
        assert torch.isnan(point).any() == False, "point should not be nan"
        assert torch.isnan(polyline).any() == False, "polyline should not be nan"
        start_points = polyline[:, :-1]
        end_points = polyline[:, 1:]
        
        # 计算线段向量: end - start
        seg_vectors = end_points - start_points
        
        # 计算点到线段起点的向量: point - start
        point_vectors = point.unsqueeze(1) - start_points
        
        # 计算叉积: seg_x * point_y - seg_y * point_x
        cross_products = seg_vectors[..., 0] * point_vectors[..., 1] - seg_vectors[..., 1] * point_vectors[..., 0]
        
        # 对于每个点，检查是否在所有线段的左侧（叉积>0）
        # 或者检查是否在边界的左侧区域
        # 这里我们取最靠近点的线段的叉积符号
        closest_seg_idx = torch.argmin(torch.norm(point_vectors, dim=-1), dim=1)
        closest_cross = torch.gather(cross_products, 1, closest_seg_idx.unsqueeze(1)).squeeze(1)
        
        return closest_cross > 0


def get_frenet_distances_between_agents(agent_s):
    s1 = agent_s.unsqueeze(2)            # [B, n_agents, 1]
    s2 = agent_s.unsqueeze(1)            # [B, 1, n_agents]
    mutual_frenet_distances = torch.abs(s1 - s2)  # [B, n_agents, n_agents]
    mutual_frenet_distances.diagonal(dim1=-2, dim2=-1).fill_(mutual_frenet_distances.max() + 1)
    return mutual_frenet_distances

def polynomial_decreasing_fcn(x, x0, x1, power):
    x_clamped = torch.clamp(x, min=x0, max=x1)
    denominator = x1 - x0
    if denominator == 0:
        return torch.ones_like(x_clamped)
    normalized_x = (x_clamped - x0) / denominator
    y = torch.pow(1 - normalized_x, power)
    return y
def calculate_max_min_acceleration(ref_v, center_cum_len):
    """
    基于v-s曲线计算离散点的最大/最小加速度
    Args:
        ref_v: tensor, shape=[M] 参考速度（m/s）
        center_cum_len: tensor, shape=[M] 累计路程（m）
    Returns:
        max_acc: float 最大加速度（m/s²）
        min_acc: float 最小加速度（m/s²）
    """
    # 1. 校验输入维度（避免维度不匹配）
    assert ref_v.ndim == 1 and center_cum_len.ndim == 1, "输入必须是一维张量"
    assert len(ref_v) == len(center_cum_len), "ref_v和center_cum_len长度必须一致"
    assert len(ref_v) >= 2, "至少需要2个离散点才能计算加速度"

    # 2. 计算相邻点的速度平方差和路程差
    v_sq = ref_v **2  # 速度平方 [M]
    delta_v_sq = v_sq[1:] - v_sq[:-1]  # 相邻速度平方差 [M-1]
    delta_s = center_cum_len[1:] - center_cum_len[:-1]  # 相邻路程差 [M-1]

    # 3. 过滤无效路程差（避免除以0）
    valid_mask = delta_s != 0.0
    delta_v_sq_valid = delta_v_sq[valid_mask]
    delta_s_valid = delta_s[valid_mask]

    # 4. 计算离散加速度（a = (v2² - v1²)/(2*Δs)）
    acc = delta_v_sq_valid / (2 * delta_s_valid)

    # 5. 计算最大/最小加速度
    max_acc = torch.max(acc).item()
    min_acc = torch.min(acc).item()

    # 6. 打印结果（保留4位小数，清晰易读）
    print("="*50)
    print(f"v-s曲线离散点加速度计算结果：,离散点数：{len(ref_v)} 个,最大加速度：{max_acc:.4f} m/s²,最小加速度：{min_acc:.4f} m/s²")
    print("="*50)

    return max_acc, min_acc


# ===================== 测试用例 =====================
if __name__ == "__main__":
    # 测试参数设置
    B = 1  # batch size=1
    K = 0.1  # 最小距离0.1米
    n_agent = 4
    n_pts = 4

    # 1. 构造边界：左边界x=0（竖直线），右边界x=1（竖直线），y从0到23
    ref_left_boundary = torch.zeros(B, 24, 2)
    ref_left_boundary[..., 1] = torch.arange(24)  # y轴从0到23
    ref_right_boundary = torch.ones(B, 24, 2)
    ref_right_boundary[..., 1] = torch.arange(24)  # y轴从0到23

    # 2. 构造预瞄点：3个关键测试点
    hinge_short_term = torch.zeros(B, n_agent, n_pts, 2)
    hinge_short_term[0, 0, 0] = torch.tensor([-1.0, 10.0])  # x=-1（左边界外）
    hinge_short_term[0, 0, 1] = torch.tensor([2.0, 10.0])   # x=2（右边界外）
    hinge_short_term[0, 0, 2] = torch.tensor([0.5, 10.0])   # x=0.5（边界内）

    # 3. 调用函数
    result = check_hinge_points_in_boundary(
        ref_left_boundary=ref_left_boundary,
        ref_right_boundary=ref_right_boundary,
        hinge_short_term=hinge_short_term,
        K=K
    )

    # 4. 输出测试结果
    print("===== 测试结果 =====")
    print(f"左边界：x=0，右边界：x=1，最小距离K={K}米")
    print(f"点 (x=-1, y=10) → 是否满足条件：{result[0,0,0,0].item()}（预期False）")
    print(f"点 (x=2, y=10)  → 是否满足条件：{result[0,0,1,0].item()}（预期False）")
    print(f"点 (x=0.5, y=10)→ 是否满足条件：{result[0,0,2,0].item()}（预期True）")
    print("\n完整输出形状:", result.shape)
    print("完整输出张量:\n", result)
