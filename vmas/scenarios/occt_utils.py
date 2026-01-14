from vmas.scenarios.road_traffic import CircularBuffer
import torch
from torch import Tensor
from vmas.scenarios.occt_map import OcctMap,OcctCRMap,MapBase
from vmas.simulator.core import World, Agent, Sphere, Box

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
        pos_world=None,
        v=None,
        rot=None,
        action_steering=None,
        action_vel=None,
        action_steering_rate=None,
        action_acc=None,
        distance_lanelet=None,
        distance_agent=None,
        distance_ref=None,
    ):
        self.pos = pos
        self.pos_world = pos_world
        self.v = v
        self.rot = rot
        self.action_steering = action_steering
        self.action_vel = action_vel
        self.action_steering_rate = action_steering_rate
        self.action_acc = action_acc
        self.distance_lanelet = distance_lanelet
        self.distance_agent = distance_agent
        self.distance_ref = distance_ref
class OcctRewards:
    """Rewards for moving forward, moving with high speed, etc."""
    def __init__(
        self,
        progress=None,
        weighting_ref_directions=None,
        higth_v=None,
        reach_goal=None,
        reach_intermediate_goal=None,
        reward_track_ref_vel=None,
        reward_track_ref_space=None,
        reward_track_ref_heading=None,
        reward_track_ref_path=None,
        reward_track_hinge=None,  # 新增：铰接距离奖励权重
    ):
        self.progress = progress
        self.weighting_ref_directions = weighting_ref_directions
        self.higth_v = higth_v
        self.reach_goal = reach_goal
        self.reach_intermediate_goal = reach_intermediate_goal
        self.reward_track_ref_vel = reward_track_ref_vel
        self.reward_track_ref_space = reward_track_ref_space
        self.reward_track_ref_heading = reward_track_ref_heading
        self.reward_track_ref_path = reward_track_ref_path
        self.reward_track_hinge = reward_track_hinge  # 铰接距离奖励权重

class OcctPenalties:
    """Penalties for collisions, being too close to other agents or lane boundaries, etc."""

    def __init__(
        self,
        deviate_from_ref_path=None,
        deviate_from_goal=None,
        weighting_deviate_from_ref_path=None,
        near_boundary=None,
        near_other_agents=None,
        collide_with_agents=None,
        collide_with_boundaries=None,
        collide_with_obstacles=None,
        backward=None,
        time=None,
        change_steering=None,
        change_acc=None,
        ref_vel_error=None,
        ref_space_error=None,
    ):
        self.deviate_from_ref_path = (
            deviate_from_ref_path  # Penalty for deviating from reference path
        )
        self.deviate_from_goal = (
            deviate_from_goal  # Penalty for deviating from goal position
        )
        self.weighting_deviate_from_ref_path = weighting_deviate_from_ref_path
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
        self.collide_with_obstacles = (
            collide_with_obstacles  # Penalty for colliding with obstacles
        )
        self.backward = backward  # Penalty for leaving the world
        self.time = time  # Penalty for losing time
        self.change_steering = (
            change_steering  # Penalty for changing steering direction
        )
        self.change_acc = change_acc  # Penalty for changing acceleration direction
        self.ref_vel_error = ref_vel_error  # Penalty for velocity error relative to reference velocity
        self.ref_space_error = ref_space_error  # Penalty for gap error relative to reference gap (unnormalized)
class OcctThresholds:
    """Different thresholds, such as starting from which distance agents are deemed being too close to other agents."""

    def __init__(
        self,
        deviate_from_ref_path=None,
        near_boundary_low=None,
        near_boundary_high=None,
        near_other_agents_low=None,
        near_other_agents_high=None,
        reach_goal=None,
        reach_intermediate_goal=None,
        change_steering=None,
        change_acc=None,
        no_reward_if_too_close_to_boundaries=None,
        no_reward_if_too_close_to_other_agents=None,
        distance_mask_agents=None,
        hinge_close=None,  # 新增：理想铰接距离
        hinge_far=None,    # 新增：最大有效铰接距离
    ):
        self.deviate_from_ref_path = deviate_from_ref_path
        self.near_boundary_low = near_boundary_low
        self.near_boundary_high = near_boundary_high
        self.near_other_agents_low = near_other_agents_low
        self.near_other_agents_high = near_other_agents_high
        self.reach_goal = reach_goal  # Threshold less than which agents are considered at their goal positions
        self.reach_intermediate_goal = reach_intermediate_goal  # Threshold less than which agents are considered at their intermediate goal positions
        self.change_steering = change_steering  # Threshold above which agents will be penalized for changing steering too quick [degree]
        self.change_acc = change_acc  # Threshold above which agents will be penalized for changing acceleration too quick [m/s^2]
        self.no_reward_if_too_close_to_boundaries = no_reward_if_too_close_to_boundaries  # Agents get no reward if they are too close to lanelet boundaries
        self.no_reward_if_too_close_to_other_agents = no_reward_if_too_close_to_other_agents  # Agents get no reward if they are too close to other agents
        self.distance_mask_agents = (
            distance_mask_agents  # Threshold above which nearing agents will be masked
        )
        self.hinge_close = hinge_close  # 理想铰接距离（米），reward=1
        self.hinge_far = hinge_far      # 最大有效铰接距离（米），reward=0
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
        error_space: CircularBuffer = None,
        error_hinge_dis=None,  # 铰接距离误差（米）
        agent_s=None,
        past_pos: CircularBuffer = None,
        past_rot: CircularBuffer = None,
        past_vertices: CircularBuffer = None,
        past_vel: CircularBuffer = None,
        past_short_term_ref_points: CircularBuffer = None,
        past_short_term_hinge_points: CircularBuffer = None,
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
        self.error_vel = error_vel  # Velocity error relative to reference velocity
        self.error_space = error_space  # Gap error relative to reference gap (unnormalized)
        self.error_hinge_dis = error_hinge_dis  # 铰接距离误差（米）
        self.agent_s = agent_s  # Arc length position
        
        self.past_pos = past_pos  # Past positions
        self.past_rot = past_rot  # Past rotations
        self.past_vertices = past_vertices  # Past vertices
        self.past_vel = past_vel  # Past velocites

        self.past_short_term_ref_points = (
            past_short_term_ref_points  # Past short-term reference points
        )
        self.past_short_term_hinge_points = (
            past_short_term_hinge_points  # Past short-term hinge points
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

def get_short_term_hinge_path_by_s(
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
