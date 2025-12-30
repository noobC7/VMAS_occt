from vmas.scenarios.road_traffic import CircularBuffer
import torch
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
        reward_track_ref_path=None,
    ):
        self.progress = progress
        self.weighting_ref_directions = weighting_ref_directions
        self.higth_v = higth_v
        self.reach_goal = reach_goal
        self.reach_intermediate_goal = reach_intermediate_goal
        self.reward_track_ref_vel = reward_track_ref_vel
        self.reward_track_ref_space = reward_track_ref_space
        self.reward_track_ref_path = reward_track_ref_path

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
        agent_s=None,
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
        self.error_vel = error_vel  # Velocity error relative to reference velocity
        self.error_space = error_space  # Gap error relative to reference gap (unnormalized)
        self.agent_s = agent_s  # Arc length position
        
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
    def check_validity(self):
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, torch.Tensor) and torch.isnan(attr_value).any():
                nan_indices = torch.nonzero(torch.isnan(attr_value), as_tuple=False)
                raise ValueError(f"NaN found in self.{attr_name}, index:{nan_indices}")
