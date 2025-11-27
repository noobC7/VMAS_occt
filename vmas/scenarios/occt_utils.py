from vmas.scenarios.road_traffic import CircularBuffer

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
        track_reference_vel=None,
        track_reference_spacing=None,
    ):
        self.progress = progress
        self.weighting_ref_directions = weighting_ref_directions
        self.higth_v = higth_v
        self.reach_goal = reach_goal
        self.reach_intermediate_goal = reach_intermediate_goal
        self.track_reference_vel = track_reference_vel
        self.track_reference_spacing = track_reference_spacing
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
        error_space=None,
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
        self.error_space = error_space  # Gap error relative to reference gap
        
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