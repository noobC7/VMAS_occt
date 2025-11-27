#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.


import torch

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics

class DynamicKinematicBicycle(Dynamics):
    # Dynamic Kinematic Bicycle model with acceleration and steering rate as actions
    # and velocity and steering angle as part of the state
    def __init__(
        self,
        world: vmas.simulator.core.World,
        width: float,
        l_f: float,
        l_r: float,
        max_steering_angle: float,
        max_acceleration: float = 5.0,  # Maximum acceleration in m/s^2
        max_deceleration: float = -10.0,  # Maximum deceleration in m/s^2 (negative)
        max_steering_rate: float = torch.pi ,  # Maximum steering rate in rad/s
        integration: str = "rk4",  # one of "euler", "rk4"
    ):
        super().__init__()
        assert integration in (
            "rk4",
            "euler",
        ), "Integration method must be 'euler' or 'rk4'."
        self.width = width
        self.l_f = l_f  # Distance between the front axle and the center of gravity
        self.l_r = l_r  # Distance between the rear axle and the center of gravity
        self.max_steering_angle = max_steering_angle
        self.max_acceleration = max_acceleration
        self.max_deceleration = max_deceleration
        self.max_steering_rate = max_steering_rate
        self.dt = world.dt
        self.integration = integration
        self.world = world
        
        # Additional state variables
        self.steering_angle = None  # Steering angle state
        self.velocity = None  # Linear velocity state

    def f(self, state, acceleration, steering_rate):
        # State now includes: [x, y, theta, v, delta]
        # where delta is steering angle
        theta = state[:, 2]  # Yaw angle
        v = state[:, 3]  # Linear velocity
        delta = state[:, 4]  # Steering angle
        
        # Calculate slip angle
        beta = torch.atan2(
            torch.tan(delta) * self.l_r / (self.l_f + self.l_r),
            torch.tensor(1, device=self.world.device),
        )
        
        # State derivatives
        dx = v * torch.cos(theta + beta)
        dy = v * torch.sin(theta + beta)
        dtheta = (v / (self.l_f + self.l_r)) * torch.cos(beta) * torch.tan(delta)
        dv = acceleration  # Velocity derivative is acceleration
        ddelta = steering_rate  # Steering angle derivative is steering rate
        
        return torch.stack((dx, dy, dtheta, dv, ddelta), dim=1)  # [batch_size,5]

    def euler(self, state, acceleration, steering_rate):
        # Euler integration method
        return self.dt * self.f(state, acceleration, steering_rate)

    def runge_kutta(self, state, acceleration, steering_rate):
        # Fourth-order Runge-Kutta integration method
        k1 = self.f(state, acceleration, steering_rate)
        k2 = self.f(state + self.dt * k1 / 2, acceleration, steering_rate)
        k3 = self.f(state + self.dt * k2 / 2, acceleration, steering_rate)
        k4 = self.f(state + self.dt * k3, acceleration, steering_rate)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    @property
    def needed_action_size(self) -> int:
        return 2  # Action size remains 2: [acceleration, steering_rate]

    def process_action(self):
        # Initialize additional state variables if not already done
        batch_size = self.agent.state.pos.shape[0]
        if self.steering_angle is None:
            self.steering_angle = torch.zeros((batch_size, 1), device=self.world.device)
        if self.velocity is None:
            # Initialize velocity as the magnitude of current velocity vector
            self.velocity = torch.norm(self.agent.state.vel, dim=1, keepdim=True)
        
        # Extract acceleration and steering rate from actions
        acceleration = self.agent.action.u[:, 0]
        steering_rate = self.agent.action.u[:, 1]
        
        # Apply constraints to acceleration and steering rate
        acceleration = torch.clamp(
            acceleration, self.max_deceleration, self.max_acceleration
        )
        steering_rate = torch.clamp(
            steering_rate, -self.max_steering_rate, self.max_steering_rate
        )
        
        # Current state including additional state variables
        pos = self.agent.state.pos  # [x, y]
        theta = self.agent.state.rot  # [theta]
        v = self.velocity  # [v]
        delta = self.steering_angle  # [delta]
        
        # Create full state vector: [x, y, theta, v, delta]
        state = torch.cat((pos, theta, v, delta), dim=1)
        
        # Select integration method to calculate state derivative
        if self.integration == "euler":
            delta_state = self.euler(state, acceleration, steering_rate)
        else:
            delta_state = self.runge_kutta(state, acceleration, steering_rate)
        
        # Update additional state variables
        new_velocity = v + delta_state[:, 3].unsqueeze(1)
        new_steering_angle = delta + delta_state[:, 4].unsqueeze(1)
        
        # Apply constraints to new state variables
        new_velocity = torch.clamp(new_velocity, 0.0, None)  # Ensure velocity is non-negative
        new_steering_angle = torch.clamp(
            new_steering_angle, -self.max_steering_angle, self.max_steering_angle
        )
        
        # Update state variables
        self.velocity = new_velocity
        self.steering_angle = new_steering_angle
        
        # Calculate new slip angle
        beta = torch.atan2(
            torch.tan(new_steering_angle).squeeze(1) * self.l_r / (self.l_f + self.l_r),
            torch.tensor(1, device=self.world.device),
        )
        
        # Calculate new velocity components
        new_vel_x = new_velocity.squeeze(1) * torch.cos(theta.squeeze(1) + beta)
        new_vel_y = new_velocity.squeeze(1) * torch.sin(theta.squeeze(1) + beta)
        
        # Calculate required acceleration components
        acc_x = (new_vel_x - self.agent.state.vel[:, 0]) / self.dt
        acc_y = (new_vel_y - self.agent.state.vel[:, 1]) / self.dt
        acc_angular = (delta_state[:, 2] - self.agent.state.ang_vel[:, 0] * self.dt) / self.dt**2
        
        # Calculate forces and torque
        force_x = self.agent.mass * acc_x
        force_y = self.agent.mass * acc_y
        torque = self.agent.moment_of_inertia * acc_angular
        
        # Update physical forces and torque
        self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        self.agent.state.torque = torque.unsqueeze(-1)
        
        # Update the state variables in agent's state for visualization and other purposes
        # This creates the state with velocity and steering angle appended at the end
        if hasattr(self.agent.state, 'custom_state'):
            # If custom_state already exists, update it
            self.agent.state.custom_state = torch.cat((
                state,
                new_velocity,
                new_steering_angle
            ), dim=1)
        else:
            # Create custom_state attribute
            self.agent.state.custom_state = torch.cat((
                state,
                new_velocity,
                new_steering_angle
            ), dim=1)