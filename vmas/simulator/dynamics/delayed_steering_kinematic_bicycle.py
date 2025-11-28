#  Copyright (c) ProrokLab.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.


import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

import vmas.simulator.core
import vmas.simulator.utils
from vmas.simulator.dynamics.common import Dynamics

class DelayedSteeringKinematicBicycle(Dynamics):
    # Kinematic Bicycle model with acceleration and target steering angle as actions
    # and steering actuator delay modeled as first-order inertia
    def __init__(
        self,
        world: vmas.simulator.core.World,
        width: float,
        l_f: float,
        l_r: float,
        max_steering_angle: float,
        max_acceleration: float = 5.0,  # Maximum acceleration in m/s^2
        max_deceleration: float = -5.0,  # Maximum deceleration in m/s^2 (negative)
        steering_time_constant: float = 0.1,  # Time constant for steering delay (s)
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
        self.steering_time_constant = steering_time_constant  # Time constant for first-order inertia
        self.dt = world.dt
        self.integration = integration
        self.world = world
        
        # Additional state variables
        self.steering_angle = None  # Actual steering angle state (with delay)
        self.target_steering_angle = None  # Target steering angle from action
        self.velocity = None  # Linear velocity state
        
        # For debugging and visualization
        self.history = {
            'pos': [],
            'vel': [],
            'steering_angle': [],
            'target_steering_angle': [],
            'yaw': []
        }

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
        return 2  # Action size remains 2: [acceleration, target_steering_angle]

    def process_action(self):
        # Initialize additional state variables if not already done
        batch_size = self.agent.state.pos.shape[0]
        if self.steering_angle is None:
            self.steering_angle = torch.zeros((batch_size, 1), device=self.world.device)
        if self.target_steering_angle is None:
            self.target_steering_angle = torch.zeros((batch_size, 1), device=self.world.device)
        if self.velocity is None:
            # Initialize velocity as the magnitude of current velocity vector
            self.velocity = torch.norm(self.agent.state.vel, dim=1, keepdim=True)
        
        # Extract acceleration and target steering angle from actions
        acceleration = self.agent.action.u[:, 0]
        target_steering_angle = self.agent.action.u[:, 1].unsqueeze(1)
        
        # Apply constraints to acceleration and target steering angle
        acceleration = torch.clamp(
            acceleration, self.max_deceleration, self.max_acceleration
        )
        target_steering_angle = torch.clamp(
            target_steering_angle, -self.max_steering_angle, self.max_steering_angle
        )
        
        # Update target steering angle
        self.target_steering_angle = target_steering_angle
        
        # Calculate steering rate using first-order inertia model
        # Model: d(steering_angle)/dt = (target_steering_angle - steering_angle) / time_constant
        steering_rate = (target_steering_angle - self.steering_angle) / self.steering_time_constant
        
        # Current state including additional state variables
        pos = self.agent.state.pos  # [x, y]
        theta = self.agent.state.rot  # [theta]
        v = self.velocity  # [v]
        delta = self.steering_angle  # [delta]
        
        # Create full state vector: [x, y, theta, v, delta]
        state = torch.cat((pos, theta, v, delta), dim=1)
        
        # Select integration method to calculate state derivative
        if self.integration == "euler":
            delta_state = self.euler(state, acceleration, steering_rate.squeeze(1))
        else:
            delta_state = self.runge_kutta(state, acceleration, steering_rate.squeeze(1))
        
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
                new_steering_angle,
                target_steering_angle
            ), dim=1)
        else:
            # Create custom_state attribute
            self.agent.state.custom_state = torch.cat((
                state,
                new_velocity,
                new_steering_angle,
                target_steering_angle
            ), dim=1)
        
        # Store history for debugging
        if batch_size == 1:  # Only store for single batch case
            self.history['pos'].append(pos.cpu().numpy().copy()[0])
            self.history['vel'].append(np.array([new_vel_x.cpu().numpy()[0], new_vel_y.cpu().numpy()[0]]))
            self.history['steering_angle'].append(new_steering_angle.cpu().numpy()[0][0])
            self.history['target_steering_angle'].append(target_steering_angle.cpu().numpy()[0][0])
            self.history['yaw'].append(theta.cpu().numpy()[0][0])

    def reset_history(self):
        """Reset the history for a new simulation run"""
        self.history = {
            'pos': [],
            'vel': [],
            'steering_angle': [],
            'target_steering_angle': [],
            'yaw': []
        }

    def plot_trajectory(self):
        """Plot the vehicle trajectory and states"""
        if not self.history['pos']:
            print("No history data to plot")
            return
        
        # Convert history to numpy arrays
        positions = np.array(self.history['pos'])
        velocities = np.array(self.history['vel'])
        steering_angles = np.array(self.history['steering_angle'])
        target_steering_angles = np.array(self.history['target_steering_angle'])
        yaw_angles = np.array(self.history['yaw'])
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot trajectory
        axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-')
        axes[0, 0].set_title('Vehicle Trajectory')
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')
        
        # Plot velocity
        time = np.arange(len(velocities)) * self.dt
        speed = np.linalg.norm(velocities, axis=1)
        axes[0, 1].plot(time, speed, 'r-')
        axes[0, 1].set_title('Vehicle Speed')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Speed (m/s)')
        axes[0, 1].grid(True)
        
        # Plot steering angle
        axes[1, 0].plot(time, np.rad2deg(steering_angles), 'g-', label='Actual Steering')
        axes[1, 0].plot(time, np.rad2deg(target_steering_angles), 'r--', label='Target Steering')
        axes[1, 0].set_title('Steering Angle with Delay')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Angle (degrees)')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Plot yaw angle
        axes[1, 1].plot(time, np.rad2deg(yaw_angles), 'm-')
        axes[1, 1].set_title('Yaw Angle')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Angle (degrees)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()


# 模拟环境和World类的简化版本，用于测试
class MockWorld:
    def __init__(self, dt=0.1):
        self.dt = dt
        self.device = torch.device('cpu')

class MockAgent:
    def __init__(self, world):
        self.world = world
        self.mass = 1500.0  # 典型车辆质量（kg）
        self.moment_of_inertia = 2000.0  # 转动惯量
        
        # 初始状态
        self.state = type('obj', (), {
            'pos': torch.zeros((1, 2), device=world.device),  # 位置 [x, y]
            'rot': torch.zeros((1, 1), device=world.device),  # 旋转角度
            'vel': torch.zeros((1, 2), device=world.device),  # 速度
            'ang_vel': torch.zeros((1, 1), device=world.device),  # 角速度
            'force': torch.zeros((1, 2), device=world.device),  # 力
            'torque': torch.zeros((1, 1), device=world.device)  # 扭矩
        })
        
        # 动作
        self.action = type('obj', (), {
            'u': torch.zeros((1, 2), device=world.device)  # 动作 [acceleration, target_steering_angle]
        })


def simulate_simple_test():
    """简单测试函数，模拟车辆运动并可视化"""
    # 创建模拟世界
    world = MockWorld(dt=0.05)
    
    # 创建车辆动力学模型
    # 典型轿车参数：轴距约2.8米，前轮距和后轮距约1.6米
    l_f = 1.4  # 前轮到重心距离
    l_r = 1.4  # 后轮到重心距离
    width = 1.6  # 车辆宽度
    max_steering_angle = np.deg2rad(35)  # 最大转向角度35度
    
    dynamics = DelayedSteeringKinematicBicycle(
        world=world,
        width=width,
        l_f=l_f,
        l_r=l_r,
        max_steering_angle=max_steering_angle,
        max_acceleration=5.0,
        max_deceleration=-10.0,
        steering_time_constant=0.2,  # 转向执行器时间常数，调整此值可改变延迟程度
        integration="rk4"
    )
    
    # 创建模拟智能体
    agent = MockAgent(world)
    dynamics.agent = agent
    
    # 运行模拟
    steps = 200
    for i in range(steps):
        # 设置不同的测试动作
        if i < 50:
            # 前50步：加速向前，不转向
            agent.action.u[0, 0] = 2.0  # 2 m/s^2 加速度
            agent.action.u[0, 1] = 0.0   # 0度转向角
        elif i < 100:
            # 接下来50步：保持速度，转向30度
            agent.action.u[0, 0] = 0.0   # 保持速度
            agent.action.u[0, 1] = np.deg2rad(30)  # 30度转向角
        elif i < 150:
            # 接下来50步：保持速度，转向-30度
            agent.action.u[0, 0] = 0.0   # 保持速度
            agent.action.u[0, 1] = -np.deg2rad(30)  # -30度转向角
        else:
            # 最后50步：减速停止，回正方向盘
            agent.action.u[0, 0] = -5.0  # 5 m/s^2 减速度
            agent.action.u[0, 1] = 0.0   # 0度转向角
        
        # 处理动作
        dynamics.process_action()
        
        # 更新智能体状态（简化版本，实际应该由物理引擎处理）
        agent.state.pos += agent.state.vel * world.dt
        agent.state.rot += agent.state.ang_vel * world.dt
        agent.state.vel += (agent.state.force / agent.mass) * world.dt
        agent.state.ang_vel += (agent.state.torque / agent.moment_of_inertia) * world.dt
    
    # 绘制轨迹和状态
    dynamics.plot_trajectory()
    
    # 创建动画展示车辆运动
    animate_vehicle_motion(dynamics, world.dt)

def animate_vehicle_motion(dynamics, dt):
    """创建车辆运动的动画"""
    if not dynamics.history['pos']:
        print("No history data to animate")
        return
    
    positions = np.array(dynamics.history['pos'])
    yaw_angles = np.array(dynamics.history['yaw'])
    
    # 车辆尺寸
    length = dynamics.l_f + dynamics.l_r
    width = dynamics.width
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(np.min(positions[:, 0]) - 5, np.max(positions[:, 0]) + 5)
    ax.set_ylim(np.min(positions[:, 1]) - 5, np.max(positions[:, 1]) + 5)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title('Vehicle Motion Animation (Delayed Steering)')
    
    # 初始化车辆表示
    vehicle, = ax.plot([], [], 'b-', linewidth=2)
    direction, = ax.plot([], [], 'r->', linewidth=2)
    trajectory, = ax.plot([], [], 'g--', alpha=0.5)
    
    def init():
        vehicle.set_data([], [])
        direction.set_data([], [])
        trajectory.set_data([], [])
        return vehicle, direction, trajectory
    
    def update(frame):
        # 绘制轨迹
        trajectory.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
        
        # 计算车辆轮廓点
        x, y = positions[frame]
        theta = yaw_angles[frame]
        
        # 车辆四个角的相对位置
        corners_rel = np.array([
            [length/2, width/2],
            [-length/2, width/2],
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2]  # 闭合图形
        ])
        
        # 旋转矩阵
        rot_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        
        # 旋转并平移车辆轮廓点
        corners = np.dot(corners_rel, rot_matrix.T) + np.array([x, y])
        
        # 绘制车辆
        vehicle.set_data(corners[:, 0], corners[:, 1])
        
        # 绘制方向指示器
        direction_end = np.array([x, y]) + np.array([length/2 * np.cos(theta), length/2 * np.sin(theta)])
        direction.set_data([x, direction_end[0]], [y, direction_end[1]])
        
        return vehicle, direction, trajectory
    
    ani = FuncAnimation(
        fig, update, frames=len(positions), init_func=init,
        blit=True, interval=dt*1000, repeat=False
    )
    
    plt.show()


if __name__ == "__main__":
    # 运行简单测试并可视化
    simulate_simple_test()