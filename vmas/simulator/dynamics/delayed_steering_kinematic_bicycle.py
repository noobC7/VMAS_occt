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
        self.max_delta = max_steering_angle
        self.max_acceleration = max_acceleration
        self.steering_time_constant = steering_time_constant  # Time constant for first-order inertia
        self.dt = world.dt
        self.integration = integration
        self.world = world
        
        # Additional state variables
        self.cur_delta = None  # Actual steering angle state (with delay)

        # For debugging and visualization
        self.reset_history()

    def f(self, state, acceleration, target_delta):
        assert torch.isnan(state).any() == False, f"state is nan"
        assert torch.isnan(acceleration).any() == False, f"acceleration is nan"
        assert torch.isnan(target_delta).any() == False, f"target_delta is nan"
        # State now includes: [x, y, theta, v, delta]
        theta = state[:, 2]  # Yaw angle
        v = state[:, 3]  # Linear velocity
        delta = state[:, 4]  # Steering angle
        
        beta = torch.atan2(
            torch.tan(delta) * self.l_r / (self.l_f + self.l_r),
            torch.tensor(1, device=self.world.device),
        )
        
        dx = v * torch.cos(theta + beta)
        dy = v * torch.sin(theta + beta)
        dtheta = (v / (self.l_f + self.l_r)) * torch.cos(beta) * torch.tan(delta)
        dv = acceleration
        # Calculate steering rate using first-order inertia model
        ddelta = (target_delta - delta) / self.steering_time_constant
        new_state = torch.stack((dx, dy, dtheta, dv, ddelta), dim=1)  # [batch_size,5]
        assert torch.isnan(new_state).any() == False, f"new_state is nan"
        return new_state
    
    def euler(self, state, acceleration, target_delta):
        # Calculate the change in state using Euler's method
        # For Euler's method, see https://math.libretexts.org/Bookshelves/Calculus/Book%3A_Active_Calculus_(Boelkins_et_al.)/07%3A_Differential_Equations/7.03%3A_Euler's_Method (the full link may not be recognized properly, please copy and paste in your browser)
        return self.dt * self.f(state, acceleration, target_delta)
    
    def runge_kutta(self, state, acceleration, target_delta):
        # Calculate the change in state using fourth-order Runge-Kutta method
        # For Runge-Kutta method, see https://math.libretexts.org/Courses/Monroe_Community_College/MTH_225_Differential_Equations/3%3A_Numerical_Methods/3.3%3A_The_Runge-Kutta_Method
        k1 = self.f(state, acceleration, target_delta)
        k2 = self.f(state + self.dt * k1 / 2, acceleration, target_delta)
        k3 = self.f(state + self.dt * k2 / 2, acceleration, target_delta)
        k4 = self.f(state + self.dt * k3, acceleration, target_delta)
        return (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    @property
    def needed_action_size(self) -> int:
        return 2  # Action size remains 2: [acceleration, target_delta]

    def process_action(self):
        # Initialize additional state variables if not already done
        batch_size = self.agent.state.pos.shape[0]
        if self.cur_delta is None:
            self.cur_delta = torch.zeros((batch_size, 1), device=self.world.device)
        # Extract acceleration and target steering angle from actions
        acceleration = self.agent.action.u[:, 0]
        target_delta = self.agent.action.u[:, 1]
        
        # Apply constraints to acceleration and target steering angle
        acceleration = torch.clamp(
            acceleration, -self.max_acceleration, self.max_acceleration
        )
        target_delta = torch.clamp(
            target_delta, -self.max_delta, self.max_delta
        )
        # Current state including additional state variables
        pos = self.agent.state.pos  # [x, y]
        theta = self.agent.state.rot  # [theta]
        vel_mag = torch.norm(self.agent.state.vel, dim=1, keepdim=True)  # 速度大小（恒正）
        vel_dir = self.agent.state.vel / (vel_mag + 1e-8)  # 速度单位向量（避免除零）
        heading_vec = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)  # 航向方向向量（x=cosθ, y=sinθ）
        direction_sign = torch.sign(torch.sum(vel_dir * heading_vec, dim=1, keepdim=True))  # 点积判断方向（1=同向，-1=反向）
        cur_v = vel_mag * direction_sign  # 带正负号的标量速度（正=前进，负=倒车）
        cur_delta = self.cur_delta  # [delta]
        
        # Create full state vector: [x, y, theta, v, delta]
        state = torch.cat((pos, theta, cur_v, cur_delta), dim=1)
        
        # Store history for debugging
        if batch_size == 1:  # Only store for single batch case
            self.history['pos'].append(pos.cpu().numpy().copy()[0])
            self.history['yaw'].append(theta.cpu().numpy().copy()[0][0])
            self.history['vel'].append(cur_v.cpu().numpy().copy()[0][0])
            self.history['delta'].append(cur_delta.cpu().numpy().copy()[0][0])
            self.history['target_delta'].append(target_delta.cpu().numpy().copy()[0])
            self.history['acc'].append(acceleration.cpu().numpy().copy()[0])

        # Select integration method to calculate state derivative
        if self.integration == "euler":
            delta_state = self.euler(state, acceleration, target_delta)
        else:
            delta_state = self.runge_kutta(state, acceleration, target_delta)

        v_cur_x = self.agent.state.vel[:, 0]  # Current velocity in x-direction
        v_cur_y = self.agent.state.vel[:, 1]  # Current velocity in y-direction
        v_cur_angular = self.agent.state.ang_vel[:, 0]  # Current angular velocity

        # Calculate the accelerations required to achieve the change in state.
        acceleration_x = (delta_state[:, 0] - v_cur_x * self.dt) / self.dt**2
        acceleration_y = (delta_state[:, 1] - v_cur_y * self.dt) / self.dt**2
        acceleration_angular = (
            delta_state[:, 2] - v_cur_angular * self.dt
        ) / self.dt**2

        # Calculate the forces required for the linear accelerations
        force_x = self.agent.mass * acceleration_x
        force_y = self.agent.mass * acceleration_y

        # Calculate the torque required for the angular acceleration
        torque = self.agent.moment_of_inertia * acceleration_angular

        # Update the physical force and torque required for the user inputs
        self.agent.state.force[:, vmas.simulator.utils.X] = force_x
        self.agent.state.force[:, vmas.simulator.utils.Y] = force_y
        self.agent.state.torque = torque.unsqueeze(-1)

        # Update additional state variables
        self.cur_delta = cur_delta + delta_state[:, 4].unsqueeze(1)
        

    def reset_history(self):
        """Reset the history for a new simulation run"""
        self.history = {
            'pos': [],
            'yaw': [],
            'vel': [],
            'delta': [],
            'target_delta': [],
            'acc': [],
        }

    def plot_trajectory(self):
        """Plot the vehicle trajectory and states"""
        if not self.history['pos']:
            print("No history data to plot")
            return
        
        # Convert history to numpy arrays
        positions = np.array(self.history['pos'])
        velocities = np.array(self.history['vel'])
        deltas = np.array(self.history['delta'])
        target_deltas = np.array(self.history['target_delta'])
        yaw_angles = np.array(self.history['yaw'])
        accelerations = np.array(self.history['acc'])  # 获取加速度数据
        
        # 调整布局为 3x2 以容纳加速度曲线
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        
        # Plot trajectory
        axes[0, 0].plot(positions[:, 0], positions[:, 1], 'b-')
        axes[0, 0].set_title('Vehicle Trajectory')
        axes[0, 0].set_xlabel('X Position (m)')
        axes[0, 0].set_ylabel('Y Position (m)')
        axes[0, 0].grid(True)
        axes[0, 0].axis('equal')
        
        # Plot velocity
        time = np.arange(len(velocities)) * self.dt
        axes[0, 1].plot(time, velocities, 'r-')
        axes[0, 1].set_title('Vehicle Speed')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Speed (m/s)')
        axes[0, 1].grid(True)
        
        # Plot steering angle
        axes[1, 0].plot(time, np.rad2deg(deltas), 'g-', label='Actual Steering')
        axes[1, 0].plot(time, np.rad2deg(target_deltas), 'r--', label='Target Steering')
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
        
        # 添加加速度曲线
        time = np.arange(len(accelerations)) * self.dt
        axes[2, 0].plot(time, accelerations, 'c-')
        axes[2, 0].set_title('Vehicle Acceleration')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Acceleration (m/s²)')
        axes[2, 0].grid(True)
        
        # 隐藏多余的子图
        axes[2, 1].axis('off')
        
        plt.tight_layout()
        plt.show()



# 模拟环境和World类的简化版本，用于测试
class MockWorld:
    def __init__(self, dt=0.01):
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
            'u': torch.zeros((1, 2), device=world.device)  # 动作 [acceleration, target_delta]
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
    max_delta = np.deg2rad(35)  # 最大转向角度35度
    
    dynamics = DelayedSteeringKinematicBicycle(
        world=world,
        width=width,
        l_f=l_f,
        l_r=l_r,
        max_steering_angle=max_delta,
        max_acceleration=5.0,
        steering_time_constant=0.1,  # 转向执行器时间常数，调整此值可改变延迟程度
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