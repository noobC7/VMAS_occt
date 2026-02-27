# def reward(self, agent: Agent):
#         agent_index = self.world.agents.index(agent)
#         if agent_index == 0:
#             self.env_current_step += 1
#         # Initialize
#         reward_details=self.reward_details
#         self.rew[:] = 0
#         # we exclude the front vehicle and end vehicle
#         # [update] mutual distances between agents, vertices of each agent, and collision matrices
#         t0=time.time()
#         self.update_state_before_rewarding(agent, agent_index)
#         t1=time.time()
#         #print(f"update_state_before_rewarding, agent_index: {agent_index}, time: {t1-t0:.6f}s")
    
#         if self.task_class == TaskClass.OCCT_PLATOON and agent_index in self.TRACTOR_SLICE:
#             self.update_state_after_rewarding(agent_index)
#             return self.rew
#         # [penalty] close to other agents
#         mutual_distance_exp_fcn = exponential_decreasing_fcn(
#             x=self.distances.agents[:, agent_index, :],
#             x0=self.thresholds.near_other_agents_low,
#             x1=self.thresholds.near_other_agents_high,
#         )
#         penalty_near_other_agents = (
#             torch.sum(mutual_distance_exp_fcn, dim=1) * self.penalties.near_other_agents
#         )
#         reward_details["penalty_near_other_agents"][:,agent_index] = penalty_near_other_agents
#         self.rew += penalty_near_other_agents


#         # [penalty] changing steering too quick
#         steering_current = self.observations.past_action_steering.get_latest(n=1)[
#             :, agent_index
#         ]
#         steering_past = self.observations.past_action_steering.get_latest(n=2)[
#             :, agent_index
#         ]
#         steering_change = torch.clamp(
#             (steering_current - steering_past).abs() * self.normalizers.action_steering
#             - self.thresholds.change_steering,  # Not forget to denormalize
#             min=0,
#         )
#         if self.observations.past_action_steering.valid_size==self.observations.n_stored_steps:
#             penalty_change_steering = (
#                 (steering_change/torch.deg2rad(torch.tensor(3,device=self.device)))**2 * self.penalties.change_steering
#             )
#             penalty_change_steering = torch.clamp(penalty_change_steering,min=-5,max=0)
#         else:
#             penalty_change_steering = 0.0
#         reward_details["penalty_change_steering"][:,agent_index] = penalty_change_steering
#         self.rew += penalty_change_steering


#         # [penalty] changing acc too quick
#         acc_current = self.observations.past_action_acc.get_latest(n=1)[
#             :, agent_index
#         ]
#         acc_past = self.observations.past_action_acc.get_latest(n=2)[
#             :, agent_index
#         ]

#         acc_change = torch.clamp(
#             (acc_current - acc_past).abs() * self.normalizers.action_acc
#             - self.thresholds.change_acc,  # Not forget to denormalize
#             min=0,
#         )
#         acc_nor=0.1
#         if self.observations.past_action_acc.valid_size==self.observations.n_stored_steps:
#             penalty_change_acc = (
#                 (acc_change/acc_nor)**2 * self.penalties.change_acc
#             )
#             penalty_change_acc = torch.clamp(penalty_change_acc,min=-5,max=0)
#         else:
#             penalty_change_acc = 0.0
#         reward_details["penalty_change_acc"][:,agent_index] = penalty_change_acc
#         self.rew += penalty_change_acc

#         # [penalty] colliding with other agents
#         is_collide_with_agents = self.collisions.with_agents[:, agent_index]
#         penalty_collide_with_agents = (
#             is_collide_with_agents.any(dim=-1) * self.penalties.collide_with_agents
#         )
#         reward_details["penalty_collide_with_agents"][:,agent_index] = penalty_collide_with_agents
#         self.rew += penalty_collide_with_agents

#         # [penalty] colliding with lanelet boundaries
#         is_collide_with_lanelets = self.collisions.with_lanelets[:, agent_index]
#         penalty_outside_boundaries = (
#             is_collide_with_lanelets * self.penalties.collide_with_boundaries
#         )
#         reward_details["penalty_outside_boundaries"][:,agent_index] = penalty_outside_boundaries
#         self.rew += penalty_outside_boundaries

#         # [penalty] close to lanelet boundaries
#         current_lane_width = torch.linalg.norm(self.ref_paths_agent_related.nearing_points_left_boundary[:, agent_index, 1] -\
#               self.ref_paths_agent_related.nearing_points_right_boundary[:, agent_index, 1],dim=-1)
#         penalty_near_boundary = (
#             torch.max(exponential_decreasing_fcn(
#                 x=self.distances.boundaries[:, agent_index]/current_lane_width,
#                 x0=self.thresholds.near_boundary_low,
#                 x1=self.thresholds.near_boundary_high,
#             ),is_collide_with_lanelets.float())
#             * self.penalties.near_boundary
#         )
#         reward_details["penalty_near_boundary"][:,agent_index] = penalty_near_boundary
#         self.rew += penalty_near_boundary

#         ref_points_vecs = self.ref_paths_agent_related.short_term[:, agent_index, 1:, 0:2] -\
#               self.ref_paths_agent_related.short_term[:, agent_index, :-1, 0:2] 
#         v_proj = torch.sum(agent.state.vel.unsqueeze(1) * ref_points_vecs, dim=-1).mean(
#             -1
#         )
#         backward_penalty = (
#             torch.where(v_proj <= 0, 1, 0)
#             * self.penalties.backward
#         )
#         reward_details["penalty_backward"][:,agent_index] = backward_penalty
#         self.rew += backward_penalty

#         # [reward] forward movement
#         latest_state = self.state_buffer.get_latest(n=1)
#         move_vec = (agent.state.pos - latest_state[:, agent_index, 0:2]).unsqueeze(
#             1
#         )  # Vector of the current movement

#         move_projected = torch.sum(move_vec * ref_points_vecs, dim=-1)
#         move_projected_weighted = torch.matmul(
#             move_projected, self.rewards.weighting_ref_directions
#         )  # Put more weights on nearing reference points

#         reward_progress = (
#             move_projected_weighted
#             / (agent.max_speed * self.world.dt)
#             * self.rewards.progress
#         )
#         reward_details["reward_progress"][:,agent_index] = reward_progress
#         self.rew += reward_progress  # Relative to the maximum possible movement

#         # [reward] high velocity
#         reward_vel = v_proj / agent.max_speed * self.rewards.higth_v
#         reward_details["reward_vel"][:,agent_index] = reward_vel
#         self.rew += reward_vel

#         # [reward] reach goal
#         reward_goal = (
#             self.collisions.with_exit_segments[:, agent_index] * self.rewards.reach_goal
#         )
#         reward_details["reward_goal"][:,agent_index] = reward_goal
#         self.rew += reward_goal  # Relative to the maximum possible movement

#         ref_vel = self.ref_paths_agent_related.short_term[:, agent_index, 0, 2]
#         agent_vel = torch.linalg.norm(agent.state.vel, dim=-1)
#         error_vel = agent_vel - ref_vel
#         reward_track_ref_vel = torch.clamp(1 - self.rewards.reward_track_ref_vel* error_vel**2, min=0)
#         reward_details["reward_track_ref_vel"][:,agent_index] =  reward_track_ref_vel
#         space_errors = self.observations.error_space.get_latest(n=1)[:, agent_index, 0]
#         reward_track_ref_space = torch.clamp(1 - self.rewards.reward_track_ref_space * space_errors**2, min=0)
#         reward_details["reward_track_ref_space"][:,agent_index] = reward_track_ref_space
#         self.rew += reward_track_ref_space
#         # [reward] 横向跟踪
#         ref_vector = torch.mean(ref_points_vecs,dim=1) # or ref_points_vecs[:,0,:]
#         ref_vector_normalized = ref_vector / (torch.norm(ref_vector, dim=-1, keepdim=True) + 1e-8)
#         move_vector = move_vec[:,0,:]
#         move_vector_normalized = move_vector/ (torch.norm(move_vector, dim=-1, keepdim=True) + 1e-8)
#         max_delta_angle=torch.deg2rad(torch.tensor(15, device=self.device, dtype=torch.float32))
#         constant_k=1/(1-torch.cos(max_delta_angle))
#         costant_b=1-constant_k
#         reward_track_ref_heading = torch.clamp(self.rewards.reward_track_ref_heading * \
#                                    (constant_k*torch.sum(ref_vector_normalized * move_vector_normalized, dim=-1)+costant_b),-1.0,1.0)
#         reward_details["reward_track_ref_heading"][:,agent_index] =  reward_track_ref_heading
#         self.rew += reward_track_ref_heading
#         ratio=0.7
#         weighted_ref_dis = ratio*self.distances.lookahead_pts[:, agent_index, 0]+(1-ratio)*self.distances.lookahead_pts[:, agent_index, 1]
#         reward_track_ref_path = 1 - self.rewards.reward_track_ref_path * weighted_ref_dis**2
#         reward_details["reward_track_ref_path"][:,agent_index] = reward_track_ref_path
#         self.rew += reward_track_ref_path
#         reward_details["reward_total"][:,agent_index] = self.rew
#         self.update_state_after_rewarding(agent_index)
#         return self.rew

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager as fm
from commonroad.visualization.mp_renderer import MPRenderer, DynamicObstacleParams
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
font_path = '/usr/share/fonts/truetype/msttcorefonts/SongTi.ttf'
font_prop = fm.FontProperties(fname=font_path, size=12)

# 字体大小统一配置（论文常用尺寸）
font_size_label = 16    # 坐标轴标签字体大小
font_size_tick = 14     # 刻度字体大小（数字用新罗马）
font_size_legend = 14   # 图例字体大小
font_size_cbar = 14     # 颜色条标签字体大小
def exponential_decreasing_fcn(x, x0, x1):
    """
    实现你提供的截断指数函数逻辑：
    当 x 从 x0 增加到 x1 时，y 从 1 指数衰减到 0。
    x < x0 时截断为 x0 (输出1)，x > x1 时截断为 x1 (输出0)。
    """
    x_clamped = np.clip(x, x0, x1)
    # 公式：(e^( -(x-x0)/(x1-x0) ) - e^-1) / (1 - e^-1)
    y = (np.exp(-(x_clamped - x0) / (x1 - x0)) - np.exp(-1)) / (1 - np.exp(-1))
    return y

# ==========================================
# 图 1：道路中心线偏移与奖励关系图 (2D)
# ==========================================
plt.figure(figsize=(8, 5))

# 设定道路几何参数以满足 [-2, 2] 安全区的要求
# 假设道路半宽为 3.0m (总宽 6m)，这样 offset=2 时距离边界为 1m
half_width = 2.3
offset = np.linspace(-half_width, half_width, 400) 
agent_width = 1.5
# 1. 路径跟踪奖励 (抛物线)
k_track = 0.5
track_reward = 1.0 - k_track * offset**2

# 2. 边界惩罚 (修正后的逻辑)
# 计算距离边界的距离
dist_to_boundary = half_width - np.abs(offset)

# 设定阈值：
# x1 (高阈值): 当距离 > 1.0m (即 offset < 2.0m) 时，惩罚为 0
# x0 (低阈值): 当距离 = 0.0m (即边界处) 时，惩罚最大
x1_bound = agent_width/2
x0_bound = 0.0

# 计算惩罚系数 (0~1)
penalty_factor = exponential_decreasing_fcn(dist_to_boundary, x0_bound, x1_bound)

# 施加惩罚权重 (假设为 -1.0 以符合"下降到 -1"的描述)
boundary_penalty = -1.0 * penalty_factor

# 3. 总奖励
total_curve = track_reward + boundary_penalty

# 绘图
plt.plot(offset, track_reward, label='跟踪奖励', color='blue', linestyle='-')
plt.plot(offset, boundary_penalty, label='边界惩罚', color='red', linestyle='-')
plt.plot(offset, total_curve, label='加权奖励', color='green', linewidth=2.5, linestyle='--')

# 辅助线
plt.axvline(x=-(half_width-agent_width/2), color='gray', linestyle=':', alpha=0.5, label='安全区上限')
plt.axvline(x=(half_width-agent_width/2), color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=-half_width, color='k', linestyle='-', alpha=0.5, label='车道边界')
plt.axvline(x=half_width, color='k', linestyle='-', alpha=0.5)

plt.xlabel('道路中心线偏移量 (m)', fontproperties=font_prop)
plt.ylabel('奖励值', fontproperties=font_prop)
plt.legend(prop=font_prop)
plt.grid(True, alpha=0.3)
plt.savefig('reward_track_path.pdf')

# ==========================================
# 图 2：障碍物势场三维网格图 (3D)
# ==========================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 网格设置
x = np.linspace(-4, 4, 100)
y = np.linspace(-4, 4, 100)
X, Y = np.meshgrid(x, y)

# 障碍物尺寸: 长3.0 x 宽1.5
obs_L = 3.0
obs_W = 1.5

# 计算到矩形的有符号距离 (外部距离)
def dist_to_rect(x, y, l, w):
    dx = np.maximum(np.abs(x) - l/2.0, 0)
    dy = np.maximum(np.abs(y) - w/2.0, 0)
    return np.sqrt(dx**2 + dy**2)

D = dist_to_rect(X, Y, obs_L, obs_W)

# 势场计算 (使用相同的 exp 函数形式)
# x0=0: 接触障碍物时势场最大
# x1=2.0: 距离障碍物 2m 外势场为 0
x0_pot = 0.0
x1_pot = 2.0
A_pot = 5.0 # 势场高度系数

# 计算势场 Z
Z = A_pot * exponential_decreasing_fcn(D, x0_pot, x1_pot)

# 绘图
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
ax.set_xlabel('X (m)', fontproperties=font_prop)
ax.set_ylabel('Y (m)', fontproperties=font_prop)
ax.set_zlabel('障碍惩罚值', fontproperties=font_prop)
ax.view_init(elev=45, azim=45) # 3D 视角 45度

plt.savefig('obstacle_penalty.pdf')
print("Files generated: reward_structure_corrected.pdf, obstacle_potential_corrected.pdf")