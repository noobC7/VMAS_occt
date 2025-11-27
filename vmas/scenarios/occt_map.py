import torch
from torch import Tensor
from typing import List, Tuple, Optional

class OcctRoad:
    def __init__(self,
                 batch_dim: int,
                 device: torch.device,
                 pts_gap: float = 1.0,
                 lane_width: float = 10.0,
                 road_pts: Optional[Tensor] = None):
        """
        初始化道路类
        
        Args:
            batch_dim: 批量环境维度
            device: 计算设备
            pts_gap: 道路点间距
            lane_width: 道路宽度
            road_pts: 可选的预定义道路点 [N, 2]，如果提供则使用它而不是生成新的道路
        """
        self.device = device
        self.batch_dim = batch_dim
        self.lane_width = lane_width
        
        # 生成道路中心线
        if road_pts is None:
            # 使用新的road_pts_gen函数生成道路点
            straight_length=50.0
            radius=25.0
            road_pts = self.road_pts_gen(
                road_segments=[
                    [straight_length, 0], 
                    [3.14*radius, 1/radius], 
                    [straight_length, 0],
                    [3.14*radius, -1/radius],
                    [straight_length, 0],
                    [3.14*radius*3, -1/3/radius],
                    [straight_length, 0],
                    [3.14*radius, -1/radius],
                ],
                start_pos=(0.0, 0.0),
                start_heading=0.0,
                pts_gap=pts_gap
            )
        
        # 扩展到batch维度
        self.road_pts = road_pts.expand(batch_dim, -1, 2)  # [B, N, 2]
        
        # 计算累积弧长
        seg = self.road_pts[:, 1:, :] - self.road_pts[:, :-1, :]  # [B, N-1, 2]
        seg_len = torch.linalg.norm(seg, dim=-1)  # [B, N-1]
        zero = torch.zeros(batch_dim, 1, device=device)
        self.road_cum_s = torch.cat([zero, torch.cumsum(seg_len, dim=-1)], dim=-1)  # [B, N]
        self.s_start = self.road_cum_s[:, 0]  # [B]
        self.s_end = self.road_cum_s[:, -1]  # [B]
        
        # 计算道路边界
        self._compute_boundaries()
    def get_s_max(self) -> Tensor:
        """
        获取最大弧长参数s_max
        
        返回:
            s_max: [B] 最大弧长参数
        """
        return self.road_cum_s[:, -1]  # [B]
    def _compute_boundaries(self):
        """
        计算道路左右边界
        """
        # 计算切线向量
        tangents = self.road_pts[:, 1:, :] - self.road_pts[:, :-1, :]  # [B, N-1, 2]
        norm_tangents = torch.linalg.norm(tangents, dim=-1, keepdim=True) + 1e-8  # [B, N-1, 1]
        unit_tangents = tangents / norm_tangents  # [B, N-1, 2]
        
        # 计算法线向量（逆时针旋转90度）
        normals = torch.stack([-unit_tangents[..., 1], unit_tangents[..., 0]], dim=-1)  # [B, N-1, 2]
        
        # 为每个点计算法线（端点使用相邻线段的法线，中间点使用左右线段法线的平均值）
        point_normals = torch.zeros_like(self.road_pts)  # [B, N, 2]
        point_normals[:, 0, :] = normals[:, 0, :]
        mid_normals = (normals[:, :-1, :] + normals[:, 1:, :]) / 2  # [B, N-2, 2]
        point_normals[:, 1:-1, :] = mid_normals
        point_normals[:, -1, :] = normals[:, -1, :]
        
        # 归一化法线向量
        point_normals = point_normals / torch.linalg.norm(point_normals, dim=-1, keepdim=True) + 1e-8
        
        # 计算左右边界点
        self.road_left_pts = self.road_pts + point_normals * self.lane_width / 2  # [B, N, 2]
        self.road_right_pts = self.road_pts - point_normals * self.lane_width / 2  # [B, N, 2]
    
    def get_pts(self, s: Tensor) -> Tensor:
        """
        根据弧长参数s获取道路上的坐标
        
        输入:
            s: [B] 或 [B,K] 的弧长参数，单位需与 self.road_cum_s 一致
        输出:
            p: [B,2] 或 [B,K,2] 上对应的坐标（分段线性插值）
        说明:
            - 使用 torch.searchsorted 在最后维度上找段索引（向量化、支持 GPU）
            - 自动夹取到合法弧长范围 [s_min, s_max]
        """
        cum_s = self.road_cum_s              # [B, N]
        pts = self.road_pts                  # [B, N, 2]
        B, N = cum_s.shape
        eps = 1e-8

        # 兼容 [B] 或 [B,K]
        s_in = s
        if s.dim() == 1:
            s = s[:, None]                   # -> [B,1]
            squeeze_back = True
        else:
            squeeze_back = False

        # 每环境有效范围
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
        p = p0 + t[..., None] * (p1 - p0)                                     # [B,K,2]

        if squeeze_back:
            p = p[:, 0, :]                                                    # [B,2]
        return p
    
    def get_tangent_vector(self, s: Tensor) -> Tensor:
        """
        计算道路上弧长s处的切线方向向量
        
        Args:
            s: [B] 或 [B,K] 弧长参数
        Returns:
            tangent_vec: [B,2] 或 [B,K,2] 切线单位向量
        """
        # 使用小扰动法计算切线向量
        epsilon = 1e-3  # 小扰动值
        # 确保max参数的维度与s匹配
        max_values = self.road_cum_s[:, -1] - 1e-6  # [B]
        if s.dim() > 1:  # 如果s是[B,K]形状
            # 扩展max_values到[B,1]以支持广播
            max_values = max_values.unsqueeze(-1)  # [B,1]
        s_plus = torch.clamp(s + epsilon, max=max_values)
        pos_plus = self.get_pts(s_plus)  # 调用get_pts而非road_C
        pos = self.get_pts(s)  # 调用get_pts而非road_C
        tangent_vec = pos_plus - pos
        # 归一化切线向量
        tangent_vec = tangent_vec / torch.linalg.norm(tangent_vec, dim=-1, keepdim=True) + 1e-8
        return tangent_vec
    
    def get_tangent_heading(self, s: Tensor) -> Tensor:
        """
        计算道路上弧长s处的切线航向角
        
        Args:
            s: [B] 或 [B,K] 弧长参数
        Returns:
            tangent_theta: [B] 或 [B,K] 切线方向角（弧度）
        """
        tangent_vec = self.get_tangent_vector(s)
        # 计算切线方向角（弧度）
        tangent_theta = torch.atan2(tangent_vec[..., 1], tangent_vec[..., 0])
        return tangent_theta
    
    def get_normal_vector(self, s: Tensor) -> Tensor:
        """
        计算道路上弧长s处的法线方向向量
        
        Args:
            s: [B] 或 [B,K] 弧长参数
        Returns:
            normal_vec: [B,2] 或 [B,K,2] 法线单位向量（逆时针旋转90度）
        """
        # 获取切线向量
        tangent_vec = self.get_tangent_vector(s)
        # 法线向量 = 切线向量逆时针旋转90度
        normal_vec = torch.stack([-tangent_vec[..., 1], tangent_vec[..., 0]], dim=-1)
        # 归一化法线向量
        normal_vec = normal_vec / torch.linalg.norm(normal_vec, dim=-1, keepdim=True) + 1e-8
        return normal_vec
    
    def get_normal_heading(self, s: Tensor) -> Tensor:
        """
        计算道路上弧长s处的法线航向角
        
        Args:
            s: [B] 或 [B,K] 弧长参数
        Returns:
            normal_theta: [B] 或 [B,K] 法线方向角（弧度）
        """
        normal_vec = self.get_normal_vector(s)
        # 计算法线方向角（弧度）
        normal_theta = torch.atan2(normal_vec[..., 1], normal_vec[..., 0])
        return normal_theta
        
    # 保留原有的get_tangent和get_normal函数以保持向后兼容，但更新实现
    def get_tangent(self, s: Tensor) -> Tensor:
        """
        计算道路上弧长s处的切线方向角（向后兼容）
        
        Args:
            s: [B] 弧长参数
        Returns:
            tangent_theta: [B] 切线方向角（弧度）
        """
        return self.get_tangent_heading(s)
    
    def get_normal(self, s: Tensor) -> Tensor:
        """
        计算道路上弧长s处的法线方向角（向后兼容）
        
        Args:
            s: [B] 弧长参数
        Returns:
            normal_theta: [B] 法线方向角（弧度）
        """
        return self.get_normal_heading(s)
    def solve_delta_s(self, s_front: Tensor, L: Tensor, *, max_iter: int = 20) -> Tuple[Tensor, Tensor]:
        """
        固定弦长：批量求 Δs（向量化二分）
        
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
        p_f = self.get_pts(s_front)                                            # [B,2]

        # 不可解判断：最大 Δs=hi 时的弦长仍 < L
        p_r_hi = self.get_pts(s_front - hi)                                    # [B,2]
        chord_max = torch.linalg.norm(p_f - p_r_hi, dim=-1)                   # [B]
        infeasible = (chord_max + 1e-6) < L                                   # [B] 需要外部策略处理

        # 向量化二分
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)                                             # [B]
            p_r = self.get_pts(s_front - mid)                                  # [B,2]
            chord = torch.linalg.norm(p_f - p_r, dim=-1)                      # [B]
            go_left = chord > L                                               # [B]
            hi = torch.where(go_left, mid, hi)
            lo = torch.where(go_left, lo, mid)

        delta_s = 0.5 * (lo + hi)                                             # [B]
        # 对不可解样本，保持 delta_s 为 0（或可改为 hi），由上层降速/脱开处理
        delta_s = torch.where(infeasible, torch.zeros_like(delta_s), delta_s)
        return delta_s, infeasible

    def road_pts_gen(
        self,
        road_segments: List[List[float]],
        start_pos: Tuple[float, float] = (0.0, 0.0),
        start_heading: float = 0.0,
        pts_gap: float = 1.0
    ) -> Tensor:
        """
        生成道路中心线点集

        参数:
            road_segments: 道路段参数序列，每个元素为[长度, 曲率]，曲率=1/半径
            start_pos: 起始点位置 (x, y)
            start_heading: 起始航向角（弧度）
            pts_gap: 道路点间距

        返回:
            road_pts: [N, 2] 道路中心线点集
        """
        points = []
        current_x, current_y = start_pos
        current_heading = start_heading
        prev_end_point = None

        for segment in road_segments:
            length, curvature = segment
            n_points = int(length // pts_gap)
            if n_points <= 0:
                continue

            if curvature == 0:
                # 直线段
                x = torch.linspace(1.0, length-1.0, n_points, device=self.device)
                y = torch.zeros(n_points, device=self.device)
                segment_pts = torch.stack([x, y], dim=-1)
            else:
                # 曲线段，曲率=1/半径
                radius = 1.0 / curvature
                angle = length / radius
                theta = torch.linspace(0.0, angle, n_points, device=self.device)
                # 计算圆弧上的点
                x = radius * torch.sin(theta)
                y = radius - radius * torch.cos(theta)
                segment_pts = torch.stack([x, y], dim=-1)

            # 应用旋转变换
            cos_heading = torch.cos(torch.tensor(current_heading))
            sin_heading = torch.sin(torch.tensor(current_heading))
            rotation_matrix = torch.tensor([[cos_heading, -sin_heading],
                                           [sin_heading, cos_heading]], device=self.device)
            rotated_pts = torch.matmul(segment_pts, rotation_matrix)

            # 平移到当前位置
            translated_pts = rotated_pts + torch.tensor([current_x, current_y], device=self.device)

            # 确保首尾衔接且不重复
            if prev_end_point is not None:
                # 移除第一个点以避免重复
                translated_pts = translated_pts[1:]
            points.append(translated_pts)

            # 更新当前位置和航向
            if len(translated_pts) > 0:
                current_pos = translated_pts[-1]
                current_x, current_y = current_pos[0], current_pos[1]
                prev_end_point = current_pos

            # 更新航向角
            current_heading += angle if curvature != 0 else 0

        # 合并所有点
        road_pts = torch.cat(points, dim=0) if points else torch.empty((0, 2), device=self.device)
        return road_pts
    
# Debug 绘图函数，用于可视化道路
import matplotlib.pyplot as plt

def plot_road_debug():
    """
    简单的debug绘图函数，用于可视化道路中心线和边界
    """
    # 创建OcctRoad实例
    device = torch.device("cpu")
    road = OcctRoad(batch_dim=1, device=device)
    
    # 获取道路数据
    road_pts = road.road_pts[0].cpu().numpy()  # 取第一个batch的道路点
    left_pts = road.road_left_pts[0].cpu().numpy()
    right_pts = road.road_right_pts[0].cpu().numpy()
    
    # 绘制道路
    plt.figure(figsize=(12, 8))
    
    # 绘制道路中心线
    plt.plot(road_pts[:, 0], road_pts[:, 1], 'b-', label='Road Center Line')
    
    # 绘制道路边界
    plt.plot(left_pts[:, 0], left_pts[:, 1], 'r--', label='Left Boundary')
    plt.plot(right_pts[:, 0], right_pts[:, 1], 'g--', label='Right Boundary')
    
    # 标记起点和终点
    plt.scatter(road_pts[0, 0], road_pts[0, 1], c='green', s=100, label='Start')
    plt.scatter(road_pts[-1, 0], road_pts[-1, 1], c='red', s=100, label='End')
    
    # 设置图表
    plt.title('Road Visualization (Debug)')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    
    # 显示图表
    plt.show()


if __name__ == "__main__":
    # 运行debug绘图函数
    plot_road_debug()