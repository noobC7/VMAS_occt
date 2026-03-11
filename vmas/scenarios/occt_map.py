import torch
from torch import Tensor
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
import os
import pickle
import glob
from collections import deque
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.scenario.scenario import Scenario
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torchcubicspline import(natural_cubic_spline_coeffs, 
                                NaturalCubicSpline)
import time
import torch.nn.functional as F
from vmas.scenarios.occt_boundary import OcctBoundaryCalculator

from scipy.interpolate import CubicSpline  # 若用GPU可替换为torch版本
import numpy as np
def smooth_road_centerline(
    centerline_pts: np.ndarray,
    sample_step: float = 0.05,  # 平滑后点的步长（适配你的校准精度）
    smooth_density: int = 5,    # 平滑密度（越大越平滑，默认5倍原始点数）
    min_gap: float = 0.2        # 原始点保留阈值：间隔≥0.2才保留该点
) -> np.ndarray:
    """
    平滑道路中心线（新增原始点筛选：仅保留间隔≥0.2的点，避免回环）
    Args:
        centerline_pts: 原始中心线顶点，格式为[N, 2]的numpy数组/列表（N≥3）
        sample_step: 平滑后点的均匀步长（单位m），默认0.05m
        smooth_density: 插值密度因子，原始点数×该值=插值后点数，默认5
        min_gap: 原始点保留阈值，相邻点间隔≥此值才保留，默认0.2m
    Returns:
        smooth_centerline_pts: 平滑后的中心线点，[M, 2]的numpy数组
    """
    # 1. 格式统一为numpy数组
    centerline_pts = np.array(centerline_pts, dtype=np.float32)
    assert len(centerline_pts.shape) == 2 and centerline_pts.shape[1] == 2, \
        f"centerline_pts必须是[N, 2]维度，当前为{centerline_pts.shape}"
    N = len(centerline_pts)
    if N < 3:  # 点数过少无需平滑，直接返回原数据
        return centerline_pts
    
    # 2. 原始点预处理：仅保留间隔≥min_gap的点（核心逻辑）
    filtered_pts = [centerline_pts[0]]  # 先保留第一个点
    last_pt = centerline_pts[0]        # 记录上一个保留的点
    
    for i in range(1, N):
        current_pt = centerline_pts[i]
        # 计算当前点与上一个保留点的欧氏距离（弧长）
        gap = np.linalg.norm(current_pt - last_pt)
        # 仅当间隔≥0.2时，保留该点并更新上一个保留点
        if gap >= min_gap:
            filtered_pts.append(current_pt)
            last_pt = current_pt
    
    # 确保至少保留3个点（样条插值的最小要求）
    if len(filtered_pts) < 3:
        filtered_pts = centerline_pts.copy()  # 不足则恢复原始点
    filtered_pts = np.array(filtered_pts)
    
    # 3. 对筛选后的点计算累积弧长
    cum_dist = np.zeros(len(filtered_pts))
    for i in range(1, len(filtered_pts)):
        cum_dist[i] = cum_dist[i-1] + np.linalg.norm(filtered_pts[i] - filtered_pts[i-1])
    
    # 4. 三次B样条插值（添加端点夹紧约束，避免振荡）
    cs_x = CubicSpline(cum_dist, filtered_pts[:, 0], bc_type='clamped')
    cs_y = CubicSpline(cum_dist, filtered_pts[:, 1], bc_type='clamped')
    
    # 5. 生成密集的插值点（基于筛选后点的弧长范围）
    t_smooth = np.linspace(cum_dist[0], cum_dist[-1], len(filtered_pts) * smooth_density)
    interp_x = cs_x(t_smooth)
    interp_y = cs_y(t_smooth)
    interp_pts = np.stack([interp_x, interp_y], axis=-1)
    
    # 6. 计算插值点的累积弧长，用于最终均匀采样
    cum_dist_interp = np.zeros(len(interp_pts))
    for i in range(1, len(interp_pts)):
        cum_dist_interp[i] = cum_dist_interp[i-1] + np.linalg.norm(interp_pts[i] - interp_pts[i-1])
    
    # 7. 按固定步长重新采样（线性插值避免三次样条振荡）
    target_dist = np.arange(0, cum_dist_interp[-1], sample_step)
    final_x = np.interp(target_dist, cum_dist_interp, interp_pts[:, 0])
    final_y = np.interp(target_dist, cum_dist_interp, interp_pts[:, 1])
    smooth_centerline_pts = np.stack([final_x, final_y], axis=-1)
    
    return smooth_centerline_pts
class MapBase(ABC):
    @abstractmethod
    def get_s_max(self) -> Tensor:
        #获取最大弧长参数s_max
        pass
    @abstractmethod
    def get_s_max_idx(self) -> Tensor:
        #获取最大弧长参数s_max的索引
        pass
    @abstractmethod
    def get_road_center_pts(self) -> Tensor:
        #获取道路中心线离散路径点
        pass
    @abstractmethod
    def get_road_left_pts(self) -> Tensor:
        #获取道路左侧离散路径点
        pass
    @abstractmethod
    def get_road_right_pts(self) -> Tensor:
        #获取道路右侧离散路径点
        pass
    @abstractmethod
    # 以下函数输入s为弧长参数，支持[B] 或 [B,K] ，输出为[B,2] 或 [B,K,2] 
    @abstractmethod
    def get_pts(self, s: Tensor) -> Tensor:
        #根据弧长参数s获取路径点
        pass
    @abstractmethod
    def get_ref_v(self, s: Tensor) -> Tensor:
        #根据弧长参数s获取参考速度
        pass
    def get_tangent_vector(self, s: Tensor) -> Tensor:
        #根据弧长参数s获取道路切线单位向量
        pass
    def get_tangent_heading(self, s: Tensor) -> Tensor:
        tangent_vec = self.get_tangent_vector(s)
        tangent_theta = torch.atan2(tangent_vec[..., 1], tangent_vec[..., 0])
        return tangent_theta
    def get_normal_vector(self, s: Tensor) -> Tensor:
        tangent_vec = self.get_tangent_vector(s)
        normal_vec = torch.stack([-tangent_vec[..., 1], tangent_vec[..., 0]], dim=-1)
        return normal_vec
    def get_normal_heading(self, s: Tensor) -> Tensor:
        normal_vec = self.get_normal_vector(s)
        normal_theta = torch.atan2(normal_vec[..., 1], normal_vec[..., 0])
        return normal_theta
    
class OcctMap(MapBase):
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
            radius=35.0
            road_pts = self._road_pts_gen(
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
                start_pos=(-40.0, -30.0),
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
        
        # 计算道路边界
        self._compute_boundaries()
    
    def _road_pts_gen(
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
        cum_s = self.road_cum_s              # [B, N]
        pts = self.road_pts                  # [B, N, 2]
        B, N = cum_s.shape
        eps = 1e-8
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
    def get_ref_v(self, s: Tensor) -> Tensor:
        raise NotImplementedError("get_ref_v is not implemented for this map type.")
    def get_road_center_pts(self) -> Tensor:
        return self.road_pts
    
    def get_road_left_pts(self) -> Tensor:
        return self.road_left_pts
    
    def get_road_right_pts(self) -> Tensor:
        return self.road_right_pts
    
    def get_s_max(self) -> Tensor:
        return self.road_cum_s[:, -1]  # [B]
    def get_s_max_idx(self) -> Tensor:
        raise NotImplementedError("get_s_max_idx is not implemented for this map type.")
    def get_tangent_vector(self, s: Tensor) -> Tensor:
        epsilon = 1e-3  # 小扰动值
        max_values = self.road_cum_s[:, -1] - 1e-6  # [B]
        if s.dim() > 1: 
            max_values = max_values.unsqueeze(-1)  # [B,1]
        s_plus = torch.clamp(s + epsilon, max=max_values)
        pos_plus = self.get_pts(s_plus)  # 调用get_pts而非road_C
        pos = self.get_pts(s)  # 调用get_pts而非road_C
        tangent_vec = pos_plus - pos
        tangent_vec = tangent_vec / torch.linalg.norm(tangent_vec, dim=-1, keepdim=True) + 1e-8
        return tangent_vec
    
    
def get_cr_scenario(scenario_path):
    scenario, _ = CommonRoadFileReader(scenario_path).open()
    for dyn_obs in scenario.dynamic_obstacles:
        scenario.remove_obstacle(dyn_obs)
    return scenario
class OcctCRMap(MapBase):
    def __init__(self,
                 batch_dim: int,
                 device: torch.device,
                 cr_map_dir: str = "vmas/scenarios_data/cr_maps/debug",
                 sample_gap: float = 1,
                 min_lane_width: float = 2.1,
                 min_lane_len: float = 70,
                 max_ref_v: float = 20/3.6,
                 is_constant_ref_v: bool = False,
                 rod_len = None,
                 extend_len = None,
                 n_agents: int = 4): # 采样间隔
        """
        初始化道路类，使用CommonRoad地图并基于torchcubicspline实现路径表示
        
        Args:
            batch_dim: 批量环境维度
            device: 计算设备
            cr_map_dir: 外部CommonRoad地图所在的文件夹路径（必填）
        """
        self.device = device
        self.batch_dim = batch_dim
        self.cr_map_dir = cr_map_dir
        self.vis_dir = os.path.join(cr_map_dir, "vis")
        os.makedirs(self.vis_dir, exist_ok=True)
        self.sample_gap = sample_gap
        self.min_lane_width = min_lane_width
        self.min_lane_len = min_lane_len
        self.max_ref_v = max_ref_v # max ref vel in m/s
        self.is_constant_ref_v = is_constant_ref_v # if True, then ref_v is constant and equal to max_ref_v
        self.rod_len = rod_len # 车辆长度
        self.extend_len = rod_len if extend_len is None else 0.0
        self.start_end_distance_threshold = 25 # 起始点和结束点的距离阈值，小于该阈值的路径会被过滤
        self.n_agents = n_agents
        
        # 初始化路径库
        self.path_library = []
        self.scenario_library = dict[str, Scenario]()
        self.path_splines = []
        self.path_s_max = []
        self.max_path_length = 0
        self.max_path_s_list = None
        
        # 处理CommonRoad地图
        if cr_map_dir.split('/')[-1]=="chapter4":
            self._cr_map_process(cr_map_dir)
            #self._cr_map_process_chapter4(cr_map_dir,extend_left_boundary=True)
        else:
            self._cr_map_process(cr_map_dir)
        
        print(f"[OcctCRMap]共{len(self.path_library)}条路径数据,最长为{self.max_path_length:.2f}米,顶点有{len(self.max_path_s_list)}个,平均宽度为{self.get_lane_width():.2f}米")
        # 确保有路径数据
        if len(self.path_library) == 0:
            raise ValueError("No paths found in the provided CommonRoad map directory")
        
        # 初始化路径样条
        self.reset_splines()

    def get_lane_width(self,type="mean") -> Tensor:
        """
        获取所有路径的车道宽度
        
        Args:
            type: "mean","min"或"max"，表示返回平均车道宽度还是最小/最大车道宽度
        
        Returns:
            mean_lane_width: [B] 平均车道宽度
        """
        lane_widths = torch.hstack([torch.tensor(path["lane_width"],device=self.device) for path in self.path_library])
        if type == "mean":
            return lane_widths.mean(dim=-1).item()
        elif type == "min":
            return lane_widths.min(dim=-1).item()
        elif type == "max":
            return lane_widths.max(dim=-1).item()
        else:
            raise ValueError(f"type must be 'mean','min' or 'max', but got {type}")

    def _get_cum_len(self, vertices: Tensor) -> Tensor:
        """
        计算路径点的累计长度
        
        Args:
            vertices: [N, 2] 路径点
        
        Returns:
            cum_len: [N] 累计长度
        """
        # 计算路径点的累计长度
        if type(vertices) == np.ndarray:
            vertices = torch.tensor(vertices,device=vertices.device)
        seg = vertices[1:] - vertices[:-1]  # [N-1, 2]
        seg_len = torch.linalg.norm(seg, dim=-1)  # [N-1]
        cum_len = torch.cat([torch.zeros(1, device=vertices.device), torch.cumsum(seg_len, dim=0)])  # [N]
        return cum_len
    
    def _resample_path(self, vertices: Tensor, M: int = None) -> Tuple[Tensor, Tensor]:
        """
        对路径点进行重采样
        
        Args:
            vertices: [N, 2] 原始路径点
        
        Returns:
            resampled_vertices: [M, 2] 重采样后的路径点
            s: [M] 重采样后的弧长参数
        """
        return_ndarray=False
        if type(vertices) == np.ndarray:
            return_ndarray=True
            vertices = torch.tensor(vertices,device=vertices.device)
        original_s = self._get_cum_len(vertices)  # [N]
        device = vertices.device
        # 计算重采样点数量
        s_max = original_s[-1]
        if M is None:
            M = max(2, int(torch.floor(s_max / self.sample_gap)))
        
        # 生成重采样的弧长参数
        s = torch.linspace(0.0, s_max, M, device=device)
        
        # 使用线性插值获取重采样点
        resampled_vertices = torch.zeros(M, 2, device=device)
        
        # 找到每个t对应的区间
        idx = torch.searchsorted(original_s, s, right=True) - 1
        idx = torch.clamp(idx, 0, len(original_s) - 2)
        
        # 线性插值
        s0 = original_s[idx]
        s1 = original_s[idx + 1]
        p0 = vertices[idx]
        p1 = vertices[idx + 1]
        
        alpha = (s - s0) / (s1 - s0 + 1e-8)
        resampled_vertices = p0 + alpha.unsqueeze(1) * (p1 - p0)
        if return_ndarray:
            resampled_vertices = resampled_vertices.cpu().numpy()
            s = s.cpu().numpy()
        return resampled_vertices, s
    
    def _enrich_vertices_sampling(self, center_vertices, left_vertices, right_vertices):
        """
        对直路路径点进行重采样,确保采样点数量为sample_num
        
        Args:
            vertices: [N, 2] 原始路径点
        
        Returns:
            resampled_vertices: [M, 2] 重采样后的路径点
            s: [M] 重采样后的弧长参数
        """
        center_seg_lengths = np.linalg.norm(center_vertices[1:] - center_vertices[:-1], axis=1)
        if np.max(center_seg_lengths) > 2*self.sample_gap:
            # make sure the resample pts num is the same for center, left, right
            center_length=self._get_cum_len(center_vertices)
            sample_num = max(2, int(torch.floor(center_length[-1] / self.sample_gap)))
            resampled_vertices = [np.zeros((sample_num, 2)),np.zeros((sample_num, 2)),np.zeros((sample_num, 2))]
            for resampled, vertices in zip(resampled_vertices, [center_vertices, left_vertices, right_vertices]):
                segment_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
                cum_lengths = np.zeros(len(vertices))
                cum_lengths[1:] = np.cumsum(segment_lengths)
                target_s = np.linspace(0, cum_lengths[-1], sample_num)
                for i in range(2):
                    resampled[:, i] = np.interp(target_s, cum_lengths, vertices[:, i])
            return resampled_vertices[0], resampled_vertices[1], resampled_vertices[2]
        return center_vertices, left_vertices, right_vertices
    def _detect_loop(self, points: List[Tuple[float, float]], tol: float = 1.0) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        检测离散曲线是否形成回环
        
        参数:
        points: 离散点列表，每个点是一个(x, y)元组
        tol: 容差距离（米），默认1米
        
        返回:
        (has_loop, loop_pairs): 
            has_loop - 是否检测到回环
            loop_pairs - 所有形成回环的点对索引列表
        """
        
        n = len(points)
        if n < 4:  # 至少需要4个点才可能形成非平凡的环
            return False, []
        
        points_array = torch.tensor(points,device=self.device)
        loop_pairs = []
        
        # 检查每对点的距离（避免相邻点和最近邻点）
        for i in range(n - 3):  # 留出足够间隔
            for j in range(i + 3, n):  # j从i+3开始，避免相邻和接近的点
                # 计算欧几里得距离
                distance = torch.linalg.norm(points_array[i] - points_array[j])
                
                # 如果距离小于容差，说明形成回环
                if distance <= tol:
                    loop_pairs.append((i, j))
        
        has_loop = len(loop_pairs) > 0
        return has_loop, loop_pairs
    @staticmethod
    def extend_trajectory(trajectory, head_extend_len, tail_extend_len):
        """
        为轨迹数组首尾各添加一个延伸点（长度为0时返回原轨迹）
        :param trajectory: 轨迹数组，shape为(N, 2)
        :param head_extend_len: 头部延伸长度（0则不延伸）
        :param tail_extend_len: 尾部延伸长度（0则不延伸）
        :return: 扩展后的轨迹数组
        """
        traj = np.array(trajectory, dtype=np.float64)
        if len(traj) < 2:
            raise ValueError("轨迹数组至少需要包含2个点才能计算增量")
        
        # 初始化延伸后的轨迹为原轨迹
        extended_traj = traj.copy()
        
        # ===== 头部延伸：仅当长度>0时添加 =====
        if head_extend_len > 1e-9:  # 浮点精度容错，避免0值误判
            p0, p1 = traj[0], traj[1]
            head_delta = p1 - p0
            head_unit_vec = head_delta / np.linalg.norm(head_delta)
            head_extend_point = p0 - head_unit_vec * head_extend_len
            extended_traj = np.vstack([head_extend_point, extended_traj])
        
        # ===== 尾部延伸：仅当长度>0时添加 =====
        if tail_extend_len > 1e-9:
            p_last2, p_last1 = traj[-2], traj[-1]
            tail_delta = p_last1 - p_last2
            tail_unit_vec = tail_delta / np.linalg.norm(tail_delta)
            tail_extend_point = p_last1 + tail_unit_vec * tail_extend_len
            extended_traj = np.vstack([extended_traj, tail_extend_point])
        
        return extended_traj
    @staticmethod
    def extend_road(center_vertices, left_vertices, right_vertices, head_extend_len, tail_extend_len):
        """
        同步延伸道路的中心线、左边界、右边界（长度为0时返回原轨迹）
        :return: (extend_center, extend_left, extend_right)
        """
        center = np.array(center_vertices, dtype=np.float64)
        left = np.array(left_vertices, dtype=np.float64)
        right = np.array(right_vertices, dtype=np.float64)
        
        if not (len(center) == len(left) == len(right)):
            raise ValueError("中心线、左边界、右边界的点数量必须一致")
        if len(center) < 2:
            raise ValueError("轨迹至少需要2个点才能延伸")
        
        # 初始化延伸后的轨迹为原轨迹
        extend_center = center.copy()
        extend_left = left.copy()
        extend_right = right.copy()
        
        # 计算中心线延伸向量（仅当长度>0时）
        head_extend_vec = np.zeros(2)
        tail_extend_vec = np.zeros(2)
        if head_extend_len > 1e-9:
            p0, p1 = center[0], center[1]
            head_delta = p1 - p0
            head_unit_vec = head_delta / np.linalg.norm(head_delta)
            head_extend_vec = head_unit_vec * head_extend_len  # 头部延伸向量（原首点 - 延伸点）
            # 头部延伸：中心线+左+右
            head_center = center[0] - head_extend_vec
            head_left = left[0] - head_extend_vec
            head_right = right[0] - head_extend_vec
            extend_center = np.vstack([head_center, extend_center])
            extend_left = np.vstack([head_left, extend_left])
            extend_right = np.vstack([head_right, extend_right])
        
        if tail_extend_len > 1e-9:
            p_last2, p_last1 = center[-2], center[-1]
            tail_delta = p_last1 - p_last2
            tail_unit_vec = tail_delta / np.linalg.norm(tail_delta)
            tail_extend_vec = tail_unit_vec * tail_extend_len  # 尾部延伸向量（延伸点 - 原尾点）
            # 尾部延伸：中心线+左+右
            tail_center = center[-1] + tail_extend_vec
            tail_left = left[-1] + tail_extend_vec
            tail_right = right[-1] + tail_extend_vec
            extend_center = np.vstack([extend_center, tail_center])
            extend_left = np.vstack([extend_left, tail_left])
            extend_right = np.vstack([extend_right, tail_right])
    
        return extend_center, extend_left, extend_right
    # @staticmethod
    # def extend_road(center_vertices, left_vertices, right_vertices, head_extend_len, tail_extend_len):
    #     """
    #     同步延伸道路的中心线、左边界、右边界（长度为0时返回原轨迹）
    #     根据左右边界端点在中心线切线方向的投影偏移，自动调整各自延伸长度
    #     :return: (extend_center, extend_left, extend_right)
    #     """
    #     center = np.array(center_vertices, dtype=np.float64)
    #     left = np.array(left_vertices, dtype=np.float64)
    #     right = np.array(right_vertices, dtype=np.float64)

    #     if not (len(center) == len(left) == len(right)):
    #         raise ValueError("中心线、左边界、右边界的点数量必须一致")
    #     if len(center) < 2:
    #         raise ValueError("轨迹至少需要2个点才能延伸")

    #     # 初始化延伸后的轨迹为原轨迹
    #     extend_center = center.copy()
    #     extend_left = left.copy()
    #     extend_right = right.copy()

    #     # 头部延伸
    #     if head_extend_len > 1e-9:
    #         p0, p1 = center[0], center[1]
    #         head_delta = p1 - p0
    #         head_unit_vec = head_delta / np.linalg.norm(head_delta)

    #         # 计算左右边界端点相对于中心线端点在切线方向的投影偏移
    #         left_offset = np.dot(left[0] - center[0], head_unit_vec)
    #         right_offset = np.dot(right[0] - center[0], head_unit_vec)
    #         # 计算延伸后的中心点位置
    #         head_center = center[0] - head_unit_vec * head_extend_len
    #         head_left = head_center + (left[0] - center[0]) - left_offset * head_unit_vec
    #         head_right = head_center + (right[0] - center[0]) - right_offset * head_unit_vec
    #         extend_center = np.vstack([head_center, extend_center])
    #         extend_left = np.vstack([head_left, extend_left])
    #         extend_right = np.vstack([head_right, extend_right])

    #     # 尾部延伸
    #     if tail_extend_len > 1e-9:
    #         p_last2, p_last1 = center[-2], center[-1]
    #         tail_delta = p_last1 - p_last2
    #         tail_unit_vec = tail_delta / np.linalg.norm(tail_delta)

    #         # 计算左右边界端点相对于中心线端点在切线方向的投影偏移
    #         left_offset = np.dot(left[-1] - center[-1], tail_unit_vec)
    #         right_offset = np.dot(right[-1] - center[-1], tail_unit_vec)
    #         # 计算延伸后的中心点位置
    #         tail_center = center[-1] + tail_unit_vec * tail_extend_len
    #         tail_left = tail_center + (left[-1] - center[-1]) - left_offset * tail_unit_vec
    #         tail_right = tail_center + (right[-1] - center[-1]) - right_offset * tail_unit_vec
    #         # 验证：三个点在切线方向的投影应该相同
    #         proj_left = np.dot(tail_left - tail_center, tail_unit_vec)
    #         proj_right = np.dot(tail_right - tail_center, tail_unit_vec)

    #         extend_center = np.vstack([extend_center, tail_center])
    #         extend_left = np.vstack([extend_left, tail_left])
    #         extend_right = np.vstack([extend_right, tail_right])

    #     return extend_center, extend_left, extend_right
    
    def _cr_map_process(self, map_dir: str) -> None:
        """
        处理CommonRoad地图和车道信息
        
        Args:
            map_dir: 地图所在的文件夹路径
        """
        boundary_calculator = OcctBoundaryCalculator()
        dump_file = os.path.join(self.cr_map_dir, "map_data.pkl")
        if os.path.exists(dump_file):
            self.scenario_library, self.path_library,\
                self.max_path_length,self.max_path_s_list = pickle.load(open(dump_file, "rb"))
            self.max_path_length=self.max_path_length.to(self.device)
            self.max_path_s_list=self.max_path_s_list.to(self.device)
            return
        # 递归读取文件夹中所有XML文件
        map_files = glob.glob(os.path.join(map_dir, "**/*.xml"), recursive=True)
        
        assert self.rod_len is not None, "请先设置货物长度 L"
        # 打印地图库信息
        print(f"找到 {len(map_files)} 个地图文件:")
        for i, map_file in enumerate(map_files):
            print(f"  {i+1}. {os.path.basename(map_file)}")
        
        # 处理每个地图文件
        for map_file in map_files:
            map_name = os.path.basename(map_file)
            print(f"\n处理地图: {map_name}")
            
            # 读取地图场景
            scenario = get_cr_scenario(map_file)

            self.scenario_library[map_name] = scenario
            
            # 获取所有车道段
            lanelets = scenario.lanelet_network.lanelets
            
            # 找到所有起点车道段（predecessor为空）
            start_lanelets = [lanelet for lanelet in lanelets if not lanelet.predecessor]
            
            # 初始化路径ID库
            path_id_library = []
            
            # 使用BFS获取所有可能的路径
            for start_lanelet in start_lanelets:
                queue = deque()
                queue.append([start_lanelet.lanelet_id])
                
                while queue:
                    current_path = queue.popleft()
                    current_lanelet_id = current_path[-1]
                    
                    # 获取当前车道段对象
                    current_lanelet = scenario.lanelet_network.find_lanelet_by_id(current_lanelet_id)
                    
                    # 如果没有后继，说明是路径终点
                    if not current_lanelet.successor:
                        path_id_library.append(current_path)
                    else:
                        # 遍历所有后继
                        for successor_id in current_lanelet.successor:
                        # 检查后继车道段是否已经在当前路径中，如果是则跳过（避免回环）
                            if successor_id not in current_path:
                                new_path = current_path.copy()
                                new_path.append(successor_id)
                                queue.append(new_path)
            
            # 打印路径ID库信息
            print(f"找到 {len(path_id_library)} 条路径:")
            for i, path in enumerate(path_id_library):
                print(f"  路径 {i+1}: {path}")
            
            # 处理每条路径，生成路径数据
            for path_ids in tqdm(path_id_library, desc="处理路径"):
                path_data = {
                    "center_vertices": [],
                    "left_vertices": [],
                    "right_vertices": []
                }
                # if not((path_ids[0]==128 and path_ids[-1]==106) or \
                #    (path_ids[0]==102 and path_ids[-1]==175) or \
                #    (path_ids[0]==189 and path_ids[-1]==175)):
                # # if not(path_ids[0]==128 and path_ids[-1]==106):
                # # if not((path_ids[0]==128 and path_ids[-1]==106) or \
                # #    (path_ids[0]==102 and path_ids[-1]==175)):
                #     continue
                # if ((path_ids[0]==100 and path_ids[-1]==129) or \
                #    (path_ids[0]==108 and path_ids[-1]==166) or \
                #    (path_ids[0]==128 and path_ids[-1]==106) or \
                #     (path_ids[0]==189 and path_ids[-1]==103)):
                #if not (path_ids[0]==102 and path_ids[-1]==164):
                # if not(path_ids[0]==128 and path_ids[-1]==106):
                # if not((path_ids[0]==100 and path_ids[-1]==169)):
                #     continue
                # for chapter 2 paper illustration
                # if not ((path_ids[0]==177 and path_ids[-1]==129) or \
                #    (path_ids[0]==153 and path_ids[-1]==175)):
                #     continue
                if (path_ids[0]==112 and path_ids[-1]==129):
                    continue
                if map_name == "USA_Roundabout_EP_repaired.xml":
                    if (path_ids[-1]==124) or \
                        (path_ids[0]==100 and path_ids[-1]==169) or \
                            (path_ids[0]==149 and path_ids[-1]==157) or \
                            (path_ids[0]==149 and path_ids[-1]==124) or \
                            (path_ids[0]==127 and path_ids[-1]==124) or \
                            (path_ids[0]==127 and path_ids[-1]==157) or \
                            (path_ids[0]==149 and path_ids[-1]==132):
                        continue
                for i, lanelet_id in enumerate(path_ids):
                    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                    center_vertices = np.array(lanelet.center_vertices)
                    if i == 0 and len(center_vertices) > 2:
                        center_vertices[1] = (center_vertices[2] + center_vertices[0]) / 2
                    if i == len(path_ids)-1 and len(center_vertices) > 2:
                        center_vertices[-2] = (center_vertices[-1] + center_vertices[-3]) / 2

                    center_vertices, left_vertices, right_vertices = OcctCRMap.extend_road(center_vertices,
                                                        lanelet.left_vertices,
                                                        lanelet.right_vertices, 
                                                        head_extend_len=self.extend_len if i==0 else 0, 
                                                        tail_extend_len=self.extend_len if i==len(path_ids)-1 else 0)
                    # center_vertices, _ = self._resample_path(center_vertices)
                    # left_vertices, _ = self._resample_path(left_vertices, M=len(center_vertices))
                    # right_vertices, _ = self._resample_path(right_vertices, M=len(center_vertices))
                    # center_vertices, left_vertices, right_vertices = center_vertices, lanelet.left_vertices, lanelet.right_vertices
                    # if lanelet.adj_left:
                    #     left_lane=scenario.lanelet_network.find_lanelet_by_id(lanelet.adj_left)
                    #     left_vertices = left_lane.left_vertices if lanelet.adj_left_same_direction else left_lane.right_vertices[::-1]
                    
                    # if lanelet.adj_right:
                    #     right_lane=scenario.lanelet_network.find_lanelet_by_id(lanelet.adj_right)
                    #     right_vertices = right_lane.right_vertices if lanelet.adj_right_same_direction else right_lane.left_vertices[::-1]
                    # center_vertices, left_vertices, right_vertices = \
                    #     self._enrich_vertices_sampling(center_vertices, left_vertices, right_vertices)
                    
                    slice_range = slice(None) if i == len(path_ids) - 1 else slice(-1)
                    for key, vertices in zip(
                        ["center_vertices", "left_vertices", "right_vertices"],
                        [center_vertices, left_vertices, right_vertices]
                    ):
                        path_data[key].extend([v.tolist() for v in vertices[slice_range]])
                # calculate correspond boundary pts
                center_vertices = np.array(path_data["center_vertices"])
                center_vertices = smooth_road_centerline(
                    center_vertices,
                    sample_step=1  # 与校准精度一致
                )
                center_vertices, _ = self._resample_path(center_vertices)
                left_vertices = np.array(path_data["left_vertices"])
                right_vertices = np.array(path_data["right_vertices"])
                left_vertices, right_vertices = boundary_calculator._calculate_boundary_pts(center_vertices, left_vertices, right_vertices)

                lane_width=([np.linalg.norm(left_vertices[i]-right_vertices[i]) for i in range(len(center_vertices))])
                center_vertices = torch.tensor(center_vertices, device=self.device, dtype=torch.float32)
                left_vertices = torch.tensor(left_vertices, device=self.device, dtype=torch.float32)
                right_vertices = torch.tensor(right_vertices, device=self.device, dtype=torch.float32)
                center_cum_len = self._get_cum_len(center_vertices)  # [N]
                if min(lane_width)<self.min_lane_width or center_cum_len[-1]<self.min_lane_len:
                    print(f"path:{path_ids} is too short or too narrow, continue")
                    continue
                if torch.linalg.norm(center_vertices[0]-center_vertices[-1])<self.start_end_distance_threshold:
                    print(f"path:{path_ids} start and end point is too close, continue")
                    continue
                #对中心路径进行重采样
                coeffs = natural_cubic_spline_coeffs(center_cum_len, left_vertices)
                left_splines = NaturalCubicSpline(coeffs)
                coeffs = natural_cubic_spline_coeffs(center_cum_len, right_vertices)
                right_splines = NaturalCubicSpline(coeffs)
                is_loop,_=self._detect_loop(center_vertices)
                if is_loop:
                    print(f"path:{path_ids} is a loop, continue")
                    continue
                resampled_lane_width=torch.linalg.norm(left_vertices-right_vertices, dim=-1)
                 # caculate ref vel according to curvature
                coeffs = natural_cubic_spline_coeffs(center_cum_len, center_vertices)
                center_splines = NaturalCubicSpline(coeffs)
                center_curvature = self.compute_curvature_2d(center_splines, center_cum_len, smooth_distance=10)
                factor=0.15 #factor=0.3 # for INTERACTION
                if self.is_constant_ref_v:
                    ref_v = self.max_ref_v * torch.ones_like(center_curvature)
                else:
                    ref_v = torch.clamp_max(factor * 1.0 / torch.sqrt(center_curvature+1e-8)**2, self.max_ref_v) 
                    ref_v = self.gaussian_smooth_1d(ref_v, sigma=8.0)
                assert len(center_vertices) == len(ref_v), "重采样后的中心路径长度与参考速度长度不一致"
                from occt_utils import calculate_max_min_acceleration
                max_acc, min_acc = calculate_max_min_acceleration(ref_v, center_cum_len)
                # 保存重采样后的路径数据
                hinge_status, hinge_trajs = self._detect_hinge_status(center_cum_len, center_splines, left_splines, right_splines)
                if (hinge_status==1).all():
                    print(f"path:{path_ids} has no corner, continue")
                    # means no corner, we dont want this path
                    continue

                # TODO: how the define hinge reward and status
                # we want to make hinge ready only pass the corner, but cant make sure all hinge has 0 status through the corner
                # make hinge status consistent through the corner
                corner_begin_s = center_cum_len[-1]
                corner_end_s = center_cum_len[0]
                for hinge_idx in range(hinge_status.shape[0]):
                    if (hinge_status[hinge_idx]==0).any():#exclude first and last hinge pts
                        begin_idx = torch.where(hinge_status[hinge_idx] == 0)[0][0]
                        pass_corner_idx = torch.where(hinge_status[hinge_idx] == 0)[0][-1]
                        hinge_status[hinge_idx, begin_idx:pass_corner_idx] = 0
                        corner_begin_s = min(center_cum_len[begin_idx],corner_begin_s)
                        corner_end_s = max(center_cum_len[pass_corner_idx],corner_end_s)
                # pass_corner_idx = torch.where(hinge_status == 0)[0][-1]
                # hinge_status[:pass_corner_idx] = 0
                self.path_library.append({
                    "map_name": map_name,
                    "path_ids": path_ids,
                    "center_vertices": center_vertices,
                    "left_vertices": left_vertices,
                    "right_vertices": right_vertices,
                    "s": center_cum_len,
                    "ref_v": ref_v,
                    "lane_width": resampled_lane_width,
                    "hinge_status": hinge_status,
                    "hinge_trajs": hinge_trajs,
                    "corner_begin_s": corner_begin_s,
                    "corner_end_s": corner_end_s,
                })
                assert hinge_status.shape[0]==self.n_agents, "hinge个数必须与n_agents一致"
                assert hinge_trajs.shape[0]==self.n_agents and hinge_trajs.shape[-1]==2, "hinge轨迹必须为2维"
                # 更新最大路径长度
                if center_cum_len[-1] > self.max_path_length:
                    self.max_path_length = center_cum_len[-1]
                    self.max_path_s_list = center_cum_len
        dump_file = os.path.join(self.cr_map_dir, "map_data.pkl")
        pickle.dump((self.scenario_library,self.path_library,\
                     self.max_path_length,self.max_path_s_list), open(dump_file, "wb"))
    def _cr_map_process_chapter4(self, map_dir: str, extend_left_boundary=False) -> None:
        """
        处理CommonRoad地图和车道信息
        
        Args:
            map_dir: 地图所在的文件夹路径
        """
        boundary_calculator = OcctBoundaryCalculator()
        dump_file = os.path.join(self.cr_map_dir, "map_data.pkl")
        if os.path.exists(dump_file):
            self.scenario_library, self.path_library,\
                self.max_path_length,self.max_path_s_list = pickle.load(open(dump_file, "rb"))
            self.max_path_length=self.max_path_length.to(self.device)
            self.max_path_s_list=self.max_path_s_list.to(self.device)
            return
        # 递归读取文件夹中所有XML文件
        map_files = glob.glob(os.path.join(map_dir, "**/*.xml"), recursive=True)
        
        assert self.rod_len is not None, "请先设置货物长度 L"
        # 打印地图库信息
        print(f"找到 {len(map_files)} 个地图文件:")
        for i, map_file in enumerate(map_files):
            print(f"  {i+1}. {os.path.basename(map_file)}")
        
        # 处理每个地图文件
        for map_file in map_files:
            map_name = os.path.basename(map_file)
            print(f"\n处理地图: {map_name}")
            
            # 读取地图场景
            scenario = get_cr_scenario(map_file)

            self.scenario_library[map_name] = scenario
            
            # 获取所有车道段
            lanelets = scenario.lanelet_network.lanelets
            
            # 找到所有起点车道段（predecessor为空）
            start_lanelets = [lanelet for lanelet in lanelets if not lanelet.predecessor]
            
            # 初始化路径ID库
            path_id_library = []
            
            # 使用BFS获取所有可能的路径
            for start_lanelet in start_lanelets:
                queue = deque()
                queue.append([start_lanelet.lanelet_id])
                
                while queue:
                    current_path = queue.popleft()
                    current_lanelet_id = current_path[-1]
                    
                    # 获取当前车道段对象
                    current_lanelet = scenario.lanelet_network.find_lanelet_by_id(current_lanelet_id)
                    
                    # 如果没有后继，说明是路径终点
                    if not current_lanelet.successor:
                        path_id_library.append(current_path)
                    else:
                        # 遍历所有后继
                        for successor_id in current_lanelet.successor:
                        # 检查后继车道段是否已经在当前路径中，如果是则跳过（避免回环）
                            if successor_id not in current_path:
                                new_path = current_path.copy()
                                new_path.append(successor_id)
                                queue.append(new_path)
            
            # 打印路径ID库信息
            print(f"找到 {len(path_id_library)} 条路径:")
            for i, path in enumerate(path_id_library):
                print(f"  路径 {i+1}: {path}")
            
            # 处理每条路径，生成路径数据
            for path_ids in tqdm(path_id_library, desc="处理路径"):
                path_data = {
                    "center_vertices": [],
                    "left_vertices": [],
                    "right_vertices": []
                }
                
                # for chapter 4
                if not ((path_ids[0]==127 and path_ids[-1]==102) or \
                   (path_ids[0]==100 and path_ids[-1]==129)):
                    continue
                for i, lanelet_id in enumerate(path_ids):
                    lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
                    center_vertices = np.array(lanelet.center_vertices)
                    if i == 0 and len(center_vertices) > 2:
                        center_vertices[1] = (center_vertices[2] + center_vertices[0]) / 2
                    if i == len(path_ids)-1 and len(center_vertices) > 2:
                        center_vertices[-2] = (center_vertices[-1] + center_vertices[-3]) / 2

                    center_vertices, left_vertices, right_vertices = OcctCRMap.extend_road(center_vertices,
                                                        lanelet.left_vertices,
                                                        lanelet.right_vertices, 
                                                        head_extend_len=self.extend_len if i==0 else 0, 
                                                        tail_extend_len=self.extend_len if i==len(path_ids)-1 else 0)
                    slice_range = slice(None) if i == len(path_ids) - 1 else slice(-1)
                    for key, vertices in zip(
                        ["center_vertices", "left_vertices", "right_vertices"],
                        [center_vertices, left_vertices, right_vertices]
                    ):
                        path_data[key].extend([v.tolist() for v in vertices[slice_range]])
                raw_left_vertices = np.array(path_data["left_vertices"])
                raw_right_vertices = np.array(path_data["right_vertices"])
                # calculate correspond boundary pts
                center_vertices = np.array(path_data["center_vertices"])
                center_vertices = smooth_road_centerline(
                    center_vertices,
                    sample_step=1  # 与校准精度一致
                )
                center_vertices, _ = self._resample_path(center_vertices)

                right_vertices = np.array(path_data["right_vertices"])
                # revise road boundary for [127, 128, 131, 133, 136, 137, 142, 145, 147, 151, 106, 102]
                if path_ids[0]==127 and extend_left_boundary:
                    left_vertices = torch.tensor(self.path_library[0]["raw_right_vertices"]).to(right_vertices.device)
                    left_vertices = torch.flip(left_vertices, dims=[0])

                    right_vertices, _ = self._resample_path(right_vertices, len(center_vertices))
                    left_vertices, _ = self._resample_path(left_vertices, len(right_vertices))
                    left_vertices = left_vertices.cpu().numpy()
                else:
                    left_vertices = np.array(path_data["left_vertices"])

                left_vertices, right_vertices = boundary_calculator._calculate_boundary_pts(center_vertices, left_vertices, right_vertices)

                lane_width=([np.linalg.norm(left_vertices[i]-right_vertices[i]) for i in range(len(center_vertices))])
                center_vertices = torch.tensor(center_vertices, device=self.device, dtype=torch.float32)
                left_vertices = torch.tensor(left_vertices, device=self.device, dtype=torch.float32)
                right_vertices = torch.tensor(right_vertices, device=self.device, dtype=torch.float32)
                center_cum_len = self._get_cum_len(center_vertices)  # [N]
                if min(lane_width)<self.min_lane_width or center_cum_len[-1]<self.min_lane_len:
                    print(f"path:{path_ids} is too short or too narrow, continue")
                    continue
                if torch.linalg.norm(center_vertices[0]-center_vertices[-1])<self.start_end_distance_threshold:
                    print(f"path:{path_ids} start and end point is too close, continue")
                    continue
                #对中心路径进行重采样
                coeffs = natural_cubic_spline_coeffs(center_cum_len, left_vertices)
                left_splines = NaturalCubicSpline(coeffs)
                coeffs = natural_cubic_spline_coeffs(center_cum_len, right_vertices)
                right_splines = NaturalCubicSpline(coeffs)
                is_loop,_=self._detect_loop(center_vertices)
                if is_loop:
                    print(f"path:{path_ids} is a loop, continue")
                    continue
                resampled_lane_width=torch.linalg.norm(left_vertices-right_vertices, dim=-1)
                 # caculate ref vel according to curvature
                coeffs = natural_cubic_spline_coeffs(center_cum_len, center_vertices)
                center_splines = NaturalCubicSpline(coeffs)
                center_curvature = self.compute_curvature_2d(center_splines, center_cum_len, smooth_distance=10)
                factor=0.15 #factor=0.3 # for INTERACTION
                if self.is_constant_ref_v:
                    ref_v = self.max_ref_v * torch.ones_like(center_curvature)
                else:
                    ref_v = torch.clamp_max(factor * 1.0 / torch.sqrt(center_curvature+1e-8)**2, self.max_ref_v) 
                    ref_v = self.gaussian_smooth_1d(ref_v, sigma=8.0)
                assert len(center_vertices) == len(ref_v), "重采样后的中心路径长度与参考速度长度不一致"
                # 保存重采样后的路径数据
                hinge_status, hinge_trajs = self._detect_hinge_status(center_cum_len, center_splines, left_splines, right_splines)
                if (hinge_status==1).all():
                    print(f"path:{path_ids} has no corner, continue")
                    # means no corner, we dont want this path
                    continue

                # TODO: how the define hinge reward and status
                # we want to make hinge ready only pass the corner, but cant make sure all hinge has 0 status through the corner
                # make hinge status consistent through the corner
                corner_begin_s = center_cum_len[-1]
                corner_end_s = center_cum_len[0]
                for hinge_idx in range(hinge_status.shape[0]):
                    if (hinge_status[hinge_idx]==0).any():#exclude first and last hinge pts
                        begin_idx = torch.where(hinge_status[hinge_idx] == 0)[0][0]
                        pass_corner_idx = torch.where(hinge_status[hinge_idx] == 0)[0][-1]
                        hinge_status[hinge_idx, begin_idx:pass_corner_idx] = 0
                        corner_begin_s = min(center_cum_len[begin_idx],corner_begin_s)
                        corner_end_s = max(center_cum_len[pass_corner_idx],corner_end_s)
                # pass_corner_idx = torch.where(hinge_status == 0)[0][-1]
                # hinge_status[:pass_corner_idx] = 0
                self.path_library.append({
                    "map_name": map_name,
                    "path_ids": path_ids,
                    "center_vertices": center_vertices,
                    "left_vertices": left_vertices,
                    "raw_left_vertices": raw_left_vertices,
                    "right_vertices": right_vertices,
                    "raw_right_vertices": raw_right_vertices,
                    "s": center_cum_len,
                    "ref_v": ref_v,
                    "lane_width": resampled_lane_width,
                    "hinge_status": hinge_status,
                    "hinge_trajs": hinge_trajs,
                    "corner_begin_s": corner_begin_s,
                    "corner_end_s": corner_end_s,
                })
                assert hinge_status.shape[0]==self.n_agents, "hinge个数必须与n_agents一致"
                assert hinge_trajs.shape[0]==self.n_agents and hinge_trajs.shape[-1]==2, "hinge轨迹必须为2维"
                # 更新最大路径长度
                if path_ids[0]==127:
                    self.max_path_length = center_cum_len[-1]
                    self.max_path_s_list = center_cum_len
        dump_file = os.path.join(self.cr_map_dir, "map_data.pkl")
        self.path_library.remove(self.path_library[0])
        pickle.dump((self.scenario_library,self.path_library,\
                     self.max_path_length,self.max_path_s_list), open(dump_file, "wb"))
    def _detect_hinge_status(self, s, center_splines: NaturalCubicSpline, left_splines: NaturalCubicSpline, right_splines: NaturalCubicSpline):
        hinge_status = torch.zeros((self.n_agents, s.shape[0]), device=self.device, dtype=torch.float32)
        hinge_trajs = torch.zeros((self.n_agents, s.shape[0], 2), device=self.device, dtype=torch.float32)
        for s_idx, s_i in enumerate(s):
            pts_first_hinge = center_splines.evaluate(s_i)
            delta_s = self.solve_delta_s_expand_single(s_i,self.rod_len,center_splines)
            s_last_hinge = s_i - delta_s
            pts_last_hinge = center_splines.evaluate(s_last_hinge)
            rod_vector = pts_last_hinge - pts_first_hinge
            for agent_idx in range(self.n_agents):
                hinge_pts = pts_first_hinge + agent_idx/(self.n_agents-1)*rod_vector
                hinge_trajs[agent_idx, s_idx, :] = hinge_pts
                correspond_s = s_i + agent_idx/(self.n_agents-1)*(s_last_hinge - s_i)
                correspond_pts = center_splines.evaluate(correspond_s)
                correspond_width = torch.linalg.norm(left_splines.evaluate(correspond_s)-right_splines.evaluate(correspond_s), dim=-1)
                # 1 means ready to hinge, 0 means not ready
                hinge_status[agent_idx, s_idx] = 1 if torch.linalg.norm(hinge_pts-correspond_pts)<max(correspond_width/2-1, 0.75) else 0
        return hinge_status, hinge_trajs
    def gaussian_smooth_1d(self, x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
        """
        对1D/批量张量做高斯平滑（保持形状不变，适配任意批量维度）
        """
        # 展平为2D（批量维度 + 序列维度），方便处理
        orig_shape = x.shape
        if x.dim() == 0:
            return x
        # 仅对最后一维做平滑
        seq_dim = x.shape[-1]
        batch_dims = x.shape[:-1]
        x_flat = x.reshape(-1, seq_dim)  # (B, L)
        
        # 生成高斯核
        kernel_size = int(2 * round(3 * sigma) + 1)  # 高斯核大小（3σ原则）
        if kernel_size > seq_dim:
            kernel_size = seq_dim if seq_dim % 2 == 1 else seq_dim - 1  # 保证奇数核
        if kernel_size < 3:
            return x  # 序列过短，无需平滑
        
        # 构建1D高斯核
        kernel = torch.arange(kernel_size, device=x.device, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel = torch.exp(-kernel ** 2 / (2 * sigma ** 2))
        kernel = kernel / torch.sum(kernel)  # 归一化
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1, 1, kernel_size)
        
        # 填充边界（避免边缘值失真）
        x_padded = F.pad(x_flat.unsqueeze(1), pad=[kernel_size//2, kernel_size//2], mode='replicate')
        
        # 卷积平滑
        smooth_x = F.conv1d(x_padded, kernel, stride=1).squeeze(1)  # (B, L)
        
        # 恢复原始形状
        smooth_x = smooth_x.reshape(orig_shape)
        return smooth_x

    def get_scenario_by_env_index(self, env_index):
        """
        获取指定环境索引的场景
        """
        map_name = self.batch_map_name[env_index]
        return self.scenario_library[map_name]
    def reset_splines(self):
        """
        生成batch_dim长度的随机整型tensor，范围0到道路库数量-1
        使用torchcubicspline初始化路径样条
        """
        start_time=time.time()
        self.batch_id = torch.randint(0, len(self.path_library), (self.batch_dim,), device=self.device)
        # 260128 revise
        for i in range(self.batch_dim):
            self.batch_id[i] = torch.tensor(i % len(self.path_library), dtype=torch.int64, device=self.device)
        #self.batch_id[0] = 3
        # 准备batch数据
        B = self.batch_dim
        max_path_pts_num = len(self.max_path_s_list)
        
        # 初始化batch数据
        self.batch_s = torch.empty(B, max_path_pts_num, device=self.device).fill_(float('nan'))
        self.batch_center_vertices = torch.empty(B, max_path_pts_num, 2, device=self.device).fill_(float('nan'))
        self.batch_left_vertices = torch.empty(B, max_path_pts_num, 2, device=self.device).fill_(float('nan'))
        self.batch_right_vertices = torch.empty(B, max_path_pts_num, 2, device=self.device).fill_(float('nan'))
        self.batch_ref_v = torch.empty(B, max_path_pts_num, 1, device=self.device).fill_(float('nan'))
        #self.batch_hinge_status = torch.ones(B, max_path_pts_num, self.n_agents, device=self.device)
        self.batch_s_max = torch.empty(B, device=self.device).fill_(float('nan'))
        self.batch_map_name = [None]*B
        self.batch_corner_s = torch.zeros(B, device=self.device)
        max_s_len=0
        max_s_len_id=0
        for batch_idx, path_id in enumerate(self.batch_id):
            path_id = path_id.item()
            self.batch_map_name[batch_idx] = self.path_library[path_id]["map_name"]
            path_data = self.path_library[path_id]
            s = path_data["s"]
            s_max = s[-1]
            length = len(s)
            if length > max_s_len:
                max_s_len = length
                max_s_len_id = path_id
            self.batch_s[batch_idx, :length] = s
            self.batch_center_vertices[batch_idx, :length] = path_data["center_vertices"]
            self.batch_left_vertices[batch_idx, :length] = path_data["left_vertices"]
            self.batch_right_vertices[batch_idx, :length] = path_data["right_vertices"]
            self.batch_s_max[batch_idx] = s_max
            self.batch_ref_v[batch_idx, :length, 0] = path_data["ref_v"]
            #self.batch_hinge_status[batch_idx, :length, :] = path_data["hinge_status"].transpose(0, 1) #[length, n_agents]
            self.batch_corner_s[batch_idx] = (path_data["corner_begin_s"] + path_data["corner_end_s"]) / 2
        # 对长度不足的道路样本进行延伸填充
        for batch_idx in range(B):
            current_length = torch.count_nonzero(~torch.isnan(self.batch_center_vertices[batch_idx, :, 0])).item()
            if current_length < max_path_pts_num:
                extend_points = max_path_pts_num - current_length
                for vertex_type in ["center", "left", "right"]:
                    vertices = getattr(self, f"batch_{vertex_type}_vertices")
                    last_point = vertices[batch_idx, current_length - 1]
                    second_last_point = vertices[batch_idx, current_length - 2]
                    direction = last_point - second_last_point
                    for i in range(1, extend_points + 1):
                        new_point = last_point + direction * i
                        vertices[batch_idx, current_length - 1 + i] = new_point
                last_ref_v = self.batch_ref_v[batch_idx, current_length - 1]
                self.batch_ref_v[batch_idx, current_length:] = last_ref_v


        
        longest_s = self.max_path_s_list
        coeffs = natural_cubic_spline_coeffs(longest_s, self.batch_center_vertices)
        self.center_splines = NaturalCubicSpline(coeffs)
        coeffs = natural_cubic_spline_coeffs(longest_s, self.batch_left_vertices)
        self.left_splines = NaturalCubicSpline(coeffs)
        coeffs = natural_cubic_spline_coeffs(longest_s, self.batch_right_vertices)
        self.right_splines = NaturalCubicSpline(coeffs)
        coeffs = natural_cubic_spline_coeffs(longest_s, self.batch_ref_v)
        self.ref_v_splines = NaturalCubicSpline(coeffs)
        # coeffs = natural_cubic_spline_coeffs(longest_s, self.batch_hinge_status)
        # self.hinge_status_splines = NaturalCubicSpline(coeffs)
        end_time=time.time()
        #print(f"结束reset_splines, cost time: {end_time-start_time}")

    def compute_curvature_2d(
        self, 
        spline: NaturalCubicSpline, 
        t: torch.Tensor, 
        smooth_distance: float = 5.0  # 曲率平滑范围（米），参数化控制
    ) -> torch.Tensor:
        """
        计算 2D 三次样条曲线在参数 t 处的曲率（平滑版：当前点曲率=指定距离范围内点的曲率均值）。
        
        参数:
            spline: NaturalCubicSpline 实例（需保证spline能通过参数t输出物理坐标）
            t: 要计算曲率的参数值（可以是批量输入），shape: (...)
            smooth_distance: 曲率平滑的物理距离范围（米），即当前点前后smooth_distance米内的均值
                            建议取值：3~10米（值越大越平滑，值越小越接近原始曲率）
        
        返回:
            curvature: t 处的平滑曲率（与 t 同形状）
        """
        # 边界保护：平滑距离不能为负
        smooth_distance = max(0.1, smooth_distance)
        
        # -------------------------- 步骤1：预计算样条的参数-距离映射（密集采样） --------------------------
        # 1.1 生成覆盖所有输入t的密集参数采样点（保证距离计算精度）
        t_min = torch.min(t) - 0.1  # 扩展边界，避免边缘点范围越界
        t_max = torch.max(t) + 0.1
        # 采样密度：每米至少10个点，保证smooth_distance范围内有足够采样点
        sample_density = 10  # 每米采样点数
        total_t_range = t_max - t_min
        total_length_estimate = total_t_range * torch.norm(spline.evaluate(t_max) - spline.evaluate(t_min)) / total_t_range
        steps = max(500, int(total_length_estimate * sample_density))  # 最少500个点
        t_samples = torch.linspace(t_min, t_max, steps=steps, device=t.device)
        
        # 1.2 计算采样点的物理坐标（米）
        pos_samples = spline.evaluate(t_samples)  # shape: (steps, 2)
        
        # 1.3 计算采样点的累计物理距离（从第一个采样点开始，单位：米）
        pos_diff = pos_samples[1:] - pos_samples[:-1]  # 相邻点坐标差
        seg_lengths = torch.norm(pos_diff, dim=-1)  # 相邻点的物理距离（米）
        cum_lengths = torch.cat([
            torch.tensor([0.0], device=t.device),
            torch.cumsum(seg_lengths, dim=0)
        ])  # 累计距离，shape: (steps,)
        
        # 1.4 预计算所有采样点的原始曲率
        dr_dt_samples = spline.derivative(t_samples, order=1)  # (steps, 2)
        d2r_dt2_samples = spline.derivative(t_samples, order=2)  # (steps, 2)
        
        # 原始曲率计算（与原逻辑一致）
        dr_norm_sq_samples = torch.sum(dr_dt_samples ** 2, dim=-1)
        dr_norm_samples = torch.sqrt(dr_norm_sq_samples)
        dr_norm_cube_samples = dr_norm_samples ** 3
        numerator_samples = torch.abs(
            dr_dt_samples[:, 0] * d2r_dt2_samples[:, 1] - dr_dt_samples[:, 1] * d2r_dt2_samples[:, 0]
        )
        curvature_samples = numerator_samples / (dr_norm_cube_samples + 1e-8)  # (steps,)
        
        # -------------------------- 步骤2：批量计算输入t对应的物理距离 --------------------------
        # 展平批量维度，方便批量处理
        t_flat = t.flatten()  # shape: (N,)
        pos_t = spline.evaluate(t_flat)  # shape: (N, 2)
        
        # 批量插值：计算每个t对应的累计物理距离（米）
        # 步骤2.1：找到每个t_flat在t_samples中的最近索引
        t_samples_expand = t_samples.unsqueeze(0)  # (1, steps)
        t_flat_expand = t_flat.unsqueeze(1)        # (N, 1)
        idx_nearest = torch.argmin(torch.abs(t_flat_expand - t_samples_expand), dim=1)  # (N,)
        
        # 步骤2.2：计算当前点与最近采样点的物理距离差
        pos_nearest = pos_samples[idx_nearest]  # (N, 2)
        pos_diff = pos_t - pos_nearest          # (N, 2)
        dist_diff = torch.norm(pos_diff, dim=-1)  # (N,)
        
        # 步骤2.3：计算每个t对应的累计物理距离
        cum_lengths_nearest = cum_lengths[idx_nearest]  # (N,)
        length_t = cum_lengths_nearest + dist_diff      # (N,) 每个t对应的累计距离（米）
        
        # -------------------------- 步骤3：批量计算指定距离范围内的曲率均值 --------------------------
        # 3.1 构建距离范围掩码（N, steps）：每个t对应的[length_t-smooth_distance, length_t+smooth_distance]
        length_t_expand = length_t.unsqueeze(1)  # (N, 1)
        cum_lengths_expand = cum_lengths.unsqueeze(0)  # (1, steps)
        mask = (cum_lengths_expand >= (length_t_expand - smooth_distance)) & \
            (cum_lengths_expand <= (length_t_expand + smooth_distance))  # (N, steps)
        
        # 3.2 计算每个t范围内的曲率均值（处理空值情况）
        # 给mask加极小值，避免除以0
        mask_float = mask.float()
        sum_weights = torch.sum(mask_float, dim=1)  # (N,) 每个t的有效点数
        curvature_samples_expand = curvature_samples.unsqueeze(0)  # (1, steps)
        
        # 计算加权和（仅有效点参与）
        curvature_sum = torch.sum(mask_float * curvature_samples_expand, dim=1)  # (N,)
        
        # 均值计算：有效点>0则取均值，否则用最近采样点的曲率
        curvature_nearest = curvature_samples[idx_nearest]  # (N,) 兜底曲率
        smooth_curvature_flat = torch.where(
            sum_weights > 0,
            curvature_sum / sum_weights,
            curvature_nearest
        )
        
        # 3.3 恢复原始形状
        smooth_curvature = smooth_curvature_flat.reshape(t.shape)
        
        return smooth_curvature
    
    def get_pts(self, s: Tensor, env_j: int = None, line: str = "center") -> Tensor:
        if line == "center":
            p = self.center_splines.evaluate(s)
        elif line == "left":
            p = self.left_splines.evaluate(s)
        elif line == "right":
            p = self.right_splines.evaluate(s)
        else:
            raise ValueError(f"未知的line参数: {line}")
        if s.dim()==0 or s.dim()==1:
            if type(env_j) == int:
                return p[env_j]
            return p
        if s.dim()==2:
            # get pts for batch
            assert self.batch_dim == s.shape[0], "s的批量维度必须与样条批量维度一致"
            p = p[torch.arange(s.shape[0]), torch.arange(s.shape[0])]
            return p

    def get_ref_v(self, s: Tensor, env_j: int = None) -> Tensor:
        ref_v = self.ref_v_splines.evaluate(s)
        if s.dim()==0 or s.dim()==1:
            if type(env_j) == int:
                return ref_v[env_j]
            return ref_v
        if s.dim()==2:
            # get pts for batch
            assert self.batch_dim == s.shape[0], "s的批量维度必须与样条批量维度一致"
            ref_v = ref_v[torch.arange(s.shape[0]), torch.arange(s.shape[0])]
            return ref_v
        
    # def get_hinge_status(self, s: Tensor, env_j: int = None) -> Tensor:
    #     hinge_status = self.hinge_status_splines.evaluate(s)
    #     if s.dim()==0 or s.dim()==1:
    #         if type(env_j) == int:
    #             return hinge_status[env_j]
    #         return hinge_status
    #     if s.dim()==2:
    #         # get pts for batch
    #         assert self.batch_dim == s.shape[0], "s的批量维度必须与样条批量维度一致"
    #         hinge_status = hinge_status[torch.arange(s.shape[0]), torch.arange(s.shape[0])]
    #         return hinge_status
    
    def get_tangent_vector(self, s: Tensor) -> Tensor:
        assert self.center_splines._a.shape[0] == s.shape[0], "s的批量维度必须与样条批量维度一致"
        tangent_vec = self.center_splines.derivative(s)
        tangent_vec = tangent_vec[torch.arange(s.shape[0]), torch.arange(s.shape[0])]
        tangent_vec = tangent_vec / (torch.linalg.norm(tangent_vec, dim=-1, keepdim=True) + 1e-8)
        return tangent_vec
    
    def get_s_max(self) -> Tensor:
        return self.batch_s_max
    def get_s_max_idx(self, env_j: int = None) -> Tensor:
        s_max_mask = ~torch.isnan(self.batch_s)
        s_max_idx = self.batch_s.shape[1] - 1 - torch.argmax(torch.flip(s_max_mask.int(), dims=[1]), dim=1)
        if env_j is not None:
            s_max_idx = s_max_idx[env_j]
        return s_max_idx
    def get_road_center_pts(self) -> Tensor:
        return self.batch_center_vertices
    
    def get_road_left_pts(self) -> Tensor:
        return self.batch_left_vertices
    
    def get_road_right_pts(self) -> Tensor:
        return self.batch_right_vertices 
    def solve_delta_s(self, s_reference: Tensor, L: Tensor, backward: bool = True, max_iter: int = 50, tol: float = 1e-3) -> Tuple[Tensor, Tensor]:
        """
        固定弦长二分法求弧长。
        input:
            s_reference: [B] 参考点弧长（backward=True 时为前端点，backward=False 时为后端点）
            L:           [B] 固定弦长（货物长度）
            backward:    bool 搜索方向，True 为向后搜索，False 为向前搜索
            max_iter:      int 最大迭代次数
            tol:           float 收敛阈值
            splines:       Splines 样条对象，用于计算点坐标(仅在_cr_map_process中使用)
        return:
            delta_s:        [B] 解出的 Δs (>=0)
            infeasible:     [B] 是否无解（弦长过短）
        """
        # 根据搜索方向设置参考点和计算最大可能弦长
        if backward:
            # 向后搜索：参考点是前端点，搜索空间是从0到s_reference
            p_fixed = self.get_pts(s_reference[:, None]).squeeze(1)
            s_boundary = torch.zeros_like(s_reference)
            p_boundary = self.get_pts(s_boundary[:, None]).squeeze(1)
            chord_max = torch.linalg.norm(p_fixed - p_boundary, dim=-1)
            hi = s_reference
        else:
            # 向前搜索：参考点是后端点，搜索空间是从s_reference到self.get_s_max()
            p_fixed = self.get_pts(s_reference[:, None]).squeeze(1)
            s_max = self.get_s_max()
            s_boundary = s_max
            p_boundary = self.get_pts(s_boundary[:, None]).squeeze(1)
            chord_max = torch.linalg.norm(p_boundary - p_fixed, dim=-1)
            hi = s_max - s_reference
        
        
        lo = torch.zeros_like(s_reference)
        # 二分搜索循环
        for _ in range(max_iter):
            interval_length = hi - lo
            
            if torch.all(interval_length < tol):
                break
            
            mid = 0.5 * (lo + hi)  # [B]
            
            # 根据搜索方向计算另一个点的位置
            if backward:
                s_other = s_reference - mid
            else:
                s_other = s_reference + mid
            
            p_other = self.get_pts(s_other[:, None]).squeeze(1)  # [B, 2]
            chord = torch.linalg.norm(p_fixed - p_other, dim=-1)  # [B]
            
            go_left = chord > L  # [B]
            hi = torch.where(go_left, mid, hi)
            lo = torch.where(go_left, lo, mid)
        
        delta_s = 0.5 * (lo + hi)
        infeasible = interval_length >= tol
        delta_s = torch.where(infeasible, torch.zeros_like(delta_s), delta_s)
        return delta_s, infeasible
    def solve_delta_s_expand_single(self, s_reference: Tensor, L: Tensor, splines: NaturalCubicSpline, backward: bool = True, max_iter: int = 50, tol: float = 1e-3) -> Tuple[Tensor, Tensor]:
        """
        固定弦长二分法求弧长。
        input:
            s_reference: [B] 参考点弧长（backward=True 时为前端点，backward=False 时为后端点）
            L:           [B] 固定弦长（货物长度）
            backward:    bool 搜索方向，True 为向后搜索，False 为向前搜索
            max_iter:      int 最大迭代次数
            tol:           float 收敛阈值
            splines:       Splines 样条对象，用于计算点坐标(仅在_cr_map_process中使用)
        return:
            delta_s:        [B] 解出的 Δs (>=0)
            infeasible:     [B] 是否无解（弦长过短）
        """
        p_fixed = splines.evaluate(s_reference)
        lo = torch.zeros_like(s_reference)
        hi = s_reference + 1.5*L if backward else self.get_s_max() + 1.5*L - s_reference
        
        for _ in range(max_iter):
            interval_length = hi - lo
            if torch.all(interval_length < tol):
                break
            mid = 0.5 * (lo + hi)
            s_other = s_reference - mid if backward else s_reference + mid
            p_other = splines.evaluate(s_other)
            chord = torch.linalg.norm(p_fixed - p_other, dim=-1)
            go_left = chord > L
            hi = torch.where(go_left, mid, hi)
            lo = torch.where(go_left, lo, mid)
        
        delta_s = 0.5 * (lo + hi)
        return delta_s
    
    def plot_road_debug(self):
        from commonroad.visualization.mp_renderer import MPRenderer, DynamicObstacleParams
        from matplotlib.collections import LineCollection
        from matplotlib.colors import Normalize
        import matplotlib.pyplot as plt
        font_size = 16
        # 创建图形和MPRenderer
        fig1, ax1 = plt.subplots(figsize=(8, 7))  # 增加宽度以容纳颜色条
        rnd = MPRenderer(ax=ax1)
        fig2, ax2 = plt.subplots(figsize=(8, 7))
        fig3, ax3 = plt.subplots(figsize=(8, 7))
        # 设置渲染参数
        rnd.draw_params.dynamic_obstacle.draw_icon = True
        rnd.draw_params.dynamic_obstacle.draw_bounding_box = True
        rnd.draw_params.dynamic_obstacle.show_label = False
        rnd.draw_params.dynamic_obstacle.state.draw_arrow=False
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "white"
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#0000CD"  # darkblue
        ego_params = DynamicObstacleParams()
        ego_params.draw_icon = True
        ego_params.vehicle_shape.occupancy.shape.facecolor = "white"
        ego_params.vehicle_shape.occupancy.shape.edgecolor = "#006400"  # darkgreen
        ego_params.vehicle_shape.occupancy.shape.zorder = 20
        
        # 遍历所有路径
        for path_id, path_data in enumerate(self.path_library):
            map_name = path_data["map_name"]
            path_ids = path_data["path_ids"]
            center_vertices = path_data["center_vertices"].detach().cpu().numpy()
            left_vertices = path_data["left_vertices"].detach().cpu().numpy()
            right_vertices = path_data["right_vertices"].detach().cpu().numpy()
            ref_v = path_data["ref_v"].detach().cpu().numpy()
            s = path_data["s"].detach().cpu().numpy()
            lane_width = path_data["lane_width"].detach().cpu().numpy()
            hinge_trajs = path_data["hinge_trajs"].detach().cpu().numpy()  # [n_agents, length, 2]
            hinge_status = path_data["hinge_status"].detach().cpu().numpy()  # [n_agents, length]
            # 获取场景并绘制
            scenario = self.scenario_library[map_name]
            for i in range(0, 1):
                plt.figure(fig1)
                rnd.draw_params.time_begin = i
                rnd.draw_params.time_end = i
                ego_params.time_begin = i
                ego_params.time_end = i
                scenario.draw(rnd)
                rnd.render(show=True)
                linewidth = 0.3
                # 绘制左右边界（保持原有颜色）
                rnd.ax.scatter(left_vertices[:, 0], left_vertices[:, 1], s = 0.5, c = 'r', label='Left Boundary', zorder=20, linewidth=linewidth)
                rnd.ax.scatter(right_vertices[:, 0], right_vertices[:, 1], s = 0.5, c = 'b', label='Right Boundary', zorder=20, linewidth=linewidth)
                rnd.ax.scatter(center_vertices[:, 0], center_vertices[:, 1], s = 0.5, c = 'gray', label='Center', zorder=20, linewidth=linewidth, alpha=0.5)
                
                # 绘制带颜色映射的中心线
                # 准备线段集合
                points = center_vertices.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)
                
                # 创建颜色映射（彩虹色系）
                norm = Normalize(vmin=ref_v.min(), vmax=ref_v.max())
                lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=linewidth, linestyle='--', zorder=19)
                
                # 根据ref_v为线段分配颜色
                lc.set_array(ref_v)
                
                # 添加到轴上
                line = rnd.ax.add_collection(lc)
                
                # 添加颜色条
                cbar = fig1.colorbar(line, ax=ax1, shrink=0.7, pad=0.05)
                cbar.set_label('Reference Velocity (m/s)', fontsize=font_size)
                cbar.ax.tick_params(labelsize=14)
                
                # 设置坐标轴
                rnd.ax.set_xlabel("x/m", fontsize=font_size)
                rnd.ax.set_ylabel("y/m", fontsize=font_size)
                rnd.ax.tick_params(axis='both', direction='in', labelsize=16, top=False, right=False)
                rnd.ax.set_title(f"{map_name} Path {path_id + 1}, Width=({min(lane_width):.2f},{np.mean(lane_width):.2f},{max(lane_width):.2f})m, Len={s[-1]:.2f}m\npath_ids={path_ids}",
                                  fontsize=font_size, zorder=20)
                # 暂停以显示
                plt.pause(0.01)
                
                # 保存图像
                fig1_path = f"{self.vis_dir}/Path{path_id:04d}_map_{i:04d}.svg"
                fig1.savefig(fig1_path, dpi=300, format="svg", bbox_inches='tight')
                
                # 清除颜色条和中心线，准备下一个路径
                cbar.remove()
                line.remove()
                plt.figure(fig2)
                ax2.clear()
                ax2.plot(s, ref_v, 'b', label='Reference Velocity', zorder=20)
                ax2.set_xlabel("s/m", fontsize=20)
                ax2.set_ylabel("v/m/s", fontsize=20)
                ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=False, right=False)
                ax2.set_title(f"Path {path_id + 1}: Reference Velocity", fontsize=font_size)
                ax2.legend(fontsize=14)
                plt.pause(0.01)
                fig2_path = f"{self.vis_dir}/Path{path_id:04d}_ref_v_{i:04d}.svg"
                fig2.savefig(fig2_path, dpi=300, format="svg", bbox_inches='tight')

                # 绘制 fig3: hinge 轨迹可视化
                plt.figure(fig3)
                ax3.clear()

                # 绘制左右边界
                ax3.scatter(left_vertices[:, 0], left_vertices[:, 1],
                           s=0.5, c='r', label='Left Boundary', zorder=20, linewidth=0.3)
                ax3.scatter(right_vertices[:, 0], right_vertices[:, 1],
                           s=0.5, c='b', label='Right Boundary', zorder=20, linewidth=0.3)

                # 绘制 hinge 轨迹
                n_agents = hinge_trajs.shape[0]
                cmap = plt.get_cmap('tab10')  # 为每个agent分配颜色

                for agent_idx in range(n_agents):
                    agent_color = cmap(agent_idx % 10)

                    # 绘制完整轨迹线
                    traj_x = hinge_trajs[agent_idx, :, 0]
                    traj_y = hinge_trajs[agent_idx, :, 1]
                    ax3.plot(traj_x, traj_y,
                            color=agent_color,
                            linewidth=0.5,
                            alpha=0.7,
                            linestyle='-',
                            zorder=18,
                            label=f'Hinge {agent_idx}')

                    # 标注 hinge_status == 0 的异常点
                    out_of_road_mask = hinge_status[agent_idx, :] == 0
                    if np.any(out_of_road_mask):
                        out_x = traj_x[out_of_road_mask]
                        out_y = traj_y[out_of_road_mask]
                        ax3.scatter(out_x, out_y,
                                   marker='x',
                                   s=2,
                                   c=agent_color,
                                   linewidths=0.5,
                                   zorder=22)

                # 设置坐标轴和标题
                ax3.set_xlabel("x/m", fontsize=font_size)
                ax3.set_ylabel("y/m", fontsize=font_size)
                ax3.tick_params(axis='both', direction='in', labelsize=16, top=False, right=False)
                ax3.set_title(f"Path {path_id + 1}: Hinge Trajectories\npath_ids={path_ids}",
                             fontsize=font_size)
                ax3.axis('equal')
                ax3.legend(fontsize=10, framealpha=0.8)
                plt.pause(0.01)
                fig3_path = f"{self.vis_dir}/Path{path_id:04d}_hinge_{i:04d}.svg"
                fig3.savefig(fig3_path, dpi=300, format="svg", bbox_inches='tight')
    def plot_road_for_paper(self,vis_dir):
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from commonroad.visualization.mp_renderer import MPRenderer, DynamicObstacleParams
        from scipy.interpolate import UnivariateSpline  # 引入UnivariateSpline
        
        def calc_traj_curvature(traj_xy, delta=3):
            """
            修改版：使用平滑样条拟合代替硬插值，消除直线段的曲率噪点
            """
            delta_xy = np.diff(traj_xy, axis=0)
            segment_lengths = np.linalg.norm(delta_xy, axis=1)
            s_arr = np.concatenate([[0.0], np.cumsum(segment_lengths)])
            
            x = traj_xy[:, 0]
            y = traj_xy[:, 1]

            # -------------------------- 核心修改开始 --------------------------
            # s=5: 经验平滑因子
            smooth_factor = 5  
            # k=3: 三次样条
            sx = UnivariateSpline(s_arr, x, k=3, s=smooth_factor)
            sy = UnivariateSpline(s_arr, y, k=3, s=smooth_factor)
            # -------------------------- 核心修改结束 --------------------------

            s_max = s_arr[-1]
            s_eval_arr = np.linspace(0, s_max, int(s_max))
            curv_arr = np.zeros_like(s_eval_arr)

            for idx, s in enumerate(s_eval_arr):
                if s <= delta:
                    s0, s1, s2 = s, s + delta, s + 2 * delta
                elif s >= s_max - delta:
                    s0, s1, s2 = s - 2 * delta, s - delta, s
                else:
                    s0, s1, s2 = s - delta, s, s + delta

                x0, y0 = sx(s0), sy(s0)
                x1, y1 = sx(s1), sy(s1)
                x2, y2 = sx(s2), sy(s2)

                if s <= delta:
                    dx = (x2 - x1) / delta
                    dy = (y2 - y1) / delta
                elif s >= s_max - delta:
                    dx = (x1 - x0) / delta
                    dy = (y1 - y0) / delta
                else:
                    dx = (x2 - x0) / (2 * delta)
                    dy = (y2 - y0) / (2 * delta)

                if s <= delta:
                    ddx = (sx(s2) - 2 * sx(s1) + sx(s0)) / (delta ** 2)
                    ddy = (sy(s2) - 2 * sy(s1) + sy(s0)) / (delta ** 2)
                elif s >= s_max - delta:
                    ddx = (sx(s2) - 2 * sx(s1) + sx(s0)) / (delta ** 2)
                    ddy = (sy(s2) - 2 * sy(s1) + sy(s0)) / (delta ** 2)
                else:
                    ddx = (sx(s + delta) - 2 * sx(s) + sx(s - delta)) / (delta ** 2)
                    ddy = (sy(s + delta) - 2 * sy(s) + sy(s - delta)) / (delta ** 2)

                denominator = (dx ** 2 + dy ** 2) ** (3 / 2)
                if abs(denominator) < 1e-6:
                    curv_arr[idx] = 0.0
                else:
                    curv_arr[idx] = (dx * ddy - ddx * dy) / denominator

            return np.abs(curv_arr), s_eval_arr

        # -------------------------- 核心配置：适配 3x3 小尺寸画布 --------------------------
        # 1. 强制设定画布尺寸 (英寸)
        fig_size_small = (4, 3)  
        
        # 2. 缩小字号以适配小画布
        font_size_label = 8      # 坐标轴标签
        font_size_tick = 6       # 刻度数字
        font_size_legend = 5     # 图例文字
        
        font_path = '/usr/share/fonts/truetype/msttcorefonts/SongTi.ttf'
        font_prop_chinese = fm.FontProperties(fname=font_path, size=font_size_label)

        # 全局参数调整
        plt.rcParams["figure.dpi"] = 300
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["legend.frameon"] = False
        plt.rcParams["axes.labelsize"] = font_size_label
        plt.rcParams["xtick.labelsize"] = font_size_tick
        plt.rcParams["ytick.labelsize"] = font_size_tick
        plt.rcParams["legend.fontsize"] = font_size_legend
        
        # -------------------------- 创建画布 --------------------------
        fig1, ax1 = plt.subplots(figsize=fig_size_small)
        fig2, ax2 = plt.subplots(figsize=fig_size_small)
        
        rnd = MPRenderer(ax=ax1)
        rnd.draw_params.dynamic_obstacle.draw_icon = True
        rnd.draw_params.dynamic_obstacle.draw_bounding_box = True
        rnd.draw_params.dynamic_obstacle.show_label = False
        rnd.draw_params.dynamic_obstacle.state.draw_arrow = False
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.facecolor = "white"
        rnd.draw_params.dynamic_obstacle.vehicle_shape.occupancy.shape.edgecolor = "#0000CD"
        
        ego_params = DynamicObstacleParams()
        ego_params.draw_icon = True
        ego_params.vehicle_shape.occupancy.shape.facecolor = "white"
        ego_params.vehicle_shape.occupancy.shape.edgecolor = "#006400"
        ego_params.vehicle_shape.occupancy.shape.zorder = 20
        
        xy_limit={0:((20,80),(-40,20)), 1:((30,100),(-40,30))}

        for path_id, path_data in enumerate(self.path_library):
            
            map_name = path_data["map_name"]
            center_vertices = path_data["center_vertices"].detach().cpu().numpy()
            left_vertices = path_data["left_vertices"].detach().cpu().numpy()
            right_vertices = path_data["right_vertices"].detach().cpu().numpy()
            s = path_data["s"].detach().cpu().numpy()
            hinge_trajs = path_data["hinge_trajs"].detach().cpu().numpy()
            hinge_status = path_data["hinge_status"].detach().cpu().numpy()
            scenario = self.scenario_library[map_name]

            # 计算曲率
            center_curv, center_s_arr = calc_traj_curvature(center_vertices)
            hinge2_curv, hinge2_s_arr = calc_traj_curvature(hinge_trajs[2])
            hinge3_curv, hinge3_s_arr = calc_traj_curvature(hinge_trajs[3])

            for i in range(0, 1):
                # -------------------------- 绘制地图 (fig1) --------------------------
                plt.figure(fig1.number)
                rnd.draw_params.time_begin = i
                rnd.draw_params.time_end = i
                ego_params.time_begin = i
                ego_params.time_end = i
                scenario.draw(rnd)
                rnd.render(show=True)
                
                linewidth = 0.8 # 线宽也相应改细
                rnd.ax.plot(left_vertices[:, 0], left_vertices[:, 1], c='r', label='左边界', zorder=20, linewidth=linewidth)
                rnd.ax.plot(right_vertices[:, 0], right_vertices[:, 1], c='b', label='右边界', zorder=20, linewidth=linewidth)
                rnd.ax.plot(center_vertices[:, 0], center_vertices[:, 1], c='purple', label='中心线', zorder=20, linestyle='--', linewidth=linewidth)
                
                rnd.ax.set_xlabel("x/m", fontproperties=font_prop_chinese, fontsize=font_size_label)
                rnd.ax.set_ylabel("y/m", fontproperties=font_prop_chinese, fontsize=font_size_label)
                rnd.ax.set_xlim(xy_limit[path_id][0])
                rnd.ax.set_ylim(xy_limit[path_id][1])
                rnd.ax.tick_params(
                    axis="both", 
                    labelsize=font_size_tick, 
                    pad=1,
                    direction='in',
                    top=False, right=False,
                    labelfontfamily='Times New Roman'
                )

                # 绘制 hinge 轨迹
                n_agents = hinge_trajs.shape[0]
                cmap = plt.get_cmap('tab10') 
                for agent_idx in range(1,n_agents-1):
                    agent_color = cmap(agent_idx % 10)
                    traj_x = hinge_trajs[agent_idx, :, 0]
                    traj_y = hinge_trajs[agent_idx, :, 1]
                    rnd.ax.plot(traj_x, traj_y, color=agent_color, linewidth=0.5, alpha=0.7, linestyle='-', zorder=18, label=f'铰接点{agent_idx+1}')
                    
                    out_of_road_mask = hinge_status[agent_idx, :] == 0
                    if np.any(out_of_road_mask):
                        out_x = traj_x[out_of_road_mask]
                        out_y = traj_y[out_of_road_mask]
                        rnd.ax.scatter(out_x, out_y, marker='x', s=2, c=agent_color, linewidths=0.5, zorder=22, label=f'不可结合')

                legend = rnd.ax.legend(fontsize=font_size_legend, prop=font_prop_chinese, frameon=False)
                legend.set_zorder(100)
                plt.pause(0.01)
                fig1_path = f"{vis_dir}/Path{path_id:04d}_map_{i:04d}.pdf"
                fig1.savefig(fig1_path, dpi=300, format="pdf", bbox_inches='tight')
                
                # -------------------------- 绘制曲率 (fig2) --------------------------
                plt.figure(fig2.number)
                ax2.clear()
                ax2.plot(center_s_arr, center_curv, c='purple', linestyle='--', linewidth=0.8, label='中心线曲率')
                ax2.plot(hinge2_s_arr, hinge2_curv, c='#FF6347', linewidth=0.8, label='铰接点2曲率')
                ax2.plot(hinge3_s_arr, hinge3_curv, c='#FFD700', linewidth=0.8, label='铰接点3曲率')
                
                # [关键] 强制正方形比例
                ax2.set_box_aspect(1)
                
                ax2.set_xlabel("累计弧长/m", fontproperties=font_prop_chinese, fontsize=font_size_label)
                ax2.set_ylabel("曲率/$m^{-1}$", fontproperties=font_prop_chinese, fontsize=font_size_label)
                ax2.legend(fontsize=font_size_legend, prop=font_prop_chinese, frameon=False)
                ax2.tick_params(
                    axis="both",
                    labelsize=font_size_tick,
                    pad=1,
                    direction='in',
                    top=False, right=False,
                    labelfontfamily='Times New Roman'
                )
                
                all_curv = np.concatenate([center_curv, hinge2_curv, hinge3_curv])
                y_min, y_max = np.min(all_curv) * 0.9, np.max(all_curv) * 1.1
                ax2.set_ylim(y_min, y_max)
                ax2.set_xlim(30,160)

                fig2_path = f"{vis_dir}/Path{path_id:04d}_curvature_{i:04d}.pdf"
                fig2.savefig(fig2_path, dpi=300, format="pdf", bbox_inches='tight')
                
                rnd.ax.clear()
# try separate each path, but more time cost, deprecated
# class OcctCRMapNew(MapBase):
#     def __init__(
#         self,
#         batch_dim: int,
#         device: torch.device,
#         cr_map_dir: str = "vmas/scenarios_data/cr_maps/debug",
#         sample_gap: float = 1,
#         min_lane_width: float = 2.1,
#         min_lane_len: float = 70,
#         max_ref_v: float = 20/3.6,
#         is_constant_ref_v: bool = False
#     ):
#         """
#         初始化道路类，预生成所有路径的样条并缓存
#         """
#         self.device = device
#         self.batch_dim = batch_dim
#         self.cr_map_dir = cr_map_dir
#         self.vis_dir = os.path.join(cr_map_dir, "vis")
#         os.makedirs(self.vis_dir, exist_ok=True)
#         self.sample_gap = sample_gap
#         self.min_lane_width = min_lane_width
#         self.min_lane_len = min_lane_len
#         self.max_ref_v = max_ref_v
#         self.is_constant_ref_v = is_constant_ref_v
#         self.start_end_distance_threshold = 25

#         # 路径库（缓存所有路径的原始数据+预生成的样条）
#         self.path_library = []
#         self.scenario_library = dict[str, Scenario]()
#         self.max_path_length = 0.0
#         self.max_path_s_list = None

#         # 批量管理变量（列表形式，长度=batch_dim）
#         self.batch_id = torch.zeros(self.batch_dim, dtype=torch.int64, device=self.device)
#         self.batch_map_name = [None] * self.batch_dim
#         self.batch_s_max = torch.zeros(self.batch_dim, device=self.device)
        
#         # 核心：每个batch对应一个预生成的样条实例（列表）
#         self.center_splines = [None] * self.batch_dim
#         self.left_splines = [None] * self.batch_dim
#         self.right_splines = [None] * self.batch_dim
#         self.ref_v_splines = [None] * self.batch_dim

#         # 一次性处理地图并预生成所有路径的样条
#         self._cr_map_process(cr_map_dir)
#         print(
#             f"[OcctCRMap]共{len(self.path_library)}条路径数据,"
#             f"最长为{self.max_path_length:.2f}米,"
#             f"平均宽度为{self.get_lane_width():.2f}米"
#         )
        
#         if len(self.path_library) == 0:
#             raise ValueError("No paths found in the provided CommonRoad map directory")
#         self.reset_splines()

#     def get_lane_width(self, type="mean") -> float:
#         lane_widths = torch.hstack([
#             torch.tensor(path["lane_width"], device=self.device) 
#             for path in self.path_library
#         ])
#         if type == "mean":
#             return lane_widths.mean().item()
#         elif type == "min":
#             return lane_widths.min().item()
#         elif type == "max":
#             return lane_widths.max().item()
#         else:
#             raise ValueError(f"type must be 'mean','min' or 'max', but got {type}")

#     def _get_cum_len(self, vertices: Tensor) -> Tensor:
#         if isinstance(vertices, np.ndarray):
#             vertices = torch.tensor(vertices, device=self.device)
#         seg = vertices[1:] - vertices[:-1]
#         seg_len = torch.linalg.norm(seg, dim=-1)
#         cum_len = torch.cat([torch.zeros(1, device=self.device), torch.cumsum(seg_len, dim=0)])
#         return cum_len

#     def _resample_path(self, vertices: Tensor) -> Tuple[Tensor, Tensor]:
#         original_s = self._get_cum_len(vertices)
#         s_max = original_s[-1]
#         M = max(2, int(torch.floor(s_max / self.sample_gap)))
#         s = torch.linspace(0.0, s_max, M, device=self.device)
        
#         idx = torch.searchsorted(original_s, s, right=True) - 1
#         idx = torch.clamp(idx, 0, len(original_s) - 2)
        
#         s0 = original_s[idx]
#         s1 = original_s[idx + 1]
#         p0 = vertices[idx]
#         p1 = vertices[idx + 1]
        
#         alpha = (s - s0) / (s1 - s0 + 1e-8)
#         resampled_vertices = p0 + alpha.unsqueeze(1) * (p1 - p0)
        
#         return resampled_vertices, s

#     def get_s_max_idx(self) -> Tensor:
#         return torch.argmax(self.batch_s_max)
    
#     def _enrich_vertices_sampling(self, center_vertices, left_vertices, right_vertices):
#         center_seg_lengths = np.linalg.norm(center_vertices[1:] - center_vertices[:-1], axis=1)
#         if np.max(center_seg_lengths) > 2 * self.sample_gap:
#             center_length = self._get_cum_len(center_vertices)
#             sample_num = max(2, int(torch.floor(center_length[-1] / self.sample_gap)))
#             resampled_vertices = [np.zeros((sample_num, 2)), np.zeros((sample_num, 2)), np.zeros((sample_num, 2))]
            
#             for resampled, vertices in zip(resampled_vertices, [center_vertices, left_vertices, right_vertices]):
#                 segment_lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
#                 cum_lengths = np.zeros(len(vertices))
#                 cum_lengths[1:] = np.cumsum(segment_lengths)
#                 target_s = np.linspace(0, cum_lengths[-1], sample_num)
#                 for i in range(2):
#                     resampled[:, i] = np.interp(target_s, cum_lengths, vertices[:, i])
#             return resampled_vertices[0], resampled_vertices[1], resampled_vertices[2]
#         return center_vertices, left_vertices, right_vertices

#     def _detect_loop(self, points: List[Tuple[float, float]], tol: float = 1.0) -> Tuple[bool, List[Tuple[int, int]]]:
#         n = len(points)
#         if n < 4:
#             return False, []
        
#         points_array = np.array(points)
#         loop_pairs = []
        
#         for i in range(n - 3):
#             for j in range(i + 3, n):
#                 distance = np.linalg.norm(points_array[i] - points_array[j])
#                 if distance <= tol:
#                     loop_pairs.append((i, j))
        
#         return len(loop_pairs) > 0, loop_pairs

#     def _cr_map_process(self, map_dir: str) -> None:
#         """
#         核心修改：处理地图时预生成所有路径的样条并缓存到path_library
#         """
#         dump_file = os.path.join(self.cr_map_dir, "map_data_with_splines.pkl")
#         # 优先加载缓存（包含预生成的样条）
#         if os.path.exists(dump_file):
#             self.scenario_library, self.path_library, self.max_path_length, self.max_path_s_list = pickle.load(
#                 open(dump_file, "rb")
#             )
#             self.max_path_length = self.max_path_length.to(self.device)
#             self.max_path_s_list = self.max_path_s_list.to(self.device)
#             return
        
#         map_files = glob.glob(os.path.join(map_dir, "**/*.xml"), recursive=True)
#         print(f"找到 {len(map_files)} 个地图文件:")
#         for i, map_file in enumerate(map_files):
#             print(f"  {i+1}. {os.path.basename(map_file)}")
        
#         for map_file in map_files:
#             map_name = os.path.basename(map_file)
#             print(f"\n处理地图: {map_name}")
            
#             scenario = get_cr_scenario(map_file)
#             self.scenario_library[map_name] = scenario
#             lanelets = scenario.lanelet_network.lanelets
#             start_lanelets = [lanelet for lanelet in lanelets if not lanelet.predecessor]
            
#             path_id_library = []
#             for start_lanelet in start_lanelets:
#                 queue = deque()
#                 queue.append([start_lanelet.lanelet_id])
                
#                 while queue:
#                     current_path = queue.popleft()
#                     current_lanelet_id = current_path[-1]
#                     current_lanelet = scenario.lanelet_network.find_lanelet_by_id(current_lanelet_id)
                    
#                     if not current_lanelet.successor:
#                         path_id_library.append(current_path)
#                     else:
#                         for successor_id in current_lanelet.successor:
#                             if successor_id not in current_path:
#                                 new_path = current_path.copy()
#                                 new_path.append(successor_id)
#                                 queue.append(new_path)
            
#             print(f"找到 {len(path_id_library)} 条路径:")
#             for i, path in enumerate(path_id_library):
#                 print(f"  路径 {i+1}: {path}")
            
#             for path_ids in path_id_library:
#                 if not (path_ids[0] == 128 and path_ids[-1] == 106):
#                     continue
                
#                 path_data = {"center_vertices": [], "left_vertices": [], "right_vertices": []}
#                 for i, lanelet_id in enumerate(path_ids):
#                     lanelet = scenario.lanelet_network.find_lanelet_by_id(lanelet_id)
#                     center_vertices = lanelet.center_vertices
#                     left_vertices = lanelet.left_vertices
#                     right_vertices = lanelet.right_vertices
                    
#                     center_vertices, left_vertices, right_vertices = self._enrich_vertices_sampling(
#                         center_vertices, left_vertices, right_vertices
#                     )
                    
#                     slice_range = slice(None) if i == len(path_ids) - 1 else slice(-1)
#                     for key, vertices in zip(
#                         ["center_vertices", "left_vertices", "right_vertices"],
#                         [center_vertices, left_vertices, right_vertices]
#                     ):
#                         path_data[key].extend([v.tolist() for v in vertices[slice_range]])
                
#                 # 基础过滤
#                 lane_width = [np.linalg.norm(left_vertices[i] - right_vertices[i]) for i in range(len(center_vertices))]
#                 center_vertices = torch.tensor(path_data["center_vertices"], device=self.device, dtype=torch.float32)
#                 center_cum_len = self._get_cum_len(center_vertices)
                
#                 if (
#                     min(lane_width) < self.min_lane_width
#                     or center_cum_len[-1] < self.min_lane_len
#                     or torch.linalg.norm(center_vertices[0] - center_vertices[-1]) < self.start_end_distance_threshold
#                 ):
#                     continue
                
#                 # 重采样路径
#                 resampled_center, s = self._resample_path(center_vertices)
#                 is_loop, _ = self._detect_loop(resampled_center.tolist())
#                 if is_loop:
#                     continue
                
#                 # 预生成左/右边界样条并插值重采样点
#                 coeffs_left = natural_cubic_spline_coeffs(center_cum_len, torch.tensor(path_data["left_vertices"], device=self.device))
#                 left_spline = NaturalCubicSpline(coeffs_left)
#                 resampled_left = left_spline.evaluate(s)
                
#                 coeffs_right = natural_cubic_spline_coeffs(center_cum_len, torch.tensor(path_data["right_vertices"], device=self.device))
#                 right_spline = NaturalCubicSpline(coeffs_right)
#                 resampled_right = right_spline.evaluate(s)
                
#                 resampled_lane_width = [
#                     torch.linalg.norm(resampled_left[i] - resampled_right[i]) 
#                     for i in range(len(resampled_center))
#                 ]
                
#                 # 预生成中心路径样条（用于计算曲率）
#                 coeffs_center = natural_cubic_spline_coeffs(s, resampled_center)
#                 center_spline = NaturalCubicSpline(coeffs_center)
                
#                 # 计算参考速度
#                 center_curvature = self.compute_curvature_2d(center_spline, s, smooth_distance=10)
#                 if self.is_constant_ref_v:
#                     ref_v = self.max_ref_v * torch.ones_like(center_curvature)
#                 else:
#                     factor = 0.2
#                     ref_v = torch.clamp_max(factor / torch.sqrt(center_curvature + 1e-8) ** 2, self.max_ref_v)
#                     ref_v = self.gaussian_smooth_1d(ref_v, sigma=5.0)
                
#                 # 预生成参考速度样条
#                 coeffs_ref_v = natural_cubic_spline_coeffs(s, ref_v.unsqueeze(-1))
#                 ref_v_spline = NaturalCubicSpline(coeffs_ref_v)
                
#                 # ===================== 核心：缓存预生成的样条 =====================
#                 self.path_library.append({
#                     "map_name": map_name,
#                     "path_ids": path_ids,
#                     "center_vertices": resampled_center,
#                     "left_vertices": resampled_left,
#                     "right_vertices": resampled_right,
#                     "s": s,
#                     "s_max": s[-1],
#                     "ref_v": ref_v,
#                     "lane_width": resampled_lane_width,
#                     # 预生成的样条实例（核心优化点）
#                     "coeffs_center_spline": coeffs_center,
#                     "coeffs_left_spline": coeffs_left,
#                     "coeffs_right_spline": coeffs_right,
#                     "coeffs_ref_v_spline": coeffs_ref_v,
#                     "center_spline": center_spline,
#                     "left_spline": left_spline,
#                     "right_spline": right_spline,
#                     "ref_v_spline": ref_v_spline,
#                 })
                
#                 # 更新最长路径信息
#                 if s[-1] > self.max_path_length:
#                     self.max_path_length = s[-1]
#                     self.max_path_s_list = s
        
#         # 保存包含预生成样条的缓存（避免重复计算）
#         pickle.dump(
#             (self.scenario_library, self.path_library, self.max_path_length, self.max_path_s_list),
#             open(dump_file, "wb")
#         )

#     def gaussian_smooth_1d(self, x: torch.Tensor, sigma: float = 2.0) -> torch.Tensor:
#         orig_shape = x.shape
#         if x.dim() == 0:
#             return x
        
#         seq_dim = x.shape[-1]
#         batch_dims = x.shape[:-1]
#         x_flat = x.reshape(-1, seq_dim)
        
#         kernel_size = int(2 * round(3 * sigma) + 1)
#         if kernel_size > seq_dim:
#             kernel_size = seq_dim if seq_dim % 2 == 1 else seq_dim - 1
#         if kernel_size < 3:
#             return x
        
#         kernel = torch.arange(kernel_size, device=x.device, dtype=torch.float32) - (kernel_size - 1) / 2
#         kernel = torch.exp(-kernel ** 2 / (2 * sigma ** 2))
#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.unsqueeze(0).unsqueeze(0)
        
#         x_padded = F.pad(x_flat.unsqueeze(1), pad=[kernel_size//2, kernel_size//2], mode='replicate')
#         smooth_x = F.conv1d(x_padded, kernel, stride=1).squeeze(1)
        
#         return smooth_x.reshape(orig_shape)

#     def get_scenario_by_env_index(self, env_index):
#         return self.scenario_library[self.batch_map_name[env_index]]

#     def reset_splines(self, env_index: Optional[int] = None):
#         """
#         核心优化：仅查表替换样条，不生成新样条
#         - env_index=None: 全量重置所有batch
#         - env_index=int: 仅重置指定batch
#         """
#         start_time = time.time()
#         print(f"开始reset_splines, env_index: {env_index}")
        
#         # 1. 确定需要更新的batch索引
#         if env_index is None:
#             update_indices = list(range(self.batch_dim))
#             # 全量更新时重新生成batch_id
#             self.batch_id = torch.randint(0, len(self.path_library), (self.batch_dim,), device=self.device)
#         else:
#             update_indices = [env_index]
#             # 仅更新指定env的batch_id
#             self.batch_id[env_index] = torch.randint(0, len(self.path_library), (1,), device=self.device).item()
        
#         # 2. 仅做查表替换（核心：无样条生成，仅指针赋值）
#         for batch_idx in update_indices:
#             path_id = self.batch_id[batch_idx].item()
#             path_data = self.path_library[path_id]
            
#             # 基础信息更新
#             self.batch_map_name[batch_idx] = path_data["map_name"]
#             self.batch_s_max[batch_idx] = path_data["s_max"]
            
#             # 样条替换（仅赋值，无计算）
#             self.center_splines[batch_idx] = path_data["center_spline"]
#             self.left_splines[batch_idx] = path_data["left_spline"]
#             self.right_splines[batch_idx] = path_data["right_spline"]
#             self.ref_v_splines[batch_idx] = path_data["ref_v_spline"]
        
#         end_time = time.time()
#         print(f"结束reset_splines, cost time: {end_time - start_time}")

#     def compute_curvature_2d(
#         self,
#         spline: NaturalCubicSpline,
#         t: torch.Tensor,
#         smooth_distance: float = 5.0
#     ) -> torch.Tensor:
#         smooth_distance = max(0.1, smooth_distance)
        
#         t_min = torch.min(t) - 0.1
#         t_max = torch.max(t) + 0.1
#         sample_density = 10
#         total_t_range = t_max - t_min
#         total_length_estimate = total_t_range * torch.norm(spline.evaluate(t_max) - spline.evaluate(t_min)) / total_t_range
#         steps = max(500, int(total_length_estimate * sample_density))
#         t_samples = torch.linspace(t_min, t_max, steps=steps, device=t.device)
        
#         pos_samples = spline.evaluate(t_samples)
#         pos_diff = pos_samples[1:] - pos_samples[:-1]
#         seg_lengths = torch.norm(pos_diff, dim=-1)
#         cum_lengths = torch.cat([torch.tensor([0.0], device=t.device), torch.cumsum(seg_lengths, dim=0)])
        
#         dr_dt_samples = spline.derivative(t_samples, order=1)
#         d2r_dt2_samples = spline.derivative(t_samples, order=2)
        
#         dr_norm_sq_samples = torch.sum(dr_dt_samples ** 2, dim=-1)
#         dr_norm_samples = torch.sqrt(dr_norm_sq_samples)
#         dr_norm_cube_samples = dr_norm_samples ** 3
#         numerator_samples = torch.abs(
#             dr_dt_samples[:, 0] * d2r_dt2_samples[:, 1] - dr_dt_samples[:, 1] * d2r_dt2_samples[:, 0]
#         )
#         curvature_samples = numerator_samples / (dr_norm_cube_samples + 1e-8)
        
#         t_flat = t.flatten()
#         pos_t = spline.evaluate(t_flat)
        
#         t_samples_expand = t_samples.unsqueeze(0)
#         t_flat_expand = t_flat.unsqueeze(1)
#         idx_nearest = torch.argmin(torch.abs(t_flat_expand - t_samples_expand), dim=1)
        
#         pos_nearest = pos_samples[idx_nearest]
#         pos_diff = pos_t - pos_nearest
#         dist_diff = torch.norm(pos_diff, dim=-1)
        
#         cum_lengths_nearest = cum_lengths[idx_nearest]
#         length_t = cum_lengths_nearest + dist_diff
        
#         length_t_expand = length_t.unsqueeze(1)
#         cum_lengths_expand = cum_lengths.unsqueeze(0)
#         mask = (cum_lengths_expand >= (length_t_expand - smooth_distance)) & \
#                (cum_lengths_expand <= (length_t_expand + smooth_distance))
        
#         mask_float = mask.float()
#         sum_weights = torch.sum(mask_float, dim=1)
#         curvature_samples_expand = curvature_samples.unsqueeze(0)
        
#         curvature_sum = torch.sum(mask_float * curvature_samples_expand, dim=1)
#         curvature_nearest = curvature_samples[idx_nearest]
#         smooth_curvature_flat = torch.where(
#             sum_weights > 0,
#             curvature_sum / sum_weights,
#             curvature_nearest
#         )
        
#         return smooth_curvature_flat.reshape(t.shape)

#     # ========== 查询函数：直接使用预生成的样条 ==========
#     def get_pts(self, s: Tensor, env_j: Optional[int] = None) -> Tensor:
#         """获取指定弧长的道路中心点"""
#         if s.dim() == 0:
#             assert env_j is not None, "单值s必须指定env_j"
#             return self.center_splines[env_j].evaluate(s)
#         else:
#             assert s.shape[0] == self.batch_dim, "s的批量维度需与batch_dim一致"
#             return torch.stack([
#                 self.center_splines[i].evaluate(s[i]) for i in range(self.batch_dim)
#             ], dim=0)

#     def get_ref_v(self, s: Tensor, env_j: Optional[int] = None) -> Tensor:
#         """获取指定弧长的参考速度"""
#         if s.dim() == 0:
#             assert env_j is not None, "单值s必须指定env_j"
#             return self.ref_v_splines[env_j].evaluate(s)
#         else:
#             assert s.shape[0] == self.batch_dim, "s的批量维度需与batch_dim一致"
#             return torch.stack([
#                 self.ref_v_splines[i].evaluate(s[i]) for i in range(self.batch_dim)
#             ], dim=0)

#     def get_tangent_vector(self, s: Tensor, env_j: Optional[int] = None) -> Tensor:
#         """获取指定弧长的切向量"""
#         if s.dim() == 0:
#             assert env_j is not None, "单值s必须指定env_j"
#             tangent = self.center_splines[env_j].derivative(s, order=1)
#         else:
#             assert s.shape[0] == self.batch_dim, "s的批量维度需与batch_dim一致"
#             tangent = torch.stack([
#                 self.center_splines[i].derivative(s[i], order=1) for i in range(self.batch_dim)
#             ], dim=0)
        
#         return tangent / (torch.linalg.norm(tangent, dim=-1, keepdim=True) + 1e-8)

#     def get_s_max(self, env_j: Optional[int] = None) -> Tensor:
#         """获取道路最大弧长"""
#         if env_j is not None:
#             return self.batch_s_max[env_j]
#         return self.batch_s_max

#     def get_road_center_pts(self, env_j: Optional[int] = None) -> Tensor:
#         """获取道路中心点列表"""
#         if env_j is not None:
#             path_id = self.batch_id[env_j].item()
#             return self.path_library[path_id]["center_vertices"]
#         return torch.stack([
#             self.path_library[self.batch_id[i].item()]["center_vertices"] for i in range(self.batch_dim)
#         ], dim=0)

#     def get_road_left_pts(self, env_j: Optional[int] = None) -> Tensor:
#         """获取道路左边界点列表"""
#         if env_j is not None:
#             path_id = self.batch_id[env_j].item()
#             return self.path_library[path_id]["left_vertices"]
#         return torch.stack([
#             self.path_library[self.batch_id[i].item()]["left_vertices"] for i in range(self.batch_dim)
#         ], dim=0)

#     def get_road_right_pts(self, env_j: Optional[int] = None) -> Tensor:
#         """获取道路右边界点列表"""
#         if env_j is not None:
#             path_id = self.batch_id[env_j].item()
#             return self.path_library[path_id]["right_vertices"]
#         return torch.stack([
#             self.path_library[self.batch_id[i].item()]["right_vertices"] for i in range(self.batch_dim)
#         ], dim=0)

if __name__ == "__main__chapter23":
    device = torch.device("cuda")
    # road = OcctMap(batch_dim=1, device=device)
    # road.plot_road_debug()

    road = OcctCRMap(batch_dim=200, 
                     cr_map_dir="vmas/scenarios_data/cr_maps/debug",
                     max_ref_v=15/3.6 ,
                     min_lane_width=2.9, 
                     min_lane_len=120,
                     device=device, 
                     sample_gap=1, 
                     is_constant_ref_v=False,
                     rod_len=30.0)
    
    road.plot_road_debug()
    road.plot_road_for_paper(vis_dir="vmas/scenarios_data/cr_maps/debug/vis_paper")
if __name__ == "__main__chapter4_no_extra_hinge":
    device = torch.device("cuda")
    road = OcctCRMap(batch_dim=200, 
                     cr_map_dir="vmas/scenarios_data/cr_maps/chapter4",
                     max_ref_v=15/3.6 ,
                     min_lane_width=2.4, 
                     min_lane_len=80,
                     device=device, 
                     sample_gap=1, 
                     is_constant_ref_v=False,
                     rod_len=18.0,
                     extend_len=None,#0.0,
                     n_agents=4)
    
    road.plot_road_debug()
if __name__ == "__main__":
    device = torch.device("cuda")
    road = OcctCRMap(batch_dim=200, 
                     cr_map_dir="vmas/scenarios_data/cr_maps/chapter4",
                     max_ref_v=15/3.6 ,
                     min_lane_width=2.4, 
                     min_lane_len=80,
                     device=device, 
                     sample_gap=1, 
                     is_constant_ref_v=False,
                     rod_len=18.0,
                     extend_len=None,#0.0,
                     n_agents=4)
    
    road.plot_road_debug()