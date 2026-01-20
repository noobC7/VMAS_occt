import numpy as np

class OcctBoundaryCalculator:
    def _calculate_boundary_pts(self, center_vertices, left_vertices, right_vertices):
        """
        最终版：首段（0-1 + 反向延伸-1-0）、尾段（0-1 + 正向延伸1-2）、中间仅0-1

        核心规则：
        1. 首段边界线段：t∈[-1, 1]（线段内+反向延伸）
        2. 尾段边界线段：t∈[0, 2]（线段内+正向延伸）
        3. 中间线段：仅线段内交点（t∈[0,1]）
        4. 所有交点满足方向约束（不跨中心线）

        Args:
            center_vertices: [N, 2] 中心路径点
            left_vertices: [M, 2] 左侧边界点
            right_vertices: [M, 2] 右侧边界点

        Returns:
            left_boundary_pts: [N, 2] 左侧边界点
            right_boundary_pts: [N, 2] 右侧边界点
        """
        center_pts = np.array(center_vertices, dtype=np.float64)
        left_pts = np.array(left_vertices, dtype=np.float64)
        right_pts = np.array(right_vertices, dtype=np.float64)
        
        N = center_pts.shape[0]
        M = left_pts.shape[0]
        left_boundary_pts = np.zeros_like(center_pts)
        right_boundary_pts = np.zeros_like(center_pts)

        for i in range(N):
            # 计算切线与法线
            if i == 0:
                tangent = center_pts[i+1] - center_pts[i]
            elif i == N-1:
                tangent = center_pts[i] - center_pts[i-1]
            else:
                tangent = center_pts[i+1] - center_pts[i-1]
            
            normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-8:
                normal = np.array([0, 1], dtype=np.float64)
            else:
                normal = normal / normal_norm

            # 构建法线直线方程
            A = normal[1]
            B = -normal[0]
            C = normal[0] * center_pts[i, 1] - normal[1] * center_pts[i, 0]

            # 计算交点
            left_boundary_pts[i] = self._find_nearest_segment_intersection(
                A, B, C, left_pts, center_pts[i], normal, is_left=True, total_segs=M-1
            )
            right_boundary_pts[i] = self._find_nearest_segment_intersection(
                A, B, C, right_pts, center_pts[i], normal, is_left=False, total_segs=M-1
            )

        return left_boundary_pts, right_boundary_pts

    def _find_nearest_segment_intersection(self, A, B, C, boundary_pts, center_pt, normal, is_left, total_segs):
        valid_intersections = []
        seg_distances = []

        for seg_idx in range(total_segs):
            p1 = boundary_pts[seg_idx]
            p2 = boundary_pts[seg_idx+1]

            # 根据线段类型选择有效t范围
            if seg_idx == 0:
                # 首段：线段内(0-1) + 反向延伸(-1-0) → t∈[-1, 1]
                intersection = self._get_intersection_with_range(A, B, C, p1, p2, t_range=(-1.0, 1.0))
            elif seg_idx == total_segs - 1:
                # 尾段：线段内(0-1) + 正向延伸(1-2) → t∈[0, 2]
                intersection = self._get_intersection_with_range(A, B, C, p1, p2, t_range=(0.0, 2.0))
            else:
                # 中间段：仅线段内(0-1)
                intersection = self._get_intersection_with_range(A, B, C, p1, p2, t_range=(0.0, 1.0))

            if intersection is not None and self._check_direction(center_pt, intersection, normal, is_left):
                valid_intersections.append(intersection)
                seg_mid = (p1 + p2) / 2
                seg_distances.append(np.linalg.norm(seg_mid - center_pt))

        # 选择最优交点
        if valid_intersections:
            valid_intersections = np.array(valid_intersections)
            seg_distances = np.array(seg_distances)
            seg_sorted_idx = np.argsort(seg_distances)
            nearest_seg_intersections = valid_intersections[seg_sorted_idx]
            pt_distances = np.linalg.norm(nearest_seg_intersections - center_pt, axis=1)
            final_idx = np.argmin(pt_distances)
            return nearest_seg_intersections[final_idx]
        
        return self._get_nearest_valid_boundary_pt(center_pt, boundary_pts, normal, is_left)

    def _get_intersection_with_range(self, A, B, C, p1, p2, t_range):
        """根据指定t范围计算交点"""
        x1, y1 = p1
        x2, y2 = p2
        denominator = A * (x2 - x1) + B * (y2 - y1)
        
        if abs(denominator) < 1e-8:
            return None
        
        t = -(A * x1 + B * y1 + C) / denominator
        t_min, t_max = t_range

        if t_min - 1e-8 <= t <= t_max + 1e-8:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return np.array([x, y], dtype=np.float64)
        return None

    def _check_direction(self, center_pt, intersection, normal, is_left):
        dir_vec = intersection - center_pt
        dot_product = np.dot(dir_vec, normal)
        if is_left:
            return dot_product > -1e-8
        else:
            return dot_product < 1e-8

    def _get_nearest_valid_boundary_pt(self, center_pt, boundary_pts, normal, is_left):
        distances = np.linalg.norm(boundary_pts - center_pt, axis=1)
        sorted_idx = np.argsort(distances)
        for idx in sorted_idx:
            pt = boundary_pts[idx]
            if self._check_direction(center_pt, pt, normal, is_left):
                return pt
        return boundary_pts[sorted_idx[0]]

# ------------------------ 测试验证 ------------------------
if __name__ == "__main__":
    calculator = OcctBoundaryCalculator()
    
    center_vertices = np.array([[0,0], [1,0], [2,0], [3,0], [4,0]])
    left_vertices = np.array([[0,-1], [1,-1.2], [2,-1], [3,-0.8], [4,-1]])
    right_vertices = np.array([[0,1], [1,1.2], [2,1], [3,0.8], [4,1]])
    
    left_pts, right_pts = calculator._calculate_boundary_pts(
        center_vertices, left_vertices, right_vertices
    )
    
    print("左侧边界点（首段含反向延伸，尾段含正向延伸）：")
    print(left_pts)
    print("右侧边界点（首段含反向延伸，尾段含正向延伸）：")
    print(right_pts)