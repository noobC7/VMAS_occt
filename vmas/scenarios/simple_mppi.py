from typing import Optional, Tuple

import torch
from torch import Tensor


class SimpleMPPIController:
    """Lightweight MPPI controller for short-horizon path tracking."""

    def __init__(
        self,
        num_agents: int,
        device: torch.device,
        dt: float,
        l_f: float,
        l_r: float,
        max_steer_abs,
        max_accel_abs: float,
        max_speed: float,
        horizon_step_T: int = 2,
        number_of_samples_K: int = 256,
        param_exploration: float = 0.1,
        param_lambda: float = 10.0,
        param_alpha: float = 1.0,
        sigma: Optional[Tensor] = None,
        stage_cost_weight: Optional[Tensor] = None,
        terminal_cost_weight: Optional[Tensor] = None,
        debug_top_k: int = 8,
    ) -> None:
        self.dim_x = 4
        self.dim_u = 2
        self.num_agents = num_agents
        self.device = device
        self.dt = dt
        self.l_f = l_f
        self.l_r = l_r
        self.wheel_base = l_f + l_r
        self.max_steer_abs = float(torch.as_tensor(max_steer_abs).item())
        self.max_accel_abs = float(max_accel_abs)
        self.max_speed = float(max_speed)

        self.T = int(horizon_step_T)
        self.K = int(number_of_samples_K)
        self.param_exploration = float(param_exploration)
        self.param_lambda = float(param_lambda)
        self.param_alpha = float(param_alpha)
        self.param_gamma = self.param_lambda * (1.0 - self.param_alpha)
        self.debug_top_k = int(debug_top_k)

        if sigma is None:
            sigma = torch.tensor([0.20, 0.80], device=device, dtype=torch.float32)
        self.sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32)
        if self.sigma.shape != (self.dim_u,):
            raise ValueError("sigma must have shape [2]")
        self.inv_sigma_diag = 1.0 / torch.clamp(self.sigma**2, min=1e-6)

        if stage_cost_weight is None:
            stage_cost_weight = torch.tensor(
                [40.0, 8.0, 12.0, 0.05, 0.20], device=device, dtype=torch.float32
            )
        if terminal_cost_weight is None:
            terminal_cost_weight = torch.tensor(
                [80.0, 12.0, 16.0], device=device, dtype=torch.float32
            )
        self.stage_cost_weight = torch.as_tensor(
            stage_cost_weight, device=device, dtype=torch.float32
        )
        self.terminal_cost_weight = torch.as_tensor(
            terminal_cost_weight, device=device, dtype=torch.float32
        )
        if self.stage_cost_weight.shape != (5,):
            raise ValueError("stage_cost_weight must have shape [5]")
        if self.terminal_cost_weight.shape != (3,):
            raise ValueError("terminal_cost_weight must have shape [3]")

        self.u_prev = torch.zeros(
            (num_agents, self.T, self.dim_u), device=device, dtype=torch.float32
        )
        self.last_debug = {}

    def reset(self, agent_idx: Optional[int] = None) -> None:
        if agent_idx is None:
            self.u_prev.zero_()
            self.last_debug = {}
        else:
            self.u_prev[agent_idx].zero_()
            self.last_debug.pop(agent_idx, None)

    def command(
        self,
        agent_idx: int,
        observed_x: Tensor,
        ref_points: Tensor,
        ref_speeds: Tensor,
        stage_cost_weight: Optional[Tensor] = None,
        terminal_cost_weight: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        observed_x = torch.as_tensor(
            observed_x, device=self.device, dtype=torch.float32
        ).reshape(-1)
        if observed_x.shape[0] != self.dim_x:
            raise ValueError("observed_x must have shape [4]")

        ref_points = self._prepare_ref_points(ref_points)
        ref_speeds = self._prepare_ref_speeds(ref_speeds)
        stage_w = self._prepare_stage_weights(stage_cost_weight)
        terminal_w = self._prepare_terminal_weights(terminal_cost_weight)

        u = self.u_prev[agent_idx].clone()
        epsilon = self._calc_epsilon()
        controls = self._sample_controls(u, epsilon)
        costs = self._rollout_costs(
            observed_x=observed_x,
            nominal_controls=u,
            sampled_controls=controls,
            ref_points=ref_points,
            ref_speeds=ref_speeds,
            stage_weight=stage_w,
            terminal_weight=terminal_w,
        )

        weights = self._compute_weights(costs)
        w_epsilon = torch.sum(weights[:, None, None] * epsilon, dim=0)
        w_epsilon = self._moving_average_filter(
            w_epsilon, window_size=min(5, self.T)
        )
        u = self._clamp_controls(u + w_epsilon)
        optimal_traj = self.rollout_nominal(observed_x, u)
        self._update_debug_cache(
            agent_idx=agent_idx,
            observed_x=observed_x,
            ref_points=ref_points,
            ref_speeds=ref_speeds,
            costs=costs,
            sampled_controls=controls,
            optimal_traj=optimal_traj,
        )

        self.u_prev[agent_idx, :-1] = u[1:]
        self.u_prev[agent_idx, -1] = u[-1]
        return u[0], u, optimal_traj

    def rollout_nominal(self, observed_x: Tensor, controls: Tensor) -> Tensor:
        x = observed_x.clone()
        traj = torch.zeros((self.T, self.dim_x), device=self.device, dtype=x.dtype)
        for t in range(self.T):
            x = self._F(x.unsqueeze(0), controls[t].unsqueeze(0)).squeeze(0)
            traj[t] = x
        return traj

    def rollout_samples(self, observed_x: Tensor, controls: Tensor) -> Tensor:
        if controls.ndim != 3 or controls.shape[-1] != self.dim_u:
            raise ValueError("controls must have shape [N, T, 2]")
        batch_size = controls.shape[0]
        x = observed_x.unsqueeze(0).expand(batch_size, -1).clone()
        traj = torch.zeros(
            (batch_size, self.T, self.dim_x), device=self.device, dtype=x.dtype
        )
        for t in range(self.T):
            x = self._F(x, controls[:, t])
            traj[:, t] = x
        return traj

    def _prepare_ref_points(self, ref_points: Tensor) -> Tensor:
        ref_points = torch.as_tensor(
            ref_points, device=self.device, dtype=torch.float32
        )
        if ref_points.ndim != 2 or ref_points.shape[1] != 2:
            raise ValueError("ref_points must have shape [N, 2]")
        target_len = self.T + 1
        if ref_points.shape[0] >= target_len:
            return ref_points[:target_len]
        pad = ref_points[-1:].expand(target_len - ref_points.shape[0], -1)
        return torch.cat([ref_points, pad], dim=0)

    def _prepare_ref_speeds(self, ref_speeds: Tensor) -> Tensor:
        ref_speeds = torch.as_tensor(
            ref_speeds, device=self.device, dtype=torch.float32
        ).reshape(-1)
        target_len = self.T + 1
        if ref_speeds.numel() == 1:
            return ref_speeds.expand(target_len)
        if ref_speeds.shape[0] >= target_len:
            return ref_speeds[:target_len]
        pad = ref_speeds[-1:].expand(target_len - ref_speeds.shape[0])
        return torch.cat([ref_speeds, pad], dim=0)

    def _prepare_stage_weights(self, weights: Optional[Tensor]) -> Tensor:
        if weights is None:
            return self.stage_cost_weight
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        if weights.shape != (5,):
            raise ValueError("stage_cost_weight must have shape [5]")
        return weights

    def _prepare_terminal_weights(self, weights: Optional[Tensor]) -> Tensor:
        if weights is None:
            return self.terminal_cost_weight
        weights = torch.as_tensor(weights, device=self.device, dtype=torch.float32)
        if weights.shape != (3,):
            raise ValueError("terminal_cost_weight must have shape [3]")
        return weights

    def _calc_epsilon(self) -> Tensor:
        noise = torch.randn(
            (self.K, self.T, self.dim_u), device=self.device, dtype=torch.float32
        )
        return noise * self.sigma.view(1, 1, -1)

    def _sample_controls(self, nominal_controls: Tensor, epsilon: Tensor) -> Tensor:
        controls = torch.empty_like(epsilon)
        exploit_count = int((1.0 - self.param_exploration) * self.K)
        if exploit_count > 0:
            controls[:exploit_count] = nominal_controls.unsqueeze(0) + epsilon[
                :exploit_count
            ]
        if exploit_count < self.K:
            controls[exploit_count:] = epsilon[exploit_count:]
        return self._clamp_controls(controls)

    def _clamp_controls(self, controls: Tensor) -> Tensor:
        controls = controls.clone()
        controls[..., 0] = torch.clamp(
            controls[..., 0], -self.max_steer_abs, self.max_steer_abs
        )
        controls[..., 1] = torch.clamp(
            controls[..., 1], -self.max_accel_abs, self.max_accel_abs
        )
        return controls

    def _rollout_costs(
        self,
        observed_x: Tensor,
        nominal_controls: Tensor,
        sampled_controls: Tensor,
        ref_points: Tensor,
        ref_speeds: Tensor,
        stage_weight: Tensor,
        terminal_weight: Tensor,
    ) -> Tensor:
        x = observed_x.unsqueeze(0).expand(self.K, -1).clone()
        costs = torch.zeros(self.K, device=self.device, dtype=torch.float32)
        ref_headings = self._compute_ref_headings(ref_points)

        for t in range(self.T):
            control_t = sampled_controls[:, t]
            x = self._F(x, control_t)
            segment_dist_sq = self._segment_distance_sq(
                x[:, :2], ref_points[t], ref_points[t + 1]
            )
            heading_error = self._angle_diff(x[:, 2], ref_headings[t])
            speed_error = x[:, 3] - ref_speeds[t + 1]
            control_cost = torch.sum(control_t**2, dim=-1)
            prev_control = (
                sampled_controls[:, t - 1]
                if t > 0
                else nominal_controls[t].unsqueeze(0).expand_as(control_t)
            )
            smooth_cost = torch.sum((control_t - prev_control) ** 2, dim=-1)
            nominal_cost = self.param_gamma * torch.sum(
                nominal_controls[t].unsqueeze(0) * control_t * self.inv_sigma_diag,
                dim=-1,
            )
            costs = costs + (
                stage_weight[0] * segment_dist_sq
                + stage_weight[1] * heading_error**2
                + stage_weight[2] * speed_error**2
                + stage_weight[3] * control_cost
                + stage_weight[4] * smooth_cost
                + nominal_cost
            )

        terminal_heading = ref_headings[-1]
        terminal_pos_error = torch.sum((x[:, :2] - ref_points[-1]) ** 2, dim=-1)
        terminal_heading_error = self._angle_diff(x[:, 2], terminal_heading)
        terminal_speed_error = x[:, 3] - ref_speeds[-1]
        costs = costs + (
            terminal_weight[0] * terminal_pos_error
            + terminal_weight[1] * terminal_heading_error**2
            + terminal_weight[2] * terminal_speed_error**2
        )
        return costs

    def _compute_ref_headings(self, ref_points: Tensor) -> Tensor:
        deltas = ref_points[1:] - ref_points[:-1]
        zero_mask = torch.linalg.norm(deltas, dim=-1) < 1e-6
        if zero_mask.any():
            deltas = deltas.clone()
            deltas[zero_mask] = torch.tensor(
                [1.0, 0.0], device=self.device, dtype=torch.float32
            )
        return torch.atan2(deltas[:, 1], deltas[:, 0])

    def _segment_distance_sq(
        self, points: Tensor, segment_start: Tensor, segment_end: Tensor
    ) -> Tensor:
        segment = segment_end - segment_start
        seg_norm_sq = torch.clamp(torch.sum(segment**2), min=1e-6)
        rel = points - segment_start.unsqueeze(0)
        proj = torch.sum(rel * segment.unsqueeze(0), dim=-1) / seg_norm_sq
        proj = torch.clamp(proj, min=0.0, max=1.0)
        closest = segment_start.unsqueeze(0) + proj.unsqueeze(-1) * segment.unsqueeze(0)
        return torch.sum((points - closest) ** 2, dim=-1)

    def _F(self, x_t: Tensor, u_t: Tensor) -> Tensor:
        x = x_t[:, 0]
        y = x_t[:, 1]
        yaw = x_t[:, 2]
        v = x_t[:, 3]
        steer = u_t[:, 0]
        accel = u_t[:, 1]

        beta = torch.atan2(
            torch.tan(steer) * self.l_r / self.wheel_base,
            torch.ones_like(steer),
        )
        new_x = x + v * torch.cos(yaw + beta) * self.dt
        new_y = y + v * torch.sin(yaw + beta) * self.dt
        new_yaw = yaw + v / self.wheel_base * torch.cos(beta) * torch.tan(steer) * self.dt
        new_v = torch.clamp(v + accel * self.dt, min=0.0, max=self.max_speed)
        return torch.stack([new_x, new_y, new_yaw, new_v], dim=-1)

    def _compute_weights(self, costs: Tensor) -> Tensor:
        rho = torch.min(costs)
        normalized = -(costs - rho) / max(self.param_lambda, 1e-6)
        return torch.softmax(normalized, dim=0)

    def _moving_average_filter(self, xx: Tensor, window_size: int) -> Tensor:
        if window_size <= 1 or xx.shape[0] <= 1:
            return xx
        out = torch.zeros_like(xx)
        half_window = window_size // 2
        for t in range(xx.shape[0]):
            begin = max(0, t - half_window)
            end = min(xx.shape[0], t + half_window + 1)
            out[t] = xx[begin:end].mean(dim=0)
        return out

    def _angle_diff(self, angle_a: Tensor, angle_b) -> Tensor:
        angle_b = torch.as_tensor(angle_b, device=self.device, dtype=torch.float32)
        return torch.atan2(torch.sin(angle_a - angle_b), torch.cos(angle_a - angle_b))

    def _update_debug_cache(
        self,
        agent_idx: int,
        observed_x: Tensor,
        ref_points: Tensor,
        ref_speeds: Tensor,
        costs: Tensor,
        sampled_controls: Tensor,
        optimal_traj: Tensor,
    ) -> None:
        top_k = min(self.debug_top_k, sampled_controls.shape[0])
        if top_k <= 0:
            self.last_debug[agent_idx] = {
                "ref_points": ref_points.detach().clone(),
                "ref_speeds": ref_speeds.detach().clone(),
                "optimal_traj": optimal_traj.detach().clone(),
                "sampled_trajs": torch.empty(
                    (0, self.T, self.dim_x), device=self.device, dtype=torch.float32
                ),
                "costs": torch.empty(0, device=self.device, dtype=torch.float32),
            }
            return

        top_indices = torch.argsort(costs)[:top_k]
        top_controls = sampled_controls[top_indices]
        top_trajs = self.rollout_samples(observed_x, top_controls)
        self.last_debug[agent_idx] = {
            "ref_points": ref_points.detach().clone(),
            "ref_speeds": ref_speeds.detach().clone(),
            "optimal_traj": optimal_traj.detach().clone(),
            "sampled_trajs": top_trajs.detach().clone(),
            "costs": costs[top_indices].detach().clone(),
        }
