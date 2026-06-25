from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# X-MoD v2 (re-parametrization): target-speed denormalization cap, m/s.
# The network predicts target_speed_norm in [0,1]; downstream it is multiplied by V_MAX_MPS
# and converted to throttle/brake by a classical PID (xmod_v2/longitudinal_controller.py).
# KEEP IN SYNC with train_xmod_m1.py and xmod_b2d_agent.py.
V_MAX_MPS = 22.0  # covers the PDM-Lite teacher target_speed range (observed max ~20 m/s) with headroom


def _build_backbone(name: str = "resnet34", pretrained: bool = False) -> nn.Module:
    try:
        import torchvision.models as tv_models

        if name == "resnet34":
            weights = tv_models.ResNet34_Weights.DEFAULT if pretrained else None
            m = tv_models.resnet34(weights=weights)
            m.fc = nn.Linear(m.fc.in_features, 256)
            return m
    except Exception:
        pass

    # Fallback encoder if torchvision is unavailable.
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 256),
    )


class MLPExpert(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, out_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class XMoDVLA(nn.Module):
    def __init__(
        self,
        backbone_name: str = "resnet34",
        pretrained_backbone: bool = False,
        expert_input_mode: str = "partial",
        ego_dim: int = 2,
        router_hidden: int = 128,
        expert_hidden: int = 64,
        architecture: str = "moe",
    ):
        super().__init__()
        self.backbone = _build_backbone(backbone_name, pretrained_backbone)
        if expert_input_mode not in {"partial", "full"}:
            raise ValueError("expert_input_mode must be 'partial' or 'full'")
        if architecture not in {"moe", "separated", "targetspeed"}:
            raise ValueError("architecture must be 'moe', 'separated', or 'targetspeed'")
        self.expert_input_mode = expert_input_mode
        self.ego_dim = int(ego_dim)
        self.router_hidden = int(router_hidden)
        self.expert_hidden = int(expert_hidden)
        self.architecture = architecture
        if self.ego_dim <= 0:
            raise ValueError("ego_dim must be positive")
        if self.router_hidden <= 0:
            raise ValueError("router_hidden must be positive")
        if self.expert_hidden <= 0:
            raise ValueError("expert_hidden must be positive")

        # 256 visual feature + ego-state. The historical checkpoints use
        # ego_dim=2: (speed, route_angle). Route-conditioned M1 variants widen
        # this vector while keeping the first two columns backward compatible.
        shared_dim = 256 + self.ego_dim
        self.router = nn.Sequential(
            nn.Linear(shared_dim, self.router_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.router_hidden, 4),
        )

        expert_in_dim = 4 if expert_input_mode == "partial" else shared_dim
        self.expert_safety = MLPExpert(expert_in_dim, hidden=self.expert_hidden)
        self.expert_legality = MLPExpert(expert_in_dim, hidden=self.expert_hidden)
        self.expert_comfort = MLPExpert(expert_in_dim, hidden=self.expert_hidden)
        self.expert_efficiency = MLPExpert(expert_in_dim, hidden=self.expert_hidden)
        self.mobility_head = MLPExpert(shared_dim, hidden=self.expert_hidden, out_dim=2)
        self.hazard_dim = 14
        self.brake_head = MLPExpert(self.hazard_dim, hidden=self.expert_hidden, out_dim=1)

    def _ego_col(self, ego_state: torch.Tensor, idx: int, default: float = 0.0) -> torch.Tensor:
        if ego_state.shape[1] <= idx:
            if torch.is_tensor(default):
                return default.to(device=ego_state.device, dtype=ego_state.dtype).reshape(ego_state.shape[0], 1)
            return torch.full(
                (ego_state.shape[0], 1),
                float(default),
                device=ego_state.device,
                dtype=ego_state.dtype,
            )
        return ego_state[:, idx : idx + 1]

    def _hazard_context(self, ego_state: torch.Tensor) -> torch.Tensor:
        speed_norm = (self._ego_col(ego_state, 0) / 50.0).clamp(0.0, 2.0)
        route_abs = (self._ego_col(ego_state, 1).abs() / 3.141592653589793).clamp(0.0, 1.0)
        actor_distance = self._ego_col(ego_state, 24, 1.0).clamp(0.0, 1.0)
        actor_forward = self._ego_col(ego_state, 25, 1.0).clamp(0.0, 1.0)
        actor_lateral_abs = self._ego_col(ego_state, 26, 1.0).abs().clamp(0.0, 1.0)
        actor_vehicle = self._ego_col(ego_state, 27, 0.0).clamp(0.0, 1.0)
        actor_walker = self._ego_col(ego_state, 28, 0.0).clamp(0.0, 1.0)
        actor_closeness = self._ego_col(ego_state, 35, 1.0 - actor_distance).clamp(0.0, 1.0)
        stopline_close = (1.0 - self._ego_col(ego_state, 32, 1.0)).clamp(0.0, 1.0)
        traffic_light_close = (1.0 - self._ego_col(ego_state, 33, 1.0)).clamp(0.0, 1.0)
        stop_sign_close = (1.0 - self._ego_col(ego_state, 34, 1.0)).clamp(0.0, 1.0)
        junction_ahead = self._ego_col(ego_state, 36, 0.0).clamp(0.0, 1.0)
        rel_speed = self._ego_col(ego_state, 41, 0.0).clamp(-1.0, 1.0)
        closing_speed = self._ego_col(ego_state, 42, rel_speed.clamp_min(0.0)).clamp(0.0, 1.0)
        return torch.cat(
            [
                speed_norm,
                route_abs,
                actor_distance,
                actor_forward,
                actor_lateral_abs,
                actor_vehicle,
                actor_walker,
                actor_closeness,
                stopline_close,
                traffic_light_close,
                stop_sign_close,
                junction_ahead,
                rel_speed,
                closing_speed,
            ],
            dim=1,
        )

    def forward(
        self,
        image: torch.Tensor,
        ego_state: torch.Tensor,
        partial_states: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        v = self.backbone(image)
        if ego_state.shape[1] != self.ego_dim:
            raise ValueError(f"expected ego_state dim {self.ego_dim}, got {ego_state.shape[1]}")
        router_in = torch.cat([v, ego_state], dim=1)
        gate_logits = self.router(router_in)
        gate_probs = F.softmax(gate_logits, dim=1)

        if self.architecture == "targetspeed":
            # X-MoD v2: the policy chooses steer + a target speed (not raw pedals).
            # action_out = [steer, target_speed_norm, 0]; a classical PID downstream maps
            # target_speed_norm * V_MAX_MPS -> throttle/brake. The router/gate is preserved
            # for explainability. This is the single untested axis from the B2D negative result.
            mobility = self.mobility_head(router_in)
            steer = torch.tanh(mobility[:, 0:1])
            target_speed_norm = torch.sigmoid(mobility[:, 1:2])
            action_out = torch.cat([steer, target_speed_norm, torch.zeros_like(steer)], dim=1)
            experts = {
                "gate_logits": gate_logits,
                "mobility": mobility,
                "target_speed_norm": target_speed_norm,
            }
            return action_out, gate_probs, experts

        if self.architecture == "separated":
            mobility = self.mobility_head(router_in)
            hazard_context = self._hazard_context(ego_state)
            brake_raw = self.brake_head(hazard_context)
            steer = torch.tanh(mobility[:, 0:1])
            throttle = torch.sigmoid(mobility[:, 1:2])
            brake = torch.sigmoid(brake_raw)
            action_out = torch.cat([steer, throttle, brake], dim=1)
            experts = {
                "gate_logits": gate_logits,
                "mobility": mobility,
                "brake": brake_raw,
                "hazard_context": hazard_context,
            }
            return action_out, gate_probs, experts

        if self.expert_input_mode == "partial":
            safety_in = partial_states["safety"]
            legality_in = partial_states["legality"]
            comfort_in = partial_states["comfort"]
            efficiency_in = partial_states["efficiency"]
        else:
            # C2-off ablation: every expert consumes the full shared state feature.
            safety_in = router_in
            legality_in = router_in
            comfort_in = router_in
            efficiency_in = router_in

        a_safety = self.expert_safety(safety_in)
        a_legality = self.expert_legality(legality_in)
        a_comfort = self.expert_comfort(comfort_in)
        a_efficiency = self.expert_efficiency(efficiency_in)

        action = (
            gate_probs[:, 0:1] * a_safety
            + gate_probs[:, 1:2] * a_legality
            + gate_probs[:, 2:3] * a_comfort
            + gate_probs[:, 3:4] * a_efficiency
        )

        # Map to control ranges.
        steer = torch.tanh(action[:, 0:1])
        throttle = torch.sigmoid(action[:, 1:2])
        brake = torch.sigmoid(action[:, 2:3])
        action_out = torch.cat([steer, throttle, brake], dim=1)

        experts = {
            "safety": a_safety,
            "legality": a_legality,
            "comfort": a_comfort,
            "efficiency": a_efficiency,
            "gate_logits": gate_logits,
        }
        return action_out, gate_probs, experts


class XMoDLoss(nn.Module):
    def __init__(
        self,
        lambda_act: float = 1.0,
        lambda_align: float = 0.5,
        lambda_sparse: float = 0.01,
        lambda_risk: float = 0.1,
        router_loss: str = "ce",
    ):
        super().__init__()
        self.lambda_act = lambda_act
        self.lambda_align = lambda_align
        self.lambda_sparse = lambda_sparse
        self.lambda_risk = lambda_risk
        self.router_loss = router_loss

    def forward(
        self,
        pred_action: torch.Tensor,
        gt_action: torch.Tensor,
        gate_probs: torch.Tensor,
        gate_logits: torch.Tensor,
        drive_single_idx: torch.Tensor,
        drive_multihot: torch.Tensor,
        safety_event: torch.Tensor,
        ttc: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_action = F.mse_loss(pred_action, gt_action)
        if self.router_loss == "bce":
            loss_align = F.binary_cross_entropy(gate_probs, drive_multihot.float())
        else:
            loss_align = F.cross_entropy(gate_logits, drive_single_idx)
        # Negative entropy penalization -> sparse routing.
        entropy = -(gate_probs * (gate_probs.clamp_min(1e-8).log())).sum(dim=1).mean()
        loss_sparse = entropy

        # Risk loss: during safety event and low TTC, throttle should be small.
        throttle = pred_action[:, 1]
        risk_weight = torch.where(ttc < 2.5, torch.ones_like(ttc), 0.2 * torch.ones_like(ttc))
        loss_risk = ((throttle * safety_event.float()) * risk_weight).mean()

        total = (
            self.lambda_act * loss_action
            + self.lambda_align * loss_align
            + self.lambda_sparse * loss_sparse
            + self.lambda_risk * loss_risk
        )
        return {
            "total": total,
            "action": loss_action.detach(),
            "align": loss_align.detach(),
            "sparse": loss_sparse.detach(),
            "risk": loss_risk.detach(),
        }
