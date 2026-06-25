from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROUTE_V1_EGO_DIM = 29
ROUTE_V2_EGO_DIM = 38
ROUTE_V3_EGO_DIM = 41
ROUTE_V4_EGO_DIM = 43
COMMAND_CLASSES = 7


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def image_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _finite_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(number):
        return default
    return number


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _clamp_signed(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _command_id(value) -> int:
    try:
        command = int(float(value))
    except (TypeError, ValueError):
        command = 4
    return max(0, min(COMMAND_CLASSES - 1, command))


def _command_onehot(value) -> list[float]:
    command = _command_id(value)
    out = [0.0] * COMMAND_CLASSES
    out[command] = 1.0
    return out


def ego_dim_for_mode(mode: str) -> int:
    if mode == "base":
        return 2
    if mode == "route_v1":
        return ROUTE_V1_EGO_DIM
    if mode == "route_v2":
        return ROUTE_V2_EGO_DIM
    if mode == "route_v3":
        return ROUTE_V3_EGO_DIM
    if mode == "route_v4":
        return ROUTE_V4_EGO_DIM
    raise ValueError(f"Unknown ego_mode: {mode}")


def _turn_command(value) -> float:
    command = _command_id(value)
    return 0.0 if command == 4 else 1.0


def build_ego_features(state: dict, partial: dict, route_flow: dict | None, mode: str) -> list[float]:
    speed_kmh = _finite_float(state.get("speed_kmh", partial.get("v", 0.0)))
    route_angle = _finite_float(
        state.get("route_angle_8m", state.get("target_wp_angle", partial.get("route_angle_8m", partial.get("wp_angle", 0.0))))
    )
    if mode == "base":
        return [speed_kmh, route_angle]

    route_flow = route_flow or {}
    route_angle_16m = _finite_float(
        state.get("route_angle_16m", partial.get("route_angle_16m", route_flow.get("route_angle_16m", route_angle)))
    )
    command = partial.get("command", route_flow.get("command", 4))
    next_command = partial.get("next_command", route_flow.get("next_command", command))
    route_len = _finite_float(partial.get("route_len", route_flow.get("route_len", 0.0)))
    route_original_len = _finite_float(partial.get("route_original_len", route_flow.get("route_original_len", route_len)))
    actor_distance = _finite_float(partial.get("runtime_front_actor_distance", 999.0), 999.0)
    actor_forward = _finite_float(partial.get("runtime_front_actor_forward", actor_distance), 999.0)
    actor_lateral = _finite_float(partial.get("runtime_front_actor_lateral", 999.0), 999.0)

    features = [
        speed_kmh,
        route_angle,
        float(np.sin(route_angle_16m)),
        float(np.cos(route_angle_16m)),
        _command_id(command) / float(COMMAND_CLASSES - 1),
        _command_id(next_command) / float(COMMAND_CLASSES - 1),
    ]
    features.extend(_command_onehot(command))
    features.extend(_command_onehot(next_command))
    features.extend(
        [
            float(partial.get("junction", 0.0)),
            float(partial.get("changed_route", 0.0)),
            _clamp01(route_len / 200.0),
            _clamp01(route_original_len / 200.0),
            _clamp01(actor_distance / 50.0),
            _clamp01(actor_forward / 50.0),
            max(-1.0, min(1.0, actor_lateral / 10.0)),
            float(partial.get("runtime_front_actor_is_vehicle", 0.0)),
            float(partial.get("runtime_front_actor_is_walker", 0.0)),
        ]
    )
    if len(features) != ROUTE_V1_EGO_DIM:
        raise RuntimeError(f"route_v1 feature dim mismatch: {len(features)} != {ROUTE_V1_EGO_DIM}")
    if mode == "route_v1":
        return features

    d_traffic_light = _finite_float(partial.get("distance_to_next_traffic_light", 999.0), 999.0)
    d_stop_sign = _finite_float(partial.get("distance_to_next_stop_sign", 999.0), 999.0)
    d_stopline = _finite_float(partial.get("d_stopline", min(d_traffic_light, d_stop_sign)), 999.0)
    stopline_norm = _clamp01(d_stopline / 50.0)
    traffic_light_norm = _clamp01(d_traffic_light / 80.0)
    stop_sign_norm = _clamp01(d_stop_sign / 80.0)
    junction_ahead = max(float(partial.get("junction", 0.0)), _turn_command(command), _turn_command(next_command))
    junction_ahead_distance = 0.0 if junction_ahead >= 0.5 else 999.0
    features.extend(
        [
            _clamp01(abs(route_angle_16m) / np.pi),
            _turn_command(command),
            _turn_command(next_command),
            stopline_norm,
            traffic_light_norm,
            stop_sign_norm,
            1.0 - _clamp01(actor_distance / 50.0),
            1.0 if junction_ahead >= 0.5 else 0.0,
            _clamp01(junction_ahead_distance / 50.0),
        ]
    )
    if len(features) != ROUTE_V2_EGO_DIM:
        raise RuntimeError(f"route_v2 feature dim mismatch: {len(features)} != {ROUTE_V2_EGO_DIM}")
    if mode == "route_v2":
        return features
    route_progress = _finite_float(partial.get("route_progress", route_flow.get("route_progress", 0.0)))
    progress_norm = _clamp01(route_progress / 1000.0)
    progress_phase = 2.0 * np.pi * progress_norm
    features.extend(
        [
            progress_norm,
            float(np.sin(progress_phase)),
            float(np.cos(progress_phase)),
        ]
    )
    if len(features) != ROUTE_V3_EGO_DIM:
        raise RuntimeError(f"route_v3 feature dim mismatch: {len(features)} != {ROUTE_V3_EGO_DIM}")
    if mode == "route_v3":
        return features
    actor_visible = actor_distance < 50.0 or actor_forward < 50.0
    v_rel_raw = _finite_float(
        partial.get("runtime_front_actor_rel_speed_mps", partial.get("v_rel", 0.0)),
        0.0,
    )
    # Existing converted datasets often carry v_rel=0. When no measured
    # relative velocity is available, use ego speed as a conservative closing
    # proxy only for frames that actually have a front actor.
    v_rel_mps = v_rel_raw / 3.6 if abs(v_rel_raw) > 35.0 else v_rel_raw
    if abs(v_rel_mps) < 1e-6 and actor_visible:
        v_rel_mps = speed_kmh / 3.6
    closing_mps = max(0.0, v_rel_mps) if actor_visible else 0.0
    features.extend(
        [
            _clamp_signed(v_rel_mps / 20.0),
            _clamp01(closing_mps / 20.0),
        ]
    )
    if len(features) != ROUTE_V4_EGO_DIM:
        raise RuntimeError(f"route_v4 feature dim mismatch: {len(features)} != {ROUTE_V4_EGO_DIM}")
    return features


def load_compatible_checkpoint(model, ckpt_path: Path):
    raw = torch.load(ckpt_path, map_location="cpu")
    source = raw.get("model", raw) if isinstance(raw, dict) else raw
    target = model.state_dict()
    patched = {}
    widened = []
    skipped = []
    for key, value in source.items():
        if key not in target:
            continue
        if target[key].shape == value.shape:
            patched[key] = value
            continue
        if (
            value.ndim == 2
            and target[key].ndim == 2
            and target[key].shape[0] == value.shape[0]
            and target[key].shape[1] > value.shape[1]
        ):
            widened_value = torch.zeros_like(target[key])
            widened_value[:, : value.shape[1]] = value
            patched[key] = widened_value
            widened.append(key)
            if not hasattr(model, "_xmod_widened_source_dims"):
                model._xmod_widened_source_dims = {}
            model._xmod_widened_source_dims[key] = int(value.shape[1])
            continue
        skipped.append((key, tuple(value.shape), tuple(target[key].shape)))
    missing, unexpected = model.load_state_dict(patched, strict=False)
    return missing, unexpected, widened, skipped


def enable_extra_column_training_only(model, scope: str = "all", source_dim_override: int | None = None) -> dict[str, int]:
    expert_scopes = {
        "expert_safety": "expert_safety.",
        "expert_legality": "expert_legality.",
        "expert_comfort": "expert_comfort.",
        "expert_efficiency": "expert_efficiency.",
    }
    if scope not in {"all", "router", "experts", *expert_scopes}:
        raise ValueError(f"Unknown extra-column training scope: {scope}")
    source_dims = dict(getattr(model, "_xmod_widened_source_dims", {}))
    if source_dim_override is not None:
        source_dims = {}
        for name, param in model.named_parameters():
            if param.ndim != 2 or param.shape[1] <= source_dim_override:
                continue
            if not (name.startswith("router.") or name.startswith("expert_")):
                continue
            if name != "router.0.weight" and not name.endswith(".net.0.weight"):
                continue
            source_dims[name] = int(source_dim_override)
    if not source_dims:
        raise RuntimeError("No widened parameters are available for extra-column-only training")
    for _, param in model.named_parameters():
        param.requires_grad = False
    enabled = {}
    for name, param in model.named_parameters():
        if name not in source_dims:
            continue
        is_router = name.startswith("router.")
        if scope == "router" and not is_router:
            continue
        if scope == "experts" and is_router:
            continue
        if scope in expert_scopes and not name.startswith(expert_scopes[scope]):
            continue
        if param.ndim != 2:
            continue
        source_dim = int(source_dims[name])
        if source_dim >= param.shape[1]:
            continue
        mask = torch.zeros_like(param)
        mask[:, source_dim:] = 1.0
        param.requires_grad = True
        param.register_hook(lambda grad, mask=mask: grad * mask.to(device=grad.device, dtype=grad.dtype))
        enabled[name] = int(param.shape[1] - source_dim)
    if not enabled:
        raise RuntimeError("No extra columns were enabled for training")
    return enabled


# X-MoD v2 target-speed denormalization cap (m/s). KEEP IN SYNC with models.x_mod_vla.V_MAX_MPS.
TARGET_SPEED_V_MAX_MPS = 22.0  # KEEP IN SYNC with models.x_mod_vla.V_MAX_MPS (covers teacher max ~20 m/s)


class XModJsonDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict],
        image_size: int,
        ego_mode: str,
        cache: bool = False,
        hold_action_weight: float = 1.0,
        recovery_action_weight: float = 1.0,
        hold_speed_kmh: float = 2.0,
        hold_brake_threshold: float = 0.5,
        hold_front_distance: float = 10.0,
        go_speed_kmh: float = 3.0,
        go_front_distance: float = 10.0,
        go_min_throttle: float = 0.8,
        go_max_brake: float = 0.05,
        slow_front_distance: float = 10.0,
        slow_min_brake: float = 0.5,
        slow_max_throttle: float = 0.2,
        target_speed_mode: bool = False,
    ):
        self.root = root
        self.rows = rows
        self.image_size = image_size
        self.ego_mode = ego_mode
        self.hold_action_weight = float(hold_action_weight)
        self.recovery_action_weight = float(recovery_action_weight)
        self.hold_speed_kmh = float(hold_speed_kmh)
        self.hold_brake_threshold = float(hold_brake_threshold)
        self.hold_front_distance = float(hold_front_distance)
        self.go_speed_kmh = float(go_speed_kmh)
        self.go_front_distance = float(go_front_distance)
        self.go_min_throttle = float(go_min_throttle)
        self.go_max_brake = float(go_max_brake)
        self.slow_front_distance = float(slow_front_distance)
        self.slow_min_brake = float(slow_min_brake)
        self.slow_max_throttle = float(slow_max_throttle)
        self.target_speed_mode = bool(target_speed_mode)
        self.cache = [self._build(i) for i in range(len(rows))] if cache else None

    def __len__(self):
        return len(self.rows)

    def _build(self, idx: int):
        row = self.rows[idx]
        state = load_json(self.root / row["state"])
        partial = load_json(self.root / row["partial"])
        action = load_json(self.root / row["action"])
        label = load_json(self.root / row["label"])
        route_flow = load_json(self.root / row["route_flow"]) if row.get("route_flow") else {}
        speed_kmh = _finite_float(state.get("speed_kmh", partial.get("v", 0.0)))
        brake = _finite_float(action.get("brake", 0.0))
        target_speed_zero = _finite_float(partial.get("teacher_target_speed_zero", 0.0))
        d_front = min(
            _finite_float(partial.get("runtime_front_actor_distance", 999.0), 999.0),
            _finite_float(partial.get("d_front", 999.0), 999.0),
        )
        d_ped = _finite_float(partial.get("d_ped", 999.0), 999.0)
        stop_or_actor = (
            target_speed_zero >= 0.5
            or d_front < self.hold_front_distance
            or d_ped < self.hold_front_distance
        )
        hold_frame = (
            speed_kmh < self.hold_speed_kmh
            and brake > self.hold_brake_threshold
            and stop_or_actor
        )
        recovery_frame = (
            _finite_float(partial.get("recovery_label", 0.0)) >= 0.5
            or bool((label.get("event_flags") or {}).get("recovery_event", False))
            or str(row.get("scenario", "")).endswith("lowmotion_recovery")
        )
        no_close_actor = d_front >= self.go_front_distance and d_ped >= self.go_front_distance
        go_frame = (
            speed_kmh <= self.go_speed_kmh
            and no_close_actor
            and target_speed_zero < 0.5
            and _finite_float(action.get("throttle", 0.0)) >= self.go_min_throttle
            and _finite_float(action.get("brake", 0.0)) <= self.go_max_brake
        )
        slow_teacher = (
            target_speed_zero >= 0.5
            or _finite_float(action.get("brake", 0.0)) >= self.slow_min_brake
            or _finite_float(action.get("throttle", 0.0)) <= self.slow_max_throttle
        )
        slow_frame = d_front <= self.slow_front_distance and slow_teacher
        action_weight = 1.0
        if hold_frame:
            action_weight *= self.hold_action_weight
        if recovery_frame:
            action_weight *= self.recovery_action_weight

        if self.target_speed_mode:
            # X-MoD v2: supervise [steer, target_speed_norm, 0] instead of raw pedals.
            ts_mps = _finite_float(partial.get("teacher_target_speed", 0.0))
            ts_norm = max(0.0, min(1.0, ts_mps / TARGET_SPEED_V_MAX_MPS))
            act_vec = [float(action["steer"]), ts_norm, 0.0]
        else:
            act_vec = [float(action["steer"]), float(action["throttle"]), float(action["brake"])]

        return {
            "image": image_tensor(self.root / row["image"], self.image_size),
            "ego": torch.tensor(build_ego_features(state, partial, route_flow, self.ego_mode), dtype=torch.float32),
            "action": torch.tensor(act_vec, dtype=torch.float32),
            "drive_single_idx": torch.tensor(int(label["drive_single_idx"]), dtype=torch.long),
            "drive_multihot": torch.tensor([float(x) for x in label.get("drive_multihot", [0, 0, 0, 1])], dtype=torch.float32),
            "safety_event": torch.tensor(float(label["event_flags"]["safety_event"]), dtype=torch.float32),
            "ttc": torch.tensor(float(partial.get("ttc", 999.0)), dtype=torch.float32),
            "action_weight": torch.tensor(float(action_weight), dtype=torch.float32),
            "hold_frame": torch.tensor(float(hold_frame), dtype=torch.float32),
            "recovery_frame": torch.tensor(float(recovery_frame), dtype=torch.float32),
            "go_frame": torch.tensor(float(go_frame), dtype=torch.float32),
            "slow_frame": torch.tensor(float(slow_frame), dtype=torch.float32),
            "route_id": str(row.get("route_id", "?")),
            "scenario": str(row.get("scenario", "?")),
        }

    def __getitem__(self, idx: int):
        if self.cache is not None:
            return self.cache[idx]
        return self._build(idx)


def collate(batch: list[dict]) -> dict:
    out = {}
    for key in [
        "image",
        "ego",
        "action",
        "drive_single_idx",
        "drive_multihot",
        "safety_event",
        "ttc",
        "action_weight",
        "hold_frame",
        "recovery_frame",
        "go_frame",
        "slow_frame",
    ]:
        out[key] = torch.stack([item[key] for item in batch])
    out["route_id"] = [item["route_id"] for item in batch]
    out["scenario"] = [item["scenario"] for item in batch]
    return out


def split_rows(rows: list[dict], val_ratio: float, seed: int):
    rng = random.Random(seed)
    shuffled = rows[:]
    rng.shuffle(shuffled)
    val_n = max(1, int(round(len(shuffled) * val_ratio))) if len(shuffled) > 1 else 0
    val_rows = shuffled[:val_n]
    train_rows = shuffled[val_n:] if val_n else shuffled
    return train_rows, val_rows


def parse_route_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    routes = {item.strip() for item in value.split(",") if item.strip()}
    return routes or None


def filter_rows_by_route(rows: list[dict], include: set[str] | None, exclude: set[str] | None) -> list[dict]:
    out = []
    for row in rows:
        route = str(row.get("route_id", ""))
        if include is not None and route not in include:
            continue
        if exclude is not None and route in exclude:
            continue
        out.append(row)
    return out


def make_sampler(dataset: XModJsonDataset, mode: str):
    labels = []
    routes = []
    for row in dataset.rows:
        labels.append(int(load_json(dataset.root / row["label"])["drive_single_idx"]))
        routes.append(str(row.get("route_id", "?")))
    counts = Counter(labels)
    if mode == "uniform":
        return None, counts
    if mode == "route":
        route_counts = Counter(routes)
        weights = [1.0 / max(route_counts[route], 1) for route in routes]
    elif mode == "label":
        weights = [1.0 / max(counts[label], 1) for label in labels]
    else:
        raise ValueError(f"Unknown sampler mode: {mode}")
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True), counts


def run_epoch(
    model,
    loss_fn,
    loader,
    device,
    optimizer=None,
    lambda_moving: float = 0.0,
    weighted_action_lambda: float = 0.0,
    retention_model=None,
    retention_routes: set[str] | None = None,
    lambda_retention_action: float = 0.0,
    lambda_retention_gate: float = 0.0,
    retention_moving_only: bool = False,
    retention_min_throttle: float = 0.2,
    retention_max_brake: float = 0.5,
    lambda_recovery_brake: float = 0.0,
    lambda_recovery_coactivation: float = 0.0,
    lambda_recovery_throttle_floor: float = 0.0,
    recovery_brake_target: float = 0.02,
    recovery_throttle_floor: float = 0.9,
    lambda_go_brake: float = 0.0,
    lambda_go_coactivation: float = 0.0,
    lambda_go_throttle_floor: float = 0.0,
    go_brake_target: float = 0.02,
    go_throttle_floor: float = 0.9,
    lambda_slow_throttle_ceiling: float = 0.0,
    lambda_slow_brake_floor: float = 0.0,
    slow_throttle_margin: float = 0.05,
    slow_brake_margin: float = 0.05,
):
    train = optimizer is not None
    model.train(train)
    totals = Counter()
    label_counts = Counter()
    gate_sum = torch.zeros(4)
    n = 0
    correct = 0

    for batch in loader:
        image = batch["image"].to(device)
        ego = batch["ego"].to(device)
        gt = batch["action"].to(device)
        drive_single = batch["drive_single_idx"].to(device)
        drive_multihot = batch["drive_multihot"].to(device)
        safety_event = batch["safety_event"].to(device)
        ttc = batch["ttc"].to(device)
        action_weight = batch["action_weight"].to(device)
        hold_frame = batch["hold_frame"].to(device)
        recovery_frame = batch["recovery_frame"].to(device)
        go_frame = batch["go_frame"].to(device)
        slow_frame = batch["slow_frame"].to(device)

        with torch.set_grad_enabled(train):
            pred, gate, experts = model(image, ego, {})
            losses = loss_fn(
                pred_action=pred,
                gt_action=gt,
                gate_probs=gate,
                gate_logits=experts["gate_logits"],
                drive_single_idx=drive_single,
                drive_multihot=drive_multihot,
                safety_event=safety_event,
                ttc=ttc,
            )
            if weighted_action_lambda > 0.0:
                per_sample_action = (pred - gt).pow(2).mean(dim=1)
                weighted_action = (per_sample_action * action_weight).sum() / action_weight.sum().clamp_min(1e-6)
                losses["total"] = losses["total"] + weighted_action_lambda * weighted_action
                losses["action_weighted"] = weighted_action.detach()
            moving_mask = (gt[:, 1] > 0.2) & (gt[:, 2] < 0.5)
            if lambda_moving > 0.0 and moving_mask.any():
                moving_loss = (
                    pred[moving_mask, 2].pow(2).mean()
                    + (1.0 - pred[moving_mask, 1]).pow(2).mean()
                )
                losses["total"] = losses["total"] + lambda_moving * moving_loss
                losses["moving"] = moving_loss.detach()
            else:
                losses["moving"] = torch.zeros((), device=device)
            if retention_model is not None and (lambda_retention_action > 0.0 or lambda_retention_gate > 0.0):
                route_mask = torch.tensor(
                    [
                        retention_routes is None or route in retention_routes
                        for route in batch["route_id"]
                    ],
                    dtype=torch.bool,
                    device=device,
                )
                if retention_moving_only:
                    route_mask = route_mask & (gt[:, 1] >= retention_min_throttle) & (gt[:, 2] <= retention_max_brake)
                if route_mask.any():
                    with torch.no_grad():
                        teacher_pred, teacher_gate, _ = retention_model(image, ego, {})
                    if lambda_retention_action > 0.0:
                        retention_action = (pred[route_mask] - teacher_pred[route_mask]).pow(2).mean()
                        losses["total"] = losses["total"] + lambda_retention_action * retention_action
                        losses["retention_action"] = retention_action.detach()
                    else:
                        losses["retention_action"] = torch.zeros((), device=device)
                    if lambda_retention_gate > 0.0:
                        retention_gate = (gate[route_mask] - teacher_gate[route_mask]).pow(2).mean()
                        losses["total"] = losses["total"] + lambda_retention_gate * retention_gate
                        losses["retention_gate"] = retention_gate.detach()
                    else:
                        losses["retention_gate"] = torch.zeros((), device=device)
                    losses["retention_rate"] = route_mask.float().mean().detach()
                else:
                    losses["retention_action"] = torch.zeros((), device=device)
                    losses["retention_gate"] = torch.zeros((), device=device)
                    losses["retention_rate"] = torch.zeros((), device=device)
            losses["hold_rate"] = hold_frame.mean().detach()
            losses["recovery_rate"] = recovery_frame.mean().detach()
            if recovery_frame.any():
                recovery_mask = recovery_frame > 0.5
                recovery_pred = pred[recovery_mask]
                losses["recovery_action_mae"] = (recovery_pred - gt[recovery_mask]).abs().mean().detach()
                losses["recovery_pred_throttle"] = recovery_pred[:, 1].mean().detach()
                losses["recovery_pred_brake"] = recovery_pred[:, 2].mean().detach()
                recovery_coactivation = (recovery_pred[:, 1].clamp_min(0.0) * recovery_pred[:, 2].clamp_min(0.0)).mean()
                losses["recovery_coactivation"] = recovery_coactivation.detach()
                if lambda_recovery_brake > 0.0:
                    recovery_brake_loss = torch.relu(recovery_pred[:, 2] - recovery_brake_target).pow(2).mean()
                    losses["total"] = losses["total"] + lambda_recovery_brake * recovery_brake_loss
                    losses["recovery_brake"] = recovery_brake_loss.detach()
                else:
                    losses["recovery_brake"] = torch.zeros((), device=device)
                if lambda_recovery_coactivation > 0.0:
                    losses["total"] = losses["total"] + lambda_recovery_coactivation * recovery_coactivation
                if lambda_recovery_throttle_floor > 0.0:
                    recovery_throttle_loss = torch.relu(recovery_throttle_floor - recovery_pred[:, 1]).pow(2).mean()
                    losses["total"] = losses["total"] + lambda_recovery_throttle_floor * recovery_throttle_loss
                    losses["recovery_throttle_floor"] = recovery_throttle_loss.detach()
                else:
                    losses["recovery_throttle_floor"] = torch.zeros((), device=device)
            else:
                losses["recovery_action_mae"] = torch.zeros((), device=device)
                losses["recovery_pred_throttle"] = torch.zeros((), device=device)
                losses["recovery_pred_brake"] = torch.zeros((), device=device)
                losses["recovery_coactivation"] = torch.zeros((), device=device)
                losses["recovery_brake"] = torch.zeros((), device=device)
                losses["recovery_throttle_floor"] = torch.zeros((), device=device)
            losses["go_rate"] = go_frame.mean().detach()
            if go_frame.any():
                go_mask = go_frame > 0.5
                go_pred = pred[go_mask]
                go_coactivation = (go_pred[:, 1].clamp_min(0.0) * go_pred[:, 2].clamp_min(0.0)).mean()
                go_brake_loss = torch.relu(go_pred[:, 2] - go_brake_target).pow(2).mean()
                go_throttle_loss = torch.relu(go_throttle_floor - go_pred[:, 1]).pow(2).mean()
                losses["go_pred_throttle"] = go_pred[:, 1].mean().detach()
                losses["go_pred_brake"] = go_pred[:, 2].mean().detach()
                losses["go_coactivation"] = go_coactivation.detach()
                losses["go_brake"] = go_brake_loss.detach()
                losses["go_throttle_floor"] = go_throttle_loss.detach()
                if lambda_go_brake > 0.0:
                    losses["total"] = losses["total"] + lambda_go_brake * go_brake_loss
                if lambda_go_coactivation > 0.0:
                    losses["total"] = losses["total"] + lambda_go_coactivation * go_coactivation
                if lambda_go_throttle_floor > 0.0:
                    losses["total"] = losses["total"] + lambda_go_throttle_floor * go_throttle_loss
            else:
                losses["go_pred_throttle"] = torch.zeros((), device=device)
                losses["go_pred_brake"] = torch.zeros((), device=device)
                losses["go_coactivation"] = torch.zeros((), device=device)
                losses["go_brake"] = torch.zeros((), device=device)
                losses["go_throttle_floor"] = torch.zeros((), device=device)
            losses["slow_rate"] = slow_frame.mean().detach()
            if slow_frame.any():
                slow_mask = slow_frame > 0.5
                slow_pred = pred[slow_mask]
                slow_gt = gt[slow_mask]
                throttle_ceiling = torch.relu(slow_pred[:, 1] - (slow_gt[:, 1] + slow_throttle_margin)).pow(2).mean()
                brake_floor = torch.relu((slow_gt[:, 2] - slow_brake_margin) - slow_pred[:, 2]).pow(2).mean()
                losses["slow_pred_throttle"] = slow_pred[:, 1].mean().detach()
                losses["slow_pred_brake"] = slow_pred[:, 2].mean().detach()
                losses["slow_gt_throttle"] = slow_gt[:, 1].mean().detach()
                losses["slow_gt_brake"] = slow_gt[:, 2].mean().detach()
                losses["slow_throttle_ceiling"] = throttle_ceiling.detach()
                losses["slow_brake_floor"] = brake_floor.detach()
                if lambda_slow_throttle_ceiling > 0.0:
                    losses["total"] = losses["total"] + lambda_slow_throttle_ceiling * throttle_ceiling
                if lambda_slow_brake_floor > 0.0:
                    losses["total"] = losses["total"] + lambda_slow_brake_floor * brake_floor
            else:
                losses["slow_pred_throttle"] = torch.zeros((), device=device)
                losses["slow_pred_brake"] = torch.zeros((), device=device)
                losses["slow_gt_throttle"] = torch.zeros((), device=device)
                losses["slow_gt_brake"] = torch.zeros((), device=device)
                losses["slow_throttle_ceiling"] = torch.zeros((), device=device)
                losses["slow_brake_floor"] = torch.zeros((), device=device)
            if train:
                optimizer.zero_grad(set_to_none=True)
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        bs = image.shape[0]
        n += bs
        for key, value in losses.items():
            totals[key] += float(value.detach().cpu()) * bs
        pred_gate = gate.argmax(dim=1)
        correct += int((pred_gate == drive_single).sum().detach().cpu())
        gate_sum += gate.detach().cpu().sum(dim=0)
        for label in drive_single.detach().cpu().tolist():
            label_counts[int(label)] += 1

    denom = max(n, 1)
    return {
        "n": n,
        "loss": {key: totals[key] / denom for key in sorted(totals)},
        "gate_acc": correct / denom,
        "gate_mean": [float(x) for x in (gate_sum / denom).tolist()],
        "label_counts": dict(sorted(label_counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description="Train a small X-MoD checkpoint on converted B2D PDM-Lite teacher data.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--xmod-root", default="/mnt/c/xmod_b2d", type=Path)
    parser.add_argument("--out-ckpt", required=True, type=Path)
    parser.add_argument("--init-ckpt", type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260618)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lambda-act", type=float, default=1.0)
    parser.add_argument("--lambda-align", type=float, default=0.5)
    parser.add_argument("--lambda-sparse", type=float, default=0.005)
    parser.add_argument("--lambda-risk", type=float, default=0.05)
    parser.add_argument("--lambda-moving", type=float, default=0.0)
    parser.add_argument("--lambda-recovery-brake", type=float, default=0.0)
    parser.add_argument("--lambda-recovery-coactivation", type=float, default=0.0)
    parser.add_argument("--lambda-recovery-throttle-floor", type=float, default=0.0)
    parser.add_argument("--recovery-brake-target", type=float, default=0.02)
    parser.add_argument("--recovery-throttle-floor", type=float, default=0.9)
    parser.add_argument("--lambda-go-brake", type=float, default=0.0)
    parser.add_argument("--lambda-go-coactivation", type=float, default=0.0)
    parser.add_argument("--lambda-go-throttle-floor", type=float, default=0.0)
    parser.add_argument("--go-brake-target", type=float, default=0.02)
    parser.add_argument("--go-throttle-floor", type=float, default=0.9)
    parser.add_argument("--go-speed-kmh", type=float, default=3.0)
    parser.add_argument("--go-front-distance", type=float, default=10.0)
    parser.add_argument("--go-min-throttle", type=float, default=0.8)
    parser.add_argument("--go-max-brake", type=float, default=0.05)
    parser.add_argument("--lambda-slow-throttle-ceiling", type=float, default=0.0)
    parser.add_argument("--lambda-slow-brake-floor", type=float, default=0.0)
    parser.add_argument("--slow-throttle-margin", type=float, default=0.05)
    parser.add_argument("--slow-brake-margin", type=float, default=0.05)
    parser.add_argument("--slow-front-distance", type=float, default=10.0)
    parser.add_argument("--slow-min-brake", type=float, default=0.5)
    parser.add_argument("--slow-max-throttle", type=float, default=0.2)
    parser.add_argument("--retention-ckpt", type=Path, help="Optional checkpoint to distill selected route states from.")
    parser.add_argument("--retention-routes", help="Comma-separated route IDs for retention distillation.")
    parser.add_argument("--lambda-retention-action", type=float, default=0.0)
    parser.add_argument("--lambda-retention-gate", type=float, default=0.0)
    parser.add_argument("--retention-moving-only", action="store_true")
    parser.add_argument("--retention-min-throttle", type=float, default=0.2)
    parser.add_argument("--retention-max-brake", type=float, default=0.5)
    parser.add_argument(
        "--hold-action-weight",
        type=float,
        default=1.0,
        help="If below 1, downweight low-speed hard-brake hold/stop frames in the action loss.",
    )
    parser.add_argument(
        "--recovery-action-weight",
        type=float,
        default=1.0,
        help="Multiply action loss weight for rows marked with partial.recovery_label or recovery_event.",
    )
    parser.add_argument("--hold-speed-kmh", type=float, default=2.0)
    parser.add_argument("--hold-brake-threshold", type=float, default=0.5)
    parser.add_argument("--hold-front-distance", type=float, default=10.0)
    parser.add_argument("--include-routes", help="Comma-separated route IDs to keep before train/val split.")
    parser.add_argument("--exclude-routes", help="Comma-separated route IDs to drop before train/val split.")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-router", action="store_true")
    parser.add_argument("--sampler-mode", choices=["label", "route", "uniform"], default="label")
    parser.add_argument("--ego-mode", choices=["base", "route_v1", "route_v2", "route_v3", "route_v4"], default="base")
    parser.add_argument("--model-arch", choices=["moe", "separated", "targetspeed"], default="moe")
    parser.add_argument("--router-hidden", type=int, default=128)
    parser.add_argument("--expert-hidden", type=int, default=64)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--save-init-only", action="store_true")
    parser.add_argument("--train-extra-columns-only", action="store_true")
    parser.add_argument(
        "--train-extra-columns-scope",
        choices=[
            "all",
            "router",
            "experts",
            "expert_safety",
            "expert_legality",
            "expert_comfort",
            "expert_efficiency",
        ],
        default="all",
    )
    parser.add_argument(
        "--train-extra-columns-from-dim",
        type=int,
        default=0,
        help="Explicitly train only input columns after this dimension, useful when continuing from an already widened checkpoint.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    sys.path.insert(0, str(args.xmod_root))
    from models import XMoDLoss, XMoDVLA

    rows = [
        json.loads(line)
        for line in (args.data_root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    include_routes = parse_route_set(args.include_routes)
    exclude_routes = parse_route_set(args.exclude_routes)
    rows = filter_rows_by_route(rows, include_routes, exclude_routes)
    if not rows:
        raise RuntimeError("No rows in metadata.jsonl")

    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
    dataset_kwargs = {
        "hold_action_weight": args.hold_action_weight,
        "recovery_action_weight": args.recovery_action_weight,
        "hold_speed_kmh": args.hold_speed_kmh,
        "hold_brake_threshold": args.hold_brake_threshold,
        "hold_front_distance": args.hold_front_distance,
        "go_speed_kmh": args.go_speed_kmh,
        "go_front_distance": args.go_front_distance,
        "go_min_throttle": args.go_min_throttle,
        "go_max_brake": args.go_max_brake,
        "slow_front_distance": args.slow_front_distance,
        "slow_min_brake": args.slow_min_brake,
        "slow_max_throttle": args.slow_max_throttle,
        "target_speed_mode": (args.model_arch == "targetspeed"),
    }
    train_ds = XModJsonDataset(
        args.data_root,
        train_rows,
        args.image_size,
        ego_mode=args.ego_mode,
        cache=args.cache,
        **dataset_kwargs,
    )
    val_ds = XModJsonDataset(
        args.data_root,
        val_rows,
        args.image_size,
        ego_mode=args.ego_mode,
        cache=args.cache,
        **dataset_kwargs,
    )
    sampler, train_label_counts = make_sampler(train_ds, args.sampler_mode)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        collate_fn=collate,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    device = torch.device(args.device)
    ego_dim = ego_dim_for_mode(args.ego_mode)
    model = XMoDVLA(
        backbone_name="tinycnn",
        expert_input_mode="full",
        ego_dim=ego_dim,
        router_hidden=args.router_hidden,
        expert_hidden=args.expert_hidden,
        architecture=args.model_arch,
    ).to(device)
    if args.init_ckpt:
        missing, unexpected, widened, skipped = load_compatible_checkpoint(model, args.init_ckpt)
        print(
            f"init_ckpt={args.init_ckpt} missing={len(missing)} unexpected={len(unexpected)} "
            f"widened={widened} skipped={skipped[:6]}",
            flush=True,
        )
    retention_model = None
    retention_routes = parse_route_set(args.retention_routes)
    if args.retention_ckpt:
        retention_model = XMoDVLA(
            backbone_name="tinycnn",
            expert_input_mode="full",
            ego_dim=ego_dim,
            router_hidden=args.router_hidden,
            expert_hidden=args.expert_hidden,
            architecture=args.model_arch,
        ).to(device)
        missing, unexpected, widened, skipped = load_compatible_checkpoint(retention_model, args.retention_ckpt)
        for param in retention_model.parameters():
            param.requires_grad = False
        retention_model.eval()
        print(
            f"retention_ckpt={args.retention_ckpt} routes={sorted(retention_routes) if retention_routes else 'all'} "
            f"lambda_retention_action={args.lambda_retention_action} "
            f"lambda_retention_gate={args.lambda_retention_gate} "
            f"retention_moving_only={args.retention_moving_only} "
            f"missing={len(missing)} unexpected={len(unexpected)} widened={widened} skipped={skipped[:6]}",
            flush=True,
        )
    if args.freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False
    if args.freeze_router:
        for param in model.router.parameters():
            param.requires_grad = False
    extra_column_train = {}
    if args.train_extra_columns_only:
        source_dim_override = args.train_extra_columns_from_dim or None
        extra_column_train = enable_extra_column_training_only(model, args.train_extra_columns_scope, source_dim_override)
        args.weight_decay = 0.0

    use_weighted_action = (
        abs(args.hold_action_weight - 1.0) > 1e-6
        or abs(args.recovery_action_weight - 1.0) > 1e-6
    )
    weighted_action_lambda = args.lambda_act if use_weighted_action else 0.0
    loss_fn = XMoDLoss(
        lambda_act=0.0 if use_weighted_action else args.lambda_act,
        lambda_align=args.lambda_align,
        lambda_sparse=args.lambda_sparse,
        lambda_risk=args.lambda_risk,
        router_loss="ce",
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters remain after freeze options")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best = None
    best_epoch = -1
    history = []
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    trainable_count = sum(param.numel() for param in model.parameters() if param.requires_grad)
    frozen_count = sum(param.numel() for param in model.parameters() if not param.requires_grad)
    route_counts = Counter(str(row.get("route_id", "?")) for row in rows)
    print(
        f"device={device} rows={len(rows)} train={len(train_rows)} val={len(val_rows)} "
        f"ego_mode={args.ego_mode} ego_dim={ego_dim} model_arch={args.model_arch} "
        f"router_hidden={args.router_hidden} expert_hidden={args.expert_hidden} "
        f"routes={dict(sorted(route_counts.items()))} "
        f"sampler_mode={args.sampler_mode} "
        f"hold_action_weight={args.hold_action_weight} "
        f"recovery_action_weight={args.recovery_action_weight} "
        f"lambda_recovery_brake={args.lambda_recovery_brake} "
        f"lambda_recovery_coactivation={args.lambda_recovery_coactivation} "
        f"lambda_recovery_throttle_floor={args.lambda_recovery_throttle_floor} "
        f"lambda_go_brake={args.lambda_go_brake} "
        f"lambda_go_coactivation={args.lambda_go_coactivation} "
        f"lambda_go_throttle_floor={args.lambda_go_throttle_floor} "
        f"lambda_slow_throttle_ceiling={args.lambda_slow_throttle_ceiling} "
        f"lambda_slow_brake_floor={args.lambda_slow_brake_floor} "
        f"train_label_counts={dict(sorted(train_label_counts.items()))} "
        f"trainable_params={trainable_count} frozen_params={frozen_count}",
        flush=True,
    )
    if extra_column_train:
        print(f"train_extra_columns_only={extra_column_train} scope={args.train_extra_columns_scope}", flush=True)
    if args.save_init_only:
        train_metrics = run_epoch(
            model,
            loss_fn,
            train_loader,
            device,
            None,
            args.lambda_moving,
            weighted_action_lambda,
            retention_model,
            retention_routes,
            args.lambda_retention_action,
            args.lambda_retention_gate,
            args.retention_moving_only,
            args.retention_min_throttle,
            args.retention_max_brake,
            args.lambda_recovery_brake,
            args.lambda_recovery_coactivation,
            args.lambda_recovery_throttle_floor,
            args.recovery_brake_target,
            args.recovery_throttle_floor,
            args.lambda_go_brake,
            args.lambda_go_coactivation,
            args.lambda_go_throttle_floor,
            args.go_brake_target,
            args.go_throttle_floor,
            args.lambda_slow_throttle_ceiling,
            args.lambda_slow_brake_floor,
            args.slow_throttle_margin,
            args.slow_brake_margin,
        )
        val_metrics = run_epoch(
            model,
            loss_fn,
            val_loader,
            device,
            None,
            args.lambda_moving,
            weighted_action_lambda,
            retention_model,
            retention_routes,
            args.lambda_retention_action,
            args.lambda_retention_gate,
            args.retention_moving_only,
            args.retention_min_throttle,
            args.retention_max_brake,
            args.lambda_recovery_brake,
            args.lambda_recovery_coactivation,
            args.lambda_recovery_throttle_floor,
            args.recovery_brake_target,
            args.recovery_throttle_floor,
            args.lambda_go_brake,
            args.lambda_go_coactivation,
            args.lambda_go_throttle_floor,
            args.go_brake_target,
            args.go_throttle_floor,
            args.lambda_slow_throttle_ceiling,
            args.lambda_slow_brake_floor,
            args.slow_throttle_margin,
            args.slow_brake_margin,
        )
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": 0,
                "val_total": val_metrics["loss"].get("total", float("inf")),
                "args": serializable_args,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            args.out_ckpt,
        )
        summary = {
            "best_epoch": 0,
            "best_val_total": val_metrics["loss"].get("total", float("inf")),
            "out_ckpt": str(args.out_ckpt),
            "rows": len(rows),
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
            "final": {"epoch": 0, "train": train_metrics, "val": val_metrics},
            "history": [{"epoch": 0, "train": train_metrics, "val": val_metrics}],
        }
        summary_path = args.out_ckpt.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print("save_init_only=true", flush=True)
        print(f"saved={args.out_ckpt}", flush=True)
        print(f"summary={summary_path}", flush=True)
        return

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            loss_fn,
            train_loader,
            device,
            optimizer,
            args.lambda_moving,
            weighted_action_lambda,
            retention_model,
            retention_routes,
            args.lambda_retention_action,
            args.lambda_retention_gate,
            args.retention_moving_only,
            args.retention_min_throttle,
            args.retention_max_brake,
            args.lambda_recovery_brake,
            args.lambda_recovery_coactivation,
            args.lambda_recovery_throttle_floor,
            args.recovery_brake_target,
            args.recovery_throttle_floor,
            args.lambda_go_brake,
            args.lambda_go_coactivation,
            args.lambda_go_throttle_floor,
            args.go_brake_target,
            args.go_throttle_floor,
            args.lambda_slow_throttle_ceiling,
            args.lambda_slow_brake_floor,
            args.slow_throttle_margin,
            args.slow_brake_margin,
        )
        val_metrics = run_epoch(
            model,
            loss_fn,
            val_loader,
            device,
            None,
            args.lambda_moving,
            weighted_action_lambda,
            retention_model,
            retention_routes,
            args.lambda_retention_action,
            args.lambda_retention_gate,
            args.retention_moving_only,
            args.retention_min_throttle,
            args.retention_max_brake,
            args.lambda_recovery_brake,
            args.lambda_recovery_coactivation,
            args.lambda_recovery_throttle_floor,
            args.recovery_brake_target,
            args.recovery_throttle_floor,
            args.lambda_go_brake,
            args.lambda_go_coactivation,
            args.lambda_go_throttle_floor,
            args.go_brake_target,
            args.go_throttle_floor,
            args.lambda_slow_throttle_ceiling,
            args.lambda_slow_brake_floor,
            args.slow_throttle_margin,
            args.slow_brake_margin,
        )
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(record)
        val_total = val_metrics["loss"].get("total", float("inf"))
        if best is None or val_total < best:
            best = val_total
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_total": best,
                    "args": serializable_args,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                },
                args.out_ckpt,
            )
        if epoch == 1 or epoch == args.epochs or epoch % 5 == 0:
            print(
                "epoch=%03d train_total=%.6f val_total=%.6f val_action=%.6f val_gate_acc=%.3f val_gate_mean=%s"
                % (
                    epoch,
                    train_metrics["loss"]["total"],
                    val_metrics["loss"]["total"],
                    val_metrics["loss"]["action"],
                    val_metrics["gate_acc"],
                    "[" + ",".join(f"{x:.3f}" for x in val_metrics["gate_mean"]) + "]",
                ),
                flush=True,
            )

    summary_path = args.out_ckpt.with_suffix(".summary.json")
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best,
        "out_ckpt": str(args.out_ckpt),
        "rows": len(rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "final": history[-1],
        "history": history,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"best_epoch={best_epoch} best_val_total={best:.6f}", flush=True)
    print(f"saved={args.out_ckpt}", flush=True)
    print(f"summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
