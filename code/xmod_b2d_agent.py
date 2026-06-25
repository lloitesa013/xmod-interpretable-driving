import json
import os, math, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import carla
import cv2
from leaderboard.autoagents import autonomous_agent
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

sys.path.insert(0, "/mnt/c/xmod_b2d")
from models import XMoDVLA
from models.x_mod_vla import V_MAX_MPS  # X-MoD v2 target-speed denormalization cap (m/s)

ROUTE_V1_EGO_DIM = 29
ROUTE_V2_EGO_DIM = 38
ROUTE_V3_EGO_DIM = 41
ROUTE_V4_EGO_DIM = 43
COMMAND_CLASSES = 7

def get_entry_point():
    return "XMoDAgent"

def _norm(a):
    while a > math.pi: a -= 2*math.pi
    while a < -math.pi: a += 2*math.pi
    return a

def _command_id(value):
    try:
        command = int(float(value))
    except (TypeError, ValueError):
        command = 4
    return max(0, min(COMMAND_CLASSES - 1, command))

def _command_onehot(value):
    command = _command_id(value)
    out = [0.0] * COMMAND_CLASSES
    out[command] = 1.0
    return out

def _ego_dim_for_mode(mode):
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
    raise ValueError("Unknown XMOD_EGO_MODE: %s" % mode)

def _turn_command(value):
    return 0.0 if _command_id(value) == 4 else 1.0

def _clamp_signed(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, value))

def _load_compatible_checkpoint(model, raw_state):
    source = raw_state.get("model", raw_state) if isinstance(raw_state, dict) else raw_state
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
            continue
        skipped.append((key, tuple(value.shape), tuple(target[key].shape)))
    missing, unexpected = model.load_state_dict(patched, strict=False)
    return missing, unexpected, widened, skipped


class FrontInteractionHead(nn.Module):
    def __init__(self, input_dim=ROUTE_V1_EGO_DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, ego):
        x = ego.clone()
        x[:, 0] = x[:, 0] / 50.0
        x[:, 1] = x[:, 1] / math.pi
        raw = self.net(x)
        steer = torch.tanh(raw[:, 0:1])
        throttle = torch.sigmoid(raw[:, 1:2])
        brake = torch.sigmoid(raw[:, 2:3])
        return torch.cat([steer, throttle, brake], dim=1)


class SourceBrakeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.image = nn.Sequential(
            nn.Conv2d(3, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.ego = nn.Sequential(nn.Linear(2, 16), nn.ReLU(inplace=True), nn.Linear(16, 16), nn.ReLU(inplace=True))
        self.shared = nn.Sequential(nn.Linear(112, 64), nn.ReLU(inplace=True))
        self.head = nn.Linear(64, 3)

    def forward(self, image, ego):
        image_feat = self.image(image).flatten(1)
        ego_norm = ego.clone()
        ego_norm[:, 0] = ego_norm[:, 0] / 50.0
        ego_norm[:, 1] = ego_norm[:, 1] / math.pi
        ego_feat = self.ego(ego_norm)
        return self.head(self.shared(torch.cat([image_feat, ego_feat], dim=1)))


class XMoDAgent(autonomous_agent.AutonomousAgent):
    def setup(self, path_to_conf_file, route_index=None, traffic_manager=None):
        self.track = autonomous_agent.Track.SENSORS
        self.image_size = 128
        ckpt = os.environ.get("XMOD_CKPT", "/mnt/c/xmod_b2d/ckpt_full.pt")
        sd = torch.load(ckpt, map_location="cpu")
        ckpt_args = sd.get("args", {}) if isinstance(sd, dict) else {}
        self.ego_mode = os.environ.get("XMOD_EGO_MODE") or ckpt_args.get("ego_mode", "base")
        self.ego_dim = _ego_dim_for_mode(self.ego_mode)
        self.model_arch = os.environ.get("XMOD_MODEL_ARCH") or ckpt_args.get("model_arch", "moe")
        self.router_hidden = int(os.environ.get("XMOD_ROUTER_HIDDEN") or ckpt_args.get("router_hidden", 128))
        self.expert_hidden = int(os.environ.get("XMOD_EXPERT_HIDDEN") or ckpt_args.get("expert_hidden", 64))
        self.model = XMoDVLA(
            backbone_name="tinycnn",
            expert_input_mode="full",
            ego_dim=self.ego_dim,
            router_hidden=self.router_hidden,
            expert_hidden=self.expert_hidden,
            architecture=self.model_arch,
        )
        missing, unexpected, widened, skipped = _load_compatible_checkpoint(self.model, sd)
        print(
            "[XMoD] loaded ckpt=%s ego_mode=%s ego_dim=%d model_arch=%s router_hidden=%d expert_hidden=%d missing=%d unexpected=%d widened=%s skipped=%s"
            % (
                ckpt,
                self.ego_mode,
                self.ego_dim,
                self.model_arch,
                self.router_hidden,
                self.expert_hidden,
                len(missing),
                len(unexpected),
                widened,
                skipped[:4],
            ),
            flush=True,
        )
        self.model.eval()
        self._vehicle = None
        self.step = -1
        self._route_progress_m = 0.0
        self._last_progress_loc = None
        self.enable_safety_shield = os.environ.get("XMOD_SAFETY_SHIELD", "0") == "1"
        self.enable_recovery_shield = os.environ.get("XMOD_RECOVERY_SHIELD", "0") == "1"
        self.safety_distance = float(os.environ.get("XMOD_SAFETY_DISTANCE", "9.0"))
        self.hard_brake_distance = float(os.environ.get("XMOD_HARD_BRAKE_DISTANCE", "5.5"))
        self.recovery_after = int(os.environ.get("XMOD_RECOVERY_AFTER", "35"))
        # X-MoD v2: target-speed action head -> classical PID, with §4.2 deficit floor.
        self.targetspeed_mode = (self.model_arch == "targetspeed") or os.environ.get("XMOD_TARGETSPEED", "0") == "1"
        self.pid = None
        if self.targetspeed_mode:
            from longitudinal_controller import LongitudinalPID
            self.pid = LongitudinalPID()
            self.stall_deficit = 0.0
            self.deficit_floor_enable = os.environ.get("XMOD_DEFICIT_FLOOR", "1") == "1"
            self.deficit_eps_mps = float(os.environ.get("XMOD_DEFICIT_EPS_MPS", "0.5"))
            self.deficit_gain = float(os.environ.get("XMOD_DEFICIT_GAIN", "0.3"))
            self.deficit_cap_mps = float(os.environ.get("XMOD_DEFICIT_CAP_MPS", "3.0"))
            self.deficit_front_margin = float(os.environ.get("XMOD_DEFICIT_FRONT_MARGIN", "6.0"))
            self.deficit_warmup_s = float(os.environ.get("XMOD_DEFICIT_WARMUP_S", "1.0"))
            self.ts_safety = os.environ.get("XMOD_TS_SAFETY", "1") == "1"
            print("[XMoD] v2 target-speed PID mode ON: V_MAX=%.1f deficit_floor=%s gain=%.2f cap=%.1f margin=%.1f warmup=%.1f" % (
                V_MAX_MPS, self.deficit_floor_enable, self.deficit_gain, self.deficit_cap_mps, self.deficit_front_margin, self.deficit_warmup_s), flush=True)
        self.recovery_throttle = float(os.environ.get("XMOD_RECOVERY_THROTTLE", "0.75"))
        self.trace_dir = os.environ.get("XMOD_TRACE_DIR", "")
        self.trace_every = int(os.environ.get("XMOD_TRACE_EVERY", "25"))
        self.trace_stuck_after = int(os.environ.get("XMOD_TRACE_STUCK_AFTER", "120"))
        self.trace_hazard_distance = float(os.environ.get("XMOD_TRACE_HAZARD_DISTANCE", "10.0"))
        self.trace_hazard_every = int(os.environ.get("XMOD_TRACE_HAZARD_EVERY", "5"))
        self.trace_limit = int(os.environ.get("XMOD_TRACE_LIMIT", "240"))
        self.trace_save_images = os.environ.get("XMOD_TRACE_SAVE_IMAGES", "1") == "1"
        self.interaction_enable = os.environ.get("XMOD_INTERACTION_ENABLE", "0") == "1"
        self.interaction_ckpt = os.environ.get("XMOD_INTERACTION_CKPT", "")
        self.interaction_min_distance = float(os.environ.get("XMOD_INTERACTION_MIN_DISTANCE", "0.0"))
        self.interaction_max_distance = float(os.environ.get("XMOD_INTERACTION_MAX_DISTANCE", "22.0"))
        self.interaction_blend = float(os.environ.get("XMOD_INTERACTION_BLEND", "0.55"))
        self.interaction_vehicle_only = os.environ.get("XMOD_INTERACTION_VEHICLE_ONLY", "1") == "1"
        self.interaction_max_brake = float(os.environ.get("XMOD_INTERACTION_MAX_BRAKE", "0.75"))
        self.interaction_min_throttle = float(os.environ.get("XMOD_INTERACTION_MIN_THROTTLE", "0.0"))
        self.front_recovery_enable = os.environ.get("XMOD_FRONT_RECOVERY_ENABLE", "0") == "1"
        self.front_recovery_after = int(os.environ.get("XMOD_FRONT_RECOVERY_AFTER", "90"))
        self.front_recovery_distance = float(os.environ.get("XMOD_FRONT_RECOVERY_DISTANCE", "6.5"))
        self.front_recovery_reverse_steps = int(os.environ.get("XMOD_FRONT_RECOVERY_REVERSE_STEPS", "28"))
        self.front_recovery_forward_steps = int(os.environ.get("XMOD_FRONT_RECOVERY_FORWARD_STEPS", "45"))
        self.front_recovery_reverse_throttle = float(os.environ.get("XMOD_FRONT_RECOVERY_REVERSE_THROTTLE", "0.45"))
        self.front_recovery_forward_throttle = float(os.environ.get("XMOD_FRONT_RECOVERY_FORWARD_THROTTLE", "0.70"))
        self.front_recovery_steer = float(os.environ.get("XMOD_FRONT_RECOVERY_STEER", "0.70"))
        self.source_brake_enable = os.environ.get("XMOD_SOURCE_BRAKE_ENABLE", "0") == "1"
        self.source_brake_control = os.environ.get("XMOD_SOURCE_BRAKE_CONTROL", "0") == "1"
        self.source_brake_ckpt = os.environ.get("XMOD_SOURCE_BRAKE_CKPT", "")
        self.source_brake_mode = os.environ.get("XMOD_SOURCE_BRAKE_MODE", "object")
        self.source_brake_near_threshold = float(os.environ.get("XMOD_SOURCE_BRAKE_NEAR_THRESHOLD", "0.90"))
        self.source_brake_object_threshold = float(os.environ.get("XMOD_SOURCE_BRAKE_OBJECT_THRESHOLD", "0.85"))
        self.source_brake_route_threshold = float(os.environ.get("XMOD_SOURCE_BRAKE_ROUTE_THRESHOLD", "0.95"))
        self.source_brake_brake = float(os.environ.get("XMOD_SOURCE_BRAKE_BRAKE", "0.55"))
        self.source_brake_max_throttle = float(os.environ.get("XMOD_SOURCE_BRAKE_MAX_THROTTLE", "0.15"))
        self.interaction_head = None
        self.source_brake_head = None
        if self.interaction_enable and self.interaction_ckpt:
            raw_interaction = torch.load(self.interaction_ckpt, map_location="cpu")
            hidden = int(raw_interaction.get("args", {}).get("hidden", 64)) if isinstance(raw_interaction, dict) else 64
            self.interaction_head = FrontInteractionHead(hidden=hidden)
            source = raw_interaction.get("model", raw_interaction) if isinstance(raw_interaction, dict) else raw_interaction
            self.interaction_head.load_state_dict(source)
            self.interaction_head.eval()
            print(
                "[XMoD] interaction head loaded ckpt=%s hidden=%d distance=[%.1f,%.1f] blend=%.2f vehicle_only=%s"
                % (
                    self.interaction_ckpt,
                    hidden,
                    self.interaction_min_distance,
                    self.interaction_max_distance,
                    self.interaction_blend,
                    self.interaction_vehicle_only,
                ),
                flush=True,
            )
        if self.source_brake_enable and self.source_brake_ckpt:
            raw_source_brake = torch.load(self.source_brake_ckpt, map_location="cpu")
            self.source_brake_head = SourceBrakeHead()
            source = raw_source_brake.get("model", raw_source_brake) if isinstance(raw_source_brake, dict) else raw_source_brake
            self.source_brake_head.load_state_dict(source)
            self.source_brake_head.eval()
            print(
                "[XMoD] source_brake loaded ckpt=%s control=%s mode=%s thresholds=(near %.2f object %.2f route %.2f)"
                % (
                    self.source_brake_ckpt,
                    self.source_brake_control,
                    self.source_brake_mode,
                    self.source_brake_near_threshold,
                    self.source_brake_object_threshold,
                    self.source_brake_route_threshold,
                ),
                flush=True,
            )
        self._trace_count = 0
        self._trace_handle = None
        if self.trace_dir:
            trace_path = Path(self.trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            if self.trace_save_images:
                (trace_path / "images").mkdir(parents=True, exist_ok=True)
            self._trace_handle = (trace_path / "trace.jsonl").open("a", encoding="utf-8")
            print(
                "[XMoD] trace dir=%s every=%d stuck_after=%d hazard_distance=%.1f hazard_every=%d limit=%d save_images=%s"
                % (
                    self.trace_dir,
                    self.trace_every,
                    self.trace_stuck_after,
                    self.trace_hazard_distance,
                    self.trace_hazard_every,
                    self.trace_limit,
                    self.trace_save_images,
                ),
                flush=True,
            )
        print(
            "[XMoD] shields safety=%s recovery=%s safety_distance=%.1f hard_brake_distance=%.1f recovery_after=%d"
            % (
                self.enable_safety_shield,
                self.enable_recovery_shield,
                self.safety_distance,
                self.hard_brake_distance,
                self.recovery_after,
            ),
            flush=True,
        )
        print(
            "[XMoD] front_recovery enable=%s after=%d distance=%.1f reverse_steps=%d forward_steps=%d"
            % (
                self.front_recovery_enable,
                self.front_recovery_after,
                self.front_recovery_distance,
                self.front_recovery_reverse_steps,
                self.front_recovery_forward_steps,
            ),
            flush=True,
        )

    def _maybe_trace(self, img, speed_kmh, wp_angle, steer, throttle, brake, gate, hazard, ego_tf):
        if self._trace_handle is None or self._trace_count >= self.trace_limit:
            return
        stuck = getattr(self, "_stuck", 0)
        trace_hazard = hazard
        front_actor = None
        if ego_tf is not None:
            front_actor = self._front_actor_proxy(
                ego_tf,
                max_dist=max(self.safety_distance, self.hard_brake_distance, self.trace_hazard_distance, 12.0),
            )
            if trace_hazard is None and front_actor is not None:
                trace_hazard = (front_actor["distance"], front_actor["type_id"])
        hazard_near = (
            trace_hazard is not None
            and trace_hazard[0] <= self.trace_hazard_distance
            and self.step % max(self.trace_hazard_every, 1) == 0
        )
        if self.step >= 3 and self.step % max(self.trace_every, 1) != 0 and stuck < self.trace_stuck_after and not hazard_near:
            return
        image_rel = None
        if self.trace_save_images:
            image_rel = "images/step_%06d.png" % self.step
            image_path = Path(self.trace_dir) / image_rel
            cv2.imwrite(str(image_path), img[:, :, ::-1])
        pose = None
        if ego_tf is not None:
            pose = {
                "x": float(ego_tf.location.x),
                "y": float(ego_tf.location.y),
                "z": float(ego_tf.location.z),
                "yaw": float(ego_tf.rotation.yaw),
            }
        payload = {
            "step": int(self.step),
            "speed_kmh": float(speed_kmh),
            "stuck": int(stuck),
            "wp_angle": float(wp_angle),
            "action": {
                "steer": float(steer),
                "throttle": float(throttle),
                "brake": float(brake),
            },
            "gate": [float(x) for x in gate[0].tolist()],
            "gate_argmax": int(torch.argmax(gate[0]).item()),
            "hazard": None if trace_hazard is None else {
                "distance": float(trace_hazard[0]),
                "type_id": str(trace_hazard[1]),
            },
            "interaction": getattr(self, "_last_interaction", None),
            "source_brake": getattr(self, "_last_source_brake", None),
            "front_recovery": getattr(self, "_last_front_recovery", None),
            "front_actor": front_actor,
            "pose": pose,
            "image": image_rel,
        }
        self._trace_handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        self._trace_handle.flush()
        self._trace_count += 1

    def sensors(self):
        return [
            {"type": "sensor.camera.rgb", "id": "front", "x": 1.5, "y": 0.0, "z": 2.2,
             "roll": 0.0, "pitch": 0.0, "yaw": 0.0, "width": 256, "height": 256, "fov": 90},
            {"type": "sensor.speedometer", "id": "speed"},
        ]

    def _veh(self):
        if self._vehicle is None:
            self._vehicle = CarlaDataProvider.get_hero_actor()
        return self._vehicle

    def _route_candidates(self, ego_tf):
        plan = getattr(self, "org_dense_route_world_coord", None) or getattr(self, "_global_plan_world_coord", None)
        if not plan:
            return []
        ego = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)
        fwdx, fwdy = math.cos(yaw), math.sin(yaw)
        candidates = []
        for item in plan:
            tr = item[0]
            loc = tr.location if hasattr(tr, "location") else tr
            dx, dy = loc.x - ego.x, loc.y - ego.y
            if dx * fwdx + dy * fwdy <= 0:   # behind ego
                continue
            d = math.hypot(dx, dy)
            command = item[1] if len(item) > 1 else 4
            candidates.append((d, dx, dy, command))
        return candidates

    def _route_angle(self, ego_tf, lookahead=8.0):
        chosen = None
        for d, dx, dy, _ in self._route_candidates(ego_tf):
            chosen = (dx, dy)
            if d >= lookahead:
                break
        if chosen is None:
            return 0.0
        yaw = math.radians(ego_tf.rotation.yaw)
        return _norm(math.atan2(chosen[1], chosen[0]) - yaw)

    def _route_command_pair(self, ego_tf):
        candidates = self._route_candidates(ego_tf)
        if not candidates:
            return 4, 4, 0
        current = candidates[0][3]
        future = candidates[-1][3]
        for d, _, _, command in candidates:
            if d >= 16.0:
                future = command
                break
        return self._road_option_value(current), self._road_option_value(future), len(candidates)

    def _road_option_value(self, command):
        if hasattr(command, "value"):
            return command.value
        return command

    def _front_actor_proxy(self, ego_tf, max_dist=50.0):
        veh = self._veh()
        if veh is None:
            return None
        world = veh.get_world()
        if world is None:
            return None
        ego = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)
        fwdx, fwdy = math.cos(yaw), math.sin(yaw)
        rightx, righty = -math.sin(yaw), math.cos(yaw)
        nearest = None
        for actor in world.get_actors():
            if actor.id == veh.id:
                continue
            type_id = getattr(actor, "type_id", "")
            if not (type_id.startswith("vehicle.") or type_id.startswith("walker.pedestrian.")):
                continue
            loc = actor.get_location()
            dx, dy = loc.x - ego.x, loc.y - ego.y
            forward = dx * fwdx + dy * fwdy
            if forward <= 0.0 or forward > max_dist:
                continue
            lateral = abs(dx * rightx + dy * righty)
            lane_width = max(2.0, 0.25 * forward + 1.2)
            if lateral > lane_width:
                continue
            dist = math.hypot(dx, dy)
            if nearest is None or dist < nearest["distance"]:
                tf = actor.get_transform()
                vel = actor.get_velocity()
                extent = getattr(getattr(actor, "bounding_box", None), "extent", None)
                nearest = {
                    "actor_id": int(actor.id),
                    "distance": dist,
                    "forward": forward,
                    "lateral": dx * rightx + dy * righty,
                    "type_id": type_id,
                    "pose": {
                        "x": float(tf.location.x),
                        "y": float(tf.location.y),
                        "z": float(tf.location.z),
                        "yaw": float(tf.rotation.yaw),
                    },
                    "velocity": {
                        "x": float(vel.x),
                        "y": float(vel.y),
                        "z": float(vel.z),
                    },
                }
                if extent is not None:
                    nearest["extent"] = {
                        "x": float(extent.x),
                        "y": float(extent.y),
                        "z": float(extent.z),
                    }
        return nearest

    def _front_hazard(self, ego_tf, max_dist=12.0):
        proxy = self._front_actor_proxy(ego_tf, max_dist=max_dist)
        if proxy is None:
            return None
        return (proxy["distance"], proxy["type_id"])

    def _junction_proxy(self, ego_tf):
        veh = self._veh()
        if veh is None or ego_tf is None:
            return 0.0
        try:
            waypoint = veh.get_world().get_map().get_waypoint(ego_tf.location)
            return 1.0 if waypoint is not None and waypoint.is_junction else 0.0
        except Exception:
            return 0.0

    def _junction_ahead_proxy(self, ego_tf, lookahead=35.0):
        veh = self._veh()
        if veh is None or ego_tf is None:
            return 0.0, 999.0
        world_map = veh.get_world().get_map()
        nearest = None
        for d, dx, dy, _ in self._route_candidates(ego_tf):
            if d > lookahead:
                break
            try:
                loc = carla.Location(x=ego_tf.location.x + dx, y=ego_tf.location.y + dy, z=ego_tf.location.z)
                waypoint = world_map.get_waypoint(loc)
            except Exception:
                continue
            if waypoint is not None and waypoint.is_junction:
                nearest = d
                break
        if nearest is None:
            return 0.0, 999.0
        return 1.0, float(nearest)

    def _distance_to_actor_ahead(self, ego_tf, match_fn, max_dist=80.0):
        veh = self._veh()
        if veh is None or ego_tf is None:
            return 999.0
        world = veh.get_world()
        ego = ego_tf.location
        yaw = math.radians(ego_tf.rotation.yaw)
        fwdx, fwdy = math.cos(yaw), math.sin(yaw)
        rightx, righty = -math.sin(yaw), math.cos(yaw)
        nearest = None
        for actor in world.get_actors():
            if actor.id == veh.id:
                continue
            type_id = getattr(actor, "type_id", "")
            if not match_fn(type_id):
                continue
            loc = actor.get_location()
            dx, dy = loc.x - ego.x, loc.y - ego.y
            forward = dx * fwdx + dy * fwdy
            if forward <= 0.0 or forward > max_dist:
                continue
            lateral = abs(dx * rightx + dy * righty)
            lane_width = max(4.0, 0.30 * forward + 2.5)
            if lateral > lane_width:
                continue
            dist = math.hypot(dx, dy)
            if nearest is None or dist < nearest:
                nearest = dist
        return 999.0 if nearest is None else float(nearest)

    def _stopline_proxies(self, ego_tf):
        traffic_light = self._distance_to_actor_ahead(
            ego_tf,
            lambda type_id: "traffic_light" in str(type_id),
            max_dist=80.0,
        )
        stop_sign = self._distance_to_actor_ahead(
            ego_tf,
            lambda type_id: "stop" in str(type_id).lower(),
            max_dist=80.0,
        )
        return traffic_light, stop_sign, min(traffic_light, stop_sign)

    def _update_route_progress(self, ego_tf):
        if ego_tf is None:
            return float(getattr(self, "_route_progress_m", 0.0))
        loc = ego_tf.location
        last = getattr(self, "_last_progress_loc", None)
        if last is not None:
            delta = math.hypot(float(loc.x) - last[0], float(loc.y) - last[1])
            if 0.0 <= delta <= 20.0:
                self._route_progress_m = float(getattr(self, "_route_progress_m", 0.0)) + delta
        self._last_progress_loc = (float(loc.x), float(loc.y))
        return float(getattr(self, "_route_progress_m", 0.0))

    def _ego_features(self, speed_kmh, ego_tf, wp_angle):
        if self.ego_mode == "base" or ego_tf is None:
            return [speed_kmh, wp_angle]
        angle_16 = self._route_angle(ego_tf, lookahead=16.0)
        command, next_command, route_len = self._route_command_pair(ego_tf)
        proxy = self._front_actor_proxy(ego_tf, max_dist=50.0)
        if proxy is None:
            actor_distance, actor_forward, actor_lateral = 999.0, 999.0, 999.0
            actor_vehicle, actor_walker = 0.0, 0.0
            rel_speed_mps, closing_mps = 0.0, 0.0
        else:
            actor_distance = float(proxy["distance"])
            actor_forward = float(proxy["forward"])
            actor_lateral = float(proxy["lateral"])
            actor_vehicle = 1.0 if str(proxy["type_id"]).startswith("vehicle.") else 0.0
            actor_walker = 1.0 if str(proxy["type_id"]).startswith("walker.") else 0.0
            yaw = math.radians(ego_tf.rotation.yaw)
            fwdx, fwdy = math.cos(yaw), math.sin(yaw)
            ego_forward_mps = float(speed_kmh) / 3.6
            veh = self._veh()
            if veh is not None:
                ego_vel = veh.get_velocity()
                ego_forward_mps = float(ego_vel.x) * fwdx + float(ego_vel.y) * fwdy
            actor_vel = proxy.get("velocity") or {}
            actor_forward_mps = float(actor_vel.get("x", 0.0)) * fwdx + float(actor_vel.get("y", 0.0)) * fwdy
            rel_speed_mps = ego_forward_mps - actor_forward_mps
            closing_mps = max(0.0, rel_speed_mps)
        features = [
            float(speed_kmh),
            float(wp_angle),
            math.sin(angle_16),
            math.cos(angle_16),
            _command_id(command) / float(COMMAND_CLASSES - 1),
            _command_id(next_command) / float(COMMAND_CLASSES - 1),
        ]
        features.extend(_command_onehot(command))
        features.extend(_command_onehot(next_command))
        features.extend(
            [
                self._junction_proxy(ego_tf),
                0.0,  # changed_route is a teacher-planner diagnostic; not available at runtime.
                max(0.0, min(1.0, route_len / 200.0)),
                max(0.0, min(1.0, route_len / 200.0)),
                max(0.0, min(1.0, actor_distance / 50.0)),
                max(0.0, min(1.0, actor_forward / 50.0)),
                max(-1.0, min(1.0, actor_lateral / 10.0)),
                actor_vehicle,
                actor_walker,
            ]
        )
        if len(features) != ROUTE_V1_EGO_DIM:
            raise RuntimeError("route_v1 feature dim mismatch: %d" % len(features))
        if self.ego_mode == "route_v1":
            return features
        traffic_light_dist, stop_sign_dist, stopline_dist = self._stopline_proxies(ego_tf)
        junction_ahead, junction_ahead_dist = self._junction_ahead_proxy(ego_tf)
        junction_ahead = max(junction_ahead, _turn_command(command), _turn_command(next_command))
        if junction_ahead >= 0.5 and junction_ahead_dist >= 999.0:
            junction_ahead_dist = 0.0
        features.extend(
            [
                max(0.0, min(1.0, abs(angle_16) / math.pi)),
                _turn_command(command),
                _turn_command(next_command),
                max(0.0, min(1.0, stopline_dist / 50.0)),
                max(0.0, min(1.0, traffic_light_dist / 80.0)),
                max(0.0, min(1.0, stop_sign_dist / 80.0)),
                1.0 - max(0.0, min(1.0, actor_distance / 50.0)),
                1.0 if junction_ahead >= 0.5 else 0.0,
                max(0.0, min(1.0, junction_ahead_dist / 50.0)),
            ]
        )
        if len(features) != ROUTE_V2_EGO_DIM:
            raise RuntimeError("route_v2 feature dim mismatch: %d" % len(features))
        if self.ego_mode == "route_v2":
            return features
        route_progress = float(getattr(self, "_route_progress_m", 0.0))
        progress_norm = max(0.0, min(1.0, route_progress / 1000.0))
        progress_phase = 2.0 * math.pi * progress_norm
        features.extend([progress_norm, math.sin(progress_phase), math.cos(progress_phase)])
        if len(features) != ROUTE_V3_EGO_DIM:
            raise RuntimeError("route_v3 feature dim mismatch: %d" % len(features))
        if self.ego_mode == "route_v3":
            return features
        features.extend(
            [
                _clamp_signed(rel_speed_mps / 20.0),
                max(0.0, min(1.0, closing_mps / 20.0)),
            ]
        )
        if len(features) != ROUTE_V4_EGO_DIM:
            raise RuntimeError("route_v4 feature dim mismatch: %d" % len(features))
        return features

    def _apply_interaction_head(self, ego_state, ego_features, steer, throttle, brake):
        self._last_interaction = None
        if self.interaction_head is None:
            return steer, throttle, brake
        if self.ego_mode != "route_v1" or len(ego_features) != ROUTE_V1_EGO_DIM:
            return steer, throttle, brake
        actor_distance = float(ego_features[24]) * 50.0
        actor_vehicle = float(ego_features[27]) >= 0.5
        if actor_distance < self.interaction_min_distance or actor_distance > self.interaction_max_distance:
            return steer, throttle, brake
        if self.interaction_vehicle_only and not actor_vehicle:
            return steer, throttle, brake
        with torch.no_grad():
            interaction = self.interaction_head(ego_state)[0].tolist()
        i_steer, i_throttle, i_brake = [float(x) for x in interaction]
        i_brake = min(max(i_brake, 0.0), self.interaction_max_brake)
        i_throttle = max(i_throttle, self.interaction_min_throttle)
        blend = max(0.0, min(1.0, self.interaction_blend))
        out_steer = (1.0 - blend) * steer + blend * i_steer
        out_throttle = (1.0 - blend) * throttle + blend * i_throttle
        out_brake = (1.0 - blend) * brake + blend * i_brake
        if out_brake < 0.05:
            out_brake = 0.0
        self._last_interaction = {
            "distance": actor_distance,
            "vehicle": actor_vehicle,
            "blend": blend,
            "head_action": {
                "steer": i_steer,
                "throttle": i_throttle,
                "brake": i_brake,
            },
            "base_action": {
                "steer": float(steer),
                "throttle": float(throttle),
                "brake": float(brake),
            },
            "out_action": {
                "steer": float(out_steer),
                "throttle": float(out_throttle),
                "brake": float(out_brake),
            },
        }
        return out_steer, out_throttle, out_brake

    def _apply_source_brake_head(self, image_t, speed_kmh, wp_angle, throttle, brake):
        self._last_source_brake = None
        if self.source_brake_head is None:
            return throttle, brake
        ego = torch.tensor([[float(speed_kmh), float(wp_angle)]], dtype=torch.float32)
        with torch.no_grad():
            probs = torch.sigmoid(self.source_brake_head(image_t, ego))[0].tolist()
        near_p, object_p, route_p = [float(x) for x in probs]
        near_hit = near_p >= self.source_brake_near_threshold
        object_hit = object_p >= self.source_brake_object_threshold
        route_hit = route_p >= self.source_brake_route_threshold
        mode = self.source_brake_mode
        if mode == "near":
            trigger = near_hit
        elif mode == "route":
            trigger = route_hit
        elif mode == "any":
            trigger = near_hit or object_hit or route_hit
        elif mode == "object_or_route":
            trigger = object_hit or route_hit
        else:
            trigger = object_hit
        out_throttle, out_brake = throttle, brake
        if self.source_brake_control and trigger:
            out_throttle = min(out_throttle, self.source_brake_max_throttle)
            out_brake = max(out_brake, self.source_brake_brake)
        self._last_source_brake = {
            "near": near_p,
            "object": object_p,
            "route": route_p,
            "mode": mode,
            "trigger": bool(trigger),
            "control": bool(self.source_brake_control),
        }
        return out_throttle, out_brake

    def _apply_front_recovery(self, ego_features, speed_kmh, wp_angle, steer, throttle, brake):
        self._last_front_recovery = None
        if not self.front_recovery_enable or len(ego_features) != ROUTE_V1_EGO_DIM:
            self._front_slow_ticks = 0
            return steer, throttle, brake, False
        actor_distance = float(ego_features[24]) * 50.0
        actor_vehicle = float(ego_features[27]) >= 0.5
        if not actor_vehicle or actor_distance > self.front_recovery_distance:
            self._front_slow_ticks = 0
            return steer, throttle, brake, False
        if speed_kmh < 3.0:
            self._front_slow_ticks = getattr(self, "_front_slow_ticks", 0) + 1
        elif speed_kmh > 6.0:
            self._front_slow_ticks = 0
        trigger_ticks = max(getattr(self, "_stuck", 0), getattr(self, "_front_slow_ticks", 0))
        if trigger_ticks <= self.front_recovery_after:
            return steer, throttle, brake, False
        reverse_steps = max(1, self.front_recovery_reverse_steps)
        forward_steps = max(1, self.front_recovery_forward_steps)
        period = reverse_steps + forward_steps
        phase = (trigger_ticks - self.front_recovery_after - 1) % period
        route_sign = 1.0 if wp_angle >= 0.0 else -1.0
        if phase < reverse_steps:
            reverse = True
            out_steer = -route_sign * self.front_recovery_steer
            out_throttle = self.front_recovery_reverse_throttle
            stage = "reverse"
        else:
            reverse = False
            out_steer = max(-0.85, min(0.85, 1.15 * wp_angle))
            out_throttle = max(throttle, self.front_recovery_forward_throttle)
            stage = "forward"
        out_brake = 0.0
        self._last_front_recovery = {
            "stage": stage,
            "phase": int(phase),
            "stuck": int(getattr(self, "_stuck", 0)),
            "front_slow_ticks": int(getattr(self, "_front_slow_ticks", 0)),
            "trigger_ticks": int(trigger_ticks),
            "distance": float(actor_distance),
            "steer": float(out_steer),
            "throttle": float(out_throttle),
            "reverse": bool(reverse),
        }
        return out_steer, out_throttle, out_brake, reverse

    def run_step(self, input_data, timestamp):
        self.step += 1
        arr = input_data["front"][1]
        img = arr[:, :, :3][:, :, ::-1]          # BGRA -> RGB
        img = cv2.resize(np.ascontiguousarray(img), (self.image_size, self.image_size))
        image_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        speed_kmh = float(input_data["speed"][1]["speed"] * 3.6)
        if self.step < 3 or self.step % 200 == 0:
            print("[IMG] step=%d shape=%s mean=%.3f std=%.3f min=%.1f max=%.1f" % (self.step, arr.shape, float(img.mean()), float(img.std()), float(img.min()), float(img.max())), flush=True)
        veh = self._veh()
        ego_tf = veh.get_transform() if veh is not None else None
        self._update_route_progress(ego_tf)
        wp_angle = self._route_angle(ego_tf) if ego_tf is not None else 0.0
        ego_features = self._ego_features(speed_kmh, ego_tf, wp_angle)
        ego_state = torch.tensor([ego_features], dtype=torch.float32)
        with torch.no_grad():
            action, gate, _ = self.model(image_t, ego_state, {})
        if self.targetspeed_mode:
            # X-MoD v2 (TRUE mechanism): model outputs [steer, target_speed_norm, 0]; a classical
            # PID maps target_speed -> throttle/brake. Legacy pedal shields bypassed; only the §4.2
            # cognitive-deficit floor (front-clear AND-gated) overrides for liveness.
            if self.step <= 1:
                self.pid.reset()
                self.stall_deficit = 0.0
            steer = float(action[0, 0])
            target_speed_mps = float(action[0, 1]) * V_MAX_MPS
            cur_mps = speed_kmh / 3.6
            dt = 0.05
            if cur_mps < self.deficit_eps_mps:
                self.stall_deficit += dt
            else:
                self.stall_deficit = max(0.0, self.stall_deficit - 2.0 * dt)
            ts_floor = 0.0
            if self.deficit_floor_enable and self.stall_deficit > self.deficit_warmup_s:
                front_clear = True
                if ego_tf is not None:
                    hz = self._front_hazard(ego_tf, max_dist=self.deficit_front_margin)
                    front_clear = hz is None or hz[0] > self.deficit_front_margin
                if front_clear:
                    ts_floor = min(self.deficit_gain * (self.stall_deficit - self.deficit_warmup_s), self.deficit_cap_mps)
                    target_speed_mps = max(target_speed_mps, ts_floor)
            throttle, brake = self.pid.step(cur_mps, target_speed_mps)
            # safety shield: the policy emits no brake of its own, so hard-brake for close front
            # obstacles -> the car does not bump them, it waits, then PID resumes when the path clears.
            sh = "-"
            if self.ts_safety and ego_tf is not None:
                hz2 = self._front_hazard(ego_tf, max_dist=self.safety_distance)
                if hz2 is not None:
                    d2 = hz2[0]
                    if d2 <= self.hard_brake_distance:
                        throttle, brake, sh = 0.0, 1.0, "HARD"
                    elif d2 <= self.safety_distance and speed_kmh > 10.0:
                        throttle = min(throttle, 0.25); brake = max(brake, 0.25); sh = "soft"
            if self.step % 25 == 0:
                g = [round(float(x), 2) for x in gate[0].tolist()]
                print("[XMoD] v2pid step=%d spd=%.1f tgt=%.2f floor=%.2f D=%.1f sh=%s act=(%.2f,%.2f,%.2f) gate=%s" %
                      (self.step, speed_kmh, target_speed_mps, ts_floor, self.stall_deficit, sh, steer, throttle, brake, g), flush=True)
            return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake, reverse=False)
        steer, throttle, brake = [float(x) for x in action[0].tolist()]
        if brake < 0.05:
            brake = 0.0
        steer, throttle, brake = self._apply_interaction_head(ego_state, ego_features, steer, throttle, brake)
        throttle, brake = self._apply_source_brake_head(image_t, speed_kmh, wp_angle, throttle, brake)
        hazard = None
        if self.enable_safety_shield and ego_tf is not None:
            hazard = self._front_hazard(ego_tf, max_dist=max(self.safety_distance, self.hard_brake_distance))
            if hazard is not None:
                dist, _ = hazard
                if dist <= self.hard_brake_distance:
                    throttle = 0.0
                    brake = max(brake, 0.75)
                elif dist <= self.safety_distance and speed_kmh > 10.0:
                    throttle = min(throttle, 0.25)
                    brake = max(brake, 0.25)
        # standard anti-stall creep (disclosed agent component): break stuck-braking
        self._stuck = getattr(self, "_stuck", 0) + 1 if speed_kmh < 1.0 else 0
        if self._stuck > 20:
            throttle = max(throttle, 0.55); brake = 0.0
        reverse = False
        steer, throttle, brake, reverse = self._apply_front_recovery(ego_features, speed_kmh, wp_angle, steer, throttle, brake)
        if self.enable_recovery_shield and self._stuck > self.recovery_after:
            # Targeted closed-loop ablation: when the learned policy is stationary
            # for too long, bias control toward the route direction instead of
            # repeatedly replaying the same near-zero motion.
            if not reverse and (hazard is None or hazard[0] > self.hard_brake_distance):
                steer = max(-0.75, min(0.75, 0.85 * wp_angle))
                throttle = max(throttle, self.recovery_throttle)
                brake = 0.0
        if self.step % 25 == 0:
            g = [round(float(x),2) for x in gate[0].tolist()]
            hazard_s = "none" if hazard is None else "%.1f:%s" % (hazard[0], hazard[1].split(".")[0])
            interaction = getattr(self, "_last_interaction", None)
            interaction_s = "none" if interaction is None else "%.1fm@%.2f" % (interaction["distance"], interaction["blend"])
            source_brake = getattr(self, "_last_source_brake", None)
            source_brake_s = "none" if source_brake is None else "%.2f/%.2f/%.2f:%s" % (
                source_brake["near"],
                source_brake["object"],
                source_brake["route"],
                "T" if source_brake["trigger"] else "f",
            )
            front_recovery = getattr(self, "_last_front_recovery", None)
            front_recovery_s = "none" if front_recovery is None else "%s:%d:%dt:%.1fm" % (
                front_recovery["stage"],
                front_recovery["phase"],
                front_recovery["trigger_ticks"],
                front_recovery["distance"],
            )
            print("[XMoD] step=%d spd=%.1f stuck=%d wpang=%.2f act=(%.2f,%.2f,%.2f,r=%s) gate=%s hazard=%s interact=%s srcbrake=%s frontrec=%s" %
                  (self.step, speed_kmh, self._stuck, wp_angle, steer, throttle, brake, reverse, g, hazard_s, interaction_s, source_brake_s, front_recovery_s), flush=True)
        self._maybe_trace(img, speed_kmh, wp_angle, steer, throttle, brake, gate, hazard, ego_tf)
        return carla.VehicleControl(steer=steer, throttle=throttle, brake=brake, reverse=reverse)

    def destroy(self):
        if self._trace_handle is not None:
            self._trace_handle.close()
            self._trace_handle = None
        self._vehicle = None
        self._route_progress_m = 0.0
        self._last_progress_loc = None
