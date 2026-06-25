import argparse
import gzip
import json
import math
import shutil
from collections import Counter
from pathlib import Path


def read_measurement(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=True, sort_keys=True), encoding="utf-8")


BAD_INFRACTION_KEYS = [
    "collisions_layout",
    "collisions_pedestrian",
    "collisions_vehicle",
    "red_light",
    "stop_infraction",
    "outside_route_lanes",
    "route_dev",
    "vehicle_blocked",
    "route_timeout",
    "scenario_timeouts",
    "yield_emergency_vehicle_infractions",
]


def endpoint_path_for_episode(episode_dir: Path, endpoint_root: Path) -> Path:
    return endpoint_root / f"{episode_dir.parent.name}_endpoint.json"


def summarize_endpoint(endpoint_path: Path) -> dict:
    if not endpoint_path.exists():
        return {
            "ok": False,
            "reason": "missing_endpoint",
            "endpoint": endpoint_path.name,
        }
    try:
        payload = json.loads(endpoint_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "reason": f"bad_endpoint_json:{exc.__class__.__name__}",
            "endpoint": endpoint_path.name,
        }

    checkpoint = payload.get("_checkpoint") or {}
    records = checkpoint.get("records") or []
    if not records:
        status = (checkpoint.get("global_record") or {}).get("status") or payload.get("entry_status", "unknown")
        return {
            "ok": False,
            "reason": f"no_records:{status}",
            "endpoint": endpoint_path.name,
            "status": status,
        }

    record = records[0]
    scores = record.get("scores") or {}
    infractions = record.get("infractions") or {}
    bad_counts = {
        key: len(infractions.get(key) or [])
        for key in BAD_INFRACTION_KEYS
    }
    min_speed_count = len(infractions.get("min_speed_infractions") or [])
    return {
        "ok": True,
        "endpoint": endpoint_path.name,
        "status": record.get("status", "unknown"),
        "route_id_full": record.get("route_id"),
        "scenario_name": record.get("scenario_name"),
        "save_name": record.get("save_name"),
        "score_route": float(scores.get("score_route", 0.0) or 0.0),
        "score_composed": float(scores.get("score_composed", 0.0) or 0.0),
        "score_penalty": float(scores.get("score_penalty", 0.0) or 0.0),
        "bad_infractions": bad_counts,
        "bad_infraction_total": int(sum(bad_counts.values())),
        "min_speed_infraction_count": int(min_speed_count),
    }


def endpoint_is_clean(
    summary: dict,
    min_score_route: float,
    min_score_composed: float,
    min_score_penalty: float,
    allow_min_speed_infractions: bool,
) -> tuple[bool, str]:
    if not summary.get("ok"):
        return False, str(summary.get("reason", "endpoint_not_ok"))
    if summary.get("status") != "Completed":
        return False, f"status:{summary.get('status')}"
    if float(summary.get("score_route", 0.0)) < min_score_route:
        return False, f"score_route:{summary.get('score_route')}"
    if float(summary.get("score_composed", 0.0)) < min_score_composed:
        return False, f"score_composed:{summary.get('score_composed')}"
    if float(summary.get("score_penalty", 0.0)) < min_score_penalty:
        return False, f"score_penalty:{summary.get('score_penalty')}"
    if int(summary.get("bad_infraction_total", 0)) > 0:
        return False, f"bad_infractions:{summary.get('bad_infraction_total')}"
    if not allow_min_speed_infractions and int(summary.get("min_speed_infraction_count", 0)) > 0:
        return False, f"min_speed:{summary.get('min_speed_infraction_count')}"
    return True, "clean"


def infer_route_info(episode_dir: Path) -> tuple[str, str, str]:
    parts = episode_dir.name.split("_")
    route_id = parts[0] if parts else "unknown"
    town = next((part for part in parts if part.startswith("Town")), "unknown")
    scenario = "unknown"
    if town in parts:
        idx = parts.index(town)
        scenario_parts = []
        for part in parts[idx + 1:]:
            if part.isdigit():
                break
            scenario_parts.append(part)
        if scenario_parts:
            scenario = "_".join(scenario_parts)
    return route_id, town, scenario


def parse_route_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {part.strip() for part in value.split(",") if part.strip()}


def parse_run_roots_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    roots = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value:
            continue
        roots.add(value.rstrip("/\\"))
        roots.add(Path(value).name)
    return roots


def source_kind(type_name) -> str:
    if not type_name:
        return "none"
    type_name = str(type_name)
    if type_name.startswith("vehicle."):
        return "vehicle"
    if type_name.startswith("walker.") or "pedestrian" in type_name:
        return "walker"
    if "traffic_light" in type_name:
        return "traffic_light"
    if "stop" in type_name:
        return "stop_sign"
    return "other"


def finite_distance(value, default: float = 999.0) -> float:
    try:
        if value is None:
            return default
        distance = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(distance) or distance < 0.0:
        return default
    return distance


def finite_value(value, default: float = 999.0) -> float:
    try:
        if value is None:
            return default
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def route_angle_from_local_route(route, lookahead_m: float, default: float = 0.0) -> float:
    if not route:
        return default
    chosen = None
    last = None
    for point in route:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = finite_value(point[0], default=0.0)
        y = finite_value(point[1], default=0.0)
        if not math.isfinite(x) or not math.isfinite(y):
            continue
        if x < -1.0:
            continue
        distance = math.hypot(x, y)
        if distance <= 0.05:
            continue
        last = (x, y)
        if distance >= lookahead_m:
            chosen = (x, y)
            break
    if chosen is None:
        chosen = last
    if chosen is None:
        return default
    return math.atan2(chosen[1], chosen[0])


def build_partial(measurement: dict, prev_speed_kmh: float | None, dt: float, progress_m: float) -> tuple[dict, float]:
    speed_kmh = float(measurement.get("speed", 0.0)) * 3.6
    speed_limit_kmh = float(measurement.get("speed_limit", 13.8888888889)) * 3.6
    target_speed_mps = float(measurement.get("target_speed", 0.0) or 0.0)
    target_speed_kmh = target_speed_mps * 3.6
    vehicle_hazard = bool(measurement.get("vehicle_hazard", False))
    walker_hazard = bool(measurement.get("walker_hazard", False))
    light_hazard = bool(measurement.get("light_hazard", False))
    stop_hazard = bool(measurement.get("stop_sign_hazard", False))
    stop_close = bool(measurement.get("stop_sign_close", False))
    junction = bool(measurement.get("junction", False))
    changed_route = bool(measurement.get("changed_route", False))
    command = float(measurement.get("command", 0.0) or 0.0)
    next_command = float(measurement.get("next_command", 0.0) or 0.0)
    route = measurement.get("route") or []
    route_original = measurement.get("route_original") or []
    route_len = float(len(route))
    route_original_len = float(len(route_original))
    route_angle_8m = route_angle_from_local_route(route, 8.0, math.radians(float(measurement.get("angle", 0.0))))
    route_angle_16m = route_angle_from_local_route(route, 16.0, route_angle_8m)
    speed_reduced_kind = source_kind(measurement.get("speed_reduced_by_obj_type"))
    speed_reduced_distance = finite_distance(measurement.get("speed_reduced_by_obj_distance"))
    speed_reduced_id = float(measurement.get("speed_reduced_by_obj_id", -1.0) or -1.0)
    d_traffic_light = finite_distance(measurement.get("distance_to_next_traffic_light"))
    d_stop_sign = finite_distance(measurement.get("distance_to_next_stop_sign"))
    runtime_front_kind = source_kind(measurement.get("runtime_front_actor_type"))
    runtime_front_distance = finite_distance(measurement.get("runtime_front_actor_distance"))
    runtime_front_forward = finite_distance(measurement.get("runtime_front_actor_forward"))
    runtime_front_lateral = finite_value(measurement.get("runtime_front_actor_lateral"))
    runtime_front_id = float(measurement.get("runtime_front_actor_id", -1.0) or -1.0)

    d_front = 5.0 if vehicle_hazard else 999.0
    d_ped = 5.0 if walker_hazard else 999.0
    if speed_reduced_kind == "vehicle":
        d_front = min(d_front, speed_reduced_distance)
    if speed_reduced_kind == "walker":
        d_ped = min(d_ped, speed_reduced_distance)
    ttc = 1.0 if (vehicle_hazard or walker_hazard) else 999.0
    tl_state = 1.0 if light_hazard else 0.0
    d_stopline = min(d_traffic_light, d_stop_sign)
    if light_hazard or stop_hazard or stop_close:
        d_stopline = min(d_stopline, 0.0)
    overspeed = 1.0 if speed_kmh > speed_limit_kmh else 0.0

    a_lon = 0.0
    jerk = 0.0
    if prev_speed_kmh is not None and dt > 0:
        a_lon = ((speed_kmh - prev_speed_kmh) / 3.6) / dt
        jerk = a_lon / dt

    progress_m += (speed_kmh / 3.6) * dt
    eta = (1000.0 - progress_m) / max(speed_kmh / 3.6, 0.1)

    partial = {
        "ttc": ttc,
        "d_front": d_front,
        "v_rel": 0.0,
        "d_ped": d_ped,
        "tl_state": tl_state,
        "d_stopline": d_stopline,
        "speed_limit": speed_limit_kmh,
        "overspeed": overspeed,
        "jerk": jerk,
        "yaw_rate": 0.0,
        "a_lat": 0.0,
        "a_lon": a_lon,
        "route_progress": progress_m,
        "v": speed_kmh,
        "v_ref": speed_limit_kmh,
        "eta": eta,
        "wp_angle": route_angle_8m,
        "teacher_wp_angle": math.radians(float(measurement.get("angle", 0.0))),
        "route_angle_8m": route_angle_8m,
        "route_angle_16m": route_angle_16m,
        "teacher_target_speed": target_speed_mps,
        "teacher_target_speed_kmh": target_speed_kmh,
        "teacher_target_speed_zero": 1.0 if target_speed_mps <= 0.01 else 0.0,
        "junction": 1.0 if junction else 0.0,
        "changed_route": 1.0 if changed_route else 0.0,
        "command": command,
        "next_command": next_command,
        "route_len": route_len,
        "route_original_len": route_original_len,
        "distance_to_next_traffic_light": d_traffic_light,
        "distance_to_next_stop_sign": d_stop_sign,
        "speed_reduced_by_obj_distance": speed_reduced_distance,
        "speed_reduced_by_obj_id": speed_reduced_id,
        "speed_reduced_by_vehicle": 1.0 if speed_reduced_kind == "vehicle" else 0.0,
        "speed_reduced_by_walker": 1.0 if speed_reduced_kind == "walker" else 0.0,
        "speed_reduced_by_traffic_light": 1.0 if speed_reduced_kind == "traffic_light" else 0.0,
        "speed_reduced_by_stop_sign": 1.0 if speed_reduced_kind == "stop_sign" else 0.0,
        "runtime_front_actor_distance": runtime_front_distance,
        "runtime_front_actor_forward": runtime_front_forward,
        "runtime_front_actor_lateral": runtime_front_lateral,
        "runtime_front_actor_id": runtime_front_id,
        "runtime_front_actor_is_vehicle": 1.0 if runtime_front_kind == "vehicle" else 0.0,
        "runtime_front_actor_is_walker": 1.0 if runtime_front_kind == "walker" else 0.0,
    }
    return partial, progress_m


def build_label(partial: dict) -> dict:
    safety = partial["ttc"] < 5.0 or partial["d_front"] < 15.0 or partial["d_ped"] < 25.0
    legality = partial["tl_state"] >= 0.9 or (partial["d_stopline"] < 8.0 and partial["overspeed"] > 0.5)
    comfort = abs(partial["jerk"]) > 2.5 or abs(partial["a_lat"]) > 2.0
    efficiency = partial["ttc"] > 5.0 and partial["tl_state"] < 0.5 and not safety
    stop_intent = partial.get("teacher_target_speed_zero", 0.0) >= 0.5
    multihot = [int(safety), int(legality), int(comfort), int(efficiency)]
    single = next((idx for idx, value in enumerate(multihot) if value), 3)
    return {
        "drive_multihot": multihot,
        "drive_single_idx": single,
        "event_flags": {
            "safety_event": bool(safety),
            "legality_event": bool(legality),
            "comfort_event": bool(comfort),
            "efficiency_event": bool(efficiency),
            "stop_intent_event": bool(stop_intent),
            "route_flow_event": bool(stop_intent),
        },
    }


def build_route_flow(measurement: dict, partial: dict) -> dict:
    return {
        "teacher_target_speed": partial["teacher_target_speed"],
        "teacher_target_speed_kmh": partial["teacher_target_speed_kmh"],
        "teacher_target_speed_zero": bool(partial["teacher_target_speed_zero"] >= 0.5),
        "junction": bool(partial["junction"] >= 0.5),
        "changed_route": bool(partial["changed_route"] >= 0.5),
        "command": measurement.get("command"),
        "next_command": measurement.get("next_command"),
        "route_len": int(partial["route_len"]),
        "route_original_len": int(partial["route_original_len"]),
        "route_angle_8m": partial["route_angle_8m"],
        "route_angle_16m": partial["route_angle_16m"],
        "teacher_wp_angle": partial["teacher_wp_angle"],
        "distance_to_next_traffic_light": partial["distance_to_next_traffic_light"],
        "distance_to_next_stop_sign": partial["distance_to_next_stop_sign"],
        "next_traffic_light_id": measurement.get("next_traffic_light_id"),
        "next_traffic_light_state": measurement.get("next_traffic_light_state"),
        "next_stop_sign_id": measurement.get("next_stop_sign_id"),
        "speed_reduced_by_obj_distance": partial["speed_reduced_by_obj_distance"],
        "speed_reduced_by_obj_id": measurement.get("speed_reduced_by_obj_id"),
        "speed_reduced_by_obj_type": measurement.get("speed_reduced_by_obj_type"),
        "runtime_front_actor_distance": partial["runtime_front_actor_distance"],
        "runtime_front_actor_forward": partial["runtime_front_actor_forward"],
        "runtime_front_actor_lateral": partial["runtime_front_actor_lateral"],
        "runtime_front_actor_id": measurement.get("runtime_front_actor_id"),
        "runtime_front_actor_type": measurement.get("runtime_front_actor_type"),
        "runtime_front_actor_is_vehicle": bool(partial["runtime_front_actor_is_vehicle"] >= 0.5),
        "runtime_front_actor_is_walker": bool(partial["runtime_front_actor_is_walker"] >= 0.5),
        "vehicle_affecting_id": measurement.get("vehicle_affecting_id"),
        "walker_affecting_id": measurement.get("walker_affecting_id"),
    }


def convert_episode(
    episode_dir: Path,
    out_root: Path,
    episode_index: int,
    metadata_handle,
    dt: float,
    limit: int | None,
    endpoint_summary: dict | None = None,
) -> int:
    route_id, town, scenario = infer_route_info(episode_dir)
    measurement_dir = episode_dir / "measurements"
    rgb_dir = episode_dir / "rgb"
    measurement_files = sorted(measurement_dir.glob("*.json.gz"))
    if limit is not None:
        measurement_files = measurement_files[:limit]

    written = 0
    prev_speed_kmh = None
    progress_m = 0.0
    episode_name = f"episode_{episode_index:05d}_{route_id}_{scenario}"

    for measurement_path in measurement_files:
        stem = measurement_path.name.split(".")[0]
        rgb_path = rgb_dir / f"{stem}.png"
        if not rgb_path.exists():
            continue

        measurement = read_measurement(measurement_path)
        frame_dir = out_root / episode_name / f"frame_{written:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)

        image_rel = frame_dir.relative_to(out_root) / "image_front.png"
        shutil.copy2(rgb_path, out_root / image_rel)

        speed_kmh = float(measurement.get("speed", 0.0)) * 3.6
        partial, progress_m = build_partial(measurement, prev_speed_kmh, dt, progress_m)
        target_wp_angle = partial["route_angle_8m"]
        state = {
            "speed_kmh": speed_kmh,
            "target_wp_angle": target_wp_angle,
            "route_angle_8m": partial["route_angle_8m"],
            "route_angle_16m": partial["route_angle_16m"],
            "teacher_wp_angle": partial["teacher_wp_angle"],
            "traffic_light_state": "red" if measurement.get("light_hazard") else "none",
        }
        prev_speed_kmh = speed_kmh
        action = {
            "steer": float(measurement.get("steer", 0.0)),
            "throttle": float(measurement.get("throttle", 0.0)),
            "brake": float(bool(measurement.get("brake", False)) or bool(measurement.get("control_brake", False))),
        }
        label = build_label(partial)
        route_flow = build_route_flow(measurement, partial)

        write_json(frame_dir / "state.json", state)
        write_json(frame_dir / "partial.json", partial)
        write_json(frame_dir / "action.json", action)
        write_json(frame_dir / "event_label.json", label)
        write_json(frame_dir / "route_flow.json", route_flow)

        sample = {
            "episode": episode_index,
            "frame": written,
            "teacher": "pdm_lite_b2d",
            "route_id": route_id,
            "scenario": scenario,
            "town": town,
            "frame_dir": (frame_dir.relative_to(out_root)).as_posix(),
            "state": (frame_dir.relative_to(out_root) / "state.json").as_posix(),
            "partial": (frame_dir.relative_to(out_root) / "partial.json").as_posix(),
            "action": (frame_dir.relative_to(out_root) / "action.json").as_posix(),
            "label": (frame_dir.relative_to(out_root) / "event_label.json").as_posix(),
            "route_flow": (frame_dir.relative_to(out_root) / "route_flow.json").as_posix(),
            "image": image_rel.as_posix(),
            "teacher_action": "garage_pdm_lite",
            "source_episode_dir": episode_dir.as_posix(),
        }
        if endpoint_summary is not None:
            sample.update(
                {
                    "source_endpoint": endpoint_summary.get("endpoint"),
                    "source_status": endpoint_summary.get("status"),
                    "source_score_route": endpoint_summary.get("score_route"),
                    "source_score_composed": endpoint_summary.get("score_composed"),
                    "source_score_penalty": endpoint_summary.get("score_penalty"),
                    "source_bad_infraction_total": endpoint_summary.get("bad_infraction_total"),
                    "source_min_speed_infraction_count": endpoint_summary.get("min_speed_infraction_count"),
                }
            )
        metadata_handle.write(json.dumps(sample, ensure_ascii=True) + "\n")
        written += 1

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert lightweight garage PDM-Lite datagen folders to X-MoD dataset format.")
    parser.add_argument("--garage-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-clean-endpoint", action="store_true")
    parser.add_argument("--endpoint-root", type=Path)
    parser.add_argument("--min-score-route", type=float, default=99.99)
    parser.add_argument("--min-score-composed", type=float, default=99.99)
    parser.add_argument("--min-score-penalty", type=float, default=0.999)
    parser.add_argument("--reject-min-speed-infractions", action="store_true")
    parser.add_argument("--include-routes", help="Comma-separated route IDs to include.")
    parser.add_argument("--exclude-routes", help="Comma-separated route IDs to exclude.")
    parser.add_argument(
        "--include-run-roots-file",
        type=Path,
        help="Optional text file of datagen run root paths to include; keeps only episodes whose parent run root matches.",
    )
    args = parser.parse_args()

    episode_dirs = sorted(
        path.parent
        for path in args.garage_root.rglob("measurements")
        if (path.parent / "rgb").is_dir()
    )
    args.out_root.mkdir(parents=True, exist_ok=True)
    metadata_path = args.out_root / "metadata.jsonl"
    endpoint_root = args.endpoint_root or args.garage_root
    allow_min_speed_infractions = not args.reject_min_speed_infractions
    include_routes = parse_route_filter(args.include_routes)
    exclude_routes = parse_route_filter(args.exclude_routes)
    include_run_roots = parse_run_roots_file(args.include_run_roots_file)

    total = 0
    kept = 0
    skipped = 0
    decisions = []
    skip_reasons = Counter()
    with metadata_path.open("w", encoding="utf-8") as metadata_handle:
        for episode_dir in episode_dirs:
            route_id, _town, _scenario = infer_route_info(episode_dir)
            if include_run_roots:
                run_root = episode_dir.parent
                run_root_posix = run_root.as_posix().rstrip("/\\")
                run_root_name = run_root.name
                if run_root_posix not in include_run_roots and run_root_name not in include_run_roots:
                    skipped += 1
                    decision = f"run_root_not_in_include:{run_root_name}"
                    skip_reasons[decision] += 1
                    decisions.append(
                        {
                            "episode_dir": episode_dir.as_posix(),
                            "kept": False,
                            "decision": decision,
                            "endpoint": None,
                        }
                    )
                    continue
            if include_routes and route_id not in include_routes:
                skipped += 1
                decision = f"route_not_in_include:{route_id}"
                skip_reasons[decision] += 1
                decisions.append(
                    {
                        "episode_dir": episode_dir.as_posix(),
                        "kept": False,
                        "decision": decision,
                        "endpoint": None,
                    }
                )
                continue
            if exclude_routes and route_id in exclude_routes:
                skipped += 1
                decision = f"route_excluded:{route_id}"
                skip_reasons[decision] += 1
                decisions.append(
                    {
                        "episode_dir": episode_dir.as_posix(),
                        "kept": False,
                        "decision": decision,
                        "endpoint": None,
                    }
                )
                continue

            endpoint_summary = None
            decision = "not_checked"
            if args.require_clean_endpoint:
                endpoint_summary = summarize_endpoint(endpoint_path_for_episode(episode_dir, endpoint_root))
                is_clean, decision = endpoint_is_clean(
                    endpoint_summary,
                    min_score_route=args.min_score_route,
                    min_score_composed=args.min_score_composed,
                    min_score_penalty=args.min_score_penalty,
                    allow_min_speed_infractions=allow_min_speed_infractions,
                )
                if not is_clean:
                    skipped += 1
                    skip_reasons[decision] += 1
                    decisions.append(
                        {
                            "episode_dir": episode_dir.as_posix(),
                            "kept": False,
                            "decision": decision,
                            "endpoint": endpoint_summary,
                        }
                    )
                    continue

            written = convert_episode(
                episode_dir,
                args.out_root,
                kept,
                metadata_handle,
                args.dt,
                args.limit,
                endpoint_summary=endpoint_summary,
            )
            total += written
            kept += 1
            decisions.append(
                {
                    "episode_dir": episode_dir.as_posix(),
                    "kept": True,
                    "decision": decision,
                    "frames": written,
                    "endpoint": endpoint_summary,
                }
            )

    summary = {
        "garage_root": args.garage_root.as_posix(),
        "out_root": args.out_root.as_posix(),
        "require_clean_endpoint": bool(args.require_clean_endpoint),
        "endpoint_root": endpoint_root.as_posix(),
        "include_routes": sorted(include_routes),
        "exclude_routes": sorted(exclude_routes),
        "include_run_roots_file": None if args.include_run_roots_file is None else args.include_run_roots_file.as_posix(),
        "include_run_roots_count": len(include_run_roots),
        "episodes_seen": len(episode_dirs),
        "episodes_kept": kept,
        "episodes_skipped": skipped,
        "frames": total,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "decisions": decisions,
    }
    summary_path = args.out_root / "dataset_filter_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"episodes_seen={len(episode_dirs)}")
    print(f"episodes_kept={kept}")
    print(f"episodes_skipped={skipped}")
    print(f"frames={total}")
    print(f"metadata={metadata_path}")
    print(f"filter_summary={summary_path}")
    if skip_reasons:
        print(f"skip_reasons={dict(sorted(skip_reasons.items()))}")


if __name__ == "__main__":
    main()
