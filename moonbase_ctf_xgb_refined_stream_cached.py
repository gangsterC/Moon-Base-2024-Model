#!/usr/bin/env python3
"""
Moon Base 2024 CTF refined XGBoost streamer.

Purpose:
- Stream thousands of NDJSON replays from an external drive without loading all rows into RAM.
- Build reusable CSV shards at 10 Hz.
- Train one XGBoost model per target.
- Compare against simple line/proximity equations.

Targets:
- Broad team pressure / chain value:
    red_out_base_3s / 5s / 10s
    blue_out_base_3s / 5s / 10s
  Meaning: within H seconds, does that team have the enemy flag OUT of base?
  This includes already-out states and future grab/regrab/handoff chains.

- True escape probability:
    red_escape_3s / 5s / 10s
    blue_escape_3s / 5s / 10s
  Meaning: only evaluated when that team is NOT already out; will they get out within H seconds?

- Cap probability:
    red_cap_5s / 10s / 20s / 30s
    blue_cap_5s / 10s / 20s / 30s

- FC lost / possession ends:
    red_fc_lost_1s / 2s / 3s / 5s
    blue_fc_lost_1s / 2s / 3s / 5s

Moon Base 2024 geometry:
- Detects red flag tile 3 and blue flag tile 4 from the map.
- Red FC is a red player carrying blue flag; out when they have moved far enough from blue flag toward red flag.
- Blue FC is a blue player carrying red flag; out when they have moved far enough from red flag toward blue flag.
- Default escape threshold is progress=0.28 along the diagonal flag-to-flag route.
  You can tune with --red-escape-progress and --blue-escape-progress.
"""

import argparse
import gc
import json
import math
import random
import shutil
import time
from bisect import bisect_right
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

TEAM_RED = 1
TEAM_BLUE = 2

CAP_HORIZONS = [5, 10, 20, 30]
OUT_HORIZONS = [3, 5, 10]
ESCAPE_HORIZONS = [3, 5, 10]
LOST_HORIZONS = [1, 2, 3, 5]

TARGETS = (
    [f"red_out_base_{h}s" for h in OUT_HORIZONS]
    + [f"blue_out_base_{h}s" for h in OUT_HORIZONS]
    + [f"red_escape_{h}s" for h in ESCAPE_HORIZONS]
    + [f"blue_escape_{h}s" for h in ESCAPE_HORIZONS]
    + [f"red_cap_{h}s" for h in CAP_HORIZONS]
    + [f"blue_cap_{h}s" for h in CAP_HORIZONS]
    + [f"red_fc_lost_{h}s" for h in LOST_HORIZONS]
    + [f"blue_fc_lost_{h}s" for h in LOST_HORIZONS]
)

BASE_FEATURE_COLS = [
    # Global / score / map
    "score_diff_red", "red_alive", "blue_alive", "map_w", "map_h",
    "red_flag_x", "red_flag_y", "blue_flag_x", "blue_flag_y",
    "flag_dist", "red_escape_progress_threshold", "blue_escape_progress_threshold",

    # Team powerup counts
    "red_tagpro_count", "blue_tagpro_count",
    "red_juke_count", "blue_juke_count",
    "red_rb_count", "blue_rb_count",
    "red_speed_count", "blue_speed_count",

    # Red offense: red player carrying blue flag
    "has_red_fc", "red_fc_x", "red_fc_y", "red_fc_vx", "red_fc_vy",
    "red_fc_out", "red_fc_in_base", "red_progress", "red_escape_margin", "red_vel_to_escape",
    "red_fc_tagpro", "red_fc_juke", "red_fc_rb", "red_fc_speed",
    "red_fc_nearest_enemy", "red_fc_second_enemy", "red_fc_nearest_friend", "red_fc_enemy_closing",
    "red_fc_d_redflag", "red_fc_d_blueflag", "red_fc_d_wall", "red_fc_d_spike",
    "red_fc_d_boost", "red_fc_d_bomb", "red_fc_d_pup",

    # Blue offense: blue player carrying red flag
    "has_blue_fc", "blue_fc_x", "blue_fc_y", "blue_fc_vx", "blue_fc_vy",
    "blue_fc_out", "blue_fc_in_base", "blue_progress", "blue_escape_margin", "blue_vel_to_escape",
    "blue_fc_tagpro", "blue_fc_juke", "blue_fc_rb", "blue_fc_speed",
    "blue_fc_nearest_enemy", "blue_fc_second_enemy", "blue_fc_nearest_friend", "blue_fc_enemy_closing",
    "blue_fc_d_redflag", "blue_fc_d_blueflag", "blue_fc_d_wall", "blue_fc_d_spike",
    "blue_fc_d_boost", "blue_fc_d_bomb", "blue_fc_d_pup",

    # Race-to-grab / base pressure
    "red_nearest_blueflag", "red_second_blueflag", "red_nearest_blueflag_x", "red_nearest_blueflag_y",
    "blue_nearest_blueflag", "blue_second_blueflag",
    "blue_nearest_redflag", "blue_second_redflag", "blue_nearest_redflag_x", "blue_nearest_redflag_y",
    "red_nearest_redflag", "red_second_redflag",
    "red_players_in_blue_base", "blue_players_in_red_base",
    "red_players_outside_blue_base", "blue_players_outside_red_base",
]

PLAYER_SLOT_COLS = []
for team_name in ["red", "blue"]:
    for i in range(1, 5):
        prefix = f"{team_name}{i}"
        PLAYER_SLOT_COLS += [
            f"{prefix}_present", f"{prefix}_x", f"{prefix}_y", f"{prefix}_vx", f"{prefix}_vy",
            f"{prefix}_has_flag", f"{prefix}_tagpro", f"{prefix}_juke", f"{prefix}_rb", f"{prefix}_speed",
            f"{prefix}_d_redflag", f"{prefix}_d_blueflag",
            f"{prefix}_red_progress", f"{prefix}_blue_progress",
        ]

FEATURE_COLS = BASE_FEATURE_COLS + PLAYER_SLOT_COLS
ALL_COLUMNS = ["replay_id", "t"] + FEATURE_COLS + TARGETS


def as_float(x, default=np.nan):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def dist(x1, y1, x2, y2) -> float:
    return math.hypot(float(x1) - float(x2), float(y1) - float(y2))


def read_ndjson_events(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                arr = json.loads(line)
                if isinstance(arr, list) and len(arr) >= 3:
                    yield int(arr[0]), arr[1], arr[2]
            except Exception:
                continue


def valid_position(p: Dict[str, Any]) -> bool:
    return (
        "rx" in p and "ry" in p
        and np.isfinite(as_float(p.get("rx")))
        and np.isfinite(as_float(p.get("ry")))
        and p.get("team") in (TEAM_RED, TEAM_BLUE)
    )


def valid_live_player(p: Dict[str, Any]) -> bool:
    return valid_position(p) and not bool(p.get("dead", False))


def xy(p: Dict[str, Any]) -> Tuple[float, float]:
    return as_float(p.get("rx")), as_float(p.get("ry"))


def vel(p: Dict[str, Any]) -> Tuple[float, float]:
    # lx/ly are a lightweight movement proxy in these TagPro replay files.
    return as_float(p.get("lx"), 0.0), as_float(p.get("ly"), 0.0)


def pbool(p: Dict[str, Any], key: str) -> int:
    return int(bool(p.get(key, False)))


def has_flag(p: Dict[str, Any]) -> bool:
    return p.get("flag") not in (None, False, 0, "", "none", "None") and not bool(p.get("dead", False))


def apply_player_updates(players: Dict[int, Dict[str, Any]], data: Any):
    if not isinstance(data, list):
        return
    for upd in data:
        if not isinstance(upd, dict) or "id" not in upd:
            continue
        try:
            pid = int(upd["id"])
        except Exception:
            continue
        if pid not in players:
            players[pid] = {"id": pid}
        players[pid].update(upd)


def get_live_players(players: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [p for p in players.values() if valid_live_player(p)]


def get_team_fc(live: List[Dict[str, Any]], team: int) -> Optional[Dict[str, Any]]:
    for p in live:
        if p.get("team") == team and has_flag(p):
            return p
    return None


def parse_tile_float(v):
    try:
        return float(v)
    except Exception:
        return None


def tile_base(v):
    fv = parse_tile_float(v)
    if fv is None:
        return str(v)
    return int(math.floor(fv))


def map_dimensions_from_tiles(map_data: Optional[Dict[str, Any]]):
    # TagPro map format is tiles[x][y], so width is len(tiles), height is len(tiles[x]).
    if not isinstance(map_data, dict):
        return None, None
    tiles = map_data.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        return None, None
    w = len(tiles)
    h = max(len(col) for col in tiles if isinstance(col, list))
    return float(w), float(h)


def extract_map_points(map_data: Optional[Dict[str, Any]]):
    pts = {
        "red_flags": [], "blue_flags": [], "walls": [], "boosts": [], "bombs": [], "pups": [], "spikes": []
    }
    if not isinstance(map_data, dict) or not isinstance(map_data.get("tiles"), list):
        return pts
    tiles = map_data["tiles"]
    for x, col in enumerate(tiles):
        if not isinstance(col, list):
            continue
        for y, v in enumerate(col):
            b = tile_base(v)
            cx, cy = x + 0.5, y + 0.5
            # Common TagPro codes. These categories are used as lightweight geometry features.
            if str(v).startswith("1"):
                pts["walls"].append((cx, cy))
            if b == 3:
                pts["red_flags"].append((cx, cy))
            elif b == 4:
                pts["blue_flags"].append((cx, cy))
            elif b == 5:
                pts["boosts"].append((cx, cy))
            elif b == 6:
                pts["pups"].append((cx, cy))
            elif b == 9:
                pts["spikes"].append((cx, cy))
            elif b == 10:
                pts["bombs"].append((cx, cy))
    return pts


def nearest_static_dist(x, y, points):
    if not points:
        return 999.0
    return min(dist(x, y, px, py) for px, py in points)


def make_geometry(map_data, args, live=None):
    map_w, map_h = map_dimensions_from_tiles(map_data)
    pts = extract_map_points(map_data)

    if not map_w or not map_h or not np.isfinite(map_w) or not np.isfinite(map_h):
        if live:
            xs = [xy(p)[0] for p in live if valid_position(p)]
            ys = [xy(p)[1] for p in live if valid_position(p)]
            map_w = max(max(xs) + 2, 40.0) if xs else 40.0
            map_h = max(max(ys) + 2, 30.0) if ys else 30.0
        else:
            map_w, map_h = 42.0, 30.0

    red_flag = pts["red_flags"][0] if pts["red_flags"] else (args.red_flag_x, args.red_flag_y)
    blue_flag = pts["blue_flags"][0] if pts["blue_flags"] else (args.blue_flag_x, args.blue_flag_y)
    if red_flag[0] is None or red_flag[1] is None:
        red_flag = (0.18 * map_w, 0.25 * map_h)
    if blue_flag[0] is None or blue_flag[1] is None:
        blue_flag = (0.82 * map_w, 0.75 * map_h)

    red_flag_x = args.red_flag_x if args.red_flag_x is not None else float(red_flag[0])
    red_flag_y = args.red_flag_y if args.red_flag_y is not None else float(red_flag[1])
    blue_flag_x = args.blue_flag_x if args.blue_flag_x is not None else float(blue_flag[0])
    blue_flag_y = args.blue_flag_y if args.blue_flag_y is not None else float(blue_flag[1])

    # Direction vectors for escape progress.
    rb_dx = blue_flag_x - red_flag_x
    rb_dy = blue_flag_y - red_flag_y
    flag_d2 = rb_dx * rb_dx + rb_dy * rb_dy + 1e-6
    flag_d = math.sqrt(flag_d2)

    return {
        "map_w": float(map_w), "map_h": float(map_h),
        "red_flag_x": red_flag_x, "red_flag_y": red_flag_y,
        "blue_flag_x": blue_flag_x, "blue_flag_y": blue_flag_y,
        "flag_dist": flag_d,
        "flag_d2": flag_d2,
        "red_to_blue_dx": rb_dx, "red_to_blue_dy": rb_dy,
        "blue_to_red_dx": -rb_dx, "blue_to_red_dy": -rb_dy,
        "red_escape_progress_threshold": float(args.red_escape_progress),
        "blue_escape_progress_threshold": float(args.blue_escape_progress),
        "static_points": pts,
    }


def red_escape_progress(x, y, geom):
    # 0 at blue flag, 1 at red flag.
    vx = geom["red_flag_x"] - geom["blue_flag_x"]
    vy = geom["red_flag_y"] - geom["blue_flag_y"]
    return ((x - geom["blue_flag_x"]) * vx + (y - geom["blue_flag_y"]) * vy) / (geom["flag_d2"])


def blue_escape_progress(x, y, geom):
    # 0 at red flag, 1 at blue flag.
    vx = geom["blue_flag_x"] - geom["red_flag_x"]
    vy = geom["blue_flag_y"] - geom["red_flag_y"]
    return ((x - geom["red_flag_x"]) * vx + (y - geom["red_flag_y"]) * vy) / (geom["flag_d2"])


def unit_vec(dx, dy):
    n = math.hypot(dx, dy) + 1e-6
    return dx / n, dy / n


def nearest_to_point(live: List[Dict[str, Any]], team: int, px: float, py: float):
    group = [p for p in live if p.get("team") == team]
    if not group:
        return [999.0, 999.0, 0.0, 0.0]
    vals = []
    for p in group:
        x, y = xy(p)
        vals.append((dist(px, py, x, y), x, y))
    vals.sort(key=lambda z: z[0])
    d1, x1, y1 = vals[0]
    d2 = vals[1][0] if len(vals) > 1 else 999.0
    return [d1, d2, x1, y1]


def fc_pressure_features(live: List[Dict[str, Any]], fc: Optional[Dict[str, Any]], enemy_team: int, friend_team: int):
    if fc is None:
        return [999.0, 999.0, 999.0, 0.0]
    fx, fy = xy(fc)
    fvx, fvy = vel(fc)
    enemies = [p for p in live if p.get("team") == enemy_team]
    friends = [p for p in live if p.get("team") == friend_team and int(p.get("id", -1)) != int(fc.get("id", -2))]

    enemy_ds = []
    closings = []
    for e in enemies:
        ex, ey = xy(e)
        evx, evy = vel(e)
        d = dist(fx, fy, ex, ey)
        enemy_ds.append(d)
        rx, ry = fx - ex, fy - ey
        norm = math.hypot(rx, ry) + 1e-6
        closings.append(((evx - fvx) * rx + (evy - fvy) * ry) / norm)
    enemy_ds.sort()

    friend_ds = []
    for fr in friends:
        x, y = xy(fr)
        friend_ds.append(dist(fx, fy, x, y))
    friend_ds.sort()

    return [
        enemy_ds[0] if len(enemy_ds) > 0 else 999.0,
        enemy_ds[1] if len(enemy_ds) > 1 else 999.0,
        friend_ds[0] if len(friend_ds) > 0 else 999.0,
        max(closings) if closings else 0.0,
    ]


def add_player_slots(row, live, geom):
    def fill_slot(prefix, p=None):
        if p is None:
            vals = {
                f"{prefix}_present": 0, f"{prefix}_x": 0.0, f"{prefix}_y": 0.0, f"{prefix}_vx": 0.0, f"{prefix}_vy": 0.0,
                f"{prefix}_has_flag": 0, f"{prefix}_tagpro": 0, f"{prefix}_juke": 0, f"{prefix}_rb": 0, f"{prefix}_speed": 0,
                f"{prefix}_d_redflag": 999.0, f"{prefix}_d_blueflag": 999.0,
                f"{prefix}_red_progress": 0.0, f"{prefix}_blue_progress": 0.0,
            }
        else:
            x, y = xy(p); vx, vy = vel(p)
            vals = {
                f"{prefix}_present": 1, f"{prefix}_x": x, f"{prefix}_y": y, f"{prefix}_vx": vx, f"{prefix}_vy": vy,
                f"{prefix}_has_flag": int(has_flag(p)), f"{prefix}_tagpro": pbool(p, "tagpro"),
                f"{prefix}_juke": pbool(p, "jukeJuice"), f"{prefix}_rb": pbool(p, "grip"), f"{prefix}_speed": pbool(p, "speed"),
                f"{prefix}_d_redflag": dist(x, y, geom["red_flag_x"], geom["red_flag_y"]),
                f"{prefix}_d_blueflag": dist(x, y, geom["blue_flag_x"], geom["blue_flag_y"]),
                f"{prefix}_red_progress": red_escape_progress(x, y, geom),
                f"{prefix}_blue_progress": blue_escape_progress(x, y, geom),
            }
        row.update(vals)

    red = [p for p in live if p.get("team") == TEAM_RED]
    blue = [p for p in live if p.get("team") == TEAM_BLUE]
    red.sort(key=lambda p: (dist(*xy(p), geom["blue_flag_x"], geom["blue_flag_y"]), int(p.get("id", 0))))
    blue.sort(key=lambda p: (dist(*xy(p), geom["red_flag_x"], geom["red_flag_y"]), int(p.get("id", 0))))
    for i in range(4):
        fill_slot(f"red{i+1}", red[i] if i < len(red) else None)
        fill_slot(f"blue{i+1}", blue[i] if i < len(blue) else None)


def add_fc_static_dists(row, prefix, x, y, geom):
    pts = geom["static_points"]
    row[f"{prefix}_d_wall"] = nearest_static_dist(x, y, pts["walls"])
    row[f"{prefix}_d_spike"] = nearest_static_dist(x, y, pts["spikes"])
    row[f"{prefix}_d_boost"] = nearest_static_dist(x, y, pts["boosts"])
    row[f"{prefix}_d_bomb"] = nearest_static_dist(x, y, pts["bombs"])
    row[f"{prefix}_d_pup"] = nearest_static_dist(x, y, pts["pups"])


def make_feature_row(path: Path, t_sec: float, live: List[Dict[str, Any]], score_r: int, score_b: int, geom: Dict[str, Any]):
    red_fc = get_team_fc(live, TEAM_RED)
    blue_fc = get_team_fc(live, TEAM_BLUE)
    red_alive = sum(1 for p in live if p.get("team") == TEAM_RED)
    blue_alive = sum(1 for p in live if p.get("team") == TEAM_BLUE)

    row = {
        "replay_id": str(path), "t": float(t_sec),
        "score_diff_red": int(score_r) - int(score_b),
        "red_alive": red_alive, "blue_alive": blue_alive,
        "map_w": geom["map_w"], "map_h": geom["map_h"],
        "red_flag_x": geom["red_flag_x"], "red_flag_y": geom["red_flag_y"],
        "blue_flag_x": geom["blue_flag_x"], "blue_flag_y": geom["blue_flag_y"],
        "flag_dist": geom["flag_dist"],
        "red_escape_progress_threshold": geom["red_escape_progress_threshold"],
        "blue_escape_progress_threshold": geom["blue_escape_progress_threshold"],
        "red_tagpro_count": sum(pbool(p, "tagpro") for p in live if p.get("team") == TEAM_RED),
        "blue_tagpro_count": sum(pbool(p, "tagpro") for p in live if p.get("team") == TEAM_BLUE),
        "red_juke_count": sum(pbool(p, "jukeJuice") for p in live if p.get("team") == TEAM_RED),
        "blue_juke_count": sum(pbool(p, "jukeJuice") for p in live if p.get("team") == TEAM_BLUE),
        "red_rb_count": sum(pbool(p, "grip") for p in live if p.get("team") == TEAM_RED),
        "blue_rb_count": sum(pbool(p, "grip") for p in live if p.get("team") == TEAM_BLUE),
        "red_speed_count": sum(pbool(p, "speed") for p in live if p.get("team") == TEAM_RED),
        "blue_speed_count": sum(pbool(p, "speed") for p in live if p.get("team") == TEAM_BLUE),
    }

    # Red offense: red carrying blue flag, escaping toward red flag.
    if red_fc is not None:
        x, y = xy(red_fc); vx, vy = vel(red_fc)
        prog = red_escape_progress(x, y, geom)
        ux, uy = unit_vec(geom["red_flag_x"] - geom["blue_flag_x"], geom["red_flag_y"] - geom["blue_flag_y"])
        red_out = int(prog >= geom["red_escape_progress_threshold"])
        red_id = int(red_fc.get("id", -1))
    else:
        x, y, vx, vy = geom["blue_flag_x"], geom["blue_flag_y"], 0.0, 0.0
        prog, ux, uy, red_out, red_id = 0.0, 0.0, 0.0, 0, -1
    r_ne, r_se, r_nf, r_close = fc_pressure_features(live, red_fc, TEAM_BLUE, TEAM_RED)
    row.update({
        "has_red_fc": int(red_fc is not None), "red_fc_id": red_id,
        "red_fc_x": x, "red_fc_y": y, "red_fc_vx": vx, "red_fc_vy": vy,
        "red_fc_out": red_out, "red_fc_in_base": int(red_fc is not None and not red_out),
        "red_progress": prog, "red_escape_margin": prog - geom["red_escape_progress_threshold"],
        "red_vel_to_escape": vx * ux + vy * uy,
        "red_fc_tagpro": pbool(red_fc or {}, "tagpro"), "red_fc_juke": pbool(red_fc or {}, "jukeJuice"),
        "red_fc_rb": pbool(red_fc or {}, "grip"), "red_fc_speed": pbool(red_fc or {}, "speed"),
        "red_fc_nearest_enemy": r_ne, "red_fc_second_enemy": r_se,
        "red_fc_nearest_friend": r_nf, "red_fc_enemy_closing": r_close,
        "red_fc_d_redflag": dist(x, y, geom["red_flag_x"], geom["red_flag_y"]),
        "red_fc_d_blueflag": dist(x, y, geom["blue_flag_x"], geom["blue_flag_y"]),
    })
    add_fc_static_dists(row, "red_fc", x, y, geom)

    # Blue offense: blue carrying red flag, escaping toward blue flag.
    if blue_fc is not None:
        x, y = xy(blue_fc); vx, vy = vel(blue_fc)
        prog = blue_escape_progress(x, y, geom)
        ux, uy = unit_vec(geom["blue_flag_x"] - geom["red_flag_x"], geom["blue_flag_y"] - geom["red_flag_y"])
        blue_out = int(prog >= geom["blue_escape_progress_threshold"])
        blue_id = int(blue_fc.get("id", -1))
    else:
        x, y, vx, vy = geom["red_flag_x"], geom["red_flag_y"], 0.0, 0.0
        prog, ux, uy, blue_out, blue_id = 0.0, 0.0, 0.0, 0, -1
    b_ne, b_se, b_nf, b_close = fc_pressure_features(live, blue_fc, TEAM_RED, TEAM_BLUE)
    row.update({
        "has_blue_fc": int(blue_fc is not None), "blue_fc_id": blue_id,
        "blue_fc_x": x, "blue_fc_y": y, "blue_fc_vx": vx, "blue_fc_vy": vy,
        "blue_fc_out": blue_out, "blue_fc_in_base": int(blue_fc is not None and not blue_out),
        "blue_progress": prog, "blue_escape_margin": prog - geom["blue_escape_progress_threshold"],
        "blue_vel_to_escape": vx * ux + vy * uy,
        "blue_fc_tagpro": pbool(blue_fc or {}, "tagpro"), "blue_fc_juke": pbool(blue_fc or {}, "jukeJuice"),
        "blue_fc_rb": pbool(blue_fc or {}, "grip"), "blue_fc_speed": pbool(blue_fc or {}, "speed"),
        "blue_fc_nearest_enemy": b_ne, "blue_fc_second_enemy": b_se,
        "blue_fc_nearest_friend": b_nf, "blue_fc_enemy_closing": b_close,
        "blue_fc_d_redflag": dist(x, y, geom["red_flag_x"], geom["red_flag_y"]),
        "blue_fc_d_blueflag": dist(x, y, geom["blue_flag_x"], geom["blue_flag_y"]),
    })
    add_fc_static_dists(row, "blue_fc", x, y, geom)

    # Race-to-grab and base-pressure features.
    rn_bf, rs_bf, rnbx, rnby = nearest_to_point(live, TEAM_RED, geom["blue_flag_x"], geom["blue_flag_y"])
    bn_bf, bs_bf, _, _ = nearest_to_point(live, TEAM_BLUE, geom["blue_flag_x"], geom["blue_flag_y"])
    bn_rf, bs_rf, bnrx, bnry = nearest_to_point(live, TEAM_BLUE, geom["red_flag_x"], geom["red_flag_y"])
    rn_rf, rs_rf, _, _ = nearest_to_point(live, TEAM_RED, geom["red_flag_x"], geom["red_flag_y"])

    red_players_in_blue_base = sum(1 for p in live if p.get("team") == TEAM_RED and red_escape_progress(*xy(p), geom) < geom["red_escape_progress_threshold"])
    blue_players_in_red_base = sum(1 for p in live if p.get("team") == TEAM_BLUE and blue_escape_progress(*xy(p), geom) < geom["blue_escape_progress_threshold"])
    red_players_outside_blue_base = sum(1 for p in live if p.get("team") == TEAM_RED and red_escape_progress(*xy(p), geom) >= geom["red_escape_progress_threshold"])
    blue_players_outside_red_base = sum(1 for p in live if p.get("team") == TEAM_BLUE and blue_escape_progress(*xy(p), geom) >= geom["blue_escape_progress_threshold"])

    row.update({
        "red_nearest_blueflag": rn_bf, "red_second_blueflag": rs_bf,
        "red_nearest_blueflag_x": rnbx, "red_nearest_blueflag_y": rnby,
        "blue_nearest_blueflag": bn_bf, "blue_second_blueflag": bs_bf,
        "blue_nearest_redflag": bn_rf, "blue_second_redflag": bs_rf,
        "blue_nearest_redflag_x": bnrx, "blue_nearest_redflag_y": bnry,
        "red_nearest_redflag": rn_rf, "red_second_redflag": rs_rf,
        "red_players_in_blue_base": red_players_in_blue_base,
        "blue_players_in_red_base": blue_players_in_red_base,
        "red_players_outside_blue_base": red_players_outside_blue_base,
        "blue_players_outside_red_base": blue_players_outside_red_base,
    })

    add_player_slots(row, live, geom)
    return row


def parse_replay_to_rows(path: Path, args) -> List[Dict[str, Any]]:
    players: Dict[int, Dict[str, Any]] = {}
    score_r = 0
    score_b = 0
    score_events: List[Tuple[float, str]] = []
    rows: List[Dict[str, Any]] = []
    next_sample_ms = 0
    step_ms = max(1, int(round(1000.0 / args.sample_hz)))
    map_data = None

    def take_sample(t_ms: int):
        live = get_live_players(players)
        if len(live) < 2:
            return
        geom = make_geometry(map_data, args, live=live)
        row = make_feature_row(path, t_ms / 1000.0, live, score_r, score_b, geom)
        rows.append(row)

    for t_ms, ev, data in read_ndjson_events(path):
        while next_sample_ms <= t_ms:
            take_sample(next_sample_ms)
            next_sample_ms += step_ms

        if ev == "map" and isinstance(data, dict):
            map_data = data
        elif ev == "p":
            apply_player_updates(players, data)
        elif ev == "score" and isinstance(data, dict):
            new_r = int(data.get("r", score_r) or 0)
            new_b = int(data.get("b", score_b) or 0)
            if new_r > score_r:
                for _ in range(new_r - score_r):
                    score_events.append((t_ms / 1000.0, "red"))
            if new_b > score_b:
                for _ in range(new_b - score_b):
                    score_events.append((t_ms / 1000.0, "blue"))
            score_r, score_b = new_r, new_b

    if len(rows) < 50:
        return []

    red_cap_times = sorted(t for t, team in score_events if team == "red")
    blue_cap_times = sorted(t for t, team in score_events if team == "blue")
    sample_times = [r["t"] for r in rows]
    red_out = [int(r["red_fc_out"]) for r in rows]
    blue_out = [int(r["blue_fc_out"]) for r in rows]
    red_ids = [int(r.get("red_fc_id", -1)) for r in rows]
    blue_ids = [int(r.get("blue_fc_id", -1)) for r in rows]
    red_has = [int(r.get("has_red_fc", 0)) for r in rows]
    blue_has = [int(r.get("has_blue_fc", 0)) for r in rows]

    def has_future_cap(times, t, horizon):
        idx = bisect_right(times, t)
        return int(idx < len(times) and times[idx] <= t + horizon)

    def has_future_bool(series, i, horizon):
        end_t = sample_times[i] + horizon
        j = i
        while j < len(rows) and sample_times[j] <= end_t:
            if series[j]:
                return 1
            j += 1
        return 0

    def fc_lost_at(team_has, team_ids, i, horizon):
        if not team_has[i] or team_ids[i] < 0:
            return np.nan
        end_t = sample_times[i] + horizon
        j = i + 1
        while j < len(rows) and sample_times[j] <= end_t:
            if (not team_has[j]) or team_ids[j] != team_ids[i]:
                return 1
            j += 1
        return 0

    for i, r in enumerate(rows):
        t = r["t"]
        for h in CAP_HORIZONS:
            r[f"red_cap_{h}s"] = has_future_cap(red_cap_times, t, h)
            r[f"blue_cap_{h}s"] = has_future_cap(blue_cap_times, t, h)
        for h in OUT_HORIZONS:
            r[f"red_out_base_{h}s"] = has_future_bool(red_out, i, h)
            r[f"blue_out_base_{h}s"] = has_future_bool(blue_out, i, h)
        for h in ESCAPE_HORIZONS:
            # Escape target excludes already-out states. It includes no-FC and in-base-FC states.
            r[f"red_escape_{h}s"] = np.nan if red_out[i] else has_future_bool(red_out, i, h)
            r[f"blue_escape_{h}s"] = np.nan if blue_out[i] else has_future_bool(blue_out, i, h)
        for h in LOST_HORIZONS:
            r[f"red_fc_lost_{h}s"] = fc_lost_at(red_has, red_ids, i, h)
            r[f"blue_fc_lost_{h}s"] = fc_lost_at(blue_has, blue_ids, i, h)
        r.pop("red_fc_id", None)
        r.pop("blue_fc_id", None)
    return rows


def proximity_score(df: pd.DataFrame, target: str):
    if target.startswith("red_out_base") or target.startswith("red_escape"):
        return (
            1.25 * df["has_red_fc"]
            + 2.25 * df["red_progress"]
            + 0.85 * df["red_vel_to_escape"]
            - 0.50 * df["red_fc_nearest_enemy"]
            + 0.20 * df["red_fc_nearest_friend"]
            - 0.35 * df["red_nearest_blueflag"]
            + 0.25 * df["red_players_in_blue_base"]
        ).values.reshape(-1, 1)
    if target.startswith("blue_out_base") or target.startswith("blue_escape"):
        return (
            1.25 * df["has_blue_fc"]
            + 2.25 * df["blue_progress"]
            + 0.85 * df["blue_vel_to_escape"]
            - 0.50 * df["blue_fc_nearest_enemy"]
            + 0.20 * df["blue_fc_nearest_friend"]
            - 0.35 * df["blue_nearest_redflag"]
            + 0.25 * df["blue_players_in_red_base"]
        ).values.reshape(-1, 1)
    if target.startswith("red_cap"):
        return (
            1.10 * df["has_red_fc"]
            + 0.90 * df["red_fc_out"]
            + 1.25 * df["red_progress"]
            + 0.65 * df["red_vel_to_escape"]
            - 0.50 * df["red_fc_nearest_enemy"]
            + 0.25 * df["red_fc_nearest_friend"]
            - 0.45 * df["has_blue_fc"]
            - 0.30 * df["blue_fc_out"]
        ).values.reshape(-1, 1)
    if target.startswith("blue_cap"):
        return (
            1.10 * df["has_blue_fc"]
            + 0.90 * df["blue_fc_out"]
            + 1.25 * df["blue_progress"]
            + 0.65 * df["blue_vel_to_escape"]
            - 0.50 * df["blue_fc_nearest_enemy"]
            + 0.25 * df["blue_fc_nearest_friend"]
            - 0.45 * df["has_red_fc"]
            - 0.30 * df["red_fc_out"]
        ).values.reshape(-1, 1)
    if target.startswith("red_fc_lost"):
        return (
            -1.15 * df["red_fc_nearest_enemy"]
            -0.25 * df["red_fc_second_enemy"]
            +0.75 * df["red_fc_enemy_closing"]
            -0.10 * df["red_fc_nearest_friend"]
            +0.25 * df["has_red_fc"]
        ).values.reshape(-1, 1)
    if target.startswith("blue_fc_lost"):
        return (
            -1.15 * df["blue_fc_nearest_enemy"]
            -0.25 * df["blue_fc_second_enemy"]
            +0.75 * df["blue_fc_enemy_closing"]
            -0.10 * df["blue_fc_nearest_friend"]
            +0.25 * df["has_blue_fc"]
        ).values.reshape(-1, 1)
    return np.zeros((len(df), 1), dtype=np.float32)


def list_replay_files(replay_dir: Path) -> List[Path]:
    return sorted([p for p in replay_dir.rglob("*.ndjson") if p.is_file() and not p.name.startswith(".")])


def write_csv_shard(rows: List[Dict[str, Any]], out_path: Path):
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    for c in ALL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[ALL_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, compression="gzip")
    return len(df)


def make_shards(files: List[Path], split_name: str, args) -> Tuple[List[Path], int, int]:
    shard_dir = Path(args.out_dir) / "shards" / split_name
    if shard_dir.exists() and not args.reuse_shards:
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_shards:
        shards = sorted(shard_dir.glob("*.csv.gz"))
        print(f"Reusing {len(shards)} {split_name} shards from {shard_dir}")
        return shards, -1, 0

    shards = []
    buffer = []
    total_rows = 0
    failures = 0
    shard_idx = 0
    print(f"\nCreating {split_name} shards...")

    # Optional speed/safety mode for Chromebook + external drives:
    # copy each replay to a local cache, parse it locally, then delete it.
    # This avoids slow FUSE/external-drive line-by-line parsing while keeping
    # Linux storage usage tiny (usually only one replay file at a time).
    local_cache_dir = None
    if getattr(args, "local_cache_dir", None):
        local_cache_dir = Path(args.local_cache_dir).expanduser() / "replay_cache"
        local_cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using local one-file replay cache: {local_cache_dir}")

    for p in tqdm(files, desc=f"Parse {split_name} replays"):
        cached_path = None
        parse_path = p
        try:
            if local_cache_dir is not None:
                cached_path = local_cache_dir / p.name
                # Remove stale partial copy if present.
                if cached_path.exists():
                    try:
                        cached_path.unlink()
                    except Exception:
                        pass
                shutil.copy2(p, cached_path)
                parse_path = cached_path

            rows = parse_replay_to_rows(parse_path, args)
            if not rows:
                failures += 1
                continue
            buffer.extend(rows)
            while len(buffer) >= args.shard_rows:
                to_write = buffer[:args.shard_rows]
                buffer = buffer[args.shard_rows:]
                out_path = shard_dir / f"{split_name}_{shard_idx:05d}.csv.gz"
                n = write_csv_shard(to_write, out_path)
                shards.append(out_path)
                total_rows += n
                shard_idx += 1
                del to_write
                gc.collect()
        except Exception as e:
            failures += 1
            print(f"\nWarning: failed {p.name}: {e}")
        finally:
            if cached_path is not None:
                try:
                    cached_path.unlink()
                except Exception:
                    pass

    if buffer:
        out_path = shard_dir / f"{split_name}_{shard_idx:05d}.csv.gz"
        n = write_csv_shard(buffer, out_path)
        shards.append(out_path)
        total_rows += n
        buffer.clear()
        gc.collect()

    print(f"{split_name}: wrote {total_rows:,} rows to {len(shards)} shards. Failures: {failures}")
    return shards, total_rows, failures


def clean_float_df(df: pd.DataFrame) -> np.ndarray:
    return (
        df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
        .to_numpy(copy=True)
    )


class ShardIter(xgb.DataIter):
    def __init__(self, shard_paths: List[Path], target: str, batch_rows: int, cache_prefix: Path):
        super().__init__(cache_prefix=str(cache_prefix), release_data=True, on_host=False)
        self.shard_paths = [Path(p) for p in shard_paths]
        self.target = target
        self.batch_rows = int(batch_rows)
        self._file_idx = 0
        self._reader = None

    def reset(self):
        self._file_idx = 0
        self._reader = None

    def next(self, input_data):
        while True:
            if self._reader is None:
                if self._file_idx >= len(self.shard_paths):
                    return False
                path = self.shard_paths[self._file_idx]
                self._file_idx += 1
                self._reader = pd.read_csv(path, chunksize=self.batch_rows)
            try:
                chunk = next(self._reader)
            except StopIteration:
                self._reader = None
                continue
            if self.target not in chunk.columns:
                continue
            chunk = chunk[chunk[self.target].notna()]
            if len(chunk) == 0:
                continue
            y = chunk[self.target].astype("int8").to_numpy(copy=True)
            X = clean_float_df(chunk)
            input_data(data=X, label=y, feature_names=FEATURE_COLS)
            return True


def collect_y(shards: List[Path], target: str, chunksize=200000) -> np.ndarray:
    parts = []
    for path in shards:
        for chunk in pd.read_csv(path, chunksize=chunksize, usecols=[target]):
            chunk = chunk[chunk[target].notna()]
            if len(chunk):
                parts.append(chunk[target].astype("int8").to_numpy(copy=True))
    if not parts:
        return np.array([], dtype=np.int8)
    return np.concatenate(parts)


def train_xgb(target: str, train_shards: List[Path], args) -> Optional[xgb.Booster]:
    y = collect_y(train_shards, target)
    if len(y) < 100 or len(np.unique(y)) < 2:
        print(f"Skipping {target}: not enough positive/negative examples.")
        return None
    del y
    gc.collect()

    cache_dir = Path(args.out_dir) / "xgb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_prefix = cache_dir / f"{target}_cache"
    train_iter = ShardIter(train_shards, target, args.batch_rows, cache_prefix)
    dtrain = xgb.ExtMemQuantileDMatrix(train_iter, max_bin=args.max_bin, nthread=1)
    params = {
        "objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
        "max_depth": args.max_depth, "eta": args.learning_rate, "subsample": 0.85,
        "colsample_bytree": 0.85, "min_child_weight": 10, "lambda": 2.0,
        "max_bin": args.max_bin, "nthread": 1, "verbosity": 1,
    }
    print(f"\nTraining XGBoost target: {target}")
    booster = xgb.train(params, dtrain, num_boost_round=args.rounds)
    model_dir = Path(args.out_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_dir / f"{target}.json")
    del dtrain, train_iter
    gc.collect()
    return booster


def metrics(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else np.nan}
    if len(y) == 0 or len(np.unique(y)) < 2:
        out.update({"auc": np.nan, "logloss": np.nan, "brier": np.nan, "top10_rate": np.nan, "lift_top10": np.nan})
        return out
    out["auc"] = float(roc_auc_score(y, p))
    out["logloss"] = float(log_loss(y, p))
    out["brier"] = float(brier_score_loss(y, p))
    cutoff = np.quantile(p, 0.90)
    top = y[p >= cutoff]
    out["top10_rate"] = float(top.mean()) if len(top) else np.nan
    out["lift_top10"] = float(out["top10_rate"] / out["positive_rate"]) if out["positive_rate"] > 0 else np.nan
    return out


def predict_xgb_on_shards(booster: xgb.Booster, shards: List[Path], target: str, args):
    ys, ps = [], []
    for path in shards:
        for chunk in pd.read_csv(path, chunksize=args.predict_chunk_rows):
            if target not in chunk.columns:
                continue
            chunk = chunk[chunk[target].notna()]
            if len(chunk) == 0:
                continue
            y = chunk[target].astype("int8").to_numpy(copy=True)
            X = clean_float_df(chunk)
            dm = xgb.DMatrix(X, feature_names=FEATURE_COLS, nthread=1)
            p = booster.predict(dm)
            ys.append(y); ps.append(p)
            del dm, X, y, p, chunk
            gc.collect()
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


def fit_baseline(train_shards: List[Path], target: str, args):
    Xs, ys = [], []
    for path in train_shards:
        for chunk in pd.read_csv(path, chunksize=args.predict_chunk_rows):
            if target not in chunk.columns:
                continue
            chunk = chunk[chunk[target].notna()]
            if len(chunk) == 0:
                continue
            Xs.append(proximity_score(chunk, target).astype("float32"))
            ys.append(chunk[target].astype("int8").to_numpy(copy=True))
    if not ys:
        return None
    X = np.vstack(Xs); y = np.concatenate(ys)
    if len(np.unique(y)) < 2:
        return None
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    base.fit(X, y)
    del X, y, Xs, ys
    gc.collect()
    return base


def predict_baseline_on_shards(base, test_shards: List[Path], target: str, args):
    ys, ps = [], []
    for path in test_shards:
        for chunk in pd.read_csv(path, chunksize=args.predict_chunk_rows):
            if target not in chunk.columns:
                continue
            chunk = chunk[chunk[target].notna()]
            if len(chunk) == 0:
                continue
            y = chunk[target].astype("int8").to_numpy(copy=True)
            X = proximity_score(chunk, target).astype("float32")
            p = base.predict_proba(X)[:, 1]
            ys.append(y); ps.append(p)
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


def inspect_one_replay(path: Path, args):
    map_data = None
    for _, ev, data in read_ndjson_events(path):
        if ev == "map" and isinstance(data, dict):
            map_data = data
            break
    geom = make_geometry(map_data, args, live=None)
    print("\n=== MOON BASE 2024 ESCAPE CALIBRATION ===")
    print(f"Replay: {path.name}")
    print(f"Map width x height: {geom['map_w']} x {geom['map_h']} tiles")
    print(f"Red flag point:  ({geom['red_flag_x']:.3f}, {geom['red_flag_y']:.3f})")
    print(f"Blue flag point: ({geom['blue_flag_x']:.3f}, {geom['blue_flag_y']:.3f})")
    print(f"Flag distance: {geom['flag_dist']:.3f} tiles")
    print(f"Red out threshold progress:  {geom['red_escape_progress_threshold']:.3f}")
    print(f"Blue out threshold progress: {geom['blue_escape_progress_threshold']:.3f}")
    print("Rule: progress=0 at enemy flag, progress=1 at own flag. Out once progress >= threshold.")
    print("Adjust with --red-escape-progress / --blue-escape-progress if this is too loose/tight.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-dir", required=True)
    ap.add_argument("--sample-hz", type=float, default=10.0)
    ap.add_argument("--train-replays", type=int, default=3000)
    ap.add_argument("--val-replays", type=int, default=365)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--local-cache-dir", default=None,
                    help="Optional local Linux folder used as a one-file replay cache before parsing. Useful when replay-dir is on an external drive.")
    ap.add_argument("--shuffle-seed", type=int, default=42)
    ap.add_argument("--reuse-shards", action="store_true")
    ap.add_argument("--inspect-only", action="store_true")

    # Moon Base diagonal escape progress thresholds.
    ap.add_argument("--red-escape-progress", type=float, default=0.28)
    ap.add_argument("--blue-escape-progress", type=float, default=0.28)
    ap.add_argument("--red-flag-x", type=float, default=None)
    ap.add_argument("--red-flag-y", type=float, default=None)
    ap.add_argument("--blue-flag-x", type=float, default=None)
    ap.add_argument("--blue-flag-y", type=float, default=None)

    # Streaming/XGBoost controls.
    ap.add_argument("--shard-rows", type=int, default=50000)
    ap.add_argument("--batch-rows", type=int, default=50000)
    ap.add_argument("--predict-chunk-rows", type=int, default=100000)
    ap.add_argument("--rounds", type=int, default=120)
    ap.add_argument("--max-depth", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=0.06)
    ap.add_argument("--max-bin", type=int, default=128)
    args = ap.parse_args()

    replay_dir = Path(args.replay_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    args.out_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not replay_dir.exists():
        raise SystemExit(f"Replay folder not found: {replay_dir}")
    files = list_replay_files(replay_dir)
    if not files:
        raise SystemExit(f"No .ndjson files found in {replay_dir}")
    if len(files) <= args.train_replays + args.val_replays and not args.inspect_only:
        raise SystemExit(f"Need more than train+val files. Found {len(files)}; train={args.train_replays}; val={args.val_replays}")

    print(f"Found {len(files)} files in {replay_dir}")
    inspect_one_replay(files[0], args)
    if args.inspect_only:
        return

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(files)
    train_files = files[:args.train_replays]
    val_files = files[args.train_replays: args.train_replays + args.val_replays]
    test_files = files[args.train_replays + args.val_replays:]
    print(f"Train replays: {len(train_files)}")
    print(f"Val replays:   {len(val_files)}")
    print(f"Test replays:  {len(test_files)}")

    train_shards, train_rows, train_fail = make_shards(train_files, "train", args)
    val_shards, val_rows, val_fail = make_shards(val_files, "val", args)
    test_shards, test_rows, test_fail = make_shards(test_files, "test", args)

    if not train_shards or not test_shards:
        raise SystemExit("No train/test shards created. Parser did not match files or the folder is wrong.")

    results = []
    for target in TARGETS:
        booster = train_xgb(target, train_shards, args)
        if booster is None:
            continue
        y_test, p_xgb = predict_xgb_on_shards(booster, test_shards, target, args)
        m = metrics(y_test, p_xgb)
        m.update({"target": target, "model": "learned_XGB"})
        results.append(m)
        del booster, p_xgb
        gc.collect()

        base = fit_baseline(train_shards, target, args)
        if base is not None:
            y_test2, p_base = predict_baseline_on_shards(base, test_shards, target, args)
            m2 = metrics(y_test2, p_base)
            m2.update({"target": target, "model": "proximity_line_equation"})
            results.append(m2)
            del base, p_base, y_test2
            gc.collect()

        res_df = pd.DataFrame(results)
        res_df = res_df[["target", "model", "n", "positive_rate", "auc", "logloss", "brier", "top10_rate", "lift_top10"]]
        res_df.sort_values(["target", "model"], inplace=True)
        res_df.to_csv(out_dir / "results_vs_baseline.csv", index=False)
        print("\nCurrent results:")
        print(res_df.to_string(index=False))

    metadata = {
        "replay_dir": str(replay_dir),
        "train_replays": len(train_files), "val_replays": len(val_files), "test_replays": len(test_files),
        "train_rows": train_rows, "val_rows": val_rows, "test_rows": test_rows,
        "targets": TARGETS, "feature_cols": FEATURE_COLS, "sample_hz": args.sample_hz,
        "red_escape_progress": args.red_escape_progress, "blue_escape_progress": args.blue_escape_progress,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "targets.json").write_text(json.dumps(TARGETS, indent=2))
    (out_dir / "feature_cols.json").write_text(json.dumps(FEATURE_COLS, indent=2))
    print(f"\nSaved results to: {out_dir / 'results_vs_baseline.csv'}")
    print(f"Saved models to: {out_dir / 'models'}")
    print(f"Saved shards to: {out_dir / 'shards'}")


if __name__ == "__main__":
    main()
