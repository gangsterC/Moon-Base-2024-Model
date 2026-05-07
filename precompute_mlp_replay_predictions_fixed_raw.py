#!/usr/bin/env python3
"""
Fixed RAW-replay MLP precompute script for Moon Base 2024.

Use this for brand-new .ndjson replays that were NOT already inside your fixed_xscore shards.

Why this version exists:
The first raw precompute path used the parser's raw feature geometry and only partially repaired
features. That can make the MLP see out-of-distribution inputs and produce nonsense like 99% cap
probabilities everywhere.

This script:
  1. parses the raw .ndjson replay with moonbase_ctf_xgb_refined_stream_cached.py
  2. recomputes the main Moon Base geometry features with scaled replay coordinates
  3. fills any missing training features
  4. runs the MLP
  5. applies Platt calibration
  6. saves a precomputed prediction JSON/CSV for the overlay

Outputs:
  <out-dir>/<replay-stem>_mlp_predictions.json
  <out-dir>/<replay-stem>_mlp_predictions.csv
"""

import argparse
import gc
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, n_class, n_reg, hidden):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            h = int(h)
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(0.10)]
            prev = h
        # Must match the trained checkpoint names.
        self.body = nn.Sequential(*layers)
        self.ch = nn.Linear(prev, n_class) if n_class > 0 else None
        self.rh = nn.Linear(prev, n_reg) if n_reg > 0 else None

    def forward(self, x):
        h = self.body(x)
        cl = self.ch(h) if self.ch is not None else None
        rg = self.rh(h) if self.rh is not None else None
        return cl, rg


def load_module(path):
    p = Path(path).expanduser()
    if not p.exists():
        raise SystemExit(f"Base parser script not found: {p}")
    spec = importlib.util.spec_from_file_location("base_parser", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def dist_np(x, y, px, py):
    return np.sqrt((x - px) ** 2 + (y - py) ** 2)


def nearest_static_series(x_arr, y_arr, points):
    if not points:
        return np.full(len(x_arr), 999.0, dtype=np.float32)
    out = np.full(len(x_arr), np.inf, dtype=np.float64)
    x = x_arr.astype(float)
    y = y_arr.astype(float)
    for px, py in points:
        d = (x - px) ** 2 + (y - py) ** 2
        out = np.minimum(out, d)
    return np.sqrt(out).astype(np.float32)


def second_nearest_to_point(points_x, points_y, tx, ty):
    if len(points_x) == 0:
        return 999.0, 999.0, np.nan, np.nan
    d = np.sqrt((points_x - tx) ** 2 + (points_y - ty) ** 2)
    order = np.argsort(d)
    first = order[0]
    nearest = float(d[first])
    second = float(d[order[1]]) if len(order) > 1 else 999.0
    return nearest, second, float(points_x[first]), float(points_y[first])


def read_map_data(base, replay_path):
    for _, ev, data in base.read_ndjson_events(Path(replay_path)):
        if ev == "map" and isinstance(data, dict):
            return data
    return None


def scaled_static_points(base, map_data, coord_scale):
    pts = base.extract_map_points(map_data)
    out = {}
    for k, arr in pts.items():
        out[k] = [(float(x) * coord_scale, float(y) * coord_scale) for x, y in arr]
    return out


def add_current_score_diff(df):
    if "current_score_diff" in df.columns:
        df["current_score_diff"] = df["current_score_diff"].astype(float)
        return df
    if "score_diff_red" in df.columns:
        df["current_score_diff"] = df["score_diff_red"].astype(float)
    elif "red_score_diff" in df.columns:
        df["current_score_diff"] = df["red_score_diff"].astype(float)
    elif "score_diff" in df.columns:
        df["current_score_diff"] = df["score_diff"].astype(float)
    elif "score_r" in df.columns and "score_b" in df.columns:
        df["current_score_diff"] = df["score_r"].astype(float) - df["score_b"].astype(float)
    elif "red_score" in df.columns and "blue_score" in df.columns:
        df["current_score_diff"] = df["red_score"].astype(float) - df["blue_score"].astype(float)
    else:
        print("WARNING: no score columns found; setting current_score_diff=0")
        df["current_score_diff"] = 0.0
    return df


def recompute_geometry_features(df, base, replay_path, args, feature_cols):
    map_data = read_map_data(base, replay_path)
    if map_data is None:
        print("WARNING: could not find map event. Using default Moon Base scaled flags.")
        static = {"red_flags": [(args.red_flag_x, args.red_flag_y)], "blue_flags": [(args.blue_flag_x, args.blue_flag_y)]}
        map_w, map_h = 16.8, 12.0
    else:
        static = scaled_static_points(base, map_data, args.coord_scale)
        try:
            mw, mh = base.map_dimensions_from_tiles(map_data)
            map_w, map_h = float(mw) * args.coord_scale, float(mh) * args.coord_scale
        except Exception:
            map_w, map_h = 16.8, 12.0

    red_flags = static.get("red_flags", [])
    blue_flags = static.get("blue_flags", [])
    if red_flags:
        rf_x, rf_y = red_flags[0]
    else:
        rf_x, rf_y = args.red_flag_x, args.red_flag_y
    if blue_flags:
        bf_x, bf_y = blue_flags[0]
    else:
        bf_x, bf_y = args.blue_flag_x, args.blue_flag_y

    red_thr = float(args.red_escape_progress)
    blue_thr = float(args.blue_escape_progress)

    print("\n=== Corrected inference geometry ===")
    print(f"coord_scale={args.coord_scale}")
    print(f"map_w/map_h={map_w:.3f} x {map_h:.3f}")
    print(f"red flag=({rf_x:.3f}, {rf_y:.3f})")
    print(f"blue flag=({bf_x:.3f}, {bf_y:.3f})")
    print(f"thresholds red={red_thr:.3f}, blue={blue_thr:.3f}")

    # Ensure key columns exist.
    for c in [
        "red_fc_x", "red_fc_y", "red_fc_vx", "red_fc_vy", "has_red_fc",
        "blue_fc_x", "blue_fc_y", "blue_fc_vx", "blue_fc_vy", "has_blue_fc",
    ]:
        if c not in df.columns:
            df[c] = 0.0

    vx_rb = bf_x - rf_x
    vy_rb = bf_y - rf_y
    flag_d2 = vx_rb * vx_rb + vy_rb * vy_rb + 1e-6
    flag_dist = math.sqrt(flag_d2)

    df["map_w"] = map_w
    df["map_h"] = map_h
    df["red_flag_x"] = rf_x
    df["red_flag_y"] = rf_y
    df["blue_flag_x"] = bf_x
    df["blue_flag_y"] = bf_y
    df["flag_dist"] = flag_dist
    df["red_escape_progress_threshold"] = red_thr
    df["blue_escape_progress_threshold"] = blue_thr

    red_x = df["red_fc_x"].astype(float)
    red_y = df["red_fc_y"].astype(float)
    blue_x = df["blue_fc_x"].astype(float)
    blue_y = df["blue_fc_y"].astype(float)

    red_prog = ((red_x - bf_x) * (rf_x - bf_x) + (red_y - bf_y) * (rf_y - bf_y)) / flag_d2
    blue_prog = ((blue_x - rf_x) * (bf_x - rf_x) + (blue_y - rf_y) * (bf_y - rf_y)) / flag_d2

    has_red = df["has_red_fc"].fillna(0).astype(int)
    has_blue = df["has_blue_fc"].fillna(0).astype(int)

    df["red_progress"] = red_prog
    df["blue_progress"] = blue_prog

    df["red_fc_out"] = ((has_red == 1) & (red_prog >= red_thr)).astype("int8")
    df["blue_fc_out"] = ((has_blue == 1) & (blue_prog >= blue_thr)).astype("int8")
    df["red_fc_in_base"] = ((has_red == 1) & (df["red_fc_out"] == 0)).astype("int8")
    df["blue_fc_in_base"] = ((has_blue == 1) & (df["blue_fc_out"] == 0)).astype("int8")

    df["red_escape_margin"] = red_prog - red_thr
    df["blue_escape_margin"] = blue_prog - blue_thr

    red_ux = (rf_x - bf_x) / flag_dist
    red_uy = (rf_y - bf_y) / flag_dist
    blue_ux = (bf_x - rf_x) / flag_dist
    blue_uy = (bf_y - rf_y) / flag_dist

    df["red_vel_to_escape"] = df["red_fc_vx"].astype(float) * red_ux + df["red_fc_vy"].astype(float) * red_uy
    df["blue_vel_to_escape"] = df["blue_fc_vx"].astype(float) * blue_ux + df["blue_fc_vy"].astype(float) * blue_uy

    df["red_fc_d_redflag"] = dist_np(red_x, red_y, rf_x, rf_y)
    df["red_fc_d_blueflag"] = dist_np(red_x, red_y, bf_x, bf_y)
    df["blue_fc_d_redflag"] = dist_np(blue_x, blue_y, rf_x, rf_y)
    df["blue_fc_d_blueflag"] = dist_np(blue_x, blue_y, bf_x, bf_y)

    for side, x, y in [("red", red_x, red_y), ("blue", blue_x, blue_y)]:
        df[f"{side}_fc_d_wall"] = nearest_static_series(x, y, static.get("walls", []))
        df[f"{side}_fc_d_spike"] = nearest_static_series(x, y, static.get("spikes", []))
        df[f"{side}_fc_d_boost"] = nearest_static_series(x, y, static.get("boosts", []))
        df[f"{side}_fc_d_bomb"] = nearest_static_series(x, y, static.get("bombs", []))
        df[f"{side}_fc_d_pup"] = nearest_static_series(x, y, static.get("pups", []))

    # Player slot corrected distances/progress.
    for team in ["red", "blue"]:
        for i in range(1, 5):
            prefix = f"{team}{i}"
            xcol, ycol = f"{prefix}_x", f"{prefix}_y"
            if xcol not in df.columns or ycol not in df.columns:
                continue
            x = df[xcol].astype(float)
            y = df[ycol].astype(float)
            df[f"{prefix}_d_redflag"] = dist_np(x, y, rf_x, rf_y)
            df[f"{prefix}_d_blueflag"] = dist_np(x, y, bf_x, bf_y)
            df[f"{prefix}_red_progress"] = ((x - bf_x) * (rf_x - bf_x) + (y - bf_y) * (rf_y - bf_y)) / flag_d2
            df[f"{prefix}_blue_progress"] = ((x - rf_x) * (bf_x - rf_x) + (y - rf_y) * (bf_y - rf_y)) / flag_d2

    vals = {
        "red_nearest_blueflag": [], "red_second_blueflag": [], "red_nearest_blueflag_x": [], "red_nearest_blueflag_y": [],
        "blue_nearest_blueflag": [], "blue_second_blueflag": [],
        "blue_nearest_redflag": [], "blue_second_redflag": [], "blue_nearest_redflag_x": [], "blue_nearest_redflag_y": [],
        "red_nearest_redflag": [], "red_second_redflag": [],
        "red_players_in_blue_base": [], "blue_players_in_red_base": [],
        "red_players_outside_blue_base": [], "blue_players_outside_red_base": [],
    }

    for _, row in df.iterrows():
        rx, ry, bx, by = [], [], [], []
        for idx in range(1, 5):
            if row.get(f"red{idx}_present", 0) == 1:
                rx.append(float(row.get(f"red{idx}_x", np.nan)))
                ry.append(float(row.get(f"red{idx}_y", np.nan)))
            if row.get(f"blue{idx}_present", 0) == 1:
                bx.append(float(row.get(f"blue{idx}_x", np.nan)))
                by.append(float(row.get(f"blue{idx}_y", np.nan)))

        rx = np.asarray(rx, dtype=float)
        ry = np.asarray(ry, dtype=float)
        bx = np.asarray(bx, dtype=float)
        by = np.asarray(by, dtype=float)

        rn_bf, rs_bf, rnbx, rnby = second_nearest_to_point(rx, ry, bf_x, bf_y)
        bn_bf, bs_bf, _, _ = second_nearest_to_point(bx, by, bf_x, bf_y)
        bn_rf, bs_rf, bnrx, bnry = second_nearest_to_point(bx, by, rf_x, rf_y)
        rn_rf, rs_rf, _, _ = second_nearest_to_point(rx, ry, rf_x, rf_y)

        vals["red_nearest_blueflag"].append(rn_bf); vals["red_second_blueflag"].append(rs_bf)
        vals["red_nearest_blueflag_x"].append(rnbx); vals["red_nearest_blueflag_y"].append(rnby)
        vals["blue_nearest_blueflag"].append(bn_bf); vals["blue_second_blueflag"].append(bs_bf)
        vals["blue_nearest_redflag"].append(bn_rf); vals["blue_second_redflag"].append(bs_rf)
        vals["blue_nearest_redflag_x"].append(bnrx); vals["blue_nearest_redflag_y"].append(bnry)
        vals["red_nearest_redflag"].append(rn_rf); vals["red_second_redflag"].append(rs_rf)

        if len(rx):
            red_prog_players = ((rx - bf_x) * (rf_x - bf_x) + (ry - bf_y) * (rf_y - bf_y)) / flag_d2
            vals["red_players_in_blue_base"].append(int((red_prog_players < red_thr).sum()))
            vals["red_players_outside_blue_base"].append(int((red_prog_players >= red_thr).sum()))
        else:
            vals["red_players_in_blue_base"].append(0)
            vals["red_players_outside_blue_base"].append(0)

        if len(bx):
            blue_prog_players = ((bx - rf_x) * (bf_x - rf_x) + (by - rf_y) * (bf_y - rf_y)) / flag_d2
            vals["blue_players_in_red_base"].append(int((blue_prog_players < blue_thr).sum()))
            vals["blue_players_outside_red_base"].append(int((blue_prog_players >= blue_thr).sum()))
        else:
            vals["blue_players_in_red_base"].append(0)
            vals["blue_players_outside_red_base"].append(0)

    for k, v in vals.items():
        df[k] = v

    # Fill missing features only at the end, but print how many were absent.
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"\nWARNING: {len(missing)} feature columns missing from raw parse; filling with 0.")
        print("First missing columns:", missing[:40])
        for c in missing:
            df[c] = 0.0

    return df


def clean_X(df, feature_cols):
    return (
        df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
        .astype("float32")
        .to_numpy(copy=True)
    )


def summarize(name, vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        print(f"{name:22s} no finite values")
        return
    print(
        f"{name:22s} min={vals.min(): .5f} max={vals.max(): .5f} "
        f"mean={vals.mean(): .5f} p50={np.quantile(vals,.5): .5f} p90={np.quantile(vals,.9): .5f}"
    )


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", required=True)
    ap.add_argument("--base-script", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calibrator", default=None)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sample-hz", type=float, default=5.0)
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--coord-scale", type=float, default=0.4)
    ap.add_argument("--red-escape-progress", type=float, default=0.28)
    ap.add_argument("--blue-escape-progress", type=float, default=0.28)
    ap.add_argument("--red-flag-x", type=float, default=3.0)
    ap.add_argument("--red-flag-y", type=float, default=3.0)
    ap.add_argument("--blue-flag-x", type=float, default=13.8)
    ap.add_argument("--blue-flag-y", type=float, default=9.0)
    args = ap.parse_args()

    replay_path = Path(args.replay).expanduser()
    if not replay_path.exists():
        raise SystemExit(f"Replay file not found: {replay_path}")

    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    base = load_module(args.base_script)

    ckpt_path = Path(args.model_dir).expanduser() / "best_model.pt"
    print(f"Loading model: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    feature_cols = ckpt["feature_cols"]
    class_targets = ckpt["class_targets"]
    reg_targets = ckpt["reg_targets"]
    mean = np.asarray(ckpt["mean"], dtype="float32")
    std = np.asarray(ckpt["std"], dtype="float32")
    r_mean = np.asarray(ckpt.get("r_mean", np.zeros(len(reg_targets))), dtype="float32")
    r_std = np.asarray(ckpt.get("r_std", np.ones(len(reg_targets))), dtype="float32")
    hidden = ckpt["hidden"]

    model = MLP(len(feature_cols), len(class_targets), len(reg_targets), hidden)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    calibrators = {}
    cal_path = Path(args.calibrator).expanduser() if args.calibrator else None
    if cal_path and cal_path.exists():
        print(f"Loading calibrators: {cal_path}")
        calibrators = joblib.load(cal_path).get("calibrators", {})
    else:
        print("No calibrator loaded; using raw sigmoid probabilities.")

    # First parse with corrected flag coordinates.
    ns = SimpleNamespace(
        sample_hz=float(args.sample_hz),
        red_escape_progress=float(args.red_escape_progress),
        blue_escape_progress=float(args.blue_escape_progress),
        red_flag_x=float(args.red_flag_x),
        red_flag_y=float(args.red_flag_y),
        blue_flag_x=float(args.blue_flag_x),
        blue_flag_y=float(args.blue_flag_y),
    )

    print(f"Parsing replay at {args.sample_hz} Hz: {replay_path}")
    rows = base.parse_replay_to_rows(replay_path, ns)
    if not rows:
        raise SystemExit("Parser returned zero rows.")

    df = pd.DataFrame(rows)
    df = add_current_score_diff(df)
    df = recompute_geometry_features(df, base, replay_path, args, feature_cols)

    print("\n=== Feature sanity after full geometry repair ===")
    print(f"Rows parsed: {len(df):,}")
    print(f"t range: {df['t'].min():.2f} to {df['t'].max():.2f}")
    for c in [
        "has_red_fc", "has_blue_fc", "red_fc_out", "blue_fc_out",
        "red_fc_in_base", "blue_fc_in_base",
        "red_progress", "blue_progress",
        "red_escape_margin", "blue_escape_margin",
        "current_score_diff",
    ]:
        if c in df.columns:
            summarize(c, df[c].to_numpy())

    X = clean_X(df, feature_cols)
    X = (X - mean) / std

    # Quick OOD sanity: if standardized values are absurd, print warning.
    finite = X[np.isfinite(X)]
    if finite.size:
        print("\n=== Standardized feature sanity ===")
        print(f"abs z p50={np.quantile(np.abs(finite), .50):.3f} p95={np.quantile(np.abs(finite), .95):.3f} p99={np.quantile(np.abs(finite), .99):.3f} max={np.max(np.abs(finite)):.3f}")

    class_logits_parts = []
    reg_parts = []
    for start in range(0, len(X), args.batch_size):
        xb = torch.from_numpy(X[start:start + args.batch_size])
        cl, rg = model(xb)
        if cl is not None:
            class_logits_parts.append(cl.cpu().numpy())
        if rg is not None:
            reg_parts.append(rg.cpu().numpy())

    logits = np.vstack(class_logits_parts) if class_logits_parts else np.zeros((len(X), 0), dtype="float32")
    reg_scaled = np.vstack(reg_parts) if reg_parts else np.zeros((len(X), 0), dtype="float32")
    reg = reg_scaled * r_std + r_mean if len(reg_targets) else reg_scaled

    records = []
    for i in range(len(df)):
        rec = {
            "t": float(df.iloc[i]["t"]),
            "current_score_diff": float(df.iloc[i].get("current_score_diff", 0.0)),
        }

        for j, target in enumerate(class_targets):
            raw = float(sigmoid(logits[i, j]))
            rec[target + "_raw"] = raw
            if target in calibrators:
                rec[target] = float(calibrators[target].predict_proba([[float(logits[i, j])]])[0, 1])
            else:
                rec[target] = raw

        for j, target in enumerate(reg_targets):
            val = float(reg[i, j])
            rec[target] = val
            if target == "red_score_delta_20s":
                rec["red_xscore_20s"] = rec["current_score_diff"] + val

        records.append(rec)

    pred_df = pd.DataFrame(records)

    print("\n=== Prediction sanity after full geometry repair ===")
    for c in [
        "red_cap_10s", "blue_cap_10s", "red_cap_20s", "blue_cap_20s",
        "red_out_base_5s", "blue_out_base_5s",
        "red_escape_5s", "blue_escape_5s",
        "red_fc_lost_2s", "blue_fc_lost_2s",
        "red_xscore_20s",
    ]:
        if c in pred_df.columns:
            summarize(c, pred_df[c].to_numpy())

    payload = {
        "ok": True,
        "mode": "precomputed_fixed_raw_full_geometry_repair",
        "file": replay_path.name,
        "path": str(replay_path),
        "sample_hz": float(args.sample_hz),
        "count": len(records),
        "class_targets": class_targets,
        "reg_targets": reg_targets,
        "calibrated": bool(calibrators),
        "predictions": records,
    }

    json_path = out_dir / f"{replay_path.stem}_mlp_predictions.json"
    csv_path = out_dir / f"{replay_path.stem}_mlp_predictions.csv"

    json_path.write_text(json.dumps(payload))
    pred_df.to_csv(csv_path, index=False)

    print("\nSaved:")
    print(json_path)
    print(csv_path)


if __name__ == "__main__":
    main()
