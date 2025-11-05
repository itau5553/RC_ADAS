#!/usr/bin/env python3
# analyze_ultra_front.py  (replacement)
#
# Parses ultrasonic/radar/servo logs and produces:\
#  - <basename>_timeseries.csv
#  - <basename>_summary.txt
#  - <basename>_plot.png  (two stacked axes)
#  - <basename>_latency_front_threshold.csv
#  - <basename>_latency_ev.csv
#
# Latency definitions
# -------------------
# 1) Threshold-triggered latency:
#    Trigger when FRONT distance crosses from >= threshold to < threshold.
#    Latency is time to the first servo step (|Δservo_us| >= servo_step_us)
#    within latency_window_s after the trigger.
#
# 2) EV-triggered latency:
#    Trigger at EV ts for labels in {START_US_TURN, START_US_STRAIGHTEN, START_US_CENTER_HOLD}.
#    Latency is time to the first servo step after that EV (within window).
#
# Usage example:
#   python "C:\Users\imran\Desktop\thesis\python script\analysis3\analyze_ultra_front.py" ^
#          "C:\Users\imran\Desktop\thesis\testing results\ultrasonic_servo_frontonly.txt" ^
#          --outdir "C:\Users\imran\Desktop\thesis\python script\analysis3\analysis_out" ^
#          --threshold 60 --verbose --title "Ultrasonic signal vs servo actuation"
#
import argparse, os, math, sys
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EV_LATENCY_LABELS = {"START_US_TURN","START_US_STRAIGHTEN","START_US_CENTER_HOLD"}

def parse_args():
    ap = argparse.ArgumentParser(description="Analyze ultrasonic + servo logs, plot, and latency.")
    ap.add_argument("logfile", help="Path to input .txt")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--threshold", type=float, default=60.0, help="Obstacle threshold for FRONT (cm)")
    ap.add_argument("--servo-step-us", type=float, default=200.0, help="Min |Δservo_us| to count as an actuation")
    ap.add_argument("--latency-window-s", type=float, default=3.0, help="Search window after a trigger (s)")
    ap.add_argument("--title", default="Ultrasonic signal vs servo actuation", help="Plot title")
    ap.add_argument("--verbose", action="store_true", help="Print extra info")
    ap.add_argument("--side-threshold", type=float, default=20.0,
        help="Side clearance threshold (cm) used for plotting")
    return ap.parse_args()


def to_float(x: str) -> Optional[float]:
    try:
        v = float(x)
        # Treat sentinel 9999.0 as missing
        if abs(v - 9999.0) < 1e-6:
            return np.nan
        return v
    except:
        return np.nan

def to_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except:
        return None

def parse_file(path: str, verbose: bool=False):
    """
    Returns:
      samples: List[dict] with keys: ts_ms, time_s, source, state, steer_mode, front_cm, left_cm, right_cm, servo_us
      phases:  List[(phase_name, start_ts_ms, end_ts_ms)] derived from EV markers
      ev_points: List[(ts_ms, label)] of EV markers (all)
      meta:    dict with first_ts_ms, last_ts_ms, radar_gate_counts
    """
    if verbose:
        print(f"[INFO] Reading: {path}")

    samples: List[Dict] = []
    ev_points: List[Tuple[int,str]] = []
    ev_marks = []  # shading spans
    radar_gate_on = 0
    radar_gate_off = 0

    first_ts = None
    last_ts  = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            tag = parts[0]

            if tag == "T" and len(parts) >= 13:
                # T, ts, episode, STATE, front, left, right, radar_on, radar_a, radar_b, servo_us, STEER_MODE, radar_speed
                ts = to_int(parts[1]);         state = parts[3]
                front = to_float(parts[4]);    left  = to_float(parts[5]);    right = to_float(parts[6])
                servo_us = to_float(parts[10]); steer_mode = parts[11] if len(parts)>11 else ""
                if ts is None: continue

                if first_ts is None: first_ts = ts
                last_ts = ts
                time_s = (ts - first_ts)/1000.0

                samples.append(dict(
                    ts_ms=ts, time_s=time_s, source="T", state=state, steer_mode=steer_mode,
                    front_cm=front, left_cm=left, right_cm=right, servo_us=servo_us
                ))

            elif tag == "RAW" and len(parts) >= 11:
                # RAW, ts, MODE, front, left, right, r_on, r_a, r_b, r_speed, servo_us, ...
                ts = to_int(parts[1]); mode = parts[2] if len(parts)>2 else ""
                front = to_float(parts[3]); left  = to_float(parts[4]); right = to_float(parts[5])
                servo_us = to_float(parts[10]) if len(parts)>10 else np.nan
                if ts is None: continue

                if first_ts is None: first_ts = ts
                last_ts = ts
                time_s = (ts - first_ts)/1000.0

                samples.append(dict(
                    ts_ms=ts, time_s=time_s, source="RAW", state=mode, steer_mode="",
                    front_cm=front, left_cm=left, right_cm=right, servo_us=servo_us
                ))

            elif tag == "EV" and len(parts) >= 2:
                label = parts[1]
                ts = to_int(parts[2]) if len(parts) > 2 else None
                if ts is not None:
                    ev_points.append((ts,label))
                    if label == "RADAR_GATE_ON": radar_gate_on += 1
                    if label == "RADAR_GATE_OFF": radar_gate_off += 1
                    if label in ("START_US_TURN","START_US_STRAIGHTEN","START_US_CENTER_HOLD","END_US_SEQUENCE"):
                        ev_marks.append((ts, label))
                    if first_ts is None: first_ts = ts
                    last_ts = ts

    # shading phases
    label_to_phase = {
        "START_US_TURN": "TURN",
        "START_US_STRAIGHTEN": "STRAIGHT",
        "START_US_CENTER_HOLD": "CENTER",
    }
    phases = []
    if ev_marks:
        ev_marks.sort(key=lambda x: x[0])
        for i, (ts, label) in enumerate(ev_marks):
            phase = label_to_phase.get(label, None)
            if phase is None: continue
            start_ts = ts
            end_ts = ev_marks[i+1][0] if (i+1)<len(ev_marks) else (last_ts if last_ts else start_ts+1)
            phases.append((phase, start_ts, end_ts))

    meta = dict(first_ts_ms=first_ts, last_ts_ms=last_ts,
                radar_gate_on=radar_gate_on, radar_gate_off=radar_gate_off)

    if verbose:
        print(f"[INFO] Parsed samples: {len(samples)}")
        print(f"[INFO] EV points: {len(ev_points)}  (for latency & shading)")
        print(f"[INFO] Radar gates ON={radar_gate_on}, OFF={radar_gate_off}")
        if first_ts and last_ts:
            print(f"[INFO] Time span: {(last_ts-first_ts)/1000.0:.2f} s")

    return samples, phases, ev_points, meta

def summarize_basic(df: pd.DataFrame, threshold_cm: float) -> Dict:
    out = {}
    if df.empty:
        return out
    out["n_rows"] = len(df)
    out["t_start_s"] = float(df["time_s"].min())
    out["t_end_s"]   = float(df["time_s"].max())
    out["duration_s"]= out["t_end_s"] - out["t_start_s"]

    for col in ["front_cm","left_cm","right_cm"]:
        s = df[col].astype(float)
        out[f"{col}_min"]    = float(np.nanmin(s)) if np.isfinite(np.nanmin(s)) else math.nan
        out[f"{col}_mean"]   = float(np.nanmean(s)) if np.isfinite(np.nanmean(s)) else math.nan
        out[f"{col}_median"] = float(np.nanmedian(s)) if np.isfinite(np.nanmedian(s)) else math.nan
        out[f"{col}_valid_frac"] = float(np.mean(~np.isnan(s)))

    def frac_below(col):
        s = df[col].astype(float); mask = ~np.isnan(s)
        return float(np.mean((s[mask] < threshold_cm))) if mask.sum()>0 else math.nan

    out["frac_front_below_thresh"] = frac_below("front_cm")
    out["frac_left_below_thresh"]  = frac_below("left_cm")
    out["frac_right_below_thresh"] = frac_below("right_cm")

    if "servo_us" in df and df["servo_us"].notna().any():
        out["servo_us_min"] = float(df["servo_us"].min(skipna=True))
        out["servo_us_max"] = float(df["servo_us"].max(skipna=True))
    else:
        out["servo_us_min"] = math.nan; out["servo_us_max"] = math.nan
    return out

def detect_servo_steps(df: pd.DataFrame, step_us: float) -> pd.DataFrame:
    """Return a small DataFrame with times where |Δservo_us| >= step_us."""
    s = df["servo_us"].astype(float).copy()
    d = np.abs(s.diff())
    idx = d >= step_us
    events = df.loc[idx, ["time_s","ts_ms","servo_us"]].copy()
    events.rename(columns={"time_s":"servo_change_time_s",
                           "ts_ms":"servo_change_ts_ms",
                           "servo_us":"servo_us_after"}, inplace=True)
    events["servo_us_before"] = df["servo_us"].shift(1)[idx].values
    events["servo_delta_us"] = events["servo_us_after"] - events["servo_us_before"]
    return events.reset_index(drop=True)

def detect_front_threshold_triggers(df: pd.DataFrame, threshold_cm: float) -> pd.DataFrame:
    """Triggers when front crosses from >= threshold to < threshold."""
    f = df["front_cm"].astype(float).copy()
    prev = f.shift(1)
    trig_idx = (prev >= threshold_cm) & (f < threshold_cm)
    trig = df.loc[trig_idx, ["time_s","ts_ms","front_cm"]].copy()
    trig.rename(columns={"time_s":"trigger_time_s","ts_ms":"trigger_ts_ms",
                         "front_cm":"front_cm_at_trigger"}, inplace=True)
    trig["trigger_type"] = "front<threshold"
    return trig.reset_index(drop=True)

def pair_triggers_to_servo(triggers: pd.DataFrame, servo_steps: pd.DataFrame,
                           window_s: float) -> pd.DataFrame:
    """For each trigger, find the earliest servo step within window_s after."""
    out_rows = []
    ss = servo_steps.sort_values("servo_change_time_s").reset_index(drop=True)
    for _, r in triggers.iterrows():
        t0 = r["trigger_time_s"]
        cand = ss[ss["servo_change_time_s"] >= t0]
        cand = cand[cand["servo_change_time_s"] <= t0 + window_s]
        if len(cand):
            first = cand.iloc[0]
            out_rows.append({
                **r.to_dict(),
                "servo_change_time_s": float(first["servo_change_time_s"]),
                "servo_us_before": float(first["servo_us_before"]),
                "servo_us_after": float(first["servo_us_after"]),
                "servo_delta_us": float(first["servo_delta_us"]),
                "latency_s": float(first["servo_change_time_s"] - t0)
            })
        else:
            out_rows.append({**r.to_dict(),
                             "servo_change_time_s": np.nan,
                             "servo_us_before": np.nan,
                             "servo_us_after": np.nan,
                             "servo_delta_us": np.nan,
                             "latency_s": np.nan})
    return pd.DataFrame(out_rows)

def detect_ev_triggers(ev_points: List[Tuple[int,str]], first_ts_ms: int) -> pd.DataFrame:
    rows = []
    for ts, label in ev_points:
        if label in EV_LATENCY_LABELS:
            rows.append(dict(trigger_type=label,
                             trigger_ts_ms=ts,
                             trigger_time_s=(ts - first_ts_ms)/1000.0))
    return pd.DataFrame(rows).sort_values("trigger_time_s").reset_index(drop=True)

def summarize_latency_table(df_lat: pd.DataFrame) -> Dict:
    stats = {}
    if df_lat.empty or df_lat["latency_s"].isna().all():
        return {"count": 0}
    valid = df_lat["latency_s"].dropna()
    stats["count"] = int(valid.count())
    stats["mean_s"] = float(valid.mean())
    stats["median_s"] = float(valid.median())
    stats["min_s"] = float(valid.min())
    stats["max_s"] = float(valid.max())
    return stats

def save_summary_txt(summary_basic: Dict, meta: Dict, threshold_cm: float,
                     lat_front_stats: Dict, lat_ev_stats: Dict, path_txt: str):
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write("ULTRASONIC RUN SUMMARY\n")
        f.write("======================\n")
        f.write(f"duration_s: {summary_basic.get('duration_s', float('nan')):.3f}\n")
        f.write(f"samples   : {summary_basic.get('n_rows', 0)}\n")
        f.write(f"threshold_cm: {threshold_cm}\n\n")

        f.write("Distances (cm):\n")
        for side in ["front","left","right"]:
            f.write(f"  {side}: min={summary_basic.get(f'{side}_cm_min',math.nan):.2f}, "
                    f"mean={summary_basic.get(f'{side}_cm_mean',math.nan):.2f}, "
                    f"median={summary_basic.get(f'{side}_cm_median',math.nan):.2f}, "
                    f"valid_frac={summary_basic.get(f'{side}_cm_valid_frac',math.nan):.3f}\n")

        f.write("\nFraction below threshold:\n")
        f.write(f"  front: {summary_basic.get('frac_front_below_thresh',math.nan):.3f}\n")
        f.write(f"  left : {summary_basic.get('frac_left_below_thresh',math.nan):.3f}\n")
        f.write(f"  right: {summary_basic.get('frac_right_below_thresh',math.nan):.3f}\n")

        f.write("\nServo range (µs):\n")
        f.write(f"  min: {summary_basic.get('servo_us_min',math.nan):.1f}, "
                f"max: {summary_basic.get('servo_us_max',math.nan):.1f}\n")

        f.write("\nRadar gating events (counts):\n")
        f.write(f"  RADAR_GATE_ON : {meta.get('radar_gate_on',0)}\n")
        f.write(f"  RADAR_GATE_OFF: {meta.get('radar_gate_off',0)}\n")

        f.write("\nLatency (front<threshold → first servo step):\n")
        if lat_front_stats.get("count",0)>0:
            f.write(f"  n={lat_front_stats['count']}, "
                    f"mean={lat_front_stats['mean_s']:.3f}s, "
                    f"median={lat_front_stats['median_s']:.3f}s, "
                    f"min={lat_front_stats['min_s']:.3f}s, "
                    f"max={lat_front_stats['max_s']:.3f}s\n")
        else:
            f.write("  no valid events\n")

        f.write("\nLatency (EV markers → first servo step):\n")
        if lat_ev_stats.get("count",0)>0:
            f.write(f"  n={lat_ev_stats['count']}, "
                    f"mean={lat_ev_stats['mean_s']:.3f}s, "
                    f"median={lat_ev_stats['median_s']:.3f}s, "
                    f"min={lat_ev_stats['min_s']:.3f}s, "
                    f"max={lat_ev_stats['max_s']:.3f}s\n")
        else:
            f.write("  no valid events\n")

def plot_timeseries(df: pd.DataFrame, phases: List, out_png: str, title: str, threshold: float = 60.0):
    if df.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.text(0.5, 0.5, "No samples parsed", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(14,8))

    # --- Top subplot: ultrasonic distances ---
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(df["time_s"], df["front_cm"], label="Front (cm)")
    ax1.plot(df["time_s"], df["left_cm"],  label="Left (cm)")
    ax1.plot(df["time_s"], df["right_cm"], label="Right (cm)")

    # Add threshold line (red dashed)
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'Threshold = {threshold} cm')

    ax1.set_ylabel("Distance (cm)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Shade EV phases on ax1
    for phase, start_ts, end_ts in phases:
        xs = (start_ts - df["ts_ms"].min()) / 1000.0
        xe = (end_ts   - df["ts_ms"].min()) / 1000.0
        color = dict(TURN="tab:orange", STRAIGHT="tab:green", CENTER="tab:purple").get(phase, "lightgray")
        ax1.axvspan(xs, xe, alpha=0.15, color=color)

    # --- Bottom subplot: servo signal ---
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.plot(df["time_s"], df["servo_us"], label="Servo (µs)")
    ax2.set_ylabel("Servo (µs)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    # Secondary right axis (approx degrees)
    def us_to_deg(us): return (us - 1500.0) * (90.0 / 1000.0)
    y2 = ax2.twinx()
    y_ticks = ax2.get_yticks()
    y2.set_ylim(ax2.get_ylim())
    y2.set_yticks(y_ticks)
    y2.set_yticklabels([f"{us_to_deg(y):.0f}°" for y in y_ticks])
    y2.set_ylabel("Steer (deg, approx)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    samples, phases, ev_points, meta = parse_file(args.logfile, verbose=args.verbose)
    df = pd.DataFrame(samples)
    base = os.path.splitext(os.path.basename(args.logfile))[0]

    out_csv = os.path.join(args.outdir, f"{base}_timeseries.csv")
    out_txt = os.path.join(args.outdir, f"{base}_summary.txt")
    out_png = os.path.join(args.outdir, f"{base}_plot.png")
    out_lat_front = os.path.join(args.outdir, f"{base}_latency_front_threshold.csv")
    out_lat_ev    = os.path.join(args.outdir, f"{base}_latency_ev.csv")

    if df.empty:
        print("[WARN] No samples parsed; writing placeholder outputs.")
        df.to_csv(out_csv, index=False)
        open(out_txt,"w").write("No samples parsed.\n")
        plt.figure().savefig(out_png); plt.close()
        return

    # numeric/sorted
    for c in ["time_s","front_cm","left_cm","right_cm","servo_us","ts_ms"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["time_s"]).reset_index(drop=True)

    # save timeseries
    df.to_csv(out_csv, index=False)

    # --- Latency analysis ---
    servo_steps = detect_servo_steps(df, step_us=args.servo_step_us)

    # 1) threshold triggers
    trig_front = detect_front_threshold_triggers(df, threshold_cm=args.threshold)
    lat_front  = pair_triggers_to_servo(trig_front, servo_steps, args.latency_window_s)
    lat_front.to_csv(out_lat_front, index=False)

    # 2) EV triggers
    ev_trig = detect_ev_triggers(ev_points, first_ts_ms=int(df["ts_ms"].min()))
    lat_ev  = pair_triggers_to_servo(ev_trig, servo_steps, args.latency_window_s)
    lat_ev.to_csv(out_lat_ev, index=False)

    # summaries
    summary_basic = summarize_basic(df, args.threshold)
    lat_front_stats = summarize_latency_table(lat_front)
    lat_ev_stats    = summarize_latency_table(lat_ev)

    # write TXT
    save_summary_txt(summary_basic, meta, args.threshold,
                     lat_front_stats, lat_ev_stats, out_txt)

    # plot
    plot_timeseries(df, phases, out_png, title=args.title)

    print("[OK] Saved outputs:")
    print(f"  CSV (timeseries): {out_csv}")
    print(f"  Plot PNG        : {out_png}")
    print(f"  Summary TXT     : {out_txt}")
    print(f"  Latency (front<threshold) CSV: {out_lat_front}")
    print(f"  Latency (EV) CSV             : {out_lat_ev}")

if __name__ == "__main__":
    main()
