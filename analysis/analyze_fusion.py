#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Robust fusion log analyzer with loud progress prints and safe fallbacks.

import os, sys, re, math, argparse
import numpy as np
import pandas as pd

# Force a non-interactive backend so PNG saving always works on Windows
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def p(s): print(s, flush=True)

def to_int(x):
    try: return int(x)
    except: return None

def to_float(x):
    try:
        v = float(x)
        if abs(v - 9999.0) < 1e-9:  # sentinel = missing
            return float("nan")
        return v
    except:
        return float("nan")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--front-threshold", type=float, default=60.0)
    ap.add_argument("--side-threshold", type=float, default=20.0)
    ap.add_argument("--servo-step-us", type=float, default=200.0)
    ap.add_argument("--latency-window-s", type=float, default=3.0)
    ap.add_argument("--title", default="Fusion analysis")
    return ap.parse_args()

def parse_file(path):
    # Expected RAW shape from your snippet:
    # RAW,ts,BOTH,front,left,right,rad_has,ang,dist,speed,servo,drive,av
    rows = []
    ev_points = []     # (ts, label, value or NaN)
    gate_stack = []    # for RADAR_GATE_ON/OFF spans
    gate_spans = []
    r_rows = []

    first_ts = None
    last_ts  = None

    pat = re.compile(r'^\s*(RAW|T|EV|R)\s*,')
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s: continue
            m = pat.match(s)
            if not m:
                continue
            tag = m.group(1)
            parts = s.split(",")

            if tag == "RAW":
                # be permissive with length checks
                if len(parts) < 13:
                    continue
                ts = to_int(parts[1])
                if ts is None: 
                    continue
                if first_ts is None: first_ts = ts
                last_ts = ts
                time_s = (ts - first_ts)/1000.0

                mode = parts[2]
                front = to_float(parts[3]); left = to_float(parts[4]); right = to_float(parts[5])
                rad_has = to_int(parts[6]) or 0
                rad_ang = to_float(parts[7]); rad_dist = to_float(parts[8]); rad_spd = to_float(parts[9])
                servo = to_float(parts[10])

                # if radar invalid, blank the readings so plots are clean
                if rad_has == 0:
                    rad_ang = float("nan"); rad_dist = float("nan"); rad_spd = float("nan")

                rows.append(dict(
                    ts_ms=ts, time_s=time_s, mode=mode,
                    front_cm=front, left_cm=left, right_cm=right,
                    rad_has=rad_has, rad_ang_deg=rad_ang, rad_dist_cm=rad_dist, rad_speed_cms=rad_spd,
                    servo_us=servo
                ))

            elif tag == "T":
                if len(parts) >= 3:
                    ts = to_int(parts[1])
                    if ts is not None:
                        if first_ts is None: first_ts = ts
                        last_ts = ts

            elif tag == "EV":
                # EV,LABEL,ts[,value]
                if len(parts) >= 3:
                    label = parts[1]
                    ts = to_int(parts[2])
                    val = to_float(parts[3]) if len(parts) >= 4 else float("nan")
                    if ts is not None:
                        ev_points.append((ts, label, val))
                        if label == "RADAR_GATE_ON":
                            gate_stack.append(ts)
                        elif label == "RADAR_GATE_OFF":
                            if gate_stack:
                                st = gate_stack.pop(0)
                                gate_spans.append((st, ts))

            elif tag == "R":
                # R,angle,dist?,speed  (no ts)
                if len(parts) >= 4:
                    r_rows.append(dict(
                        rad_ang_deg=to_float(parts[1]),
                        rad_dist_raw=to_float(parts[2]),
                        rad_speed_cms=to_float(parts[3])
                    ))

    # Close any unterminated gate spans at last_ts
    if gate_stack:
        end = last_ts if last_ts is not None else (gate_stack[-1] + 1)
        for st in gate_stack:
            gate_spans.append((st, end))

    df = pd.DataFrame(rows).sort_values("ts_ms").reset_index(drop=True)
    df_r = pd.DataFrame(r_rows)
    meta = dict(first_ts_ms=first_ts, last_ts_ms=last_ts)
    return df, ev_points, gate_spans, meta, df_r

def detect_servo_steps(df, step_us):
    if df.empty or "servo_us" not in df:
        return pd.DataFrame()
    s = pd.to_numeric(df["servo_us"], errors="coerce")
    d = np.abs(s.diff())
    idx = d >= step_us
    out = df.loc[idx, ["time_s","ts_ms","servo_us"]].copy()
    out.rename(columns={"time_s":"servo_change_time_s",
                        "ts_ms":"servo_change_ts_ms",
                        "servo_us":"servo_us_after"}, inplace=True)
    out["servo_us_before"] = df["servo_us"].shift(1)[idx].values
    out["servo_delta_us"]  = out["servo_us_after"] - out["servo_us_before"]
    return out.reset_index(drop=True)

def triggers_front(df, front_thresh):
    if df.empty or "front_cm" not in df: return pd.DataFrame()
    f = pd.to_numeric(df["front_cm"], errors="coerce")
    prev = f.shift(1)
    idx = (prev >= front_thresh) & (f < front_thresh)
    t = df.loc[idx, ["time_s","ts_ms","front_cm"]].copy()
    t.rename(columns={"time_s":"trigger_time_s","ts_ms":"trigger_ts_ms",
                      "front_cm":"front_cm_at_trigger"}, inplace=True)
    t["trigger_type"] = "front<threshold"
    return t.reset_index(drop=True)

def ev_triggers(ev_points, first_ts_ms):
    target = {"RADAR_GATE_ON","START_US_TURN","START_US_STRAIGHTEN","START_US_CENTER_HOLD"}
    rows = []
    for ts, label, val in ev_points:
        if label in target and first_ts_ms is not None:
            rows.append(dict(trigger_type=label,
                             trigger_ts_ms=ts,
                             trigger_time_s=(ts-first_ts_ms)/1000.0,
                             ev_value=val))
    return pd.DataFrame(rows).sort_values("trigger_time_s").reset_index(drop=True)

def pair_latency(trig_df, servo_steps, win_s):
    out = []
    if trig_df.empty: return pd.DataFrame()
    ss = servo_steps.sort_values("servo_change_time_s").reset_index(drop=True)
    for _, r in trig_df.iterrows():
        t0 = r["trigger_time_s"]
        cand = ss[(ss["servo_change_time_s"] >= t0) & (ss["servo_change_time_s"] <= t0 + win_s)]
        if len(cand):
            first = cand.iloc[0]
            d = dict(r)
            d.update({
                "servo_change_time_s": float(first["servo_change_time_s"]),
                "servo_us_before": float(first["servo_us_before"]),
                "servo_us_after": float(first["servo_us_after"]),
                "servo_delta_us": float(first["servo_delta_us"]),
                "latency_s": float(first["servo_change_time_s"] - t0)
            })
            out.append(d)
        else:
            d = dict(r); d.update({"servo_change_time_s": np.nan,
                                   "servo_us_before": np.nan,
                                   "servo_us_after": np.nan,
                                   "servo_delta_us": np.nan,
                                   "latency_s": np.nan})
            out.append(d)
    return pd.DataFrame(out)

def plot_fig(df, gate_spans, title, out_png, front_thresh, side_thresh):
    import numpy as np
    import matplotlib.pyplot as plt

    if df.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.text(0.5, 0.5, "No RAW samples parsed", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig); return

    t0 = int(df["ts_ms"].min())
    fig = plt.figure(figsize=(16,9))

    # ---- Top axes: distances + radar angle ----
    ax1 = fig.add_subplot(2,1,1)

    # Ultrasonic distances
    ax1.plot(df["time_s"], df["front_cm"], label="Front (cm)", linewidth=1.6, alpha=0.9)
    ax1.plot(df["time_s"], df["left_cm"],  label="Left (cm)",  linewidth=1.2, alpha=0.7)
    ax1.plot(df["time_s"], df["right_cm"], label="Right (cm)", linewidth=1.2, alpha=0.7)

    # Threshold guides
    ax1.axhline(front_thresh, color="red", linestyle="--", linewidth=1.1, label=f"Front<th={front_thresh:.1f} cm")
    ax1.axhline(side_thresh,  color="0.4", linestyle=":",  linewidth=1.0, label=f"Sides<th={side_thresh:.1f} cm")

    # --- NEW: radar trigger threshold line at 300 cm ---
    radar_thresh = 300.0
    ax1.axhline(radar_thresh, color="#ff9900", linestyle="--", linewidth=1.3, label=f"Radar ≤ {radar_thresh:.0f} cm")

    # Gate shading
    for st, en in gate_spans:
        xs = (st - t0)/1000.0; xe = (en - t0)/1000.0
        ax1.axvspan(xs, xe, color="g", alpha=0.08)

    ax1.set_ylabel("Distance (cm)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)

    # ---- Radar angle: thick purple continuous line ----
    ax1b = ax1.twinx()
    radar_ang_raw = pd.to_numeric(df["rad_ang_deg"], errors="coerce")
    radar_ang_connected = radar_ang_raw.copy().interpolate(method="linear", limit_direction="both")
    radar_color = "#8e44ad"  # deep purple
    ax1b.plot(df["time_s"], radar_ang_connected, color=radar_color, linewidth=2.3,
              label="Radar angle (deg)", zorder=5)
    ax1b.set_ylabel("Radar angle (deg)")

    # Combined legend
    h1,l1 = ax1.get_legend_handles_labels()
    h2,l2 = ax1b.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc="upper right")

    # ---- Bottom axes: servo ----
    ax2 = fig.add_subplot(2,1,2, sharex=ax1)
    ax2.plot(df["time_s"], df["servo_us"], linewidth=1.8, label="Servo (µs)")
    ax2.set_ylabel("Servo (µs)")
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, alpha=0.3)

    # Right axis: approximate steer degrees
    y2 = ax2.twinx()
    def us_to_deg(us): return (us - 1500.0) * (90.0/1000.0)
    yt = ax2.get_yticks()
    y2.set_ylim(ax2.get_ylim()); y2.set_yticks(yt)
    y2.set_yticklabels([f"{us_to_deg(y):.0f}°" for y in yt])
    y2.set_ylabel("Steer (deg, approx)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_four_panel(df, gate_spans, title, out_png,
                    front_thresh=60.0, side_thresh=20.0, radar_thresh=300.0):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    if df.empty:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.text(0.5, 0.5, "No RAW samples parsed", ha="center", va="center", transform=ax.transAxes)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    t0 = int(df["ts_ms"].min())
    t  = df["time_s"]

    # --- Radar series (interpolate to look continuous) ---
    rad_dist = pd.to_numeric(df.get("rad_dist_cm"), errors="coerce")
    rad_ang  = pd.to_numeric(df.get("rad_ang_deg"), errors="coerce")

    rad_dist_i = rad_dist.copy().interpolate(limit_direction="both")
    rad_ang_i  = rad_ang.copy().interpolate(limit_direction="both")

    fig = plt.figure(figsize=(16,12))

    # ============== 1) Radar distance ==============
    ax1 = fig.add_subplot(4,1,1)
    ax1.plot(t, rad_dist_i, color="#1f77b4", linewidth=2.0, label="Radar dist (cm)")
    ax1.axhline(radar_thresh, color="#ff9900", linestyle="--", linewidth=1.2, label=f"Radar ≤ {radar_thresh:.0f} cm")

    for st, en in gate_spans:
        xs = (st - t0)/1000.0; xe = (en - t0)/1000.0
        ax1.axvspan(xs, xe, color="g", alpha=0.08)

    ax1.set_ylabel("Distance (cm)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # ============== 2) Radar angle ==============
    ax2 = fig.add_subplot(4,1,2, sharex=ax1)
    ax2.plot(t, rad_ang_i, color="#8e44ad", linewidth=2.0, label="Radar angle (deg)")
    for st, en in gate_spans:
        xs = (st - t0)/1000.0; xe = (en - t0)/1000.0
        ax2.axvspan(xs, xe, color="g", alpha=0.08)
    ax2.set_ylabel("Angle (deg)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    # ============== 3) Ultrasonics ==============
    ax3 = fig.add_subplot(4,1,3, sharex=ax1)
    ax3.plot(t, df["front_cm"], label="Front (cm)", linewidth=1.8)
    ax3.plot(t, df["left_cm"],  label="Left (cm)",  linewidth=1.4, alpha=0.9)
    ax3.plot(t, df["right_cm"], label="Right (cm)", linewidth=1.4, alpha=0.9)

    # Thresholds
    ax3.axhline(front_thresh, color="red", linestyle="--", linewidth=1.1, label=f"Front<th={front_thresh:.0f} cm")
    ax3.axhline(side_thresh,  color="0.4", linestyle=":",  linewidth=1.0, label=f"Sides<th={side_thresh:.0f} cm")
    ax3.axhline(radar_thresh, color="#ff9900", linestyle="--", linewidth=1.2, label=f"Radar ≤ {radar_thresh:.0f} cm")

    for st, en in gate_spans:
        xs = (st - t0)/1000.0; xe = (en - t0)/1000.0
        ax3.axvspan(xs, xe, color="g", alpha=0.08)

    ax3.set_ylabel("Distance (cm)")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper right", ncol=2)

    # ============== 4) Servo position ==============
    ax4 = fig.add_subplot(4,1,4, sharex=ax1)
    ax4.plot(t, df["servo_us"], linewidth=1.8, label="Servo (µs)")
    ax4.set_ylabel("Servo (µs)")
    ax4.set_xlabel("Time (s)")
    ax4.grid(True, alpha=0.3)

    # Right axis in approx degrees
    ax4r = ax4.twinx()
    def us_to_deg(us): return (us - 1500.0) * (90.0/1000.0)
    yt = ax4.get_yticks()
    ax4r.set_ylim(ax4.get_ylim())
    ax4r.set_yticks(yt)
    ax4r.set_yticklabels([f"{us_to_deg(y):.0f}°" for y in yt])
    ax4r.set_ylabel("Steer (deg, approx)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
def _summarize_latency_csv(csv_path: str, label: str):
    """Read a latency CSV and compute basic stats. Returns dict or None."""
    if not os.path.exists(csv_path):
        print(f"[WARN] {label}: file not found -> {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[WARN] {label}: could not read CSV ({e}) -> {csv_path}")
        return None

    if "latency_s" not in df.columns:
        print(f"[WARN] {label}: 'latency_s' column missing in -> {csv_path}")
        return None

    vals = df["latency_s"].astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        print(f"[WARN] {label}: no valid latency samples in -> {csv_path}")
        return None

    stats = {
        "label": label,
        "count": int(vals.size),
        "mean_s": float(vals.mean()),
        "median_s": float(vals.median()),
        "std_s": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
        "min_s": float(vals.min()),
        "p10_s": float(vals.quantile(0.10)),
        "p90_s": float(vals.quantile(0.90)),
        "max_s": float(vals.max()),
        "values": vals,  # keep for plotting
        "csv_path": csv_path,
    }
    print(f"[OK] {label} latency: n={stats['count']}  "
          f"mean={stats['mean_s']:.3f}s  median={stats['median_s']:.3f}s  "
          f"min={stats['min_s']:.3f}s  p10={stats['p10_s']:.3f}s  "
          f"p90={stats['p90_s']:.3f}s  max={stats['max_s']:.3f}s")
    return stats

def append_latency_summaries(out_dir: str, out_txt: str):
    """
    Looks for the two latency CSVs in out_dir, appends summary stats to out_txt,
    and saves an optional comparison plot.
    """
    ev_csv     = os.path.join(out_dir, "fusion_latency_ev.csv")
    front_csv  = os.path.join(out_dir, "fusion_latency_front_threshold.csv")

    radar_stats = _summarize_latency_csv(ev_csv,    "Radar→Servo (EV-based)")
    ultra_stats = _summarize_latency_csv(front_csv, "Ultrasonic→Servo (front<threshold)")

    # Append to the summary text file
    with open(out_txt, "a", encoding="utf-8") as f:
        f.write("\n\nLATENCY SUMMARY (Fusion)\n")
        f.write("=====================================\n")
        if radar_stats:
            f.write("Radar→Servo (EV-based)\n")
            f.write(f"  file   : {radar_stats['csv_path']}\n")
            f.write(f"  n      : {radar_stats['count']}\n")
            f.write(f"  mean   : {radar_stats['mean_s']:.3f} s\n")
            f.write(f"  median : {radar_stats['median_s']:.3f} s\n")
            f.write(f"  std    : {radar_stats['std_s']:.3f} s\n")
            f.write(f"  min    : {radar_stats['min_s']:.3f} s\n")
            f.write(f"  p10    : {radar_stats['p10_s']:.3f} s\n")
            f.write(f"  p90    : {radar_stats['p90_s']:.3f} s\n")
            f.write(f"  max    : {radar_stats['max_s']:.3f} s\n")
        else:
            f.write("Radar→Servo (EV-based): no valid samples\n")

        f.write("\n")
        if ultra_stats:
            f.write("Ultrasonic→Servo (front<threshold)\n")
            f.write(f"  file   : {ultra_stats['csv_path']}\n")
            f.write(f"  n      : {ultra_stats['count']}\n")
            f.write(f"  mean   : {ultra_stats['mean_s']:.3f} s\n")
            f.write(f"  median : {ultra_stats['median_s']:.3f} s\n")
            f.write(f"  std    : {ultra_stats['std_s']:.3f} s\n")
            f.write(f"  min    : {ultra_stats['min_s']:.3f} s\n")
            f.write(f"  p10    : {ultra_stats['p10_s']:.3f} s\n")
            f.write(f"  p90    : {ultra_stats['p90_s']:.3f} s\n")
            f.write(f"  max    : {ultra_stats['max_s']:.3f} s\n")
        else:
            f.write("Ultrasonic→Servo (front<threshold): no valid samples\n")

    # Optional: small comparison chart (skip if no data)
    if (radar_stats and radar_stats["count"] > 0) or (ultra_stats and ultra_stats["count"] > 0):
        labels, means, meds, mins, maxs = [], [], [], [], []
        if radar_stats:
            labels.append("Radar→Servo")
            means.append(radar_stats["mean_s"]); meds.append(radar_stats["median_s"])
            mins.append(radar_stats["min_s"]);   maxs.append(radar_stats["max_s"])
        if ultra_stats:
            labels.append("US→Servo")
            means.append(ultra_stats["mean_s"]); meds.append(ultra_stats["median_s"])
            mins.append(ultra_stats["min_s"]);   maxs.append(ultra_stats["max_s"])

        fig = plt.figure(figsize=(6.5, 4))
        x = np.arange(len(labels))
        plt.bar(x - 0.15, means, width=0.30, label="mean (s)")
        plt.bar(x + 0.15, meds,  width=0.30, label="median (s)")
        # whiskers
        for i in range(len(labels)):
            plt.plot([x[i], x[i]], [mins[i], maxs[i]], marker="_")
        plt.xticks(x, labels, rotation=0)
        plt.ylabel("Latency (s)")
        plt.title("Latency comparison")
        plt.legend()
        out_latency_png = os.path.join(out_dir, "fusion_latency_summary.png")
        plt.tight_layout()
        try:
            plt.savefig(out_latency_png, dpi=160)
            print(f"[OK] Saved latency summary plot -> {out_latency_png}")
        except Exception as e:
            print(f"[WARN] Could not save latency plot: {e}")
        plt.close(fig)

        
def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.logfile))[0]
    out_csv = os.path.join(args.outdir, f"{base}_timeseries.csv")
    out_txt = os.path.join(args.outdir, f"{base}_summary.txt")
    out_png = os.path.join(args.outdir, f"{base}_plot.png")
    out_lat1 = os.path.join(args.outdir, f"{base}_latency_front_threshold.csv")
    out_lat2 = os.path.join(args.outdir, f"{base}_latency_ev.csv")

    p(f"[1/6] Reading: {args.logfile}")
    try:
        df, ev_points, gate_spans, meta, df_r = parse_file(args.logfile)
    except Exception as e:
        p(f"[ERROR] Parse failed: {e}")
        sys.exit(1)

    p(f"[2/6] RAW rows parsed: {len(df)} | EV points: {len(ev_points)} | gate spans: {len(gate_spans)} | R rows: {len(df_r)}")

    if df.empty:
        p("[WARN] No RAW rows found. Writing placeholders so you still get files.")
        pd.DataFrame().to_csv(out_csv, index=False)
        pd.DataFrame().to_csv(out_lat1, index=False)
        pd.DataFrame().to_csv(out_lat2, index=False)
        open(out_txt, "w", encoding="utf-8").write("No samples parsed.\n")
        plot_fig(df, gate_spans, args.title, out_png, args.front_threshold, args.side_threshold)
        p("[DONE] Outputs saved (placeholders).")
        return

    df = df.sort_values("time_s").reset_index(drop=True)
    df.to_csv(out_csv, index=False)
    p(f"[3/6] Wrote timeseries: {out_csv}")

    servo_steps = detect_servo_steps(df, args.servo_step_us)
    trig_front  = triggers_front(df, args.front_threshold)
    lat_front   = pair_latency(trig_front, servo_steps, args.latency_window_s)
    lat_front.to_csv(out_lat1, index=False)

    # EV-latency
    first_ts = int(df["ts_ms"].min())
    ev_trig = ev_triggers(ev_points, first_ts)
    lat_ev  = pair_latency(ev_trig, servo_steps, args.latency_window_s)
    lat_ev.to_csv(out_lat2, index=False)
    p(f"[4/6] Latency rows: front={len(lat_front)} ev={len(lat_ev)}")

    # Summary file
    def nanstats(x):
        x = pd.to_numeric(df[x], errors="coerce")
        if x.notna().any():
            return dict(min=float(np.nanmin(x)), mean=float(np.nanmean(x)),
                        median=float(np.nanmedian(x)), valid=float(np.mean(~np.isnan(x))))
        return dict(min=math.nan, mean=math.nan, median=math.nan, valid=0.0)
    summ_lines = []
    summ_lines.append(f"samples={len(df)} duration_s={df['time_s'].max()-df['time_s'].min():.3f}")
    for col in ["front_cm","left_cm","right_cm","rad_dist_cm","rad_ang_deg"]:
        s = nanstats(col)
        summ_lines.append(f"{col}: min={s['min']:.2f} mean={s['mean']:.2f} median={s['median']:.2f} valid={s['valid']:.3f}")
    if "servo_us" in df:
        smin = float(pd.to_numeric(df["servo_us"], errors="coerce").min(skipna=True))
        smax = float(pd.to_numeric(df["servo_us"], errors="coerce").max(skipna=True))
        summ_lines.append(f"servo_us_range: {smin:.1f}..{smax:.1f}")
    open(out_txt, "w", encoding="utf-8").write("\n".join(summ_lines)+"\n")
    p(f"[5/6] Wrote summary: {out_txt}")

    open(out_txt, "w", encoding="utf-8").write("\n".join(summ_lines)+"\n")
    p(f"[5/6] Wrote summary: {out_txt}")

    # --- Latency summaries ---
    append_latency_summaries(args.outdir, out_txt)

    # Plot
    plot_four_panel(
        df, gate_spans, args.title, out_png,
        front_thresh=args.front_threshold,
        side_thresh=args.side_threshold,
        radar_thresh=300.0  # as requested
    )

    p("[6/6] Plot + latency summary done.")

    # Plot
    plot_four_panel(
    df, gate_spans, args.title, out_png,
    front_thresh=args.front_threshold,
    side_thresh=args.side_threshold,
    radar_thresh=300.0  # as requested

    
)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # ensure any crash still tells you what happened
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(2)
