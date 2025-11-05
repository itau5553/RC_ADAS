#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ultrasonic Hard-Stop Analysis
Author: Imran Tauqeer (ADAS Thesis)

This script analyzes your ADAS test log (RAW + EV lines) and:
1. Detects when the front ultrasonic distance drops below a threshold.
2. Finds the next motor STOP event.
3. Computes reaction latency.
4. Saves CSVs, summary, and a plot.

Example:
  python analyze_hardstop.py "C:\\path\\hardstop_ultrasonic.txt" ^
      --outdir "C:\\path\\analysis_out" ^
      --threshold 25 --latency-window-s 3.0 --title "Ultrasonic Hard-Stop Test"
"""

import argparse, os, re, math, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure matplotlib runs headless
plt.switch_backend("Agg")

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("logfile", help="Input .txt log file (RAW+EV)")
    ap.add_argument("--outdir", required=True, help="Output folder path")
    ap.add_argument("--threshold", type=float, default=25.0, help="Front ultrasonic trigger distance (cm)")
    ap.add_argument("--latency-window-s", type=float, default=3.0, help="Maximum time window to find STOP after trigger")
    ap.add_argument("--title", default="Ultrasonic Hard-Stop Test", help="Plot title")
    return ap.parse_args()

# ---------- PARSE FILE ----------
def parse_file(path):
    rows, ev_rows = [], []
    pat_raw = re.compile(r"^RAW,")
    pat_ev = re.compile(r"^EV,")

    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if pat_raw.match(s):
                p = s.split(",")
                if len(p) < 12:
                    continue
                try:
                    rows.append([
                        int(p[1]), p[2], float(p[3]), float(p[4]), float(p[5]),
                        int(p[6]), float(p[7]), float(p[8]), float(p[9]),
                        int(p[10]), int(p[11])
                    ])
                except:
                    continue
            elif pat_ev.match(s):
                ev_rows.append(p)

    df = pd.DataFrame(rows, columns=[
        "ts_ms","mode","front_cm","left_cm","right_cm",
        "radar_has","rad_ang_deg","rad_dist_cm","rad_speed_cms",
        "servo_us","drive_state"
    ])
    if not df.empty:
        df = df.sort_values("ts_ms").reset_index(drop=True)
        t0 = df["ts_ms"].iloc[0]
        df["time_s"] = (df["ts_ms"] - t0) / 1000.0

    # Parse EV events
    evs = []
    for e in ev_rows:
        try:
            if len(e) >= 5 and e[1] == "STOP" and e[2] == "STOP_FRONT":
                evs.append({"kind": "STOP_FRONT", "ts_ms": int(e[4])})
            elif len(e) >= 3 and e[1] == "STOP_CLEAR":
                evs.append({"kind": "STOP_CLEAR", "ts_ms": int(e[2])})
        except:
            pass
    df_ev = pd.DataFrame(evs)
    if not df_ev.empty and not df.empty:
        t0 = df["ts_ms"].iloc[0]
        df_ev["time_s"] = (df_ev["ts_ms"] - t0) / 1000.0

    return df, df_ev

# ---------- TRIGGER DETECTION ----------
def detect_triggers(df, threshold):
    triggers = []
    t = df["time_s"].to_numpy()
    f = df["front_cm"].to_numpy()
    for i in range(1, len(df)):
        if f[i-1] > threshold and f[i] <= threshold:
            triggers.append((t[i], f[i]))
    return triggers

# ---------- LATENCY PAIRING ----------
def pair_latency(triggers, df, window_s):
    results = []
    t = df["time_s"].to_numpy()
    drive = df["drive_state"].to_numpy()
    for trig_t, front_val in triggers:
        stop_t = np.nan
        for j in range(len(t)):
            if t[j] > trig_t and t[j] - trig_t <= window_s and drive[j] == 0:
                stop_t = t[j]
                break
        latency = stop_t - trig_t if not math.isnan(stop_t) else np.nan
        results.append([trig_t, front_val, stop_t, latency])
    return pd.DataFrame(results, columns=["trigger_time_s","front_cm","stop_time_s","latency_s"])

# ---------- PLOTTING ----------
def plot_results(df, trig_df, df_ev, out_png, title, threshold):
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    # Extract arrays
    t = df["time_s"].to_numpy()
    f = pd.to_numeric(df["front_cm"], errors="coerce").to_numpy()
    mot = df["drive_state"].map({0: 0, 1: 255, 2: -255}).fillna(0).to_numpy()

    # --- Top panel: front ultrasonic ---
    ax[0].plot(t, f, lw=1.6, label="Front Ultrasonic (cm)")
    ax[0].axhline(threshold, color="r", ls="--", label=f"Threshold {threshold} cm")

    # Mark triggers and stops
    for _, r in trig_df.iterrows():
        ax[0].axvline(r["trigger_time_s"], color="tab:green", ls=":", lw=1)
        if not math.isnan(r["stop_time_s"]):
            ax[0].axvline(r["stop_time_s"], color="tab:orange", ls=":", lw=1)

    # Mark EV events
    if not df_ev.empty:
        for _, e in df_ev.iterrows():
            c = "tab:red" if e["kind"] == "STOP_FRONT" else "tab:blue"
            ax[0].axvline(e["time_s"], color=c, ls="--", lw=1)

    ax[0].set_ylabel("Front Distance (cm)")
    ax[0].legend()
    ax[0].grid(ls=":")

    # ✅ Limit the visible y-axis to 0–100 cm regardless of outliers
    ax[0].set_ylim(0, 100)
    ax[0].set_xlim(df["time_s"].min(), df["time_s"].max())

    # --- Bottom panel: motor proxy ---
    ax[1].plot(t, mot, lw=1.6, label="Motor state proxy (±255)")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Motor proxy")
    ax[1].grid(ls=":")
    ax[1].legend()

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Plot saved: {out_png}")

# ---------- SUMMARY ----------
def write_summary(df, trig_df, out_txt, threshold, window_s):
    lines = []
    lines.append(f"samples={len(df)} duration_s={df['time_s'].max():.3f}")
    lines.append(f"threshold_cm={threshold} latency_window_s={window_s}")
    lines.append(f"triggers_detected={len(trig_df)}")
    if not trig_df.empty:
        paired = trig_df["latency_s"].notna().sum()
        lines.append(f"paired_stops={paired}")
        if paired > 0:
            lat = trig_df["latency_s"].dropna()
            lines.append(f"latency_mean={lat.mean():.3f}")
            lines.append(f"latency_median={lat.median():.3f}")
            lines.append(f"latency_min={lat.min():.3f}")
            lines.append(f"latency_max={lat.max():.3f}")
    open(out_txt, "w", encoding="utf-8").write("\n".join(lines))
    print(f"[OK] Summary saved: {out_txt}")

# ---------- MAIN ----------
def main():
    print(">>> Script started"); sys.stdout.flush()
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.logfile))[0]
    out_csv = os.path.join(args.outdir, f"{base}_timeseries.csv")
    out_lat = os.path.join(args.outdir, f"{base}_latency.csv")
    out_txt = os.path.join(args.outdir, f"{base}_summary.txt")
    out_png = os.path.join(args.outdir, f"{base}_plot.png")

    print(f"[1/4] Reading {args.logfile}")
    df, df_ev = parse_file(args.logfile)
    if df.empty:
        print("[ERROR] No RAW data found."); return

    df.to_csv(out_csv, index=False)
    print(f"[2/4] Timeseries written: {out_csv}")

    triggers = detect_triggers(df, args.threshold)
    trig_df = pair_latency(triggers, df, args.latency_window_s)
    trig_df.to_csv(out_lat, index=False)
    print(f"[3/4] Latency written: {out_lat} (n={len(trig_df)})")

    write_summary(df, trig_df, out_txt, args.threshold, args.latency_window_s)
    plot_results(df, trig_df, df_ev, out_png, args.title, args.threshold)

    print("[4/4] Done.")

if __name__ == "__main__":
    main()
