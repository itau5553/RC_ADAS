#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse
import pandas as pd, numpy as np
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

pat_t  = re.compile(r"^T,")
pat_ev = re.compile(r"^EV,")

def parse_args():
    ap = argparse.ArgumentParser(description="Radar steer-away analysis & plots")
    ap.add_argument("logfile", help="Path to the log file")
    ap.add_argument("--outdir", required=True, help="Output directory")
    ap.add_argument("--title", default="Radar Steer-Away Test", help="Plot title")
    ap.add_argument("--servo-delta-us", type=float, default=80.0,
                    help="Servo change (µs) to count as a reaction when no STEER_CMD is logged")
    ap.add_argument("--pre-baseline-ms", type=int, default=500,
                    help="Time window before GATE_ON to compute servo baseline (ms)")
    ap.add_argument("--dist-ylim", type=float, nargs=2, default=[0, 400],
                    help="Y-limits for radar distance plot (cm), e.g. 0 400")
    ap.add_argument("--angle-ylim", type=float, nargs=2, default=[-90, 90],
                    help="Y-limits for radar angle plot (deg)")
    return ap.parse_args()

def parse_file(path):
    """Parse T-lines for telemetry and EV-lines for events. Robust to extra fields."""
    data_rows, ev_rows = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if pat_t.match(s):
                # T,<ts>,<mode_int>,<drive>,<front>,<left>,<right>,<rad_has>,<angle>,<dist>,<servo>,<us_state>,<joy>
                p = s.split(",")
                if len(p) < 11:
                    continue
                try:
                    ts    = int(p[1])
                    # p[7]=rad_has, p[8]=angle_deg, p[9]=dist_cm, p[10]=servo_us
                    ang   = float(p[8])
                    dist  = float(p[9])
                    servo = int(float(p[10]))
                    data_rows.append([ts, ang, dist, servo])
                except Exception:
                    pass

            elif pat_ev.match(s):
                # EV,TYPE,ts[,value]
                p = s.split(",")
                if len(p) >= 3:
                    etype = p[1]
                    try:
                        ets = int(p[2])
                        val = float(p[3]) if len(p) >= 4 else np.nan
                        ev_rows.append([etype, ets, val])
                    except Exception:
                        pass

    df = pd.DataFrame(data_rows, columns=["ts_ms","angle_deg","dist_cm","servo_us"])
    if not df.empty:
        df = df.sort_values("ts_ms").reset_index(drop=True)
        t0 = df["ts_ms"].iloc[0]
        df["time_s"] = (df["ts_ms"] - t0) / 1000.0

    df_ev = pd.DataFrame(ev_rows, columns=["kind","ts_ms","value"])
    if not df_ev.empty and not df.empty:
        t0 = df["ts_ms"].iloc[0]
        df_ev = df_ev.sort_values("ts_ms").reset_index(drop=True)
        df_ev["time_s"] = (df_ev["ts_ms"] - t0) / 1000.0

    return df, df_ev

def compute_latency_event_based(df_ev):
    """Pair RADAR_GATE_ON → first STEER_CMD after it, if present."""
    if df_ev.empty:
        return pd.DataFrame(columns=["gate_on_ts","steer_cmd_ts","latency_s","method"])

    ons  = df_ev[df_ev["kind"].str.contains("RADAR_GATE_ON", na=False)]
    cmds = df_ev[df_ev["kind"].str.contains("STEER_CMD", na=False)]
    pairs = []
    if ons.empty or cmds.empty:
        return pd.DataFrame(columns=["gate_on_ts","steer_cmd_ts","latency_s","method"])

    for _, r in ons.iterrows():
        later = cmds[cmds["ts_ms"] >= r["ts_ms"]]
        if not later.empty:
            t2 = int(later["ts_ms"].iloc[0])
            latency = (t2 - int(r["ts_ms"])) / 1000.0
            pairs.append([int(r["ts_ms"]), t2, latency, "events"])

    return pd.DataFrame(pairs, columns=["gate_on_ts","steer_cmd_ts","latency_s","method"])

def compute_latency_servo_based(df, df_ev, delta_us=80.0, pre_baseline_ms=500):
    """
    Pair RADAR_GATE_ON → first significant servo movement after gate_on.
    Servo baseline = median servo in [gate_on - pre_baseline_ms, gate_on).
    """
    if df.empty or df_ev.empty:
        return pd.DataFrame(columns=["gate_on_ts","steer_cmd_ts","latency_s","method"])

    pairs = []
    ons = df_ev[df_ev["kind"].str.contains("RADAR_GATE_ON", na=False)]
    if ons.empty:
        return pd.DataFrame(columns=["gate_on_ts","steer_cmd_ts","latency_s","method"])

    for _, g in ons.iterrows():
        t_on = int(g["ts_ms"])

        # baseline window
        pre = df[(df["ts_ms"] >= (t_on - pre_baseline_ms)) & (df["ts_ms"] < t_on)]
        if pre.empty:
            base = np.median(df["servo_us"])
        else:
            base = float(np.median(pre["servo_us"]))

        after = df[df["ts_ms"] >= t_on]
        if after.empty:
            continue

        diff = (after["servo_us"] - base).abs()
        moved = diff[diff > delta_us]
        if len(moved) == 0:
            continue

        first_idx = moved.index[0]
        t2 = int(after.loc[first_idx, "ts_ms"])
        latency = (t2 - t_on) / 1000.0
        pairs.append([t_on, t2, latency, "servo"])

    return pd.DataFrame(pairs, columns=["gate_on_ts","steer_cmd_ts","latency_s","method"])

def plot_results(df, df_ev, lat_df, out_png, title, dist_ylim, angle_ylim):
    if df.empty:
        return

    t = df["time_s"].to_numpy()
    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    # --- Radar distance (clip for plot only) ---
    dist = pd.to_numeric(df["dist_cm"], errors="coerce").to_numpy()
    # Treat 9999 (no target) as NaN for plotting clarity
    dist_plot = np.where(dist >= 9999.0, np.nan, dist)
    ax[0].plot(t, dist_plot, lw=1.5, label="Radar distance (cm)")
    ax[0].axhline(100, color="r", ls="--", label="100 cm")
    ax[0].set_ylabel("Distance (cm)")
    ax[0].grid(ls=":")
    ax[0].set_ylim(dist_ylim[0], dist_ylim[1])
    ax[0].legend(loc="upper right")

    # --- Radar angle ---
    ang = pd.to_numeric(df["angle_deg"], errors="coerce").to_numpy()
    ax[1].plot(t, ang, lw=1.5, label="Radar angle (°)")
    ax[1].axhline(60, color="r", ls="--")
    ax[1].axhline(-60, color="r", ls="--", label="±60°")
    ax[1].set_ylabel("Angle (°)")
    ax[1].grid(ls=":")
    ax[1].set_ylim(angle_ylim[0], angle_ylim[1])
    ax[1].legend(loc="upper right")

    # --- Servo position ---
    servo = pd.to_numeric(df["servo_us"], errors="coerce").to_numpy()
    ax[2].plot(t, servo, lw=1.5, label="Servo (µs)")
    ax[2].axhline(1500, ls="--", label="Center 1500 µs")
    ax[2].set_ylabel("Servo (µs)")
    ax[2].set_xlabel("Time (s)")
    ax[2].grid(ls=":")
    ax[2].legend(loc="upper right")

    # --- Event markers & shaded gate windows ---
    if not df_ev.empty:
        colors = {"RADAR_GATE_ON":"tab:green","RADAR_GATE_OFF":"tab:red","STEER_CMD":"tab:purple"}
        # vertical lines
        for _, e in df_ev.iterrows():
            c = colors.get(e["kind"], "gray")
            for a in ax:
                a.axvline(e["time_s"], color=c, ls=":", lw=1)

        # shaded gate regions
        ons = df_ev[df_ev["kind"] == "RADAR_GATE_ON"].reset_index(drop=True)
        offs= df_ev[df_ev["kind"] == "RADAR_GATE_OFF"].reset_index(drop=True)
        i, j = 0, 0
        while i < len(ons):
            t_on = float(ons.loc[i, "time_s"])
            # find the OFF that follows this ON
            while j < len(offs) and offs.loc[j, "time_s"] < t_on:
                j += 1
            if j < len(offs):
                t_off = float(offs.loc[j, "time_s"])
            else:
                t_off = t[-1]
            for a in ax:
                a.axvspan(t_on, t_off, color="tab:green", alpha=0.08)
            i += 1
            j += 1

    # --- Latency points ---
    if lat_df is not None and not lat_df.empty and not df.empty:
        t0 = df["ts_ms"].iloc[0]
        for _, r in lat_df.iterrows():
            t_on  = (int(r["gate_on_ts"])  - t0) / 1000.0
            t_cmd = (int(r["steer_cmd_ts"]) - t0) / 1000.0
            # show markers on the servo plot
            ax[2].axvline(t_on,  color="tab:olive",  ls="--", lw=1)
            ax[2].axvline(t_cmd, color="tab:orange", ls="--", lw=1)

    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[OK] Saved plot: {out_png}")

def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.logfile))[0]
    out_png = os.path.join(args.outdir, f"{base}_plot.png")
    out_lat = os.path.join(args.outdir, f"{base}_latency.csv")
    out_txt = os.path.join(args.outdir, f"{base}_summary.txt")
    out_csv = os.path.join(args.outdir, f"{base}_timeseries.csv")

    print(f"[1/4] Reading {args.logfile}")
    df, df_ev = parse_file(args.logfile)
    if df.empty:
        open(out_txt, "w", encoding="utf-8").write("No telemetry (T-lines) found.\n")
        print("[WARN] No telemetry found — wrote placeholder summary.")
        return

    # Save timeseries for debugging
    df.to_csv(out_csv, index=False)
    print(f"[2/4] Wrote timeseries CSV: {out_csv} (n={len(df)})")

    # Latency by events, then fallback to servo movement
    lat_evt  = compute_latency_event_based(df_ev)
    lat_ser  = compute_latency_servo_based(df, df_ev, args.servo_delta_us, args.pre_baseline_ms)
    lat_df   = pd.concat([lat_evt, lat_ser], ignore_index=True) if not lat_evt.empty or not lat_ser.empty \
               else pd.DataFrame(columns=["gate_on_ts","steer_cmd_ts","latency_s","method"])
    lat_df.to_csv(out_lat, index=False)
    print(f"[3/4] Wrote latency CSV: {out_lat} (n={len(lat_df)})")

    # Robustness metrics
    #  - valid_any: radar returned a real value (not 9999)
    #  - valid_<=100cm: target within 1 m
    dist = pd.to_numeric(df["dist_cm"], errors="coerce")
    valid_any = float(np.mean(dist < 9999.0)) if len(dist) else 0.0
    valid_100 = float(np.mean((dist > 0) & (dist <= 100.0))) if len(dist) else 0.0

    # Summary
    lines = []
    dur_s = df["time_s"].iloc[-1] - df["time_s"].iloc[0] if len(df) > 1 else 0.0
    lines.append(f"samples={len(df)} duration_s={dur_s:.3f}")
    lines.append(f"robustness_valid_any={(valid_any*100):.1f}% (radar returned a value)")
    lines.append(f"robustness_within_100cm={(valid_100*100):.1f}% (target ≤100 cm)")
    if not lat_df.empty:
        lines.append(f"latency_count={len(lat_df)}")
        lines.append(f"latency_mean_s={lat_df['latency_s'].mean():.3f}")
        lines.append(f"latency_median_s={lat_df['latency_s'].median():.3f}")
        lines.append(f"latency_min_s={lat_df['latency_s'].min():.3f}")
        lines.append(f"latency_max_s={lat_df['latency_s'].max():.3f}")
        method_counts = lat_df["method"].value_counts().to_dict()
        lines.append(f"latency_methods={method_counts}")
    else:
        lines.append("latency_count=0 (no STEER_CMD and no significant servo movement after gates)")

    open(out_txt, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print(f"[OK] Wrote summary: {out_txt}")

    # Plot
    plot_results(df, df_ev, lat_df, out_png, args.title, args.dist_ylim, args.angle_ylim)
    print("[4/4] Done.")

if __name__ == "__main__":
    main()
