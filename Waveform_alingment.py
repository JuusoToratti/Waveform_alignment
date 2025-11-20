import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from scipy.signal import find_peaks
import os

# ---------------------------------------------------------
# TIEDOSTOT
# ---------------------------------------------------------
sim_file = "I2C_all.csv"
meas_file = "REGRESSION_ AREC_X21_MADE2_I2C_1-I2C_0_P1V8(1) SDA-INC74024-2593-55585.csv"

# ---------------------------------------------------------
# DATAN LUKEMINEN
# ---------------------------------------------------------
# Simulaatio
with open(sim_file, "r", encoding="utf-8", errors="ignore") as f:
    for line_num, line in enumerate(f):
        if line.startswith("Time;"):
            headers = line.strip().split(";")
            break

sim_df_raw = pd.read_csv(
    sim_file,
    sep=";",
    decimal=",",
    skiprows=line_num + 1,
    header=None,
    names=headers,
    engine="python"
)

sim_df = sim_df_raw.copy()
for col in sim_df.columns:
    sim_df[col] = pd.to_numeric(sim_df[col], errors='coerce')
sim_df = sim_df.dropna(how='all').reset_index(drop=True)

# Mittaus
meas_df = pd.read_csv(meas_file, sep=",", header=None, names=["Time", "Voltage"])
meas_df["Time"] = pd.to_numeric(meas_df["Time"], errors='coerce')
meas_df["Voltage"] = pd.to_numeric(meas_df["Voltage"], errors='coerce')
meas_df = meas_df.dropna().reset_index(drop=True)

# ---------------------------------------------------------
# SDA-SARAKE (Serial Data Line)
# ---------------------------------------------------------
sim_sda_col = next(c for c in sim_df.columns if "INC74024" in str(c))
meas_sda_col = "Voltage"

# ---------------------------------------------------------
# REUNOJEN ETSINTÄ JA ALIGN
# ---------------------------------------------------------
def find_transition_times(t, v, level, rising=True):
    t = np.asarray(t)
    v = np.asarray(v)
    if rising:
        idx = np.where((v[:-1] <= level) & (v[1:] > level))[0]
    else:
        idx = np.where((v[:-1] > level) & (v[1:] <= level))[0]

    times = []
    for i in idx:
        dv = v[i+1] - v[i]
        if abs(dv) < 1e-12:
            continue
        frac = (level - v[i]) / dv
        times.append(t[i] + frac * (t[i+1] - t[i]))
    return np.array(times)

# 50 % threshold ja swing
v_min = min(meas_df[meas_sda_col].min(), sim_df[sim_sda_col].min())
v_max = max(meas_df[meas_sda_col].max(), sim_df[sim_sda_col].max())
threshold50 = v_min + 0.5 * (v_max - v_min)
swing = v_max - v_min

# Falling edget
meas_falling = find_transition_times(meas_df["Time"].values, meas_df[meas_sda_col].values, threshold50, rising=False)
sim_falling  = find_transition_times(sim_df["Time"].values,  sim_df[sim_sda_col].values,  threshold50, rising=False)

# Alignaus
meas_edge_idx = 0
align_t_meas = meas_falling[meas_edge_idx] if len(meas_falling) > meas_edge_idx else meas_df["Time"].iloc[0]
align_t_sim  = sim_falling[0] if len(sim_falling) > 0 else sim_df["Time"].iloc[0]

meas_df["Time"] -= align_t_meas
sim_df["Time"]  -= align_t_sim

# ---------------------------------------------------------
# NON-MONOTONICITY DETECTION
# ---------------------------------------------------------
def detect_non_monotonicity(t, v, falling_times, swing, window_ns=200e-9, min_ampl_pct=5):
    considerable = 0
    markers = []
    min_ampl = (min_ampl_pct / 100.0) * swing

    for ft in falling_times:
        mask = (t >= ft) & (t <= ft + window_ns)
        if not np.any(mask):
            continue
        v_win = v[mask]
        t_win = t[mask]

        peaks_pos, _ = find_peaks(v_win, distance=5)
        peaks_neg, _ = find_peaks(-v_win, distance=5)
        peaks = np.sort(np.concatenate((peaks_pos, peaks_neg)))

        if len(peaks) >= 2:
            ampl = np.ptp(v_win[peaks])
            if ampl > min_ampl:
                considerable += 1
                for p in peaks:
                    markers.append((t_win[p], v_win[p]))
    return considerable, markers

nonmono_meas_cnt, nonmono_meas_mark = detect_non_monotonicity(meas_df["Time"].values, meas_df[meas_sda_col].values,
                                                             meas_falling, swing)
nonmono_sim_cnt,  nonmono_sim_mark  = detect_non_monotonicity(sim_df["Time"].values,  sim_df[sim_sda_col].values,
                                                             sim_falling, swing)

# ---------------------------------------------------------
# PARAMETRIT
# ---------------------------------------------------------
def calculate_params(t, v, th50, swing):
    V_high = np.nanmean(v[v > th50])
    V_low  = np.nanmean(v[v <= th50])
    delta  = V_high - V_low if not (np.isnan(V_high) or np.isnan(V_low)) else np.nan

    l10 = V_low + 0.1 * delta
    l90 = V_low + 0.9 * delta

    rise10 = find_transition_times(t, v, l10, rising=True)
    rise90 = find_transition_times(t, v, l90, rising=True)
    fall90 = find_transition_times(t, v, l90, rising=False)
    fall10 = find_transition_times(t, v, l10, rising=False)

    rise_time = np.mean(rise90 - rise10) if len(rise90) == len(rise10) and len(rise90) > 0 else np.nan
    fall_time = np.mean(fall10 - fall90) if len(fall10) == len(fall90) and len(fall90) > 0 else np.nan

    falling = find_transition_times(t, v, th50, rising=False)
    freq = 1 / np.mean(np.diff(falling)) if len(falling) > 1 else np.nan

    band = 0.05 * delta if delta else 0
    after = t >= -1e-10
    deviations = np.abs(v[after] - V_low)
    exceed = np.where(deviations > band)[0]
    settling = t[after][exceed[-1]] if len(exceed) > 0 else 0.0

    return {
        "V_high": V_high, "V_low": V_low, "V_max": np.max(v), "V_min": np.min(v),
        "overshoot_%": (np.max(v) - V_high) / delta * 100 if delta else 0.0,
        "undershoot_%": (V_low - np.min(v)) / delta * 100 if delta else 0.0,
        "rise_ns": rise_time * 1e9, "fall_ns": fall_time * 1e9,
        "freq_kHz": freq / 1e3 if not np.isnan(freq) else np.nan,
        "settling_ns": settling * 1e9, "duty_%": np.mean(v > th50) * 100
    }

p_meas = calculate_params(meas_df["Time"].values, meas_df[meas_sda_col].values, threshold50, swing)
p_sim  = calculate_params(sim_df["Time"].values,  sim_df[sim_sda_col].values, threshold50, swing)

# ---------------------------------------------------------
# TEKSTIT
# ---------------------------------------------------------
metadata_text = f"""Mittaus: {os.path.basename(meas_file)}
Simulaatio: {os.path.basename(sim_file)}
Pisteitä: {len(meas_df):,} (mittaus) | {len(sim_df):,} (sim)
Align: falling edge #{meas_edge_idx + 1}
Threshold 50 %: {threshold50:.3f} V"""

params_text = f"""Waveform parameters (SDA INC74024)

{'Parameter':<20} {'Measured':>12} {'Simulated':>12}
{'─'*52}
V_high [V]           {p_meas['V_high']:12.4f} {p_sim['V_high']:12.4f}
V_low [V]            {p_meas['V_low']:12.4f} {p_sim['V_low']:12.4f}
V_max [V]            {p_meas['V_max']:12.4f} {p_sim['V_max']:12.4f}
V_min [V]            {p_meas['V_min']:12.4f} {p_sim['V_min']:12.4f}
Rise time [ns]       {p_meas['rise_ns']:12.1f} {p_sim['rise_ns']:12.1f}
Fall time [ns]       {p_meas['fall_ns']:12.1f} {p_sim['fall_ns']:12.1f}
Overshoot [%]        {p_meas['overshoot_%']:12.1f} {p_sim['overshoot_%']:12.1f}
Undershoot [%]       {p_meas['undershoot_%']:12.1f} {p_sim['undershoot_%']:12.1f}
Duty cycle [%]       {p_meas['duty_%']:12.1f} {p_sim['duty_%']:12.1f}
Frequency [kHz]      {p_meas['freq_kHz']:12.1f} {p_sim['freq_kHz']:12.1f}
Settling time [ns]   {p_meas['settling_ns']:12.1f} {p_sim['settling_ns']:12.1f}
Non-monotonic edges  {nonmono_meas_cnt:12d} {nonmono_sim_cnt:12d}
"""

# ---------------------------------------------------------
# PIIRTO Chartille
# ---------------------------------------------------------
plt.figure(figsize=(16, 9))

plt.plot(meas_df["Time"], meas_df[meas_sda_col], color="red", linewidth=2.5, label="Mitattu SDA")
plt.plot(sim_df["Time"], sim_df[sim_sda_col], color="blue", linewidth=2, label="Simuloitu SDA")

for col in sim_df.columns:
    if col not in ["Time", sim_sda_col]:
        plt.plot(sim_df["Time"], sim_df[col], color="lightgray", alpha=0.3, linewidth=0.6)

# Non-monotonic-merkinnät
if nonmono_meas_mark:
    tm, vm = zip(*nonmono_meas_mark)
    plt.scatter(tm, vm, color="red", s=80, zorder=6, edgecolor="white", linewidth=1.5, label="Non-monotonic (meas)")

if nonmono_sim_mark:
    ts, vs = zip(*nonmono_sim_mark)
    plt.scatter(ts, vs, color="magenta", s=60, zorder=6, edgecolor="white", linewidth=1, label="Non-monotonic (sim)")

# Akselit EngFormatterilla 
plt.gca().xaxis.set_major_formatter(EngFormatter(unit='s'))
plt.gca().yaxis.set_major_formatter(EngFormatter(unit='V'))

plt.xlabel("Aika")
plt.ylabel("Jännite")
plt.title("I2C SDA – Simulaatio vs. Mittaus")
plt.grid(True, alpha=0.3)
plt.legend(loc="upper right")

plt.text(0.02, 0.98, metadata_text, transform=plt.gca().transAxes, va='top', ha='left', fontsize=10,
         family='monospace', bbox=dict(boxstyle="round,pad=0.6", facecolor="wheat", alpha=0.9))
plt.text(0.98, 0.98, params_text, transform=plt.gca().transAxes, va='top', ha='right', fontsize=9.5,
         family='monospace', bbox=dict(boxstyle="round,pad=0.8", facecolor="lightblue", alpha=0.95))

plt.xlim(meas_df["Time"].min(), meas_df["Time"].max())
plt.tight_layout()
plt.show()