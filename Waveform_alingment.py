import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.widgets import Button
from scipy.signal import find_peaks
import os

# ---------------------------------------------------------
# TIEDOSTOT
# ---------------------------------------------------------
sim_file = "I2C_all.csv"
meas_file = "REGRESSION_ AREC_X21_MADE2_I2C_1-I2C_0_P1V8(1) SDA-INC74024-2593-55585.csv"

# ---------------------------------------------------------
# KOHINANPOISTO
# ---------------------------------------------------------
from scipy.signal import savgol_filter, medfilt

def remove_noise(t, v, method='savgol', window=11, polyorder=10):
    """
    Poistaa kohinaa signaalista.
    
    Parametrit:
    - method: 'savgol' (Savitzky-Golay), 'median', 'moving_avg', tai 'none'
    - window: ikkunan koko (pariton luku)
    - polyorder: polynomin aste (vain savgol)
    
    Palauttaa: suodatettu signaali
    """
    if method == 'none':
        return v
    
    v_array = np.asarray(v)
    
    # Varmista että window on pariton ja < datapisteiden määrä
    window = min(window, len(v_array))
    if window % 2 == 0:
        window += 1
    window = max(3, window)  # Vähintään 3
    
    if method == 'savgol':
        # Savitzky-Golay suodin - säilyttää reunat hyvin
        polyorder = min(polyorder, window - 1)
        return savgol_filter(v_array, window, polyorder)
    
    elif method == 'median':
        # Mediaanisuodin - hyvä piikkikohinalle
        return medfilt(v_array, window)
    
    elif method == 'moving_avg':
        # Liukuva keskiarvo - yksinkertainen mutta tehokas
        return np.convolve(v_array, np.ones(window)/window, mode='same')
    
    else:
        return v

# Kohinanpoiston asetukset
NOISE_FILTER_METHOD = 'savgol'  # 'savgol', 'median', 'moving_avg', tai 'none'
NOISE_FILTER_WINDOW = 500        # Ikkunan koko (pariton luku suositellaan)
NOISE_FILTER_POLYORDER = 2      # Polynomin aste (vain savgol)

# ---------------------------------------------------------
# DATAN LUKEMINEN – ROBUSTI HEADER-ETSINTÄ
# ---------------------------------------------------------
headers = None
data_start_row = 0

with open(sim_file, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    cleaned = line.strip().lstrip(",")
    fields = cleaned.split(";")
    if fields and fields[0] == "Time":
        headers = fields
        data_start_row = i + 1
        break

if headers is None:
    raise ValueError(f"Header-riviä 'Time' ei löytynyt tiedostosta {sim_file}")

sim_df_raw = pd.read_csv(
    sim_file,
    sep=";",
    decimal=",",
    skiprows=data_start_row,
    header=None,
    names=headers,
    engine="python",
    on_bad_lines='skip'
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

# Alkuperäiset ajat ja jännitteet
original_meas_time = meas_df["Time"].copy()
original_sim_time  = sim_df["Time"].copy()

# Sovella kohinanpoistoa
original_meas_v_raw = meas_df["Voltage"].copy()
original_sim_v_raw = sim_df[next(c for c in sim_df.columns if "INC74024" in str(c))].copy()

original_meas_v = remove_noise(original_meas_time, original_meas_v_raw, 
                                method=NOISE_FILTER_METHOD, 
                                window=NOISE_FILTER_WINDOW,
                                polyorder=NOISE_FILTER_POLYORDER)
original_sim_v = remove_noise(original_sim_time, original_sim_v_raw,
                               method=NOISE_FILTER_METHOD,
                               window=NOISE_FILTER_WINDOW,
                               polyorder=NOISE_FILTER_POLYORDER)

# Muunna takaisin pandas Seriesiksi jos tarpeen
if hasattr(original_meas_v_raw, 'index'):
    original_meas_v = pd.Series(original_meas_v, index=original_meas_v_raw.index)
if hasattr(original_sim_v_raw, 'index'):
    original_sim_v = pd.Series(original_sim_v, index=original_sim_v_raw.index)

# ---------------------------------------------------------
# SDA-SARAKE
# ---------------------------------------------------------
sim_sda_col = next(c for c in sim_df.columns if "INC74024" in str(c))
meas_sda_col = "Voltage"

# ---------------------------------------------------------
# REUNOJEN ETSINTÄ
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

# ---------------------------------------------------------
# NON-MONOTONICITY DETECTION
# ---------------------------------------------------------
def detect_non_monotonicity(t, v, falling_times, swing, window_ns=200e-9, min_ampl_pct=5):
    """
    Tunnistaa non-monotoniset reunat (värähtely laskevalla reunalla).
    Palauttaa löydettyjen tapausten lukumäärän ja merkintäpisteet.
    """
    considerable = 0
    markers = []
    min_ampl = (min_ampl_pct / 100.0) * swing

    for ft in falling_times:
        mask = (t >= ft) & (t <= ft + window_ns)
        if not np.any(mask):
            continue
        
        v_win = v[mask].values if hasattr(v[mask], 'values') else np.asarray(v)[mask]
        t_win = t[mask].values if hasattr(t[mask], 'values') else np.asarray(t)[mask]

        if len(v_win) < 3:
            continue

        peaks_pos, _ = find_peaks(v_win, distance=5)
        peaks_neg, _ = find_peaks(-v_win, distance=5)
        peaks = np.sort(np.concatenate((peaks_pos, peaks_neg)))

        if len(peaks) >= 2:
            v_at_peaks = v_win[peaks]
            amplitude = np.max(v_at_peaks) - np.min(v_at_peaks)
            
            if amplitude >= min_ampl:
                considerable += 1
                peak_idx = peaks[0]
                markers.append((t_win[peak_idx], v_win[peak_idx]))
    
    return considerable, markers

# ---------------------------------------------------------
# APUFUNKTIO: Etsi lähin simulointireunan
# ---------------------------------------------------------
def find_closest_sim_edge(meas_edge_time, sim_edges, max_distance=10e-6):
    """Etsi lähin simulointikäyrän reuna mittausreunan läheisyydestä"""
    if len(sim_edges) == 0:
        return None
    distances = np.abs(sim_edges - meas_edge_time)
    min_idx = np.argmin(distances)
    if distances[min_idx] < max_distance:
        return sim_edges[min_idx]
    return None

# ---------------------------------------------------------
# PARAMETRIEN LASKENTA
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

    # Robustimpi rise/fall time laskenta
    rise_times = []
    for t10 in rise10:
        candidates = rise90[rise90 > t10]
        if len(candidates) > 0:
            t90 = candidates[0]
            rt = t90 - t10
            if rt > 0 and rt < 1e-3:
                rise_times.append(rt)
    
    fall_times = []
    for t90 in fall90:
        candidates = fall10[fall10 > t90]
        if len(candidates) > 0:
            t10 = candidates[0]
            ft = t10 - t90
            if ft > 0 and ft < 1e-3:
                fall_times.append(ft)
    
    rise_time = np.mean(rise_times) if len(rise_times) > 0 else np.nan
    fall_time = np.mean(fall_times) if len(fall_times) > 0 else np.nan

    # Slew rate laskenta
    # Slew rate = jännitteen muutos / aika
    # Käytetään 10%-90% väliä (0.8 * delta)
    voltage_swing = 0.8 * delta  # 90% - 10% = 80% of delta
    
    # Lasketaan V/s, skaalataan myöhemmin
    slew_rate_rise = (voltage_swing / rise_time) if not np.isnan(rise_time) and rise_time > 0 else np.nan  # V/s
    slew_rate_fall = (voltage_swing / fall_time) if not np.isnan(fall_time) and fall_time > 0 else np.nan  # V/s

    falling = find_transition_times(t, v, th50, rising=False)
    freq = 1 / np.mean(np.diff(falling)) if len(falling) > 1 else np.nan

    band = 0.05 * delta if delta else 0
    after = t >= (falling[0] if len(falling) > 0 else t.iloc[0] if hasattr(t, 'iloc') else t[0])
    deviations = np.abs(v[after] - V_low)
    exceed = np.where(deviations > band)[0]
    settling = t[after].iloc[exceed[-1]] if len(exceed) > 0 and hasattr(t[after], 'iloc') else 0.0

    return {
        "V_high": V_high, "V_low": V_low, "V_max": np.max(v), "V_min": np.min(v),
        "overshoot_%": (np.max(v) - V_high) / delta * 100 if delta not in (None, 0, np.nan) and not np.isnan(delta) else 0.0,
        "undershoot_%": (V_low - np.min(v)) / delta * 100 if delta not in (None, 0, np.nan) and not np.isnan(delta) else 0.0,
        "rise_ns": rise_time * 1e9 if not np.isnan(rise_time) else np.nan,
        "fall_ns": fall_time * 1e9 if not np.isnan(fall_time) else np.nan,
        "slew_rate_rise": slew_rate_rise,  # V/s
        "slew_rate_fall": slew_rate_fall,  # V/s
        "freq_kHz": freq / 1e3 if not np.isnan(freq) else np.nan,
        "settling_ns": settling * 1e9 if settling else 0.0,
        "duty_%": np.mean(v > th50) * 100
    }

# ---------------------------------------------------------
# APUFUNKTIO: Automaattinen yksikön skaalaus
# ---------------------------------------------------------
def format_slew_rate(sr_value):
    """
    Muotoilee slew rate -arvon sopivalla yksiköllä.
    Syöte: V/s
    Palauttaa: (arvo, yksikkö)
    """
    if np.isnan(sr_value):
        return np.nan, "V/s"
    
    abs_val = abs(sr_value)
    
    # Valitse sopiva yksikkö suuruusluokan mukaan
    if abs_val >= 1e12:      # TV/s
        return sr_value / 1e12, "TV/s"
    elif abs_val >= 1e9:     # GV/s
        return sr_value / 1e9, "GV/s"
    elif abs_val >= 1e6:     # MV/s
        return sr_value / 1e6, "MV/s"
    elif abs_val >= 1e3:     # kV/s
        return sr_value / 1e3, "kV/s"
    elif abs_val >= 1:       # V/s
        return sr_value, "V/s"
    elif abs_val >= 1e-3:    # mV/s
        return sr_value * 1e3, "mV/s"
    elif abs_val >= 1e-6:    # µV/s
        return sr_value * 1e6, "µV/s"
    elif abs_val >= 1e-9:    # nV/s
        return sr_value * 1e9, "nV/s"
    else:                    # pV/s
        return sr_value * 1e12, "pV/s"

# ---------------------------------------------------------
# LASKENNAT
# ---------------------------------------------------------
v_min = min(original_meas_v.min(), original_sim_v.min())
v_max = max(original_meas_v.max(), original_sim_v.max())
threshold50 = v_min + 0.5 * (v_max - v_min)
swing = v_max - v_min

# Hae kaikki reunat molemmista signaaleista
meas_rising  = find_transition_times(original_meas_time, original_meas_v, threshold50, rising=True)
meas_falling = find_transition_times(original_meas_time, original_meas_v, threshold50, rising=False)
sim_rising   = find_transition_times(original_sim_time, original_sim_v, threshold50, rising=True)
sim_falling  = find_transition_times(original_sim_time, original_sim_v, threshold50, rising=False)

# Non-monotonicity detection
nonmono_meas_cnt, nonmono_meas_mark = detect_non_monotonicity(
    original_meas_time, original_meas_v, meas_falling, swing
)
nonmono_sim_cnt, nonmono_sim_mark = detect_non_monotonicity(
    original_sim_time, original_sim_v, sim_falling, swing
)

# Parametrit
p_meas = calculate_params(original_meas_time, original_meas_v, threshold50, swing)
p_sim  = calculate_params(original_sim_time, original_sim_v, threshold50, swing)

# Parametrit
p_meas = calculate_params(original_meas_time, original_meas_v, threshold50, swing)
p_sim  = calculate_params(original_sim_time, original_sim_v, threshold50, swing)

# Formatoi slew rate -arvot
sr_rise_meas_val, sr_rise_meas_unit = format_slew_rate(p_meas['slew_rate_rise'])
sr_fall_meas_val, sr_fall_meas_unit = format_slew_rate(p_meas['slew_rate_fall'])
sr_rise_sim_val, sr_rise_sim_unit = format_slew_rate(p_sim['slew_rate_rise'])
sr_fall_sim_val, sr_fall_sim_unit = format_slew_rate(p_sim['slew_rate_fall'])

params_text = f"""Waveform parameters (SDA INC74024)

{'Parameter':<22} {'Measured':>15} {'Simulated':>15}
{'─'*60}
V_high [V]             {p_meas['V_high']:15.4f} {p_sim['V_high']:15.4f}
V_low [V]              {p_meas['V_low']:15.4f} {p_sim['V_low']:15.4f}
V_max [V]              {p_meas['V_max']:15.4f} {p_sim['V_max']:15.4f}
V_min [V]              {p_meas['V_min']:15.4f} {p_sim['V_min']:15.4f}
Rise time [ns]         {p_meas['rise_ns']:15.1f} {p_sim['rise_ns']:15.1f}
Fall time [ns]         {p_meas['fall_ns']:15.1f} {p_sim['fall_ns']:15.1f}
Slew rate rise         {sr_rise_meas_val:11.3f} {sr_rise_meas_unit:>3} {sr_rise_sim_val:11.3f} {sr_rise_sim_unit:>3}
Slew rate fall         {sr_fall_meas_val:11.3f} {sr_fall_meas_unit:>3} {sr_fall_sim_val:11.3f} {sr_fall_sim_unit:>3}
Overshoot [%]          {p_meas['overshoot_%']:15.1f} {p_sim['overshoot_%']:15.1f}
Undershoot [%]         {p_meas['undershoot_%']:15.1f} {p_sim['undershoot_%']:15.1f}
Duty cycle [%]         {p_meas['duty_%']:15.1f} {p_sim['duty_%']:15.1f}
Frequency [kHz]        {p_meas['freq_kHz']:15.1f} {p_sim['freq_kHz']:15.1f}
Settling time [ns]     {p_meas['settling_ns']:15.1f} {p_sim['settling_ns']:15.1f}
Non-monotonic edges    {nonmono_meas_cnt:15d} {nonmono_sim_cnt:15d}
"""

# ---------------------------------------------------------
# INTERAKTIIVINEN NAVIGOINTI
# ---------------------------------------------------------
view_window = 50e-6

current_falling_idx = 0
current_rising_idx = 0
current_edge_time = 0.0
current_sim_offset = 0.0
manual_meas_offset = 0.0  # UUSI: Manuaalinen offset mittaukselle
manual_mode = False  # UUSI: Manuaalinen säätötila

fig = plt.figure(figsize=(16, 12))
fig.subplots_adjust(bottom=0.12, top=0.95, hspace=0.3)

ax = fig.add_subplot(2, 1, 1)
ax_params = fig.add_subplot(2, 1, 2)
ax_params.axis('off')

def update_plot(meas_edge_time, edge_type, edge_num):
    global current_edge_time, current_sim_offset
    current_edge_time = meas_edge_time
    
    # Etsi lähin simulointikäyrän reuna dynaamisesti
    if edge_type == "falling":
        sim_edge = find_closest_sim_edge(meas_edge_time, sim_falling)
    else:  # rising
        sim_edge = find_closest_sim_edge(meas_edge_time, sim_rising)
    
    # Laske offset tälle reunalle
    if sim_edge is not None:
        current_sim_offset = sim_edge - meas_edge_time
        time_diff_ns = current_sim_offset * 1e9
    else:
        time_diff_ns = current_sim_offset * 1e9

    # Käytä manuaalista offsetia jos manuaalitila päällä
    total_meas_offset = manual_meas_offset if manual_mode else 0.0
    
    # Normalisoi aika-akseli alkamaan nollasta
    t_start = min(
        (original_meas_time - meas_edge_time + total_meas_offset).min(),
        (original_sim_time - meas_edge_time - current_sim_offset).min()
    )
    
    meas_shifted = original_meas_time - meas_edge_time + total_meas_offset - t_start
    sim_shifted  = original_sim_time - meas_edge_time - current_sim_offset - t_start

    ax.cla()

    ax.plot(meas_shifted, original_meas_v, color="red", linewidth=2.5, label="Mitattu SDA", alpha=0.9)
    ax.plot(sim_shifted,  original_sim_v, color="blue", linewidth=2,   label="Simuloitu SDA", alpha=0.9)
    
    # Näytä myös raakadata haalealla jos kohinanpoisto on päällä
    if NOISE_FILTER_METHOD != 'none':
        # Laske raakadatan shift
        meas_raw_shifted = original_meas_time - meas_edge_time + total_meas_offset - t_start
        sim_raw_shifted = original_sim_time - meas_edge_time - current_sim_offset - t_start
        ax.plot(meas_raw_shifted, original_meas_v_raw, color="red", linewidth=0.8, 
                alpha=0.3, linestyle=':', label="Raaka mitattu (kohinainen)")
        ax.plot(sim_raw_shifted, original_sim_v_raw, color="blue", linewidth=0.8,
                alpha=0.3, linestyle=':', label="Raaka simuloitu (kohinainen)")

    for col in sim_df.columns:
        if col not in ["Time", sim_sda_col]:
            ax.plot(sim_shifted, sim_df[col], color="lightgray", alpha=0.3, linewidth=0.6)

    # Non-monotonic-merkinnät
    if nonmono_meas_mark:
        tm = [x - meas_edge_time + total_meas_offset - t_start for x, y in nonmono_meas_mark]
        vm = [y for x, y in nonmono_meas_mark]
        ax.scatter(tm, vm, marker='o', s=80, color="red", zorder=6, edgecolor="white", linewidth=1.5)

    if nonmono_sim_mark:
        ts = [x - meas_edge_time - current_sim_offset - t_start for x, y in nonmono_sim_mark]
        vs = [y for x, y in nonmono_sim_mark]
        ax.scatter(ts, vs, marker='o', s=60, color="magenta", zorder=6, edgecolor="white", linewidth=1)

    ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax.yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax.set_xlabel("Aika", fontsize=11)
    ax.set_ylabel("Jännite", fontsize=11)
    
    title_suffix = f" [MANUAL: {manual_meas_offset*1e9:.1f} ns]" if manual_mode else ""
    ax.set_title(f"I2C SDA – Align to {edge_type} edge #{edge_num}  (Δt = {time_diff_ns:.1f} ns){title_suffix}", 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)

    metadata = f"Mittaus: {os.path.basename(meas_file)}\nAlign: {edge_type} edge #{edge_num}\nTime difference: {time_diff_ns:.1f} ns"
    if manual_mode:
        metadata += f"\n🔧 Manual offset: {manual_meas_offset*1e9:.1f} ns"
    if NOISE_FILTER_METHOD != 'none':
        metadata += f"\n🔇 Noise filter: {NOISE_FILTER_METHOD} (window={NOISE_FILTER_WINDOW})"
    ax.text(0.02, 0.98, metadata, transform=ax.transAxes, va='top', ha='left', fontsize=9,
            family='monospace', bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.85))

    ax.axvline(-t_start, color='black', linestyle='--', alpha=0.7, label='Alignment point')
    
    # Laske näkymän rajat suhteessa nollaan
    view_start = -t_start - view_window / 2
    view_end = -t_start + view_window / 2
    ax.set_xlim(max(0, view_start), view_end)

    # Parametritaulukko
    ax_params.cla()
    ax_params.axis('off')
    ax_params.text(0.5, 0.5, params_text, transform=ax_params.transAxes, 
                   va='center', ha='center', fontsize=10,
                   family='monospace', 
                   bbox=dict(boxstyle="round,pad=1.0", facecolor="lightblue", alpha=0.95))

    fig.canvas.draw_idle()

# ---------------------------------------------------------
# NÄPPÄIMISTÖKUUNTELIJA (NUOLET)
# ---------------------------------------------------------
def on_key(event):
    global manual_meas_offset, manual_mode, current_falling_idx, current_rising_idx
    
    step_coarse = 10e-9  # 10 ns
    step_fine = 1e-9     # 1 ns
    
    if event.key == 'left':
        manual_mode = True
        manual_meas_offset -= step_coarse
        if current_falling_idx < len(meas_falling):
            update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
        elif current_rising_idx < len(meas_rising):
            update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
    
    elif event.key == 'right':
        manual_mode = True
        manual_meas_offset += step_coarse
        if current_falling_idx < len(meas_falling):
            update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
        elif current_rising_idx < len(meas_rising):
            update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
    
    elif event.key == 'shift+left':
        manual_mode = True
        manual_meas_offset -= step_fine
        if current_falling_idx < len(meas_falling):
            update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
        elif current_rising_idx < len(meas_rising):
            update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
    
    elif event.key == 'shift+right':
        manual_mode = True
        manual_meas_offset += step_fine
        if current_falling_idx < len(meas_falling):
            update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
        elif current_rising_idx < len(meas_rising):
            update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
    
    elif event.key == 'r':  # Reset manuaalinen offset
        manual_mode = False
        manual_meas_offset = 0.0
        if current_falling_idx < len(meas_falling):
            update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
        elif current_rising_idx < len(meas_rising):
            update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
        print("Manual offset reset")
    
    elif event.key == 's':  # Tallenna offset
        print(f"Current manual offset: {manual_meas_offset*1e9:.2f} ns")

fig.canvas.mpl_connect('key_press_event', on_key)

# Full View -nappi
ax_full = plt.axes([0.65, 0.03, 0.15, 0.055])
b_full = Button(ax_full, 'Full View')
def on_full_view(event):
    global current_edge_time, current_sim_offset, manual_meas_offset
    total_meas_offset = manual_meas_offset if manual_mode else 0.0
    
    # Laske t_start samalla tavalla kuin update_plot
    t_start = min(
        (original_meas_time - current_edge_time + total_meas_offset).min(),
        (original_sim_time - current_edge_time - current_sim_offset).min()
    )
    
    meas_shifted = original_meas_time - current_edge_time + total_meas_offset - t_start
    sim_shifted  = original_sim_time - current_edge_time - current_sim_offset - t_start
    all_x = np.concatenate([meas_shifted, sim_shifted])
    margin = (all_x.max() - all_x.min()) * 0.05
    ax.set_xlim(max(0, all_x.min() - margin), all_x.max() + margin)
    fig.canvas.draw_idle()
b_full.on_clicked(on_full_view)

# Alustava piirto
if len(meas_falling) > 0:
    initial_edge = meas_falling[0]
    update_plot(initial_edge, "falling", 1)
elif len(meas_rising) > 0:
    initial_edge = meas_rising[0]
    update_plot(initial_edge, "rising", 1)
else:
    initial_edge = original_meas_time.iloc[0]
    update_plot(initial_edge, "falling", 1)

# ---------------------------------------------------------
# NAPPIEN LUONTI
# ---------------------------------------------------------
ax_prev_f = plt.axes([0.05, 0.03, 0.13, 0.055])
b_prev_f = Button(ax_prev_f, '← Prev Falling')
def prev_f(event):
    global current_falling_idx
    if current_falling_idx > 0:
        current_falling_idx -= 1
        update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
b_prev_f.on_clicked(prev_f)

ax_next_f = plt.axes([0.19, 0.03, 0.13, 0.055])
b_next_f = Button(ax_next_f, 'Next Falling →')
def next_f(event):
    global current_falling_idx
    if current_falling_idx < len(meas_falling) - 1:
        current_falling_idx += 1
        update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
b_next_f.on_clicked(next_f)

ax_prev_r = plt.axes([0.35, 0.03, 0.13, 0.055])
b_prev_r = Button(ax_prev_r, '← Prev Rising')
def prev_r(event):
    global current_rising_idx
    if current_rising_idx > 0:
        current_rising_idx -= 1
        update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
b_prev_r.on_clicked(prev_r)

ax_next_r = plt.axes([0.49, 0.03, 0.13, 0.055])
b_next_r = Button(ax_next_r, 'Next Rising →')
def next_r(event):
    global current_rising_idx
    if current_rising_idx < len(meas_rising) - 1:
        current_rising_idx += 1
        update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
b_next_r.on_clicked(next_r)

# Ohjeteksti
print("\n" + "="*60)
print("KÄYTTÖOHJEET:")
print("="*60)
print("← →  : Siirrä mittausta 10 ns kerrallaan")
print("Shift + ← →  : Siirrä mittausta 1 ns kerrallaan (hieno säätö)")
print("R    : Nollaa manuaalinen offset")
print("S    : Tulosta nykyinen offset konsoliin")
print("="*60)
print(f"KOHINANPOISTO: {NOISE_FILTER_METHOD}")
if NOISE_FILTER_METHOD != 'none':
    print(f"  - Ikkuna: {NOISE_FILTER_WINDOW} pistettä")
    if NOISE_FILTER_METHOD == 'savgol':
        print(f"  - Polynomin aste: {NOISE_FILTER_POLYORDER}")
    print("  - Raakadata näkytetään haalealla viivalla vertailua varten")
print("="*60 + "\n")

plt.show()