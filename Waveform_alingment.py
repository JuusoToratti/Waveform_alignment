import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.widgets import Button
from scipy.signal import find_peaks
import os
#tiedosto nimen lyhennys
def short_name(fname, maxlen=25):
    base = os.path.basename(fname)
    return base if len(base) <= maxlen else base[:maxlen-3] + "..."

# ---------------------------------------------------------
# TIEDOSTOT
# ---------------------------------------------------------
sim_file = "I2C_all.csv"
meas_file = "REGRESSION_ AREC_X21_MADE2_I2C_1-I2C_0_P1V8(1) SDA-INC74024-2593-55585.csv"

# ---------------------------------------------------------
# KOHINANPOISTO
# ---------------------------------------------------------
from scipy.signal import savgol_filter, medfilt

def remove_noise(t, v, method='savgol', window=11, polyorder=3):
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
NOISE_FILTER_WINDOW = 11        # Ikkunan koko (pariton luku suositellaan)
NOISE_FILTER_POLYORDER = 3      # Polynomin aste (vain savgol)

#----------------------------------------------
#Kohinan poiston apufunktio vain tasaiset reunat
#----------------------------------------------
def smart_filter(t, v, rising_edges, falling_edges, window=11, polyorder=3, guard_ns=200):
    """
    Suodattaa kohinaa, mutta EI koske reunoihin.
    - rising_edges, falling_edges = listat reuna-ajoista
    - guard_ns = alue, jota EI suodateta (ns molemmin puolin)
    """

    t = np.asarray(t)
    v = np.asarray(v).copy()

    # Reunojen ympärille EI suodateta
    guard_s = guard_ns * 1e-9

    protected_mask = np.zeros(len(t), dtype=bool)

    # Merkitään alueet, joita ei saa suodattaa
    for edge in np.concatenate((rising_edges, falling_edges)):
        mask = (t >= edge - guard_s) & (t <= edge + guard_s)
        protected_mask |= mask

    # Suodatetaan kopio vain niiltä alueilta, joihin saa koskea
    v_filtered = remove_noise(t, v, method='savgol', window=window, polyorder=polyorder)

    # Korvataan vain sallitut kohdat
    for i in range(len(v)):
        if not protected_mask[i]:
            v[i] = v_filtered[i]

    return v


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
original_meas_v = original_meas_v_raw.copy()

# Automaattinen optimointi jos pyydetty
if NOISE_FILTER_METHOD == 'auto_optimize' and AUTO_OPTIMIZE_TARGET == 'sim':
    print("\n Optimoidaan suodatinparametreja simulaatioon...")
    best_window, best_polyorder, rms_error = optimize_filter_to_match_target(
        original_meas_time, original_meas_v_raw,
        original_sim_time, original_sim_v_raw,
        method='savgol'
    )
    print(f" Optimaaliset parametrit löydetty:")
    print(f"   - Window: {best_window}")
    print(f"   - Polyorder: {best_polyorder}")
    print(f"   - RMS error: {rms_error*1000:.2f} mV\n")
    
    # Käytä optimoituja parametreja
    NOISE_FILTER_WINDOW = best_window
    NOISE_FILTER_POLYORDER = best_polyorder
    actual_method = 'savgol'
else:
    actual_method = NOISE_FILTER_METHOD

original_meas_v = remove_noise(original_meas_time, original_meas_v_raw, 
                               method=actual_method, 
                              window=NOISE_FILTER_WINDOW,
                             polyorder=NOISE_FILTER_POLYORDER)
original_sim_v = original_sim_v_raw.copy()#tämä ei suodata nyt simuloitua käyrää

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
# APUFUNKTIO: Taajuuden estimointi FFT:n avulla
# Käytetään, jos reunapohjainen periodin mittaus ei onnistu
# ---------------------------------------------------------
def fft_frequency_estimate(t, v):
    t = np.asarray(t)
    v = np.asarray(v)

    # Poista DC
    v = v - np.mean(v)

    # Laske dt turvallisesti
    dt_array = np.diff(t)
    dt_array = dt_array[dt_array > 0]
    if len(dt_array) == 0:
        print(" FFT ERROR: dt ei kelpaa → freq=0")
        return 0.0

    dt = np.mean(dt_array)
    fs = 1.0 / dt

    # FFT
    spectrum = np.abs(np.fft.rfft(v))
    freqs = np.fft.rfftfreq(len(v), dt)

    if np.max(spectrum) < 1e-12:
        print(" FFT ERROR: signaali tasainen → freq=0")
        return 0.0

    # Etsi huippu (DC ohitetaan)
    peak = np.argmax(spectrum[1:]) + 1
    freq = freqs[peak]

    return freq / 1e3  # kHz



# ---------------------------------------------------------
# PARAMETRIEN LASKENTA
# ---------------------------------------------------------
def calculate_params(t, v):
    t = np.asarray(t)
    v = np.asarray(v)

    # PERUSTASOT
    v_min = v.min()
    v_max = v.max()
    delta = v_max - v_min
    th50 = v_min + 0.5 * delta
    l10 = v_min + 0.1 * delta
    l90 = v_min + 0.9 * delta

    # REUNAT
    rising = find_transition_times(t, v, th50, rising=True)
    falling = find_transition_times(t, v, th50, rising=False)

    # EDGE-BASED FREQUENCY (Tarvitaan Fall Time ja Settling Time rajoituksiin)
    if len(falling) > 1:
        periods = np.diff(falling)
        periods = periods[periods > 0]
        freq_s = np.mean(periods) if len(periods) else np.nan
        freq_kHz = 1.0 / freq_s / 1e3 if freq_s else fft_frequency_estimate(t, v)
    else:
        freq_s = np.nan
        freq_kHz = fft_frequency_estimate(t, v)
    
    # Aseta dynaaminen aikaraja jaksonajan perusteella (tai maksimi)
    max_edge_time_s = 50e-6 # 50 us on turvallinen maksimi I2C:lle

    # Jos taajuus tiedossa, käytä puolta jaksoa
    if not np.isnan(freq_s):
        max_edge_time_s = min(freq_s / 2.0, max_edge_time_s)

    # RISE / FALL TIME
    rise10 = find_transition_times(t, v, l10, rising=True)
    rise90 = find_transition_times(t, v, l90, rising=True)
    fall10 = find_transition_times(t, v, l10, rising=False)
    fall90 = find_transition_times(t, v, l90, rising=False)

    rise_times = []
    for t10 in rise10:
        cand = rise90[rise90 > t10]
        if len(cand):
            dt = cand[0] - t10
            # KORJattu: Käytä dynaamista maksimirajaa
            if 0 < dt < max_edge_time_s: 
                rise_times.append(dt)
    rise_time = np.mean(rise_times) if rise_times else np.nan

    fall_times = []
    for t90 in fall90:
        cand = fall10[fall10 > t90]
        if len(cand):
            dt = cand[0] - t90
            # KORJattu: Käytä dynaamista maksimirajaa
            if 0 < dt < max_edge_time_s: 
                fall_times.append(dt)
    fall_time = np.mean(fall_times) if fall_times else np.nan

    # UUSI SLEW RATE (Jätetään ennalleen)
    # ... (slew rate laskenta) ...
    dv = np.diff(v)
    dt_diff = np.diff(t)
    dt_diff[dt_diff == 0] = np.min(dt_diff[dt_diff > 0])
    slope = dv / dt_diff
    slope_rise = slope[slope > 0]
    slope_fall = -slope[slope < 0]
    slew_rate_rise = np.max(slope_rise) if len(slope_rise) else np.nan
    slew_rate_fall = np.max(slope_fall) if len(slope_fall) else np.nan

    # Overshoot / Undershoot (Jätetään ennalleen)
    V_high = np.nanmean(v[v > th50])
    V_low = np.nanmean(v[v <= th50])
    overshoot_pct = ((v_max - V_high) / delta * 100) if delta else 0
    undershoot_pct = ((V_low - v_min) / delta * 100) if delta else 0

    # KORJATTU SETTLING TIME
    band = 0.05 * delta
    settling_times = []
    
    # Tarkista asettuminen jokaisen reunan jälkeen erikseen
    for t_fall_start in falling:
        # Tarkkailuikkuna, esim. 2 x jaksonaika reunan jälkeen
        t_max = t_fall_start + max_edge_time_s * 2 
        
        mask = (t >= t_fall_start) & (t <= t_max)
        t_win = t[mask]
        v_win = v[mask]

        if len(t_win) < 2: continue

        deviations = np.abs(v_win - V_low)
        exceed_indices = np.where(deviations > band)[0]
        
        if len(exceed_indices) > 0:
            # Viimeinen poikkeama toleranssista
            t_settle_abs = t_win[exceed_indices[-1]] 
            # Asettumisaika suhteessa reunan alkuun
            settling_times.append(t_settle_abs - t_fall_start)

    settling_time = np.mean(settling_times) if settling_times else 0.0
    settling_ns = settling_time * 1e9

    # DUTY CYCLE (Jätetään ennalleen)
    duty = np.mean(v > th50) * 100

    # PALAUTUS
    return {
        "V_high": V_high,
        "V_low": V_low,
        "V_max": v_max,
        "V_min": v_min,
        "rise_ns": rise_time * 1e9 if rise_time else np.nan,
        "fall_ns": fall_time * 1e9 if fall_time else np.nan,
        "slew_rate_rise": slew_rate_rise,
        "slew_rate_fall": slew_rate_fall,
        "freq_kHz": freq_kHz,
        "duty_%": duty,
        "overshoot_%": overshoot_pct,
        "undershoot_%": undershoot_pct,
        "settling_ns": settling_ns
    }
# ---------------------------------------------------------
# APUFUNKTIO: Automaattinen yksikön skaalaus
# ---------------------------------------------------------
def format_slew_rate(sr_value):
    """
    Muotoilee slew raten ns-aikaskaalaan.
    Syöte: V/s
    Palauttaa arvon ja yksikön (V/ns, mV/ns, µV/ns)
    """

    if np.isnan(sr_value):
        return np.nan, "V/ns"

    # V/s → V/ns muunnos
    v_per_ns = sr_value / 1e9

    abs_val = abs(v_per_ns)

    if abs_val >= 1:
        return v_per_ns, "V/ns"
    elif abs_val >= 1e-3:
        return v_per_ns * 1e3, "mV/ns"
    else:
        return v_per_ns * 1e6, "µV/ns"
# ---------------------------------------------------------
# LASKENNAT
# ---------------------------------------------------------
sim_v_min = original_sim_v.min()
sim_v_max = original_sim_v.max()
sim_threshold50 = sim_v_min + 0.5 * (sim_v_max - sim_v_min)

meas_v_min = original_meas_v.min()
meas_v_max = original_meas_v.max()
meas_threshold50 = meas_v_min + 0.5 * (meas_v_max - meas_v_min)

swing = meas_v_max - meas_v_min

# Hae kaikki reunat molemmista signaaleista
meas_rising  = find_transition_times(original_meas_time, original_meas_v, meas_threshold50, rising=True)
meas_falling = find_transition_times(original_meas_time, original_meas_v, meas_threshold50)
sim_rising   = find_transition_times(original_sim_time, original_sim_v, sim_threshold50,rising=True)
sim_falling  = find_transition_times(original_sim_time, original_sim_v, sim_threshold50, rising=False)
# Edge-aware suodatus
original_meas_v = smart_filter(
    original_meas_time,
    original_meas_v_raw,
    rising_edges=meas_rising,
    falling_edges=meas_falling,
    window=NOISE_FILTER_WINDOW,
    polyorder=NOISE_FILTER_POLYORDER,
    guard_ns=200
)

# Non-monotonicity detection
nonmono_meas_cnt, nonmono_meas_mark = detect_non_monotonicity(
    original_meas_time, original_meas_v, meas_falling, swing
)
nonmono_sim_cnt, nonmono_sim_mark = detect_non_monotonicity(
    original_sim_time, original_sim_v, sim_falling, swing
)

# Parametrit
p_meas = calculate_params(original_meas_time, original_meas_v)
p_sim  = calculate_params(original_sim_time,  original_sim_v)


# Parametrit
p_meas = calculate_params(original_meas_time, original_meas_v)
p_sim  = calculate_params(original_sim_time,  original_sim_v)

# Formatoi slew rate -arvot
sr_rise_meas_val, sr_rise_meas_unit = format_slew_rate(p_meas['slew_rate_rise'])
sr_fall_meas_val, sr_fall_meas_unit = format_slew_rate(p_meas['slew_rate_fall'])
sr_rise_sim_val, sr_rise_sim_unit = format_slew_rate(p_sim['slew_rate_rise'])
sr_fall_sim_val, sr_fall_sim_unit = format_slew_rate(p_sim['slew_rate_fall'])

params_text = f"""Waveform parameters (SDA INC74024)


Parameter             Meas.          Sim.
---------------------------------------------------
V_high [V]        {p_meas['V_high']:>10.3f}    {p_sim['V_high']:>10.3f}
V_low [V]         {p_meas['V_low']:>10.3f}    {p_sim['V_low']:>10.3f}
V_max [V]         {p_meas['V_max']:>10.3f}    {p_sim['V_max']:>10.3f}
V_min [V]         {p_meas['V_min']:>10.3f}    {p_sim['V_min']:>10.3f}
Rise time [ns]    {p_meas['rise_ns']:>10.1f}    {p_sim['rise_ns']:>10.1f}
Fall time [ns]    {p_meas['fall_ns']:>10.1f}    {p_sim['fall_ns']:>10.1f}
Slew rise [{sr_rise_meas_unit}]  {sr_rise_meas_val:>10.3f}   {sr_rise_sim_val:>10.3f}
Slew fall [{sr_fall_meas_unit}]  {sr_fall_meas_val:>10.3f}   {sr_fall_sim_val:>10.3f}
Overshoot [%]     {p_meas['overshoot_%']:>10.1f}    {p_sim['overshoot_%']:>10.1f}
Undershoot [%]    {p_meas['undershoot_%']:>10.1f}    {p_sim['undershoot_%']:>10.1f}
Duty [%]          {p_meas['duty_%']:>10.1f}    {p_sim['duty_%']:>10.1f}
Freq [kHz]        {p_meas['freq_kHz']:>10.1f}    {p_sim['freq_kHz']:>10.1f}
Settling [ns]     {p_meas['settling_ns']:>10.1f}    {p_sim['settling_ns']:>10.1f}
Non-monotonic     {nonmono_meas_cnt:>10d}    {nonmono_sim_cnt:>10d}
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
fig.subplots_adjust(bottom=0.12, top=0.95, hspace=0.3)  # Palautettu alkuperäinen

ax = fig.add_subplot(2, 1, 1)
ax_params = fig.add_subplot(2, 1, 2)
ax_params.axis('off')

def update_plot(meas_edge_time, edge_type, edge_num):
    global current_edge_time, current_sim_offset
    global manual_meas_offset, manual_mode

    current_edge_time = meas_edge_time

    # --- 1. Selvitä simulaation vastaava reuna ---
    if edge_type == "falling":
        sim_edge = find_closest_sim_edge(meas_edge_time, sim_falling)
        fallback_edges = sim_falling
    else:
        sim_edge = find_closest_sim_edge(meas_edge_time, sim_rising)
        fallback_edges = sim_rising

    # --- 2. Jos vastaavaa sim-reunaa ei löydy ---
    if sim_edge is None:
        if len(fallback_edges) > 0:
            # fallback: simulaation ensimmäinen saman tyypin reuna
            sim_edge = fallback_edges[0]
            current_sim_offset = sim_edge - meas_edge_time
        else:
            # jos simulaatiossa ei ole reunoja lainkaan
            sim_edge = 0.0
            current_sim_offset = 0.0
    else:
        # Normaalisti päivitä offset
        current_sim_offset = sim_edge - meas_edge_time

    # --- 3. Manuaalinen offset mittaukselle ---
    total_meas_offset = manual_meas_offset if manual_mode else 0.0

    # --- 4. Kohdista signaalit niin että reuna = 0 ---
    sim_shifted = original_sim_time - sim_edge
    meas_shifted = original_meas_time - meas_edge_time + total_meas_offset

    # --- 5. Piirto ---
    ax.cla()

    ax.plot(
    meas_shifted, original_meas_v,
    color="red", linewidth=2.5,
    label=f"Meas SDA ({short_name(meas_file)})",
    alpha=0.9)

    ax.plot(
    sim_shifted, original_sim_v,
    color="blue", linewidth=2,
    label=f"Sim SDA ({short_name(sim_file)})",
    alpha=0.9)

    # Raakadata
    #if NOISE_FILTER_METHOD != 'none':
     #   meas_raw_shifted = original_meas_time - meas_edge_time + total_meas_offset
      #  ax.plot(meas_raw_shifted, original_meas_v_raw,
       #         color="red", linewidth=0.8, alpha=0.3, linestyle=':',
        #        label="Raaka mitattu (kohinainen)")

    # Muut simulaatiosarakkeet haaleina
    for col in sim_df.columns:
        if col not in ["Time", sim_sda_col]:
            ax.plot(sim_shifted, sim_df[col],
                    color="lightgray", alpha=0.3, linewidth=0.6)

    # Non-monotonic markers
    if nonmono_meas_mark:
        tm = [x - meas_edge_time + total_meas_offset for x, y in nonmono_meas_mark]
        vm = [y for x, y in nonmono_meas_mark]
        ax.scatter(tm, vm, marker='o', s=80, color="red",
                   zorder=6, edgecolor="white", linewidth=1.5)

    if nonmono_sim_mark:
        ts = [x - sim_edge for x, y in nonmono_sim_mark]
        vs = [y for x, y in nonmono_sim_mark]
        ax.scatter(ts, vs, marker='o', s=60, color="magenta",
                   zorder=6, edgecolor="white", linewidth=1)

    # --- Akseliasetukset ---
    ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax.yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax.set_xlabel("Aika (kohdistettu reunaan t=0)", fontsize=11)
    ax.set_ylabel("Voltage", fontsize=11)

    time_diff_ns = current_sim_offset * 1e9
    title_suffix = f" [MANUAL: {manual_meas_offset*1e9:.1f} ns]" if manual_mode else ""
    ax.set_title(
        f"I2C SDA – Align to {edge_type} edge #{edge_num}  (Δt = {time_diff_ns:.1f} ns){title_suffix}",
        fontsize=13, fontweight='bold'
    )

    ax.grid(True, alpha=0.3)
    ax.legend(
        bbox_to_anchor=(-0.11, -1.0),
        loc='upper left',
        fontsize=10,
        framealpha=0.95
    )

    # Parametritaulukko
    ax_params.cla()
    ax_params.axis('off')
    ax_params.text(
        0.5, 0.5, params_text,
        transform=ax_params.transAxes,
        va='center', ha='center',
        fontfamily='monospace', fontsize=10,
        bbox=dict(boxstyle="round,pad=1.0",
                  facecolor="lightblue", alpha=0.95)
    )

    # Näkymän rajaus
    ax.set_xlim(-view_window / 2, view_window / 2)

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
# Full View -nappi
ax_full = plt.axes([0.65, 0.03, 0.15, 0.055])
b_full = Button(ax_full, 'Full View')
def on_full_view(event):
    global current_edge_time, current_sim_offset, manual_meas_offset
    total_meas_offset = manual_meas_offset if manual_mode else 0.0
    
    sim_shifted = original_sim_time - current_edge_time - current_sim_offset
    meas_shifted = original_meas_time - current_edge_time + total_meas_offset
    

    
    # Etsi koko näkymän rajat
    min_x = min(meas_shifted.min(), sim_shifted.min())
    max_x = max(meas_shifted.max(), sim_shifted.max())
    margin = (max_x - min_x) * 0.05
    ax.set_xlim(min_x - margin, max_x + margin)
    fig.canvas.draw_idle()

# MUUTOS: Tämä rivi puuttui koodistasi ja kytkee napin toimintaan
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
    global current_falling_idx, manual_mode, manual_meas_offset
    if current_falling_idx > 0:
        current_falling_idx -= 1
        manual_mode = False  # Nollaa manuaalinen tila
        manual_meas_offset = 0.0  # Nollaa offset
        update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
b_prev_f.on_clicked(prev_f)

ax_next_f = plt.axes([0.19, 0.03, 0.13, 0.055])
b_next_f = Button(ax_next_f, 'Next Falling →')
def next_f(event):
    global current_falling_idx, manual_mode, manual_meas_offset
    if current_falling_idx < len(meas_falling) - 1:
        current_falling_idx += 1
        manual_mode = False  # Nollaa manuaalinen tila
        manual_meas_offset = 0.0  # Nollaa offset
        update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
b_next_f.on_clicked(next_f)

ax_prev_r = plt.axes([0.35, 0.03, 0.13, 0.055])
b_prev_r = Button(ax_prev_r, '← Prev Rising')
def prev_r(event):
    global current_rising_idx, manual_mode, manual_meas_offset
    if current_rising_idx > 0:
        current_rising_idx -= 1
        manual_mode = False  # Nollaa manuaalinen tila
        manual_meas_offset = 0.0  # Nollaa offset
        update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
b_prev_r.on_clicked(prev_r)

ax_next_r = plt.axes([0.49, 0.03, 0.13, 0.055])
b_next_r = Button(ax_next_r, 'Next Rising →')
def next_r(event):
    global current_rising_idx, manual_mode, manual_meas_offset
    if current_rising_idx < len(meas_rising) - 1:
        current_rising_idx += 1
        manual_mode = False  # Nollaa manuaalinen tila
        manual_meas_offset = 0.0  # Nollaa offset
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
if NOISE_FILTER_METHOD == 'auto_optimize':
    print(f"  Automaattinen optimointi käytössä (tavoite: {AUTO_OPTIMIZE_TARGET})")
    print(f"  - Optimoitu window: {NOISE_FILTER_WINDOW}")
    print(f"  - Optimoitu polyorder: {NOISE_FILTER_POLYORDER}")
elif NOISE_FILTER_METHOD != 'none':
    print(f"  - Ikkuna: {NOISE_FILTER_WINDOW} pistettä")
    if NOISE_FILTER_METHOD == 'savgol':
        print(f"  - Polynomin aste: {NOISE_FILTER_POLYORDER}")
    print("  - Raakadata näkytetään haalealla viivalla vertailua varten")
print("="*60)
print("\n VINKKI: Aseta NOISE_FILTER_METHOD = 'auto_optimize'")
print("   koodin alussa optimoidaksesi suodatin automaattisesti!")
print("="*60 + "\n")

plt.show()