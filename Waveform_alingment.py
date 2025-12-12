import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from matplotlib.widgets import Button
from scipy.signal import find_peaks, savgol_filter, medfilt
from tkinter import filedialog, Tk
import os
import sys

# ---------------------------------------------------------
# PyInstaller EXE path management
# ---------------------------------------------------------
def resource_path(relative_path):
    """Finds the file correctly, works with both .py and .exe"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# ---------------------------------------------------------
# AUTOMATIC CSV FORMAT DETECTION
# ---------------------------------------------------------
def detect_csv_format(filepath, max_lines=50):
    """Detects CSV file format automatically"""
    print(f"\n{'='*60}")
    print(f"Analyzing file: {os.path.basename(filepath)}")
    print(f"{'='*60}")
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = [f.readline() for _ in range(max_lines)]
    
    separators = [';', ',', '\t', '|']
    decimal_seps = ['.', ',']
    
    best_config = {
        'separator': ',',
        'decimal': '.',
        'has_header': False,
        'header_row': 0,
        'time_col': 0,
        'voltage_col': 1,
        'num_columns': 2,
        'score': 0
    }
    
    for sep in separators:
        for dec in decimal_seps:
            if sep == dec:
                continue
            
            score = 0
            has_header = False
            header_row = 0
            time_col = 0
            voltage_col = 1
            num_cols = 0
            
            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                
                parts = line.strip().split(sep)
                if len(parts) < 2:
                    continue
                
                num_cols = max(num_cols, len(parts))
                
                if i < 5:
                    if any(keyword in line.lower() for keyword in ['time', 'voltage', 'v', 't', 'signal']):
                        has_header = True
                        header_row = i
                        
                        for idx, part in enumerate(parts):
                            if 'time' in part.lower() or part.lower() == 't':
                                time_col = idx
                            if 'voltage' in part.lower() or 'v' in part.lower() or 'signal' in part.lower():
                                voltage_col = idx
                        score += 50
                        continue
                
                numeric_count = 0
                for part in parts:
                    try:
                        num_str = part.strip().replace(dec, '.')
                        float(num_str)
                        numeric_count += 1
                    except:
                        pass
                
                if numeric_count >= 2:
                    score += 1
            
            if num_cols == 2:
                score += 20
            elif num_cols > 2:
                score += 10
            
            if score > best_config['score']:
                best_config = {
                    'separator': sep,
                    'decimal': dec,
                    'has_header': has_header,
                    'header_row': header_row,
                    'time_col': time_col,
                    'voltage_col': voltage_col,
                    'num_columns': num_cols,
                    'score': score
                }
    
    print(f"  Separator: '{best_config['separator']}'")
    print(f"  Decimal separator: '{best_config['decimal']}'")
    print(f"  Headers: {'Yes' if best_config['has_header'] else 'No'}")
    if best_config['has_header']:
        print(f"  Header row: {best_config['header_row']}")
    print(f"  Number of columns: {best_config['num_columns']}")
    print(f"  Confidence score: {best_config['score']}")
    print(f"{'='*60}\n")
    
    return best_config

def load_measurement_file(filepath):
    """Load measurement file with automatic detection"""
    config = detect_csv_format(filepath)
    
    try:
        if config['has_header']:
            df = pd.read_csv(
                filepath,
                sep=config['separator'],
                decimal=config['decimal'],
                skiprows=config['header_row'],
                engine='python',
                on_bad_lines='skip'
            )
        else:
            df = pd.read_csv(
                filepath,
                sep=config['separator'],
                decimal=config['decimal'],
                header=None,
                engine='python',
                on_bad_lines='skip'
            )
            df.columns = [f'Col_{i}' for i in range(len(df.columns))]
        
        time_col = None
        voltage_col = None
        
        for col in df.columns:
            col_lower = str(col).lower()
            if 'time' in col_lower or col == 'Col_0':
                time_col = col
            if 'voltage' in col_lower or 'signal' in col_lower or col == 'Col_1':
                voltage_col = col
        
        if time_col is None or voltage_col is None:
            time_col = df.columns[0]
            voltage_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
        
        result_df = pd.DataFrame()
        result_df['Time'] = pd.to_numeric(df[time_col], errors='coerce')
        result_df['Voltage'] = pd.to_numeric(df[voltage_col], errors='coerce')
        result_df = result_df.dropna().reset_index(drop=True)
        
        print(f"  Measurement file loaded: {len(result_df)} points")
        print(f"  Time column: {time_col}")
        print(f"  Voltage column: {voltage_col}")
        print(f"  Time range: {result_df['Time'].min():.2e} - {result_df['Time'].max():.2e} s")
        print(f"  Voltage range: {result_df['Voltage'].min():.3f} - {result_df['Voltage'].max():.3f} V\n")
        
        return result_df, voltage_col
        
    except Exception as e:
        print(f" ERROR loading measurement file: {e}")
        raise

def load_simulation_file(filepath):
    """Load simulation file with automatic detection"""
    config = detect_csv_format(filepath, max_lines=100)
    
    headers = None
    data_start_row = 0
    
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        cleaned = line.strip().lstrip(",")
        fields = cleaned.split(config['separator'])
        
        if any('time' in str(f).lower() for f in fields):
            headers = fields
            data_start_row = i + 1
            print(f"  Found Time header at row {i}")
            break
    
    try:
        if headers is not None:
            df_raw = pd.read_csv(
                filepath,
                sep=config['separator'],
                decimal=config['decimal'],
                skiprows=data_start_row,
                header=None,
                names=headers,
                engine="python",
                on_bad_lines='skip'
            )
        else:
            df_raw = pd.read_csv(
                filepath,
                sep=config['separator'],
                decimal=config['decimal'],
                header=None if not config['has_header'] else 0,
                engine="python",
                on_bad_lines='skip'
            )
            
            if not config['has_header']:
                df_raw.columns = [f'Signal_{i}' if i > 0 else 'Time' for i in range(len(df_raw.columns))]
        
        df = df_raw.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(how='all').reset_index(drop=True)
        
        available_columns = [c for c in df.columns if c != 'Time']
        
        if not available_columns:
            raise ValueError("No signal columns found!")
        
        variances = {}
        for col in available_columns:
            var = df[col].var()
            if not np.isnan(var):
                variances[col] = var
        
        if variances:
            default_signal = max(variances, key=variances.get)
            print(f"  The most active signal: {default_signal} (variance: {variances[default_signal]:.2e})")
        else:
            default_signal = available_columns[0]
            print(f"  Default signal: {default_signal}")
        
        print(f"  Simulation file loaded: {len(df)} points")
        print(f"  Available signals: {len(available_columns)} pcs")
        print(f"  Time column found: {'Time' in df.columns}")
        print(f"  Time range: {df['Time'].min():.2e} - {df['Time'].max():.2e} s\n")
        
        return df, available_columns, default_signal
        
    except Exception as e:
        print(f" ERROR loading simulation file: {e}")
        raise

# ---------------------------------------------------------
# NOISE CANCELLATION
# ---------------------------------------------------------
def remove_noise(t, v, method='savgol', window=11, polyorder=3):
    """Removes noise from the signal."""
    if method == 'none':
        return v
    
    v_array = np.asarray(v)
    
    window = min(window, len(v_array))
    if window % 2 == 0:
        window += 1
    window = max(3, window)
    
    if method == 'savgol':
        polyorder = min(polyorder, window - 1)
        return savgol_filter(v_array, window, polyorder)
    elif method == 'median':
        return medfilt(v_array, window)
    elif method == 'moving_avg':
        return np.convolve(v_array, np.ones(window)/window, mode='same')
    else:
        return v

NOISE_FILTER_METHOD = 'savgol'
NOISE_FILTER_WINDOW = 11
NOISE_FILTER_POLYORDER = 3

def smart_filter(t, v, rising_edges, falling_edges, window=11, polyorder=3, guard_ns=200):
    """Filters noise, but does NOT affect edges."""
    t = np.asarray(t)
    v = np.asarray(v).copy()
    guard_s = guard_ns * 1e-9
    protected_mask = np.zeros(len(t), dtype=bool)

    for edge in np.concatenate((rising_edges, falling_edges)):
        mask = (t >= edge - guard_s) & (t <= edge + guard_s)
        protected_mask |= mask

    v_filtered = remove_noise(t, v, method='savgol', window=window, polyorder=polyorder)

    for i in range(len(v)):
        if not protected_mask[i]:
            v[i] = v_filtered[i]

    return v

# ---------------------------------------------------------
# FINDING EDGES
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

def detect_non_monotonicity(t, v, falling_times, swing, window_ns=200e-9, min_ampl_pct=5):
    """Identify non-monotonic edges."""
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

def find_closest_sim_edge(meas_edge_time, sim_edges, max_distance=10e-6):
    """Find the nearest simulation curve edge near the measurement edge"""
    if len(sim_edges) == 0:
        return None
    distances = np.abs(sim_edges - meas_edge_time)
    min_idx = np.argmin(distances)
    if distances[min_idx] < max_distance:
        return sim_edges[min_idx]
    return None

def fft_frequency_estimate(t, v):
    """Frequency estimation using FFT"""
    t = np.asarray(t)
    v = np.asarray(v)
    v = v - np.mean(v)

    dt_array = np.diff(t)
    dt_array = dt_array[dt_array > 0]
    if len(dt_array) == 0:
        return 0.0

    dt = np.mean(dt_array)
    fs = 1.0 / dt

    spectrum = np.abs(np.fft.rfft(v))
    freqs = np.fft.rfftfreq(len(v), dt)

    if np.max(spectrum) < 1e-12:
        return 0.0

    peak = np.argmax(spectrum[1:]) + 1
    freq = freqs[peak]

    return freq / 1e3

# ---------------------------------------------------------
# CALCULATION OF PARAMETERS
# ---------------------------------------------------------
def calculate_params(t, v):
    """Calculates parameters from a given time window"""
    t = np.asarray(t)
    v = np.asarray(v)

    v_min = v.min()
    v_max = v.max()
    delta = v_max - v_min
    th50 = v_min + 0.5 * delta
    l10 = v_min + 0.1 * delta
    l90 = v_min + 0.9 * delta

    rising = find_transition_times(t, v, th50, rising=True)
    falling = find_transition_times(t, v, th50, rising=False)

    if len(falling) > 1:
        periods = np.diff(falling)
        periods = periods[periods > 0]
        freq_s = np.mean(periods) if len(periods) else np.nan
        freq_kHz = 1.0 / freq_s / 1e3 if freq_s else fft_frequency_estimate(t, v)
    else:
        freq_s = np.nan
        freq_kHz = fft_frequency_estimate(t, v)
    
    max_edge_time_s = 50e-6
    if not np.isnan(freq_s):
        max_edge_time_s = min(freq_s / 2.0, max_edge_time_s)

    rise10 = find_transition_times(t, v, l10, rising=True)
    rise90 = find_transition_times(t, v, l90, rising=True)
    fall10 = find_transition_times(t, v, l10, rising=False)
    fall90 = find_transition_times(t, v, l90, rising=False)

    rise_times = []
    for t10 in rise10:
        cand = rise90[rise90 > t10]
        if len(cand):
            dt = cand[0] - t10
            if 0 < dt < max_edge_time_s: 
                rise_times.append(dt)
    rise_time = np.mean(rise_times) if rise_times else np.nan

    fall_times = []
    for t90 in fall90:
        cand = fall10[fall10 > t90]
        if len(cand):
            dt = cand[0] - t90
            if 0 < dt < max_edge_time_s: 
                fall_times.append(dt)
    fall_time = np.mean(fall_times) if fall_times else np.nan

    dv = np.diff(v)
    dt_diff = np.diff(t)
    dt_diff[dt_diff == 0] = np.min(dt_diff[dt_diff > 0]) if np.any(dt_diff > 0) else 1e-12
    slope = dv / dt_diff
    slope_rise = slope[slope > 0]
    slope_fall = -slope[slope < 0]
    slew_rate_rise = np.max(slope_rise) if len(slope_rise) else np.nan
    slew_rate_fall = np.max(slope_fall) if len(slope_fall) else np.nan

    V_high = np.nanmean(v[v > th50])
    V_low = np.nanmean(v[v <= th50])
    overshoot_pct = ((v_max - V_high) / delta * 100) if delta else 0
    undershoot_pct = ((V_low - v_min) / delta * 100) if delta else 0

    band = 0.05 * delta
    settling_times = []
    
    for t_fall_start in falling:
        t_max = t_fall_start + max_edge_time_s * 2 
        mask = (t >= t_fall_start) & (t <= t_max)
        t_win = t[mask]
        v_win = v[mask]

        if len(t_win) < 2: 
            continue

        deviations = np.abs(v_win - V_low)
        exceed_indices = np.where(deviations > band)[0]
        
        if len(exceed_indices) > 0:
            t_settle_abs = t_win[exceed_indices[-1]] 
            settling_times.append(t_settle_abs - t_fall_start)

    settling_time = np.mean(settling_times) if settling_times else 0.0
    settling_ns = settling_time * 1e9

    duty = np.mean(v > th50) * 100

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
# GLOBAL VARIABLES
# ---------------------------------------------------------
sim_df = None
meas_df = None
sim_file = None
meas_file = None
available_sim_columns = []
sim_sda_col = None
meas_sda_col = "Voltage"
current_sim_column_idx = 0

original_sim_time = None
original_meas_time = None
original_sim_v_raw = None
original_meas_v_raw = None
original_sim_v = None
original_meas_v = None

sim_v_min = 0
sim_v_max = 0
sim_threshold50 = 0
meas_v_min = 0
meas_v_max = 0
meas_threshold50 = 0
swing = 0

meas_rising = np.array([])
meas_falling = np.array([])
sim_rising = np.array([])
sim_falling = np.array([])

view_window = 50e-6
current_falling_idx = 0
current_rising_idx = 0
current_edge_time = 0.0
current_sim_offset = 0.0
manual_meas_offset = 0.0
manual_mode = False

fig = None
ax = None
ax_params = None
ax2 = None

# ---------------------------------------------------------
# FILE DOWNLOAD FUNCTIONS
# ---------------------------------------------------------
def reinitialize_analysis():
    """Recalculates everything after files are loaded"""
    global original_sim_time, original_meas_time
    global original_sim_v_raw, original_meas_v_raw
    global original_sim_v, original_meas_v
    global sim_v_min, sim_v_max, sim_threshold50
    global meas_v_min, meas_v_max, meas_threshold50, swing
    global meas_rising, meas_falling, sim_rising, sim_falling
    global current_falling_idx, current_rising_idx
    global manual_meas_offset, manual_mode
    
    if sim_df is None or meas_df is None:
        return
    
    # Reset offsets
    manual_meas_offset = 0.0
    manual_mode = False
    current_falling_idx = 0
    current_rising_idx = 0
    
    # Copy times and voltages
    original_sim_time = sim_df["Time"].copy()
    original_meas_time = meas_df["Time"].copy()
    
    original_meas_v_raw = meas_df["Voltage"].copy()
    original_sim_v_raw = sim_df[sim_sda_col].copy()
    
    #Noise reduction for measurement
    original_meas_v = remove_noise(original_meas_time, original_meas_v_raw, 
                                   method=NOISE_FILTER_METHOD, 
                                   window=NOISE_FILTER_WINDOW,
                                   polyorder=NOISE_FILTER_POLYORDER)
    original_sim_v = original_sim_v_raw.copy()
    
    if hasattr(original_meas_v_raw, 'index'):
        original_meas_v = pd.Series(original_meas_v, index=original_meas_v_raw.index)
    if hasattr(original_sim_v_raw, 'index'):
        original_sim_v = pd.Series(original_sim_v, index=original_sim_v_raw.index)
    
    # Calculate thresholds
    sim_v_min = original_sim_v.min()
    sim_v_max = original_sim_v.max()
    sim_threshold50 = sim_v_min + 0.5 * (sim_v_max - sim_v_min)
    
    meas_v_min = original_meas_v.min()
    meas_v_max = original_meas_v.max()
    meas_threshold50 = meas_v_min + 0.5 * (meas_v_max - meas_v_min)
    
    swing = meas_v_max - meas_v_min
    
    # Get the edges
    meas_rising = find_transition_times(original_meas_time, original_meas_v, meas_threshold50, rising=True)
    meas_falling = find_transition_times(original_meas_time, original_meas_v, meas_threshold50, rising=False)
    sim_rising = find_transition_times(original_sim_time, original_sim_v, sim_threshold50, rising=True)
    sim_falling = find_transition_times(original_sim_time, original_sim_v, sim_threshold50, rising=False)
    
    # Edge-aware filtering
    original_meas_v = smart_filter(
        original_meas_time,
        original_meas_v_raw,
        rising_edges=meas_rising,
        falling_edges=meas_falling,
        window=NOISE_FILTER_WINDOW,
        polyorder=NOISE_FILTER_POLYORDER,
        guard_ns=200
    )
    
    print(f" Analysis updated")
    print(f"  Measuring edges: {len(meas_rising)} rising, {len(meas_falling)} falling")
    print(f"  Simulation edges: {len(sim_rising)} rising, {len(sim_falling)} falling\n")
    
    # Refresh the view
    if len(meas_falling) > 0:
        update_plot(meas_falling[0], "falling", 1)
    elif len(meas_rising) > 0:
        update_plot(meas_rising[0], "rising", 1)

def load_new_measurement():
    """Download a new measurement file"""
    global meas_df, meas_file, meas_sda_col
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    filepath = filedialog.askopenfilename(
        title="Select the measurement file (Measurement)",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    
    if not filepath:
        print("File selection canceled")
        return
    
    try:
        meas_file = filepath
        meas_df, meas_sda_col = load_measurement_file(filepath)
        
        if sim_df is not None:
            reinitialize_analysis()
        
        print("Measurement file successfully changed!\n")
        
    except Exception as e:
        print(f" Error loading measurement file: {e}\n")

def load_new_simulation():
    """Load a new simulation file"""
    global sim_df, sim_file, available_sim_columns, sim_sda_col, current_sim_column_idx
    
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    
    filepath = filedialog.askopenfilename(
        title="Select the simulation file(Simulation)",
        filetypes=[
            ("CSV files", "*.csv"),
            ("Text files", "*.txt"),
            ("All files", "*.*")
        ]
    )
    root.destroy()
    
    if not filepath:
        print("File selection canceled")
        return
    
    try:
        sim_file = filepath
        sim_df, available_sim_columns, sim_sda_col = load_simulation_file(filepath)
        current_sim_column_idx = available_sim_columns.index(sim_sda_col)
        
        if meas_df is not None:
            reinitialize_analysis()
        
        print(" Simulation file changed successfully!\n")
        
    except Exception as e:
        print(f" Error loading simulation file: {e}\n")

# ---------------------------------------------------------
#PLOTTING FUNCTIONS
# ---------------------------------------------------------
def update_sim_signal(new_idx):
    """Updates the simulation curve being used"""
    global current_sim_column_idx, sim_sda_col, original_sim_v_raw, original_sim_v
    global sim_rising, sim_falling, sim_threshold50, sim_v_min, sim_v_max
    
    if sim_df is None:
        return
    
    current_sim_column_idx = new_idx
    sim_sda_col = available_sim_columns[current_sim_column_idx]
    original_sim_v_raw = sim_df[sim_sda_col].copy()
    original_sim_v = original_sim_v_raw.copy()
    
    if hasattr(original_sim_v_raw, 'index'):
        original_sim_v = pd.Series(original_sim_v, index=original_sim_v_raw.index)
    
    sim_v_min = original_sim_v.min()
    sim_v_max = original_sim_v.max()
    sim_threshold50 = sim_v_min + 0.5 * (sim_v_max - sim_v_min)
    
    sim_rising = find_transition_times(original_sim_time, original_sim_v, sim_threshold50, rising=True)
    sim_falling = find_transition_times(original_sim_time, original_sim_v, sim_threshold50, rising=False)
    
    print(f"Switched to simulation curve: {sim_sda_col}")
    print(f"  curve {current_sim_column_idx + 1}/{len(available_sim_columns)}")
#==============================================
#Waveform alignment update function
#==============================================
def update_plot(meas_edge_time, edge_type, edge_num):
    global current_edge_time, current_sim_offset
    global manual_meas_offset, manual_mode
    
    if sim_df is None or meas_df is None:
        print(" Download both files first!")
        return

    current_edge_time = meas_edge_time

    if edge_type == "falling":
        sim_edge = find_closest_sim_edge(meas_edge_time, sim_falling)
        fallback_edges = sim_falling
    else:
        sim_edge = find_closest_sim_edge(meas_edge_time, sim_rising)
        fallback_edges = sim_rising

    if sim_edge is None:
        if len(fallback_edges) > 0:
            sim_edge = fallback_edges[0]
            current_sim_offset = sim_edge - meas_edge_time
        else:
            sim_edge = 0.0
            current_sim_offset = 0.0
    else:
        current_sim_offset = sim_edge - meas_edge_time

    total_meas_offset = manual_meas_offset if manual_mode else 0.0

    sim_shifted = original_sim_time - sim_edge
    meas_shifted = original_meas_time - meas_edge_time + total_meas_offset

    window_margin = view_window * 1.5
    
    meas_mask = (original_meas_time >= meas_edge_time - window_margin/2) & \
                (original_meas_time <= meas_edge_time + window_margin/2)
    meas_window_t = original_meas_time[meas_mask]
    meas_window_v = original_meas_v[meas_mask]
    
    sim_mask = (original_sim_time >= sim_edge - window_margin/2) & \
               (original_sim_time <= sim_edge + window_margin/2)
    sim_window_t = original_sim_time[sim_mask]
    sim_window_v = original_sim_v[sim_mask]
    
    if len(meas_window_t) > 10 and len(meas_window_v) > 10:
        p_meas = calculate_params(meas_window_t, meas_window_v)
    else:
        p_meas = {k: np.nan for k in ["V_high", "V_low", "V_max", "V_min", 
                                       "rise_ns", "fall_ns", "slew_rate_rise", 
                                       "slew_rate_fall", "freq_kHz", "duty_%", 
                                       "overshoot_%", "undershoot_%", "settling_ns"]}
    
    if len(sim_window_t) > 10 and len(sim_window_v) > 10:
        p_sim = calculate_params(sim_window_t, sim_window_v)
    else:
        p_sim = {k: np.nan for k in ["V_high", "V_low", "V_max", "V_min", 
                                      "rise_ns", "fall_ns", "slew_rate_rise", 
                                      "slew_rate_fall", "freq_kHz", "duty_%", 
                                      "overshoot_%", "undershoot_%", "settling_ns"]}
    
    meas_falling_window = find_transition_times(meas_window_t, meas_window_v, 
                                                meas_threshold50, rising=False)
    sim_falling_window = find_transition_times(sim_window_t, sim_window_v, 
                                               sim_threshold50, rising=False)
    
    nonmono_meas_cnt, nonmono_meas_mark = detect_non_monotonicity(
        meas_window_t, meas_window_v, meas_falling_window, swing
    )
    nonmono_sim_cnt, nonmono_sim_mark = detect_non_monotonicity(
        sim_window_t, sim_window_v, sim_falling_window, swing
    )
    
    sr_rise_meas_val = p_meas['slew_rate_rise'] / 1e6
    sr_fall_meas_val = p_meas['slew_rate_fall'] / 1e6
    sr_rise_sim_val  = p_sim['slew_rate_rise']  / 1e6
    sr_fall_sim_val  = p_sim['slew_rate_fall']  / 1e6
    
    params_text = f"""Waveform parameters ({sim_sda_col})
    Time window parameters: ±{window_margin*1e6:.1f} µs around the edge

Parameter             Meas.          Sim.
---------------------------------------------------
V_high [V]        {p_meas['V_high']:>10.3f}    {p_sim['V_high']:>10.3f}
V_low [V]         {p_meas['V_low']:>10.3f}    {p_sim['V_low']:>10.3f}
V_max [V]         {p_meas['V_max']:>10.3f}    {p_sim['V_max']:>10.3f}
V_min [V]         {p_meas['V_min']:>10.3f}    {p_sim['V_min']:>10.3f}
Rise time [ns]    {p_meas['rise_ns']:>10.1f}    {p_sim['rise_ns']:>10.1f}
Fall time [ns]    {p_meas['fall_ns']:>10.1f}    {p_sim['fall_ns']:>10.1f}
Slew rise [V/µs]  {sr_rise_meas_val:>12.3f}   {sr_rise_sim_val:>12.3f}
Slew fall [V/µs]  {sr_fall_meas_val:>12.3f}   {sr_fall_sim_val:>12.3f}
Overshoot [%]     {p_meas['overshoot_%']:>10.1f}    {p_sim['overshoot_%']:>10.1f}
Undershoot [%]    {p_meas['undershoot_%']:>10.1f}    {p_sim['undershoot_%']:>10.1f}
Duty [%]          {p_meas['duty_%']:>10.1f}    {p_sim['duty_%']:>10.1f}
Freq [kHz]        {p_meas['freq_kHz']:>10.1f}    {p_sim['freq_kHz']:>10.1f}
Settling [ns]     {p_meas['settling_ns']:>10.1f}    {p_sim['settling_ns']:>10.1f}
Non-monotonic     {nonmono_meas_cnt:>10d}    {nonmono_sim_cnt:>10d}
"""

    ax.cla()
    ax2.cla()
    
    #Legends dynamically with filenames
    meas_label = f"Meas: {os.path.basename(meas_file)}" if meas_file else "Measurement"
    sim_label = f"Sim: {os.path.basename(sim_file)}" if sim_file else "Simulation"
    
    ax.plot(meas_shifted, original_meas_v,
            color="red", linewidth=2.5,
            label=meas_label,
            alpha=0.9)

    ax.plot(sim_shifted, original_sim_v,
            color="blue", linewidth=2,
            label=sim_label,
            alpha=0.9)
    ax2.plot(meas_shifted, original_meas_v,
             color="red", linewidth=2,
             alpha=0.9)
    
    ax2.plot(sim_shifted, original_sim_v,
             color="blue", linewidth=2,
             alpha=0.7)

    for col in sim_df.columns:
        if col not in ["Time", sim_sda_col]:
            ax.plot(sim_shifted, sim_df[col],
                    color="lightgray", alpha=0.3, linewidth=0.6)

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

    ax.xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax.yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax2.xaxis.set_major_formatter(EngFormatter(unit='s'))
    ax2.yaxis.set_major_formatter(EngFormatter(unit='V'))
    ax.set_xlabel("Time (aligned to the edge t=0)", fontsize=8)
    ax.set_ylabel("Voltage", fontsize=11)

    time_diff_ns = current_sim_offset * 1e9
    title_suffix = f" [MANUAL: {manual_meas_offset*1e9:.1f} ns]" if manual_mode else ""
    signal_info = f" | Signal: {sim_sda_col} ({current_sim_column_idx + 1}/{len(available_sim_columns)})"
    ax.set_title(
        f"Waveform Alignment – {edge_type} edge #{edge_num}  (Δt = {time_diff_ns:.1f} ns){title_suffix}{signal_info}",
        fontsize=13, fontweight='bold'
    )

    ax.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(0.6, -0.13), loc='upper center', fontsize=10, framealpha=0.95)

    ax_params.cla()
    ax_params.axis('off')
    ax_params.text(
        -0.05, 0.4, params_text,
        transform=ax_params.transAxes,
        va='center', ha='left',
        fontfamily='monospace', fontsize=10,
        bbox=dict(boxstyle="round,pad=1.0",
                  facecolor="lightblue", alpha=0.95)
    )

    ax.set_xlim(-view_window / 2, view_window / 2)
    fig.canvas.draw_idle()

# ---------------------------------------------------------
#USER INTERFACE
# ---------------------------------------------------------
print("\n" + "="*60)
print("WAVEFORM ALIGNMENT TOOL - Interactive version")
print("="*60)
print("Download files using the buttons on the interface!")
print("="*60 + "\n")

fig = plt.figure(figsize=(16, 12))
fig.subplots_adjust(bottom=0.12, top=0.95, left=0.06, right=0.94, hspace=0.3)
gs = fig.add_gridspec(3, 3, height_ratios=[3, 1, 2])
ax = fig.add_subplot(2, 1, 1)
ax_params = fig.add_subplot(2, 1, 2)
ax_params.axis('off')
ax2 = fig.add_subplot(gs[2, 1:3])

# Opening message
ax.text(0.5, 0.5, 'Download the files to get started\n\nUse "Load Measurement" ja "Load Simulation" -buttons',
        ha='center', va='center', fontsize=16, color='gray',
        transform=ax.transAxes)

# ---------------------------------------------------------
# KEYBOARD LISTENER
# ---------------------------------------------------------
def on_key(event):
    global manual_meas_offset, manual_mode, current_falling_idx, current_rising_idx
    
    if sim_df is None or meas_df is None:
        return
    
    step_coarse = 10e-9
    step_fine = 1e-9
    
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
    
    elif event.key == 'r':
        manual_mode = False
        manual_meas_offset = 0.0
        if current_falling_idx < len(meas_falling):
            update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
        elif current_rising_idx < len(meas_rising):
            update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
        print("Manual offset reset")
    
    elif event.key == 's':
        print(f"Current manual offset: {manual_meas_offset*1e9:.2f} ns")
    
    elif event.key == 'p':
        if current_sim_column_idx > 0:
            update_sim_signal(current_sim_column_idx - 1)
            if current_falling_idx < len(meas_falling):
                update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
            elif current_rising_idx < len(meas_rising):
                update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
    
    elif event.key == 'n':
        if current_sim_column_idx < len(available_sim_columns) - 1:
            update_sim_signal(current_sim_column_idx + 1)
            if current_falling_idx < len(meas_falling):
                update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
            elif current_rising_idx < len(meas_rising):
                update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)

fig.canvas.mpl_connect('key_press_event', on_key)

# ---------------------------------------------------------
# Buttons
# ---------------------------------------------------------

# Files Download buttons 
ax_load_meas = plt.axes([0.02, 0.485, 0.10, 0.035])
b_load_meas = Button(ax_load_meas, ' Load Measurement', color='lightcoral', hovercolor='salmon')
b_load_meas.on_clicked(lambda event: load_new_measurement())

ax_load_sim = plt.axes([0.13, 0.485, 0.10, 0.035])
b_load_sim = Button(ax_load_sim, ' Load Simulation', color='lightblue', hovercolor='skyblue')
b_load_sim.on_clicked(lambda event: load_new_simulation())

# FULL VIEW
ax_full = plt.axes([0.86, 0.39, 0.13, 0.055])
b_full = Button(ax_full, 'Full View')
def on_full_view(event):
    global current_edge_time, current_sim_offset, manual_meas_offset
    if sim_df is None or meas_df is None:
        return
    total_meas_offset = manual_meas_offset if manual_mode else 0.0
    
    sim_shifted = original_sim_time - current_edge_time - current_sim_offset
    meas_shifted = original_meas_time - current_edge_time + total_meas_offset
    
    min_x = min(meas_shifted.min(), sim_shifted.min())
    max_x = max(meas_shifted.max(), sim_shifted.max())
    margin = (max_x - min_x) * 0.05
    ax.set_xlim(min_x - margin, max_x + margin)
    fig.canvas.draw_idle()
b_full.on_clicked(on_full_view)

# SIGNAL SELECTION
ax_prev_sig = plt.axes([0.86, 0.51, 0.13, 0.055])
b_prev_sig = Button(ax_prev_sig, '← Prev Signal')
def prev_signal(event):
    global current_sim_column_idx
    if sim_df is None or current_sim_column_idx <= 0:
        return
    update_sim_signal(current_sim_column_idx - 1)
    if current_falling_idx < len(meas_falling):
        update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
    elif current_rising_idx < len(meas_rising):
        update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
b_prev_sig.on_clicked(prev_signal)

ax_next_sig = plt.axes([0.86, 0.45, 0.13, 0.055])
b_next_sig = Button(ax_next_sig, '→ Next Signal')
def next_signal(event):
    global current_sim_column_idx
    if sim_df is None or current_sim_column_idx >= len(available_sim_columns) - 1:
        return
    update_sim_signal(current_sim_column_idx + 1)
    if current_falling_idx < len(meas_falling):
        update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
    elif current_rising_idx < len(meas_rising):
        update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
b_next_sig.on_clicked(next_signal)

# EDGE NAVIGATION
ax_prev_f = plt.axes([0.320, 0.39, 0.13, 0.055])
b_prev_f = Button(ax_prev_f, '← Prev Falling')
def prev_f(event):
    global current_falling_idx, manual_mode, manual_meas_offset
    if meas_df is None or current_falling_idx <= 0:
        return
    current_falling_idx -= 1
    manual_mode = False
    manual_meas_offset = 0.0
    update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
b_prev_f.on_clicked(prev_f)

ax_next_f = plt.axes([0.456, 0.39, 0.13, 0.055])
b_next_f = Button(ax_next_f, 'Next Falling →')
def next_f(event):
    global current_falling_idx, manual_mode, manual_meas_offset
    if meas_df is None or current_falling_idx >= len(meas_falling) - 1:
        return
    current_falling_idx += 1
    manual_mode = False
    manual_meas_offset = 0.0
    update_plot(meas_falling[current_falling_idx], "falling", current_falling_idx + 1)
b_next_f.on_clicked(next_f)

ax_prev_r = plt.axes([0.590, 0.39, 0.13, 0.055])
b_prev_r = Button(ax_prev_r, '← Prev Rising')
def prev_r(event):
    global current_rising_idx, manual_mode, manual_meas_offset
    if meas_df is None or current_rising_idx <= 0:
        return
    current_rising_idx -= 1
    manual_mode = False
    manual_meas_offset = 0.0
    update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
b_prev_r.on_clicked(prev_r)

ax_next_r = plt.axes([0.724, 0.39, 0.13, 0.055])
b_next_r = Button(ax_next_r, 'Next Rising →')
def next_r(event):
    global current_rising_idx, manual_mode, manual_meas_offset
    if meas_df is None or current_rising_idx >= len(meas_rising) - 1:
        return
    current_rising_idx += 1
    manual_mode = False
    manual_meas_offset = 0.0
    update_plot(meas_rising[current_rising_idx], "rising", current_rising_idx + 1)
b_next_r.on_clicked(next_r)

# Guidetexts
print("\n" + "="*60)
print("INSTRUCTIONS:")
print("="*60)
print("FILE LOADING:")
print("   Load Measurement - Load measurement file")
print("   Load Simulation  - Load simulation file")
print("\nKEYS:")
print("  ← →  : Move measurement by 10 ns at a time")
print("  Shift + ← →  : Move measurement by 1 ns at a time")
print("  R    : Reset manual offset")
print("  S    : Print current offset to console")
print("  P    : Previous simulation curve")
print("  N    : Next simulation curve")
print("="*60)
print(f"NOISE FILTERING: {NOISE_FILTER_METHOD}")
print(f"  - Window: {NOISE_FILTER_WINDOW} points")
if NOISE_FILTER_METHOD == 'savgol':
    print(f"  - Polynomial order: {NOISE_FILTER_POLYORDER}")
print("="*60 + "\n")

plt.show()