

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Function to find first rising edge time
def find_first_rising_edge(time, voltage, threshold):
    if np.any(np.isnan(voltage)):
        print("Warning: NaN values in voltage data.")
    rising_indices = np.where((voltage[:-1] < threshold) & (voltage[1:] >= threshold))[0]
    if len(rising_indices) > 0:
        return time[rising_indices[0]]
    return 0  # Default if no edge found

# Load simulated CSV (REGRESSION... assumed simulated)
sim_file = 'REGRESSION_ AREC_X21_MADE2_I2C_1-I2C_0_P1V8(1) SDA-INC74024-2593-55585.csv'
sim_df = pd.read_csv(sim_file, header=None, delimiter=',', low_memory=False)
sim_time = sim_df.iloc[:, 0].values
sim_voltage = sim_df.iloc[:, 1].values

# Load measured CSV (I2C_all.csv assumed measured)
meas_file = 'I2C_all.csv'
meas_df = pd.read_csv(meas_file, skiprows=2, delimiter=',', low_memory=False)
meas_df = meas_df[meas_df['Time'] != '.DATA']  # Remove non-data row
meas_time = pd.to_numeric(meas_df['Time'], errors='coerce')
meas_voltage = pd.to_numeric(meas_df['V V [INC74024.1 (at pin)]'], errors='coerce')

# Handle any NaNs by interpolation or drop
meas_voltage = meas_voltage.interpolate().fillna(method='bfill').fillna(method='ffill')
meas_time = meas_time.interpolate().fillna(method='bfill').fillna(method='ffill')

# Print stats
print(f'Simulated: Time {sim_time.min()} to {sim_time.max()}, Voltage {np.nanmin(sim_voltage)} to {np.nanmax(sim_voltage)}')
print(f'Measured: Time {meas_time.min()} to {meas_time.max()}, Voltage {np.nanmin(meas_voltage)} to {np.nanmax(meas_voltage)}')

# Threshold for edges (midpoint of logic level ~0.9V for 1.8V)
threshold = 0.9

# Try cross-correlation for offset
sim_norm = (sim_voltage - np.nanmean(sim_voltage)) / (np.nanstd(sim_voltage) + 1e-8)
meas_norm = (meas_voltage - np.nanmean(meas_voltage)) / (np.nanstd(meas_voltage) + 1e-8)
corr = correlate(sim_norm, meas_norm, mode='full')
lag = np.argmax(corr) - (len(meas_norm) - 1)
dt_meas = np.nanmean(np.diff(meas_time))
offset = lag * dt_meas
print(f'Correlation offset: {offset}')

# If correlation fails (nan), fallback to edge alignment
if np.isnan(offset):
    print('Correlation failed, using edge alignment.')
    sim_first_rise = find_first_rising_edge(sim_time, sim_voltage, threshold)
    meas_first_rise = find_first_rising_edge(meas_time, meas_voltage, threshold)
    offset = sim_first_rise - meas_first_rise
    print(f'Edge offset: {offset}')

# Shift measured time
meas_time_shifted = meas_time + offset

# Min-max scaling for comparison
def min_max_scale(v):
    min_v, max_v = np.nanmin(v), np.nanmax(v)
    return (v - min_v) / (max_v - min_v + 1e-8)

sim_scaled = min_max_scale(sim_voltage)
meas_scaled = min_max_scale(meas_voltage)

# Plot aligned waveforms
plt.figure(figsize=(12, 6))
plt.plot(sim_time, sim_scaled, label='Simulated', color='blue')
plt.plot(meas_time_shifted, meas_scaled, label='Measured (aligned)', color='red', linestyle='--')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Voltage')
plt.title('Simulated vs Measured Waveforms (Rising Edges Aligned)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optional: Save the plot
# plt.savefig('aligned_waveforms.png')