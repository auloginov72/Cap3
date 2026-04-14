"""
dp_generate_prompt.py
---------------------
Runs the DP simulation and writes a structured prompt
to  'dp_fault_prompt.txt'  ready to be fed to the model.
"""

import numpy as np
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────
# SIMULATION  (same parameters as dp_simulate.py)
# ─────────────────────────────────────────────
dt         = 0.1
T          = 5 * 60
t          = np.arange(0, T, dt)
N          = len(t)

fault_time  = 4 * 60
step_size   = 2.0
noise_std   = 0.05
alarm_limit = 1.5

Kp_pos     = 0.1
Kp_speed   = 0.3
Kp_reg     = 10.0
Ki_reg     = 1.0
tau_vessel = 10.0

X_true       = np.zeros(N)
vessel_speed = np.zeros(N)
reg_integr   = np.zeros(N)
thruster_pct = np.zeros(N)
noise        = np.random.normal(0, noise_std, N)
step_dist    = np.zeros(N)
step_dist[t >= fault_time] = step_size

for i in range(1, N):
    X_meas_i      = X_true[i-1] + noise[i-1] + step_dist[i-1]
    pos_error     = 0.0 - X_meas_i
    speed_des     = Kp_pos * pos_error
    speed_error   = speed_des - vessel_speed[i-1]
    reg_integr[i] = reg_integr[i-1] + Ki_reg * speed_error * dt
    reg_output    = Kp_reg * speed_error + reg_integr[i]
    thrust_force  = Kp_speed * reg_output
    vessel_speed[i] = vessel_speed[i-1] + (dt / tau_vessel) * (thrust_force - vessel_speed[i-1])
    X_true[i]       = X_true[i-1] + vessel_speed[i] * dt
    thruster_pct[i] = np.clip(np.abs(thrust_force) * 10, 0, 100)

X_meas        = X_true + noise + step_dist
X_des         = np.zeros(N)
pos_error_sig = X_meas - X_des

# offline zero-phase filter
nyq    = 0.5 / dt
b, a   = butter(4, 0.05 / nyq, btype='low')
X_filt = filtfilt(b, a, X_meas)

alarm_active = np.abs(pos_error_sig) > alarm_limit
alarm_times  = t[alarm_active]
alarm_first  = alarm_times[0] if len(alarm_times) > 0 else None

# ─────────────────────────────────────────────
# BUILD DATA TABLE  (every 5 seconds, window around fault)
# ─────────────────────────────────────────────
window_start = max(0,  fault_time - 60)
window_end   = min(T,  fault_time + 120)
mask         = (t >= window_start) & (t <= window_end)
indices      = np.where(mask)[0][::50]   # every 5 seconds

rows = []
for i in indices:
    note = ""
    if alarm_first is not None and abs(t[i] - alarm_first) < 0.6:
        note = "<<< POSITION LIMIT ALARM TRIGGERED"
    elif alarm_active[i]:
        note = "[alarm active]"
    rows.append((t[i], X_meas[i], X_des[i], X_filt[i], thruster_pct[i], note))

# ─────────────────────────────────────────────
# WRITE PROMPT FILE
# ─────────────────────────────────────────────
output_file = "dp_fault_prompt.txt"

with open(output_file, "w") as f:

    f.write("=" * 65 + "\n")
    f.write("  SYSTEM ROLE\n")
    f.write("=" * 65 + "\n")
    f.write(
        "You are an expert DP (Dynamic Positioning) system fault analyst.\n"
        "You will be given time series data from a DP vessel that experienced\n"
        "a position fault. Analyze the data carefully and explain:\n"
        "  1. What happened step by step\n"
        "  2. What was the root cause of the alarm\n"
        "  3. What the control system did in response\n"
        "  4. What could be checked or done to prevent recurrence\n"
        "Be concise and technical. Use engineering language.\n"
    )

    f.write("\n" + "=" * 65 + "\n")
    f.write("  SYSTEM CONFIGURATION\n")
    f.write("=" * 65 + "\n")
    f.write(
        f"  Control architecture : Cascade (position outer / speed inner loop)\n"
        f"  Position controller  : Proportional, Kp = {Kp_pos}\n"
        f"  Speed controller     : PI regulator  (10s+1)/s\n"
        f"  Vessel dynamics      : First order   1/(10s+1), tau = {tau_vessel}s\n"
        f"  Reference sensors    : 1 (single sensor, X-axis only)\n"
        f"  Desired position     : 0.00 m (constant setpoint)\n"
        f"  Position alarm limit : +/- {alarm_limit} m\n"
        f"  Measurement noise    : std = {noise_std} m\n"
    )

    f.write("\n" + "=" * 65 + "\n")
    f.write("  FAULT EVENT SUMMARY\n")
    f.write("=" * 65 + "\n")
    if alarm_first:
        f.write(
            f"  Alarm time    : t = {alarm_first:.1f}s  ({alarm_first/60:.2f} min)\n"
            f"  Alarm type    : POSITION LIMIT exceeded +/- {alarm_limit}m\n"
        )
    f.write(
        f"  Max position error  : {np.abs(pos_error_sig).max():.3f} m\n"
        f"  Max thruster load   : {thruster_pct.max():.1f} %\n"
    )

    f.write("\n" + "=" * 65 + "\n")
    f.write("  SIGNAL DESCRIPTIONS\n")
    f.write("=" * 65 + "\n")
    f.write(
        "  X_meas  : Measured position [m]  — raw sensor output\n"
        "  X_des   : Desired position  [m]  — constant setpoint\n"
        "  X_filt  : Offline zero-phase filtered X_meas [m] — smoothed position estimate\n"
        "  Thrust  : Thruster load     [%]  — 0=idle, 100=saturated\n"
    )

    f.write("\n" + "=" * 65 + "\n")
    f.write("  TIME SERIES DATA  (5-second intervals around fault)\n")
    f.write("=" * 65 + "\n")
    f.write(
        f"  {'Time(s)':<10} {'X_meas(m)':<12} {'X_des(m)':<11} "
        f"{'X_filt(m)':<12} {'Thrust(%)':<12} Note\n"
    )
    f.write("  " + "-" * 62 + "\n")
    for (ts, xm, xd, xf, th, note) in rows:
        f.write(
            f"  {ts:<10.1f} {xm:<12.3f} {xd:<11.3f} "
            f"{xf:<12.3f} {th:<12.1f} {note}\n"
        )

    f.write("\n" + "=" * 65 + "\n")
    f.write("  QUESTION\n")
    f.write("=" * 65 + "\n")
    f.write(
        "Based on the time series data and system configuration above,\n"
        "explain step by step what happened and why the position limit\n"
        "alarm was triggered. What does the thruster behaviour tell you?\n"
        "What is the difference between X_meas and X_filt telling you?\n"
    )

print(f"Prompt written to:  {output_file}")
print(f"Lines in file    :  {len(open(output_file).readlines())}")
print(f"\nPreview of data table:")
print(f"  {'Time(s)':<10} {'X_meas':<12} {'X_des':<11} {'X_filt':<12} {'Thrust%':<12} Note")
print(f"  " + "-"*62)
for (ts, xm, xd, xf, th, note) in rows:
    print(f"  {ts:<10.1f} {xm:<12.3f} {xd:<11.3f} {xf:<12.3f} {th:<12.1f} {note}")
