import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────
dt          = 0.1           # time step [seconds] - smaller for accuracy
T           = 5 * 60        # total time [seconds] = 5 minutes
t           = np.arange(0, T, dt)
N           = len(t)

fault_time  = 4 * 60        # step injected at 4 minutes [seconds]
step_size   = 2.0           # position step size [meters]
noise_std   = 0.05          # measurement noise std [meters]
alarm_limit = 1.5           # position limit alarm threshold [meters]

# ─────────────────────────────────────────────
# CONTROLLER GAINS (from Simulink diagram)
# ─────────────────────────────────────────────
Kp_pos   = 0.1              # GainPos
Kp_speed = 0.3              # GainSpeed

# PI speed regulator: (10s+1)/s  -> Kp=10, Ki=1
Kp_reg   = 10.0
Ki_reg   = 1.0

# Vessel dynamic: 1/(10s+1)  -> time constant 10s
tau_vessel = 10.0

# ─────────────────────────────────────────────
# STATE VARIABLES
# ─────────────────────────────────────────────
X_true      = np.zeros(N)
vessel_speed= np.zeros(N)
reg_integr  = np.zeros(N)
thruster_pct= np.zeros(N)

X_des_val   = 0.0
noise       = np.random.normal(0, noise_std, N)

step_dist   = np.zeros(N)
step_dist[t >= fault_time] = step_size

# ─────────────────────────────────────────────
# SIMULATION LOOP
# ─────────────────────────────────────────────
for i in range(1, N):
    X_meas_i    = X_true[i-1] + noise[i-1] + step_dist[i-1]
    pos_error   = X_des_val - X_meas_i
    speed_des   = Kp_pos * pos_error
    speed_error = speed_des - vessel_speed[i-1]

    reg_integr[i] = reg_integr[i-1] + Ki_reg * speed_error * dt
    reg_output    = Kp_reg * speed_error + reg_integr[i]
    thrust_force  = Kp_speed * reg_output

    vessel_speed[i] = vessel_speed[i-1] + (dt / tau_vessel) * (thrust_force - vessel_speed[i-1])
    X_true[i]       = X_true[i-1] + vessel_speed[i] * dt
    thruster_pct[i] = np.clip(np.abs(thrust_force) * 10, 0, 100)

# ─────────────────────────────────────────────
# RECONSTRUCT SIGNALS
# ─────────────────────────────────────────────
X_meas        = X_true + noise + step_dist
X_des         = np.full(N, X_des_val)
pos_error_sig = X_meas - X_des

alarm_active  = np.abs(pos_error_sig) > alarm_limit
alarm_times   = t[alarm_active]
alarm_first   = alarm_times[0] if len(alarm_times) > 0 else None

# ─────────────────────────────────────────────
# PRINT SUMMARY
# ─────────────────────────────────────────────
print("=" * 60)
print("  DP SIMULATION SUMMARY  (Simulink-matched)")
print("=" * 60)
print(f"  Desired position    : {X_des_val:.2f} m (constant)")
print(f"  Step disturbance    : {step_size} m at t={fault_time:.0f}s ({fault_time/60:.0f} min)")
print(f"  Noise std dev       : {noise_std} m")
print(f"  GainPos             : {Kp_pos}  |  GainSpeed : {Kp_speed}")
print(f"  PI regulator        : Kp={Kp_reg}, Ki={Ki_reg}  [(10s+1)/s]")
print(f"  Vessel time const   : {tau_vessel} s  [1/(10s+1)]")
print(f"  Alarm threshold     : +/-{alarm_limit} m")
if alarm_first:
    print(f"  Alarm triggered at  : t={alarm_first:.1f}s  ({alarm_first/60:.2f} min)")
    print(f"  Delay after fault   : {alarm_first - fault_time:.1f}s")
else:
    print("  Alarm               : NOT triggered")
print(f"  Max position error  : {np.abs(pos_error_sig).max():.3f} m")
print("=" * 60)

# data table around fault
window_start = max(0, fault_time - 30)
window_end   = min(T, fault_time + 90)
mask         = (t >= window_start) & (t <= window_end)
indices      = np.where(mask)[0][::50]

print("\n  DATA WINDOW AROUND FAULT EVENT:")
print(f"  {'Time(s)':<10} {'Xmeas(m)':<12} {'Xdes(m)':<10} {'Xtrue(m)':<12} {'Thrust(%)':<12} {'Note'}")
print("  " + "-" * 70)
for i in indices:
    note = ""
    if abs(t[i] - fault_time) < dt * 2:
        note = "<-- STEP INJECTED"
    elif alarm_active[i] and alarm_first is not None and abs(t[i] - alarm_first) < 0.6:
        note = "<-- ALARM TRIGGERED"
    elif alarm_active[i]:
        note = "[ALARM ACTIVE]"
    print(f"  {t[i]:<10.1f} {X_meas[i]:<12.3f} {X_des[i]:<10.3f} {X_true[i]:<12.3f} {thruster_pct[i]:<12.1f} {note}")

# ─────────────────────────────────────────────
# OFFLINE ZERO-PHASE FILTER  (filtfilt = no delay)
# This simulates what you can do in post-analysis
# using only X_meas - no X_true needed
# ─────────────────────────────────────────────
from scipy.signal import butter, filtfilt

def butter_lowpass_filtfilt(data, cutoff_hz, fs_hz, order=4):
    """Zero-phase Butterworth low-pass filter."""
    nyq    = 0.5 * fs_hz
    normal_cutoff = cutoff_hz / nyq
    b, a   = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

fs_hz      = 1.0 / dt          # sampling frequency [Hz] = 10 Hz
cutoff_hz  = 0.05              # cut-off: pass below 0.05 Hz, reject noise above
X_filt     = butter_lowpass_filtfilt(X_meas, cutoff_hz, fs_hz)

print("\n  FILTER INFO:")
print(f"  Type        : Zero-phase Butterworth low-pass (filtfilt)")
print(f"  Order       : 4")
print(f"  Cut-off     : {cutoff_hz} Hz  (time constant ~{1/cutoff_hz:.0f}s)")
print(f"  Sampling    : {fs_hz} Hz")
print(f"  Max diff vs X_true : {np.abs(X_filt - X_true).max():.4f} m")
print(f"  Max diff vs X_meas : {np.abs(X_filt - X_meas).max():.4f} m")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(13, 10), facecolor="#0f1117")
fig.suptitle("DP System Simulation  —  Simulink-Matched Control Loop",
             color="white", fontsize=13, fontweight="bold", y=0.98)

gs  = gridspec.GridSpec(3, 1, hspace=0.48)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

for ax in [ax1, ax2, ax3]:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(True, color="#252836", linewidth=0.7)
    ax.axvline(fault_time, color="#ff5252", linewidth=1.6,
               linestyle="--", alpha=0.9, label=f"Step @ t={fault_time:.0f}s")
    if alarm_first:
        ax.axvline(alarm_first, color="#ffd740", linewidth=1.6,
                   linestyle="--", alpha=0.9, label=f"Alarm @ t={alarm_first:.1f}s")

# Plot 1: Positions
ax1.plot(t, X_meas, color="#4fc3f7", linewidth=0.9, alpha=0.7, label="X_meas  (noisy + step)")
ax1.plot(t, X_true, color="#69f0ae", linewidth=2.0,             label="X_true  (real position)")
ax1.plot(t, X_filt, color="#ff9800", linewidth=1.8, linestyle="-", label="X_filt  (offline zero-phase)")
ax1.plot(t, X_des,  color="#ffffff", linewidth=1.5, linestyle=":", label="X_des   (setpoint)")
ax1.axhline( alarm_limit, color="#ffd740", linewidth=1.0, linestyle="-.", alpha=0.5, label=f"+/-{alarm_limit}m alarm zone")
ax1.axhline(-alarm_limit, color="#ffd740", linewidth=1.0, linestyle="-.", alpha=0.5)
ax1.set_ylabel("Position [m]")
ax1.set_title("Position Signals")
ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1d27", labelcolor="white", framealpha=0.85)

# Plot 2: Position Error
ax2.fill_between(t, pos_error_sig, 0,
                 where=(np.abs(pos_error_sig) > alarm_limit),
                 color="#ff5252", alpha=0.25, label="Outside alarm limit")
ax2.plot(t, pos_error_sig, color="#ef9a9a", linewidth=1.4, label="Position error")
ax2.axhline( alarm_limit, color="#ffd740", linewidth=1.0, linestyle="-.", alpha=0.7)
ax2.axhline(-alarm_limit, color="#ffd740", linewidth=1.0, linestyle="-.", alpha=0.7)
ax2.set_ylabel("Error [m]")
ax2.set_title("Position Error  (X_meas - X_des)")
ax2.legend(loc="upper left", fontsize=8, facecolor="#1a1d27", labelcolor="white", framealpha=0.85)

# Plot 3: Thruster + Vessel Speed
ax3b = ax3.twinx()
ax3b.set_facecolor("#1a1d27")
ax3b.tick_params(colors="#ce93d8")
ax3b.yaxis.label.set_color("#ce93d8")
ax3.fill_between(t, thruster_pct, alpha=0.2, color="#80cbc4")
ax3.plot(t, thruster_pct,  color="#80cbc4", linewidth=1.4, label="Thruster load [%]")
ax3b.plot(t, vessel_speed, color="#ce93d8", linewidth=1.4, label="Vessel speed [m/s]")
ax3.set_ylim(0, 110)
ax3.set_ylabel("Thruster load [%]")
ax3b.set_ylabel("Vessel speed [m/s]")
ax3.set_xlabel("Time [seconds]")
ax3.set_title("Thruster Load  &  Vessel Speed")
lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3b.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
           fontsize=8, facecolor="#1a1d27", labelcolor="white", framealpha=0.85)

plt.savefig("dp_simulation.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.show()
print("\n  Plot saved as dp_simulation.png")