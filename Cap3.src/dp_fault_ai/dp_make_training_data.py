"""
dp_make_training_data.py
------------------------
Generates 10 training examples for LoRA fine-tuning.
Each example = prompt (data + question) + correct expert answer.
Output: dp_training_data.json
"""

import json
import numpy as np
from scipy.signal import butter, filtfilt

# ─────────────────────────────────────────────
# HELPER — run simulation with given parameters
# ─────────────────────────────────────────────
def run_simulation(step_size, fault_time, noise_std=0.05,
                   alarm_limit=1.5, seed=42):
    np.random.seed(seed)
    dt = 0.1
    T  = 5 * 60
    t  = np.arange(0, T, dt)
    N  = len(t)

    Kp_pos = 0.1;  Kp_speed = 0.3
    Kp_reg = 10.0; Ki_reg   = 1.0
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
        vessel_speed[i] = vessel_speed[i-1] + (dt/tau_vessel)*(thrust_force - vessel_speed[i-1])
        X_true[i]       = X_true[i-1] + vessel_speed[i] * dt
        thruster_pct[i] = np.clip(np.abs(thrust_force) * 10, 0, 100)

    X_meas = X_true + noise + step_dist
    X_des  = np.zeros(N)

    nyq    = 0.5 / dt
    b, a   = butter(4, 0.05 / nyq, btype='low')
    X_filt = filtfilt(b, a, X_meas)

    pos_error_sig = X_meas - X_des
    alarm_active  = np.abs(pos_error_sig) > alarm_limit
    alarm_times   = t[alarm_active]
    alarm_first   = alarm_times[0] if len(alarm_times) > 0 else None

    return t, X_meas, X_des, X_filt, X_true, thruster_pct, alarm_first, alarm_active

# ─────────────────────────────────────────────
# HELPER — build data table string
# ─────────────────────────────────────────────
def build_data_table(t, X_meas, X_des, X_filt, thruster_pct,
                     alarm_active, alarm_first, fault_time, alarm_limit):
    window_start = max(0,       fault_time - 60)
    window_end   = min(5 * 60,  fault_time + 120)
    mask         = (t >= window_start) & (t <= window_end)
    indices      = np.where(mask)[0][::50]

    header = (
        f"  {'Time(s)':<10} {'X_meas(m)':<12} {'X_des(m)':<11} "
        f"{'X_filt(m)':<12} {'Thrust(%)':<12} Note\n"
        f"  {'-'*62}\n"
    )
    rows = ""
    for i in indices:
        note = ""
        if alarm_first is not None and abs(t[i] - alarm_first) < 0.6:
            note = "<<< POSITION LIMIT ALARM TRIGGERED"
        elif alarm_active[i]:
            note = "[alarm active]"
        rows += (
            f"  {t[i]:<10.1f} {X_meas[i]:<12.3f} {X_des[i]:<11.3f} "
            f"{X_filt[i]:<12.3f} {thruster_pct[i]:<12.1f} {note}\n"
        )
    return header + rows

# ─────────────────────────────────────────────
# HELPER — build full prompt (no answer hints)
# ─────────────────────────────────────────────
def build_prompt(t, X_meas, X_des, X_filt, thruster_pct,
                 alarm_active, alarm_first, alarm_limit,
                 fault_time, noise_std):
    table = build_data_table(t, X_meas, X_des, X_filt,
                             thruster_pct, alarm_active,
                             alarm_first, fault_time, alarm_limit)
    alarm_str = (f"t = {alarm_first:.1f}s ({alarm_first/60:.2f} min)"
                 if alarm_first else "not triggered")

    return f"""=================================================================
  SYSTEM ROLE
=================================================================
You are an expert DP (Dynamic Positioning) system fault analyst.
Analyze the time series data and explain:
  1. What happened step by step
  2. What was the root cause of the alarm
  3. What the control system did in response
  4. What should be checked to prevent recurrence
Be concise and technical. Use engineering language.

=================================================================
  SYSTEM CONFIGURATION
=================================================================
  Control architecture : Cascade (position outer / speed inner loop)
  Position controller  : Proportional, Kp = 0.1
  Speed controller     : PI regulator  (10s+1)/s
  Vessel dynamics      : First order   1/(10s+1), tau = 10s
  Reference sensors    : 1 (single sensor, X-axis only)
  Desired position     : 0.00 m (constant setpoint)
  Position alarm limit : +/- {alarm_limit} m
  Measurement noise    : std = {noise_std} m

=================================================================
  FAULT EVENT SUMMARY
=================================================================
  Alarm time    : {alarm_str}
  Alarm type    : POSITION LIMIT exceeded +/- {alarm_limit}m
  Max pos error : {np.abs(X_meas - X_des).max():.3f} m
  Max thruster  : {thruster_pct.max():.1f} %

=================================================================
  SIGNAL DESCRIPTIONS
=================================================================
  X_meas  : Measured position [m]  — raw sensor output
  X_des   : Desired position  [m]  — constant setpoint
  X_filt  : Offline zero-phase filtered X_meas [m] — smoothed estimate
  Thrust  : Thruster load     [%]  — 0=idle, 100=saturated

=================================================================
  TIME SERIES DATA  (5-second intervals around alarm)
=================================================================
{table}
=================================================================
  QUESTION
=================================================================
Based on the time series data and system configuration above,
explain step by step what happened and why the position limit
alarm was triggered. What does the thruster behaviour tell you?
What is the difference between X_meas and X_filt telling you?
"""

# ─────────────────────────────────────────────
# HELPER — build correct expert answer
# ─────────────────────────────────────────────
def build_answer(step_size, fault_time, alarm_first,
                 max_error, max_thrust, alarm_limit):
    delay    = alarm_first - fault_time if alarm_first else 0
    severity = (
        "minor"    if step_size < 1.0 else
        "moderate" if step_size < 2.5 else
        "severe"
    )
    thrust_desc = (
        "remained moderate"   if max_thrust < 40 else
        "increased sharply"   if max_thrust < 70 else
        "reached high levels"
    )

    return f"""Step-by-step fault analysis:

1. SIGNAL BEHAVIOUR:
   X_meas shows a sudden {severity} step of approximately {step_size:.1f}m
   at t={fault_time:.0f}s. X_filt (zero-phase filtered signal) remains
   smooth and close to zero, confirming that no real vessel motion
   occurred. The divergence between X_meas and X_filt is the key
   diagnostic indicator of a measurement fault.

2. ROOT CAUSE:
   The position measurement signal experienced an abrupt step change
   of {step_size:.1f}m. This is characteristic of a reference sensor fault
   such as: sensor signal spike, DGPS jump, loss and re-acquisition
   of signal, or electrical interference. The vessel's true position
   did not change — only the measurement was affected.

3. CONTROL SYSTEM RESPONSE:
   The cascade controller interpreted the measurement step as a real
   position error of {max_error:.2f}m. The position controller commanded
   a corrective speed demand, which the PI speed regulator amplified.
   Thruster load {thrust_desc} to up to {max_thrust:.0f}% as the system
   tried to correct a position error that did not physically exist.
   The position limit alarm triggered {delay:.0f}s after the measurement
   step, when the indicated error exceeded +/-{alarm_limit}m.

4. RECOMMENDED CHECKS:
   - Inspect reference sensor logs for signal quality at t={fault_time:.0f}s
   - Check for DGPS outage, multipath, or satellite geometry change
   - Review sensor confidence and integrity flags at fault time
   - If second sensor available, compare — real motion would show
     on both sensors; a measurement fault shows on one only
   - Consider adding sensor voting/redundancy if single sensor only
"""

# ─────────────────────────────────────────────
# DEFINE 10 TRAINING SCENARIOS
# ─────────────────────────────────────────────
scenarios = [
    # (step_size, fault_time, noise_std, alarm_limit, seed, description)
    (2.0,  240, 0.05, 1.5, 42,  "2m step at 4min, standard noise"),
    (0.8,  180, 0.05, 1.5, 43,  "0.8m step at 3min, small step no alarm"),
    (3.5,  240, 0.05, 1.5, 44,  "3.5m step at 4min, severe"),
    (1.6,  150, 0.05, 1.5, 45,  "1.6m step at 2.5min, just over limit"),
    (2.0,  240, 0.10, 1.5, 46,  "2m step at 4min, higher noise"),
    (4.0,  200, 0.05, 1.5, 47,  "4m step at 3.3min, large step"),
    (1.0,  240, 0.03, 1.5, 48,  "1m step at 4min, low noise"),
    (2.5,  120, 0.05, 2.0, 49,  "2.5m step at 2min, wider alarm limit"),
    (3.0,  270, 0.05, 1.5, 50,  "3m step at 4.5min, late fault"),
    (2.0,  240, 0.05, 1.0, 51,  "2m step at 4min, tighter alarm limit"),
]

# ─────────────────────────────────────────────
# GENERATE TRAINING DATA
# ─────────────────────────────────────────────
training_data = []

print("Generating training examples...")
print("-" * 50)

for idx, (step_size, fault_time, noise_std,
          alarm_limit, seed, desc) in enumerate(scenarios):

    t, X_meas, X_des, X_filt, X_true, thruster_pct, alarm_first, alarm_active = \
        run_simulation(step_size, fault_time, noise_std, alarm_limit, seed)

    prompt = build_prompt(
        t, X_meas, X_des, X_filt, thruster_pct,
        alarm_active, alarm_first, alarm_limit,
        fault_time, noise_std
    )

    answer = build_answer(
        step_size, fault_time, alarm_first,
        np.abs(X_meas - X_des).max(),
        thruster_pct.max(),
        alarm_limit
    )

    # Phi-3 instruct format
    training_data.append({
        "text": f"<|user|>\n{prompt.strip()}<|end|>\n<|assistant|>\n{answer.strip()}<|end|>",
        "description": desc,
        "step_size": step_size,
        "alarm_triggered": alarm_first is not None
    })

    alarm_str = f"t={alarm_first:.0f}s" if alarm_first else "no alarm"
    print(f"  Example {idx+1:2d}: {desc}")
    print(f"            step={step_size}m  alarm={alarm_str}")

print("-" * 50)
print(f"Total examples : {len(training_data)}")

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
output_file = "dp_training_data.json"
with open(output_file, "w") as f:
    json.dump(training_data, f, indent=2)

print(f"Saved to       : {output_file}")
#print(f"\nSample prompt length  : {len(training_data[0]['text'].split('[/INST]')[0])} chars")
#print(f"Sample answer length  : {len(training_data[0]['text'].split('[/INST]')[1])} chars")

print(f"Sample prompt length  : {len(training_data[0]['text'].split('<|assistant|>')[0])} chars")
print(f"Sample answer length  : {len(training_data[0]['text'].split('<|assistant|>')[1])} chars")


# preview first example
print("\n" + "="*50)
print("PREVIEW — Example 1 answer:")
print("="*50)
#print(training_data[0]['text'].split('[/INST]')[1][:800])
print(training_data[0]['text'].split('<|assistant|>')[1][:800])
print("...")
