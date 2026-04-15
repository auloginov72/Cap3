"""
dp_compare_models.py
--------------------
Compares raw Phi-3 Mini vs LoRA fine-tuned Phi-3 Mini on a DP fault prompt.

TWO MODES — select by command line argument:

  Mode 1 — prompt check (fast, no model needed):
    python dp_compare_models.py --mode 1
    Generates a fault scenario, builds the prompt, saves it to
    dp_test_prompt.txt and prints it. Use this to verify the prompt
    looks correct before committing to the full model run.

  Mode 2 — full comparison (requires model weights + adapter):
    python dp_compare_models.py --mode 2
    Generates a NEW scenario (different from training data, seed=99),
    queries both the raw base model and the fine-tuned model,
    prints both responses side by side and saves to dp_comparison.txt.

Requirements:
  - Mode 1: numpy, scipy only
  - Mode 2: torch, transformers, peft + model at BASE_MODEL + adapter at ADAPTER_DIR
"""

import argparse
import datetime
import os
import sys
import numpy as np
from scipy.signal import butter, filtfilt 

# ─────────────────────────────────────────────
# CONFIGURATION — adjust paths if needed
# ─────────────────────────────────────────────
BASE_MODEL   = r"D:\huggingface_cache\phi3"
ADAPTER_DIR  = "./dp_phi3_adapter"
OUTPUT_FILE  = "dp_comparison.txt"
PROMPT_FILE  = "dp_test_prompt.txt"
MAX_NEW_TOKENS = 512

# Test scenario — deliberately different from all 10 training examples
# Training used seeds 42-51, step sizes 0.8-4.0, fault times 120-270s
# This uses seed=99, step=2.8m, fault at 3min — unseen combination
TEST_SCENARIO = dict(
    step_size   = 2.8,
    fault_time  = 180,
    noise_std   = 0.04,
    alarm_limit = 1.5,
    seed        = 99,
)

# ─────────────────────────────────────────────
# SIMULATION  (copied from dp_make_training_data.py)
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
        X_meas_i        = X_true[i-1] + noise[i-1] + step_dist[i-1]
        pos_error       = 0.0 - X_meas_i
        speed_des       = Kp_pos * pos_error
        speed_error     = speed_des - vessel_speed[i-1]
        reg_integr[i]   = reg_integr[i-1] + Ki_reg * speed_error * dt
        reg_output      = Kp_reg * speed_error + reg_integr[i]
        thrust_force    = Kp_speed * reg_output
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


def build_data_table(t, X_meas, X_des, X_filt, thruster_pct,
                     alarm_active, alarm_first, fault_time, alarm_limit):
    window_start = max(0,      fault_time - 60)
    window_end   = min(5 * 60, fault_time + 120)
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


def build_prompt(t, X_meas, X_des, X_filt, thruster_pct,
                 alarm_active, alarm_first, alarm_limit,
                 fault_time, noise_std):
    table     = build_data_table(t, X_meas, X_des, X_filt,
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
What is the difference between X_meas and X_filt telling you?"""


# ─────────────────────────────────────────────
# DIVIDER HELPER
# ─────────────────────────────────────────────
def divider(title=""):
    w = 65
    if title:
        pad = (w - len(title) - 2) // 2
        return f"\n{'='*pad} {title} {'='*(w-pad-len(title)-2)}\n"
    return "\n" + "=" * w + "\n"


# ─────────────────────────────────────────────
# MODE 1 — PROMPT CHECK
# ─────────────────────────────────────────────
def mode_prompt_check():
    print(divider("MODE 1 — PROMPT CHECK"))
    print(f"Scenario : step={TEST_SCENARIO['step_size']}m  "
          f"fault_time={TEST_SCENARIO['fault_time']}s  "
          f"seed={TEST_SCENARIO['seed']}")
    print("NOTE: this scenario was NOT in the training data\n")

    t, X_meas, X_des, X_filt, X_true, thruster_pct, alarm_first, alarm_active = \
        run_simulation(**TEST_SCENARIO)

    prompt = build_prompt(
        t, X_meas, X_des, X_filt, thruster_pct,
        alarm_active, alarm_first,
        TEST_SCENARIO["alarm_limit"],
        TEST_SCENARIO["fault_time"],
        TEST_SCENARIO["noise_std"],
    )

    # Phi-3 instruct wrapper
    full_prompt = f"<|user|>\n{prompt.strip()}<|end|>\n<|assistant|>\n"

    print(full_prompt)
    print(divider())

    char_count  = len(full_prompt)
    token_est   = char_count // 4
    print(f"Prompt length : {char_count} chars  (~{token_est} tokens estimated)")
    if alarm_first:
        print(f"Alarm fires   : t={alarm_first:.1f}s ({alarm_first/60:.2f} min)")
    else:
        print("Alarm fires   : not triggered")

    with open(PROMPT_FILE, "w", encoding="utf-8") as f:
        f.write(full_prompt)
    print(f"\nPrompt saved to: {os.path.abspath(PROMPT_FILE)}")
    print("If prompt looks correct, run:  python dp_compare_models.py --mode 2")


# ─────────────────────────────────────────────
# MODE 2 — FULL COMPARISON
# ─────────────────────────────────────────────
def mode_full_comparison():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(divider("MODE 2 — FULL MODEL COMPARISON"))

    # ── check paths ──────────────────────────
    if not os.path.isdir(BASE_MODEL):
        print(f"ERROR: Base model not found at {BASE_MODEL}")
        sys.exit(1)
    if not os.path.isdir(ADAPTER_DIR):
        print(f"ERROR: LoRA adapter not found at {ADAPTER_DIR}")
        print("Run dp_finetune.py first to generate the adapter.")
        sys.exit(1)

    # ── device ───────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    if device == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
    dtype = torch.float16 if device == "cuda" else torch.float32

    # ── generate prompt ───────────────────────
    print("\nGenerating test scenario...")
    print(f"  step={TEST_SCENARIO['step_size']}m  "
          f"fault_time={TEST_SCENARIO['fault_time']}s  "
          f"seed={TEST_SCENARIO['seed']}  "
          f"(NOT in training data)")

    t, X_meas, X_des, X_filt, X_true, thruster_pct, alarm_first, alarm_active = \
        run_simulation(**TEST_SCENARIO)

    prompt = build_prompt(
        t, X_meas, X_des, X_filt, thruster_pct,
        alarm_active, alarm_first,
        TEST_SCENARIO["alarm_limit"],
        TEST_SCENARIO["fault_time"],
        TEST_SCENARIO["noise_std"],
    )
    full_prompt = f"<|user|>\n{prompt.strip()}<|end|>\n<|assistant|>\n"

    # ── load tokenizer ────────────────────────
    print(f"\nLoading tokenizer from {BASE_MODEL} ...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    print(f"Prompt tokenized : {input_len} tokens")

    # ─────────────────────────────────────────
    # QUERY 1 — RAW BASE MODEL
    # ─────────────────────────────────────────
    print(divider("LOADING RAW BASE MODEL"))
    print(f"Path: {BASE_MODEL}")

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype             = dtype,
        trust_remote_code = True,
        device_map        = device,
        low_cpu_mem_usage = True,
        attn_implementation = "eager",
    )
    base_model.config.use_cache = True
    base_model.eval()

    print("Generating raw model response...")
    t0 = datetime.datetime.now()
    with torch.no_grad():
        raw_output = base_model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = False,
            temperature    = 1.0,
            pad_token_id   = tokenizer.eos_token_id,
        )
    t1 = datetime.datetime.now()

    raw_response = tokenizer.decode(
        raw_output[0][input_len:], skip_special_tokens=True
    ).strip()
    raw_time = (t1 - t0).seconds

    print(f"Raw model response generated in {raw_time}s")

    # free GPU memory before loading fine-tuned
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────
    # QUERY 2 — FINE-TUNED MODEL (base + adapter)
    # ─────────────────────────────────────────
    print(divider("LOADING FINE-TUNED MODEL"))
    print(f"Base  : {BASE_MODEL}")
    print(f"Adapter: {ADAPTER_DIR}")

    ft_base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype             = dtype,
        trust_remote_code = True,
        device_map        = device,
        low_cpu_mem_usage = True,
        attn_implementation = "eager",
    )
    ft_model = PeftModel.from_pretrained(ft_base, ADAPTER_DIR)
    ft_model.config.use_cache = True
    ft_model.eval()

    print("Generating fine-tuned model response...")
    t0 = datetime.datetime.now()
    with torch.no_grad():
        ft_output = ft_model.generate(
            **inputs,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = False,
            temperature    = 1.0,
            pad_token_id   = tokenizer.eos_token_id,
        )
    t1 = datetime.datetime.now()

    ft_response = tokenizer.decode(
        ft_output[0][input_len:], skip_special_tokens=True
    ).strip()
    ft_time = (t1 - t0).seconds

    print(f"Fine-tuned model response generated in {ft_time}s")

    # ─────────────────────────────────────────
    # PRINT AND SAVE COMPARISON
    # ─────────────────────────────────────────
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    output = []
    output.append(divider("DP FAULT ANALYSIS — MODEL COMPARISON"))
    output.append(f"Generated : {ts}")
    output.append(f"Scenario  : step={TEST_SCENARIO['step_size']}m  "
                  f"fault_time={TEST_SCENARIO['fault_time']}s  "
                  f"alarm_limit={TEST_SCENARIO['alarm_limit']}m  "
                  f"seed={TEST_SCENARIO['seed']}")
    output.append(f"Note      : this scenario was NOT in the training data\n")

    output.append(divider("PROMPT SENT TO BOTH MODELS"))
    output.append(full_prompt)

    output.append(divider(f"RESPONSE — RAW BASE MODEL  ({raw_time}s)"))
    output.append(raw_response)

    output.append(divider(f"RESPONSE — FINE-TUNED MODEL  ({ft_time}s)"))
    output.append(ft_response)

    output.append(divider("WHAT TO LOOK FOR"))
    output.append(
        "Fine-tuned model should show vs raw model:\n"
        "  ✓ Consistent 4-section structure (SIGNAL BEHAVIOUR / ROOT CAUSE /\n"
        "    CONTROL SYSTEM RESPONSE / RECOMMENDED CHECKS)\n"
        "  ✓ Explicitly compares X_meas vs X_filt divergence\n"
        "  ✓ Identifies fault as measurement fault, not vessel motion\n"
        "  ✓ References specific signal values and timestamps\n"
        "  ✓ DP engineering terminology (DGPS, sensor spike, cascade controller)\n"
    )

    full_output = "\n".join(output)
    print(full_output)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_output)
    print(f"\nFull comparison saved to: {os.path.abspath(OUTPUT_FILE)}")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare raw vs fine-tuned Phi-3 on a DP fault scenario"
    )
    parser.add_argument(
        "--mode", type=int, choices=[1, 2], default=1,
        help="1 = prompt check only (fast)  |  2 = full model comparison"
    )
    args = parser.parse_args()

    if args.mode == 1:
        mode_prompt_check()
    else:
        mode_full_comparison()
