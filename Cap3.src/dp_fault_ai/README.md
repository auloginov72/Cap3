# DP Fault AI — Project Summary

## Overview
Local AI system for **Dynamic Positioning fault analysis**.  
Given time series data from a DP vessel, the model explains step by step
what happened, what caused the alarm, and what should be checked.  
Runs **100% locally** — no internet required after setup.

---

## Architecture

```
DP System (C++ / real data)
        ↓
Signal extraction around fault event
        ↓
Prompt builder  (dp_generate_prompt.py)
        ↓
Local LLM via Ollama  (dp_ask_model.py)
        ↓
Plain-language fault explanation
```

---

## Models

| Model | Usage | Location |
|---|---|---|
| **Mistral 7B** | Baseline testing via Ollama | D:\.ollama |
| **Phi-3 Mini 3.8B** | Fine-tuning target | D:\huggingface_cache\phi3 |
| **TinyLlama 1.1B** | Baseline comparison via Ollama | D:\.ollama |

### Ollama model management
```bash
ollama pull mistral
ollama pull phi3
ollama pull tinyllama
ollama list
```

---

## Simulation

### Control loop (matches Simulink diagram)
```
DezPos (setpoint=0)
    → GainPos (Kp=0.1)       outer position loop
    → GainSpeed (Kp=0.3)     inner speed loop
    → PI regulator (10s+1)/s
    → Vessel dynamics 1/(10s+1), tau=10s
    → Speed integrator 1/s
    → X_true (real position)

Fault injection: step disturbance on X_meas at t=4min
```

### Signals
| Signal | Description |
|---|---|
| `X_des` | Desired position — constant setpoint |
| `X_meas` | Raw measured position — noisy + step disturbance |
| `X_true` | True vessel position (simulation only, not available in reality) |
| `X_filt` | Offline zero-phase filtered X_meas — best post-analysis estimate |
| `Thrust` | Thruster load [%] |

---

## Python Files

### `SimulateFaultData.py`
Simulates 5-minute DP scenario with step disturbance at 4min.  
Generates and plots all signals. Saves `dp_simulation.png`.
```bash
python SimulateFaultData.py
```

### `dp_generate_prompt.py`
Runs simulation and writes structured fault prompt to `dp_fault_prompt.txt`.  
Prompt contains only observable data — no hints about root cause.
```bash
python dp_generate_prompt.py
```

### `dp_ask_model.py`
Reads `dp_fault_prompt.txt` and sends it to Mistral and Phi-3 via Ollama.  
Streams both responses to console. Saves both to `dp_fault_response.txt`.
```bash
python dp_ask_model.py
```

### `test_mistral.py`
Simple single-model test with hardcoded DP fault scenario.  
Used for quick sanity checks.
```bash
python test_mistral.py
```

### `test_compare_models.py`
Sends the same fault prompt to Mistral, Phi-3 and TinyLlama.  
Good for quick baseline comparison before fine-tuning.
```bash
python test_compare_models.py
```

### `dp_make_training_data.py`
Generates 10 simulation scenarios with varying parameters.  
For each scenario builds a blind prompt + correct expert answer.  
Saves to `dp_training_data.json` in Phi-3 instruct format.
```bash
python dp_make_training_data.py
```

### `dp_finetune.py`
LoRA fine-tuning of Phi-3 Mini on `dp_training_data.json`.  
Saves adapter to `./dp_phi3_adapter/`.  
Expected training time: 30–90 minutes on CPU.

**First run** — model not yet cached, must set HF_HOME so download goes to D drive:
```bash
set HF_HOME=D:\huggingface_cache
python dp_finetune.py
```
**Subsequent runs** — model already cached at `D:\huggingface_cache\phi3`,  
`set HF_HOME` is not needed since `BASE_MODEL` points directly to local folder:
```bash
python dp_finetune.py
```

### `modeling_phi3.py`
Patched Phi-3 model file — fixes compatibility with transformers 5.x.  
Must be placed in same folder as `dp_finetune.py`.  
Automatically copied into HuggingFace cache at runtime.  
**Do not delete.**

---

## Training Data Scenarios

10 simulation variations covering:

| Parameter | Range |
|---|---|
| Step size | 0.8m → 4.0m |
| Fault time | 2min → 4.5min |
| Noise level | low / standard / high |
| Alarm limit | 1.0m / 1.5m / 2.0m |

Each example teaches the model to identify:
1. Sudden jump in X_meas while X_filt stays smooth → measurement fault
2. Thruster load spike → control system reacting to false error
3. Vessel did not actually move → X_filt confirms position near setpoint

---

## LoRA Fine-Tuning Parameters

| Parameter | Value |
|---|---|
| Base model | Phi-3 Mini 3.8B |
| LoRA rank (r) | 8 |
| LoRA alpha | 16 |
| Target modules | q_proj, v_proj, k_proj, o_proj |
| Epochs | 3 |
| Batch size | 1 |
| Learning rate | 2e-4 |
| Trainable params | ~1.57M of 3.82B (0.04%) |

---

## Environment Setup

```bash
# Python 3.13.1 required
pip install requests numpy matplotlib scipy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft datasets trl
pip install huggingface_hub

pip freeze > requirements.txt
```

### Key environment variables
```
OLLAMA_MODELS = D:\.ollama
HF_HOME       = D:\huggingface_cache
```

---

## File Structure

```
dp_fault_ai/
├── SimulateFaultData.py        simulation + plots
├── dp_generate_prompt.py       builds prompt file
├── dp_ask_model.py             queries models
├── dp_make_training_data.py    generates training JSON
├── dp_finetune.py              LoRA fine-tuning
├── modeling_phi3.py            patched Phi-3 model file
├── test_mistral.py             quick model test
├── test_compare_models.py      multi-model comparison
├── dp_fault_prompt.txt         generated prompt (runtime)
├── dp_fault_response.txt       model responses (runtime)
├── dp_training_data.json       10 training examples (runtime)
├── dp_simulation.png           simulation plot (runtime)
└── dp_phi3_adapter/            LoRA adapter output (after training)

D:\huggingface_cache\phi3\     Phi-3 model weights
D:\.ollama\                    Ollama models (Mistral, Phi-3, TinyLlama)
```

---

## Next Steps

- [ ] Complete LoRA fine-tuning and verify loss decreases
- [ ] Merge adapter back into model and test with `dp_ask_model.py`
- [ ] Compare fine-tuned vs baseline responses on same prompt
- [ ] Add RAG pipeline for real DP system manuals
- [ ] Wrap as FastAPI service for C++ integration
- [ ] Deploy on separate machine (Option 2 — Ethernet)
