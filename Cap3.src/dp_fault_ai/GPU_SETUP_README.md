# DP Fault AI — GPU Machine Setup Guide
### Target hardware: GTX 1080 Ti (11 GB VRAM) — Windows

---

## What you need before starting

- This repository folder (copy from USB or network share)
- Internet connection for downloads
- ~15 GB free disk space on D:\ for model weights
- ~3 GB free disk space for Python environment

**Choose your project folder before starting and use it consistently
throughout all steps. Example:**
```
WORK_FOLDER = C:\Work\Cap3.src
```
Any folder on any drive works — just be consistent.

---

## STEP 1 — Install Python 3.13

Download and install from https://www.python.org/downloads/

During install:
- ✅ Check **"Add Python to PATH"**
- ✅ Check **"Install for all users"** (optional but recommended)

Verify in a new Command Prompt:
```
python --version
```
Expected: `Python 3.13.x`

---

## STEP 2 — Install CUDA 11.8

The GTX 1080 Ti requires CUDA 11.8 for PyTorch GPU support.

Download from:
https://developer.nvidia.com/cuda-11-8-0-download-archive

Select: Windows → x86_64 → your Windows version → exe (local)

Run the installer, choose **Express installation**.

After install, verify in Command Prompt:
```
nvcc --version
```
Expected: `release 11.8`

Also check the driver is up to date — CUDA 11.8 requires driver ≥ 520.x:
```
nvidia-smi
```
If driver is older, update from https://www.nvidia.com/drivers

---

## STEP 3 — Set up Python environment

Copy the project folder to your chosen `WORK_FOLDER`, for example:
```
WORK_FOLDER\Cap3.src\
```

Open Command Prompt, navigate to the project folder:
```
cd WORK_FOLDER\Cap3.src
```

Run the GPU setup script:
```
Setup_gpu.bat
```

This will:
1. Create a Python virtual environment in `venv\`
2. Install PyTorch 2.10.0 with CUDA 11.8 support (~2 GB download)
3. Install all other packages from `requirements_gpu.txt`
4. Print a CUDA verification line at the end

**Expected last lines of Setup_gpu.bat output:**
```
CUDA available: True
GPU: NVIDIA GeForce GTX 1080 Ti
```

If you see `CUDA available: False` — CUDA 11.8 is not installed correctly.
Go back to Step 2.

---

## STEP 4 — Download the Phi-3 model weights

The model is downloaded directly from HuggingFace (~7.5 GB).
Internet connection required. Takes 10–30 minutes depending on speed.

First create the target folder on D:\ :
```
mkdir D:\huggingface_cache
```

Then activate the venv and run the download:
```
cd WORK_FOLDER\Cap3.src
venv\Scripts\activate
huggingface-cli download microsoft/Phi-3-mini-4k-instruct --local-dir D:\huggingface_cache\phi3
```

When complete, verify the folder contains these files:
```
D:\huggingface_cache\phi3\
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
    model.safetensors.index.json
```

> NOTE: modeling_phi3.py is NOT needed and should NOT be placed here.
> transformers 4.57.6 supports Phi-3 natively without any patches.

---

## STEP 5 — Generate training data

Activate the venv and generate the 10 training examples:
```
cd WORK_FOLDER\Cap3.src\dp_fault_ai
..\venv\Scripts\activate
python dp_make_training_data.py
```

Expected output:
```
Generating training examples...
  Example  1: 2m step at 4min, standard noise  ...
  ...
  Example 10: 2m step at 4min, tighter alarm limit
Total examples : 10
Saved to       : dp_training_data.json
```

---

## STEP 6 — Run fine-tuning

Use the GPU version of the script:
```
python dp_finetune.py
```

**Expected startup sequence in the log:**
```
[HH:MM:SS] GPU detected: NVIDIA GeForce GTX 1080 Ti  (11.0 GB VRAM)
[HH:MM:SS] Base model loaded OK  (dtype=torch.float16  VRAM used: 7.5 GB)
[HH:MM:SS] Training started — 30 steps expected
[HH:MM:SS] step   1/30  loss=2.xxxx  lr=...  VRAM=9.xGB  ETA=0:xx:xx
```

**Good signs:**
- Loss on step 1 is between 2.0 and 3.5
- VRAM stays below 10.5 GB
- ETA shows minutes, not hours

**Expected total training time: 15–30 minutes**

---

## STEP 7 — Monitor progress

The log is written to `dp_finetune.log` in the same folder.
You can open it in Notepad while training runs — it updates after every step.

Loss should decrease over the 30 steps, for example:
```
step  1/30  loss=2.8  ...
step  5/30  loss=2.1  ...
step 15/30  loss=1.4  ...
step 30/30  loss=0.9  ...
```

---

## STEP 8 — After training completes

The adapter is saved to:
```
dp_fault_ai\dp_phi3_adapter\
    adapter_config.json
    adapter_model.safetensors
    tokenizer files...
```

This folder (~6 MB) can be copied back to the main machine.
The original Phi-3 model weights are NOT modified.

Next step: run `dp_merge_and_test.py` to merge adapter into the model
and test the fine-tuned responses.

---

## Troubleshooting

**`CUDA not available` at runtime:**
- CUDA 11.8 not installed → go to Step 2
- PyTorch installed without CUDA → re-run Setup_gpu.bat

**`CUDA out of memory` during training:**
- Open dp_finetune.py and reduce MAX_SEQ_LENGTH from 2048 to 1536
- Also try closing other GPU applications (games, video software)

**`loss=12` or `mean_token_accuracy=0` on step 1:**
- Wrong library versions — verify with:
  `pip show transformers trl peft | findstr Version`
- Expected: transformers 4.57.6 / trl 0.15.2 / peft 0.13.2

**Script crashes immediately with import error:**
- Venv not activated — run `..\venv\Scripts\activate` first

---

## STEP 9 — Install Ollama and test baseline vs fine-tuned

This step runs on your **main machine** (not the GTX machine).
It compares the original Phi-3 response against the LoRA fine-tuned version
on the same fault prompt so you can see the difference clearly.

### 9.1 — Install Ollama

Download and install from https://ollama.com/download

Run the installer — it installs as a background service automatically.

Verify in Command Prompt:
```
ollama --version
```

### 9.2 — Pull Phi-3 into Ollama

```
ollama pull phi3
```

This downloads the Phi-3 Mini model into Ollama's own cache (~2 GB).
This is separate from the HuggingFace weights — Ollama uses its own format.

Also pull Mistral for an additional baseline comparison:
```
ollama pull mistral
```

Verify both are available:
```
ollama list
```

Expected output:
```
NAME            ID            SIZE    MODIFIED
phi3:latest     ...           2.2 GB  ...
mistral:latest  ...           4.1 GB  ...
```

### 9.3 — Copy the LoRA adapter back from the GTX machine

Copy the adapter folder from the GTX machine:
```
dp_fault_ai\dp_phi3_adapter\
```
to the same location on your main machine. This is only ~6 MB.

### 9.4 — Merge the adapter into the base model

The adapter needs to be merged into the Phi-3 weights before Ollama
can use it. Run the merge script on the main machine:
```
cd WORK_FOLDER\Cap3.src\dp_fault_ai
..\venv\Scripts\activate
python dp_merge_and_test.py
```

This creates a merged model at:
```
dp_fault_ai\dp_phi3_merged\
```

The original weights at `D:\huggingface_cache\phi3\` remain untouched.

### 9.5 — Generate a test fault prompt

Generate a fresh prompt to use for comparison:
```
python dp_generate_prompt.py
```

This creates `dp_fault_prompt.txt` — the same format your C++ system
will eventually send, containing signal data around a fault event.

### 9.6 — Query baseline model (Ollama Phi-3, no fine-tuning)

```
python dp_ask_model.py
```

This sends the prompt to the original Phi-3 via Ollama and saves the
response to `dp_fault_response.txt`. Read that file — this is your baseline.

**Typical baseline response:** generic, may miss the X_meas vs X_filt
distinction, may not identify measurement fault specifically, answer
structure varies each run.

### 9.7 — Query the fine-tuned model

Now query the merged fine-tuned model directly via HuggingFace:
```
python dp_ask_finetuned.py
```

**What a successful fine-tuned response looks like:**
```
Step-by-step fault analysis:

1. SIGNAL BEHAVIOUR:
   X_meas shows a sudden step of approximately X.Xm at t=XXXs.
   X_filt remains smooth and close to zero, confirming no real
   vessel motion occurred...

2. ROOT CAUSE:
   Measurement fault — characteristic of DGPS signal spike...

3. CONTROL SYSTEM RESPONSE:
   Thruster load increased to XX% responding to false error...

4. RECOMMENDED CHECKS:
   - Inspect reference sensor logs at t=XXXs...
```

### 9.8 — What to look for in the comparison

| | Baseline Phi-3 (Ollama) | Fine-tuned Phi-3 |
|---|---|---|
| Structure | Variable | Always 4 numbered sections |
| X_meas vs X_filt | May ignore | Always compared explicitly |
| Root cause | Generic | Identifies measurement fault |
| Recommendations | Generic | Specific to DP sensor checks |
| Language | General | DP engineering terminology |

If the fine-tuned model follows the 4-section structure and correctly
identifies the X_meas step as a measurement fault — fine-tuning worked.

If responses look similar to baseline — loss did not decrease enough
during training, consider increasing NUM_EPOCHS to 5 in dp_finetune.py
and retraining.
