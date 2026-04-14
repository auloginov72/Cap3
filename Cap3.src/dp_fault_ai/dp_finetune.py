"""
dp_finetune.py
--------------
LoRA fine-tuning of Phi-3 Mini on dp_training_data.json
Output: fine-tuned adapter saved to ./dp_phi3_adapter/

Hardware target: GTX 1080 Ti (Pascal, 11GB VRAM, CUDA 11.8)
Tested with:
  torch        == 2.10.0+cu118
  transformers == 4.57.6
  trl          == 0.15.2
  peft         == 0.13.2

Key settings:
  - float16  — native on Pascal GPU, halves VRAM vs float32
  - fp16=True in TrainingArguments — enables mixed precision
  - device_map="cuda" — runs on GPU
  - MAX_SEQ_LENGTH 2048 — prompts measured at 1318-1624 tokens, fits cleanly
  - modeling_phi3.py patch NOT needed — transformers 4.57.6 supports Phi-3 natively
"""

import json
import os
import sys
import time
import datetime
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# ─────────────────────────────────────────────
# LOG SETUP — mirrors every print() to file
# flushed after every line — survives crash / kill
# ─────────────────────────────────────────────
LOG_FILE = "dp_finetune.log"

class Tee:
    """Mirrors stdout to a log file, flushing after every write."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log      = open(filepath, "a", encoding="utf-8", buffering=1)
        self._write_header()

    def _write_header(self):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.write(f"\n{'='*55}\n  Run started: {ts}\n{'='*55}\n")
        self.log.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

sys.stdout = Tee(LOG_FILE)

def tprint(msg):
    """Print with HH:MM:SS timestamp."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

tprint(f"Logging to: {os.path.abspath(LOG_FILE)}")
tprint("Versions: torch==2.10.0+cu118  transformers==4.57.6  trl==0.15.2  peft==0.13.2")

# ─────────────────────────────────────────────
# CUDA CHECK — fail early if GPU not visible
# ─────────────────────────────────────────────
if not torch.cuda.is_available():
    tprint("ERROR: CUDA not available — is the GPU driver installed?")
    tprint("Run: python -c \"import torch; print(torch.cuda.is_available())\"")
    tprint("If False — install CUDA 11.8 from nvidia.com and retry.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
tprint(f"GPU detected: {gpu_name}  ({vram_gb:.1f} GB VRAM)")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
TRAINING_DATA_FILE = "dp_training_data.json"
OUTPUT_DIR         = "./dp_phi3_adapter"
BASE_MODEL         = r"D:\huggingface_cache\phi3"

LORA_R          = 8
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05

NUM_EPOCHS      = 3
BATCH_SIZE      = 1
MAX_SEQ_LENGTH  = 2048    # prompts 1318-1624 tokens — fits without truncation
LEARNING_RATE   = 2e-4
SAVE_STEPS      = 20

# ─────────────────────────────────────────────
# LOAD TRAINING DATA
# ─────────────────────────────────────────────
print("=" * 55)
print("  DP FAULT ANALYSER — LoRA Fine-Tuning")
print("=" * 55)

try:
    with open(TRAINING_DATA_FILE, "r") as f:
        raw_data = json.load(f)
    tprint(f"Training examples loaded : {len(raw_data)}")
except FileNotFoundError:
    tprint(f"ERROR: {TRAINING_DATA_FILE} not found.")
    tprint("Run dp_make_training_data.py first.")
    sys.exit(1)

dataset     = Dataset.from_list([{"text": d["text"]} for d in raw_data])
total_steps = (len(dataset) // BATCH_SIZE) * NUM_EPOCHS
tprint(f"Dataset ready            : {len(dataset)} examples")
tprint(f"Estimated total steps    : {total_steps}  (epochs={NUM_EPOCHS})")
tprint(f"Sequence length cap      : {MAX_SEQ_LENGTH} tokens")
tprint(f"  (measured prompt range : ~1318-1624 tokens — no truncation)")

# ─────────────────────────────────────────────
# LOAD TOKENIZER
# ─────────────────────────────────────────────
tprint(f"Loading tokenizer from   : {BASE_MODEL}")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "right"
tprint("Tokenizer loaded OK")

# ─────────────────────────────────────────────
# LOAD BASE MODEL
# ─────────────────────────────────────────────
tprint(f"Loading base model       : {BASE_MODEL}")
tprint("Loading to GPU — this will take 1-2 minutes...")

# float16: native on Pascal (GTX 1080 Ti), ~7.5GB VRAM — fits in 11GB
# Do NOT use bfloat16 — not supported on Pascal architecture
DTYPE = torch.float16

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype         = DTYPE,
    trust_remote_code   = True,
    device_map          = "cuda",       # GPU — was "cpu" on CPU machine
    low_cpu_mem_usage   = True,
    attn_implementation = "eager",
)
model.config.use_cache = False

vram_used = torch.cuda.memory_allocated(0) / 1e9
tprint(f"Base model loaded OK  (dtype={DTYPE}  VRAM used: {vram_used:.1f} GB)")

# ─────────────────────────────────────────────
# APPLY LoRA
# ─────────────────────────────────────────────
tprint("Applying LoRA adapter...")

lora_config = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    lora_dropout   = LORA_DROPOUT,
    bias           = "none",
    task_type      = TaskType.CAUSAL_LM,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# LOGGING CALLBACK
# ─────────────────────────────────────────────
class StepLogger(TrainerCallback):
    def __init__(self):
        self._t_start = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t_start = time.time()
        tprint(f"Training started — {total_steps} steps expected")

    def on_log(self, args, state: TrainerState, control: TrainerControl,
               logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        loss = logs.get("loss", logs.get("train_loss", "?"))
        lr   = logs.get("learning_rate", "?")

        if self._t_start and step > 0:
            elapsed       = time.time() - self._t_start
            secs_per_step = elapsed / step
            remaining     = (total_steps - step) * secs_per_step
            eta_str       = str(datetime.timedelta(seconds=int(remaining)))
        else:
            eta_str = "calculating..."

        loss_str = f"{loss:.4f}" if isinstance(loss, float) else str(loss)
        lr_str   = f"{lr:.2e}"   if isinstance(lr,   float) else str(lr)

        # also report VRAM usage each step so we know if we're close to limit
        vram_str = f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB"

        tprint(
            f"step {step:>3}/{total_steps}"
            f"  loss={loss_str}"
            f"  lr={lr_str}"
            f"  VRAM={vram_str}"
            f"  ETA={eta_str}"
        )

    def on_save(self, args, state, control, **kwargs):
        tprint(f"Checkpoint saved at step {state.global_step}")

    def on_train_end(self, args, state, control, **kwargs):
        elapsed = time.time() - self._t_start
        tprint(f"Training complete — total time: "
               f"{str(datetime.timedelta(seconds=int(elapsed)))}")

# ─────────────────────────────────────────────
# TRAINING ARGUMENTS
# trl 0.15.2: TrainingArguments here,
# dataset_text_field / max_seq_length / packing go in SFTTrainer() below
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = NUM_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = 2,
    learning_rate               = LEARNING_RATE,
    fp16                        = True,         # float16 mixed precision on GPU
    bf16                        = False,        # not supported on Pascal
    logging_steps               = 1,
    save_steps                  = SAVE_STEPS,
    save_total_limit            = 2,
    report_to                   = "none",
    optim                       = "adamw_torch",
    warmup_ratio                = 0.1,
    lr_scheduler_type           = "cosine",
)

# ─────────────────────────────────────────────
# TRAINER
# trl 0.15.2: tokenizer= (not processing_class)
# dataset_text_field / max_seq_length / packing go HERE
# ─────────────────────────────────────────────
tprint("Initialising trainer...")

trainer = SFTTrainer(
    model              = model,
    train_dataset      = dataset,
    args               = training_args,
    tokenizer          = tokenizer,
    dataset_text_field = "text",
    max_seq_length     = MAX_SEQ_LENGTH,
    packing            = False,
    callbacks          = [StepLogger()],
)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
tprint("STARTING TRAINING")
tprint(f"Log file : {os.path.abspath(LOG_FILE)}")
tprint("Watch loss — should start ~2-3 and decrease each step")
tprint("Watch VRAM — should stay under 10.5 GB")
print("=" * 55 + "\n")

trainer.train()

# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────
tprint("Saving LoRA adapter...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("=" * 55)
tprint(f"Adapter saved to : {OUTPUT_DIR}")
tprint("Next step        : run dp_merge_and_test.py")
print("=" * 55)
