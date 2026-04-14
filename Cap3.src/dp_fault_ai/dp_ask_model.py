"""
dp_ask_model.py
---------------
Reads dp_fault_prompt.txt and sends it to both Mistral and Phi-3.
Streams responses to console and saves both to dp_fault_response.txt 
"""

import requests
import json
from datetime import datetime

PROMPT_FILE   = "dp_fault_prompt.txt"
RESPONSE_FILE = "dp_fault_response.txt"
OLLAMA_URL    = "http://localhost:11434/api/generate"
MODELS        = ["mistral", "phi3" , "tinyllama"]

# ─────────────────────────────────────────────
# READ PROMPT FILE
# ─────────────────────────────────────────────
try:
    with open(PROMPT_FILE, "r") as f:
        prompt = f.read()
    print(f"Prompt loaded from : {PROMPT_FILE}")
    print(f"Prompt length      : {len(prompt)} characters")
except FileNotFoundError:
    print(f"ERROR: {PROMPT_FILE} not found.")
    print("Run dp_generate_prompt.py first.")
    exit(1)

# ─────────────────────────────────────────────
# QUERY FUNCTION
# ─────────────────────────────────────────────
def ask_model(model, prompt):
    print(f"\n{'=' * 65}")
    print(f"  MODEL: {model.upper()}")
    print(f"{'=' * 65}")

    start_time    = datetime.now()
    full_response = ""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.1
                }
            },
            stream=True
        )

        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                token = chunk.get("response", "")
                print(token, end="", flush=True)
                full_response += token
                if chunk.get("done", False):
                    break

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama.")
        print("Make sure Ollama is running (check system tray).")
        exit(1)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'-' * 65}")
    print(f"  Response time   : {elapsed:.1f}s")
    print(f"  Response length : {len(full_response)} characters")

    return full_response, elapsed

# ─────────────────────────────────────────────
# RUN BOTH MODELS
# ─────────────────────────────────────────────
results = {}
for model in MODELS:
    response, elapsed = ask_model(model, prompt)
    results[model] = {"response": response, "elapsed": elapsed}

# ─────────────────────────────────────────────
# SAVE BOTH RESPONSES TO FILE
# ─────────────────────────────────────────────
with open(RESPONSE_FILE, "w") as f:
    f.write(f"Run time : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 65 + "\n")
    f.write("PROMPT\n")
    f.write("=" * 65 + "\n")
    f.write(prompt)

    for model in MODELS:
        r = results[model]
        f.write(f"\n{'=' * 65}\n")
        f.write(f"  RESPONSE — {model.upper()}\n")
        f.write(f"  Time: {r['elapsed']:.1f}s\n")
        f.write(f"{'=' * 65}\n")
        f.write(r["response"])

print(f"\n{'=' * 65}")
print(f"  Both responses saved to : {RESPONSE_FILE}")
print(f"  Open it in VSCode to compare side by side")
print(f"{'=' * 65}")
