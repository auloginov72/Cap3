import requests
import json

def ask_model_stream(prompt, model, system_prompt):
    response = requests.post(
        "http://localhost:11434/api/generate", 
        json={
            "model": model,
            "stream": True,
            "system": system_prompt,
            "prompt": prompt,
            "options": {
                "temperature": 0.1
            }
        },
        stream=True
    )
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk.get("response", ""), end="", flush=True)
    print()

# ─────────────────────────────────────────────
# SAME TEST DATA AS ORIGINAL TEST
# ─────────────────────────────────────────────
data = """
Fault detected: ALM_2047 Position loss warning
Timestamp: 14:23:11

Time series data (10 second intervals before fault):

Time        HeadingErr(deg)  WindSpeed(m/s)  Thruster1(%)  Thruster2(%)
14:22:31    0.3              12.1            45.2          44.8
14:22:41    0.5              13.4            52.1          51.3
14:22:51    0.9              14.8            63.4          62.1
14:23:01    1.8              16.2            78.9          75.4
14:23:11    3.2              18.4            94.2          45.1  <-- FAULT

Question: What trend do you see in this data and what likely caused the fault? Very brief answer needed
"""

system = "You are a DP system fault analyst. Analyze the given time series data and explain what happened concisely."

# ─────────────────────────────────────────────
# TEST MISTRAL
# ─────────────────────────────────────────────
print("=" * 55)
print("  MISTRAL 7B")
print("=" * 55)
ask_model_stream(data, "mistral", system)

# ─────────────────────────────────────────────
# TEST PHI-3
# ─────────────────────────────────────────────
print("\n" + "=" * 55)
print("  PHI-3 MINI")
print("=" * 55)
ask_model_stream(data, "phi3", system)
