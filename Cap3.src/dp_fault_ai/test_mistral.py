import requests
import json

def ask_mistral_stream(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "stream": True,
            "system": "You are a DP system fault analyst. Analyze the given time series data and explain what happened concisely.",
            "prompt": prompt
        },
        stream=True
    )
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk.get("response", ""), end="", flush=True)
    print()

# Simulate some time domain data around a fault event
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

ask_mistral_stream(data)