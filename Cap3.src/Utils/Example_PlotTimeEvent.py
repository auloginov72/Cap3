import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
T1 = np.linspace(0, 10, 100)
X1 = np.sin(T1)
X2 = np.cos(T1)

# Event data: times when events change and their types
T3 = [0, 2, 5, 7, 10]  # event change times
EventType = [0, 2, 1, 3]  # event types between times

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot main data
ax.plot(T1, X1, label='X1', linewidth=2)
ax.plot(T1, X2, label='X2', linewidth=2)

# Add colored background sections for events
colors = {0: 'gray', 1: 'blue', 2: 'green', 3: 'yellow'}

for i in range(len(T3)-1):
    ax.axvspan(T3[i], T3[i+1], 
               facecolor=colors[EventType[i]], 
               alpha=0.3,
               zorder=0)

ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlabel('Time')
ax.set_ylabel('Value')
ax.set_title('Data with Event Background')

plt.show()