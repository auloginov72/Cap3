import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# Generate sample data
T1 = np.linspace(0, 10, 100)
X1 = np.sin(T1)
X2 = np.cos(T1)

# Event data
T3 = [0, 2, 5, 7, 10]
EventType = [0, 2, 1, 3]

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot main data
ax.plot(T1, X1, label='X1', linewidth=2)
ax.plot(T1, X2, label='X2', linewidth=2)

# Event colors and names
colors = {0: 'gray', 1: 'blue', 2: 'green', 3: 'yellow'}
names = {0: 'Idle', 1: 'Active', 2: 'Processing', 3: 'Error'}

# Add colored background sections
for i in range(len(T3)-1):
    ax.axvspan(T3[i], T3[i+1], 
               facecolor=colors[EventType[i]], 
               alpha=0.3,
               zorder=0)

ax.grid(True, alpha=0.3)
ax.set_ylabel('Value')
ax.set_title('Data with Event Background')
ax.legend(loc='upper right')

# Add colored rectangles and text for xlabel
xlabel_y = 0.04  # Position above bottom of figure

# Add "Time (Events: " text
fig.text(0.35, xlabel_y, 'Time (Events: ', 
         fontsize=10, 
         verticalalignment='center',
         transform=fig.transFigure)

start_x = 0.45
spacing = 0.08  # Reduced spacing between items

for idx, event_id in enumerate(sorted(colors.keys())):
    # Calculate position for each box and label
    x_pos = start_x + idx * spacing
    
    # Add colored rectangle
    rect = Rectangle((x_pos, xlabel_y - 0.005), 0.015, 0.015, 
                     transform=fig.transFigure, 
                     facecolor=colors[event_id], 
                     alpha=0.5,
                     edgecolor='black',
                     linewidth=0.5)
    fig.patches.append(rect)
    
    # Add text label (closer to rectangle)
    fig.text(x_pos + 0.018, xlabel_y, f'{names[event_id]}', 
             fontsize=9, 
             verticalalignment='center',
             transform=fig.transFigure)

# Add closing ")"
fig.text(start_x + 4 * spacing + 0.04, xlabel_y, ')', 
         fontsize=10, 
         verticalalignment='center',
         transform=fig.transFigure)

plt.subplots_adjust(bottom=0.12)
plt.show()