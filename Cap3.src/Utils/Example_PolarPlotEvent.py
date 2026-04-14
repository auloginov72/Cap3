import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

# Generate sample polar data
theta = np.linspace(0, 2*np.pi, 100)
r1 = 1 + 0.5 * np.sin(4*theta)
r2 = 1.5 + 0.3 * np.cos(3*theta)

# Event data: angular positions where events change and their types
theta_events = [0, np.pi/3, np.pi, 4*np.pi/3, 2*np.pi]  # event change angles
EventType = [0, 2, 1, 3]  # event types between angles

# Create polar plot
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

# Event colors and names
colors = {0: 'gray', 1: 'blue', 2: 'green', 3: 'yellow'}
names = {0: 'Idle', 1: 'Active', 2: 'Processing', 3: 'Error'}

# Set radial limits first (before plotting sectors)
max_r = max(np.max(r1), np.max(r2)) * 1.1  # Add 10% margin
ax.set_ylim(0, max_r)

# Add colored sectors for events
for i in range(len(theta_events)-1):
    # Create array of angles for the sector
    theta_sector = np.linspace(theta_events[i], theta_events[i+1], 50)
    
    # Fill sector from center to max radius using the fixed r_max
    ax.fill_between(theta_sector, 0, max_r,
                    facecolor=colors[EventType[i]], 
                    alpha=0.3,
                    zorder=0)

# Plot main data (after sectors, so they're on top)
ax.plot(theta, r1, label='R1', linewidth=2, zorder=2)
ax.plot(theta, r2, label='R2', linewidth=2, zorder=2)

ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.set_title('Polar Plot with Event Sectors', pad=20)
ax.grid(True, alpha=0.3)

# Add event legend at the bottom
from matplotlib.patches import Rectangle

xlabel_y = 0.05

fig.text(0.35, xlabel_y, 'Events: ', 
         fontsize=10, 
         verticalalignment='center',
         transform=fig.transFigure)

start_x = 0.43
spacing = 0.08

for idx, event_id in enumerate(sorted(colors.keys())):
    x_pos = start_x + idx * spacing
    
    # Add colored rectangle
    rect = Rectangle((x_pos, xlabel_y - 0.005), 0.015, 0.015, 
                     transform=fig.transFigure, 
                     facecolor=colors[event_id], 
                     alpha=0.5,
                     edgecolor='black',
                     linewidth=0.5)
    fig.patches.append(rect)
    
    # Add text label
    fig.text(x_pos + 0.018, xlabel_y, f'{names[event_id]}', 
             fontsize=9, 
             verticalalignment='center',
             transform=fig.transFigure)

plt.subplots_adjust(bottom=0.12)
plt.show()