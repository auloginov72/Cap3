import numpy as np
import matplotlib
import sys

# Use Agg backend if tkinter is not available
try:
    import tkinter
    matplotlib.use("TkAgg")
except ImportError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def PolarPlot(Val, Dir, options):
    """
    Create a polar plot with multiple datasets
    
    Args:
        Val: List of value arrays (radial values)
        Dir: List of direction arrays (angular values in degrees)
        options: Object containing plot configuration
    
    Returns:
        Figure object that can be saved as PNG
    """
    # Validate inputs
    if len(Dir) != len(Val):
        print("Error: Val and Dir must have same length")
        return None
    
    # Check each dataset has matching lengths
    for i in range(len(Val)):
        if len(Dir[i]) != len(Val[i]):
            print(f"Error: Dir[{i}] and Val[{i}] must have same length")
            return None
    
    # Set default options if not provided
    if not hasattr(options, 'PlotLegend') or options.PlotLegend is None:
        options.PlotLegend = [f'Data{i+1}' for i in range(len(Val))]
    
    if not hasattr(options, 'LineColor') or options.LineColor is None:
        # Default color cycle
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        options.LineColor = [colors[i % len(colors)] for i in range(len(Val))]
    
    if not hasattr(options, 'LineStyle') or options.LineStyle is None:
        # Default line styles: '-' solid, '--' dashed, ':' dotted, 'None' invisible
        options.LineStyle = ['-' for _ in range(len(Val))]
    
    if not hasattr(options, 'LineWidth') or options.LineWidth is None:
        options.LineWidth = [2.0 for _ in range(len(Val))]
    
    if not hasattr(options, 'LineMarkerColor') or options.LineMarkerColor is None:
        options.LineMarkerColor = options.LineColor  # Same as line color
    
    if not hasattr(options, 'LineMarkerStyle') or options.LineMarkerStyle is None:
        # Marker styles: 'o' circle, 's' square, '^' triangle, 'None' invisible
        options.LineMarkerStyle = ['None' for _ in range(len(Val))]
    
    if not hasattr(options, 'LineMarkerSize') or options.LineMarkerSize is None:
        options.LineMarkerSize = [6 for _ in range(len(Val))]
    
    if not hasattr(options, 'AxesLimit'):
        options.AxesLimit = 'auto'
    
    if not hasattr(options, 'NameLbl'):
        options.NameLbl = 'Graph Name'
    
    if not hasattr(options, 'ValLbl'):
        options.ValLbl = 'Name of Values'
    
    # Grid options - NEW
    if not hasattr(options, 'GridAlpha'):
        options.GridAlpha = 0.3  # Transparency (0-1, higher = more visible)
    
    if not hasattr(options, 'GridLineWidth'):
        options.GridLineWidth = 1.0  # Grid line thickness
    
    if not hasattr(options, 'GridLineStyle'):
        options.GridLineStyle = '-'  # '-' solid, '--' dashed, ':' dotted
    
    if not hasattr(options, 'GridColor'):
        options.GridColor = 'gray'  # Grid color
    
    if not hasattr(options, 'RadialGridLines'):
        options.RadialGridLines = 8  # Number of radial lines (angular divisions)
    
    if not hasattr(options, 'ConcentricGridLines'):
        options.ConcentricGridLines = 5  # Number of concentric circles
    
    if not hasattr(options, 'ShowRadialGrid'):
        options.ShowRadialGrid = True  # Show/hide radial grid lines
    
    if not hasattr(options, 'ShowConcentricGrid'):
        options.ShowConcentricGrid = True  # Show/hide concentric circles
    
    if not hasattr(options, 'FigureSize'):
        options.FigureSize = (10, 8)  # Default size as tuple
    
    # Event coloring options
    if not hasattr(options, 'EventFi'):
        options.EventFi = None  # List of event angles (in degrees)
    
    if not hasattr(options, 'EventType'):
        options.EventType = None  # List of event types corresponding to angles
    
    if not hasattr(options, 'EventColors'):
        options.EventColors = None  # Dictionary mapping event type to color
    
    #Font size , etc
    if not hasattr(options, 'TitleFontSize'):
        options.TitleFontSize = 20
    if not hasattr(options, 'LabelFontSize'):
        options.LabelFontSize = 14        
    if not hasattr(options, 'TickFontSize'):
        options.TickFontSize = 10
    if not hasattr(options, 'TickFontWeight'):
        options.TickFontWeight = 'normal'

    if not hasattr(options, 'TitleFontFamily'):
        options.TitleFontFamily = 'sans-serif'  # or 'serif', 'monospace', etc.
    if not hasattr(options, 'LabelFontFamily'):
        options.LabelFontFamily = 'sans-serif'
    if not hasattr(options, 'TickFontFamily'):
        options.TickFontFamily = 'sans-serif'
    # Optional: Font weight control
    if not hasattr(options, 'TitleFontWeight'):
        options.TitleFontWeight = 'bold'  # or 'normal'
    if not hasattr(options, 'LabelFontWeight'):
        options.LabelFontWeight = 'normal'
    
    # Radial tick label position (angular direction in degrees)
    if not hasattr(options, 'TickPositionDirection'):
        options.TickPositionDirection = 22.5  # Default matplotlib position (approximately 22.5 degrees)
    
    # Tick label z-order (controls whether labels appear above or below data)
    if not hasattr(options, 'TickLabelZOrder'):
        options.TickLabelZOrder = 'auto'  # Default: auto (matplotlib default), or specify number like 10 for above data
    
    # Indicate out-of-area data points
    if not hasattr(options, 'IndicateOutOfAreaData'):
        options.IndicateOutOfAreaData = False  # Default: do not indicate
    if not hasattr(options, 'OutOfAreaIndicatorColor'):
        options.OutOfAreaIndicatorColor = 'red'  # Color for the indicator arcs
    if not hasattr(options, 'OutOfAreaIndicatorWidth'):
        options.OutOfAreaIndicatorWidth = 4.0  # Width of the indicator arcs
    if not hasattr(options, 'OutOfAreaIndicatorAlpha'):
        options.OutOfAreaIndicatorAlpha = 0.8  # Transparency of indicator arcs
    if not hasattr(options, 'OutOfAreaSectorResolution'):
        options.OutOfAreaSectorResolution = 10  # Angular resolution in degrees for grouping out-of-area points


    # Create figure with polar projection
    fig = plt.figure(figsize=options.FigureSize)
    ax = fig.add_subplot(111, projection='polar')
    
    # Plot each dataset
    max_val = 0
    for i in range(len(Val)):
        # Convert directions from degrees to radians
        theta = np.deg2rad(Dir[i])
        r = Val[i]
        
        # Update max value for axis limits
        if len(r) > 0:
            max_val = max(max_val, np.max(np.abs(r)))
        
        # Map line style strings
        line_style = options.LineStyle[i] if i < len(options.LineStyle) else '-'
        if line_style == 'solid':
            line_style = '-'
        elif line_style == 'dash':
            line_style = '--'
        elif line_style == 'invisible':
            line_style = 'None'
        
        # Map marker style strings
        marker_style = options.LineMarkerStyle[i] if i < len(options.LineMarkerStyle) else 'None'
        if marker_style == 'round':
            marker_style = 'o'
        elif marker_style == 'square':
            marker_style = 's'
        elif marker_style == 'triangle':
            marker_style = '^'
        elif marker_style == 'invisible':
            marker_style = 'None'
        
        # Get other properties
        color = options.LineColor[i] if i < len(options.LineColor) else 'blue'
        linewidth = options.LineWidth[i] if i < len(options.LineWidth) else 2.0
        marker_color = options.LineMarkerColor[i] if i < len(options.LineMarkerColor) else color
        marker_size = options.LineMarkerSize[i] if i < len(options.LineMarkerSize) else 6
        label = options.PlotLegend[i] if i < len(options.PlotLegend) else f'Data{i+1}'
        
        # Plot the data
        ax.plot(theta, r, 
                linestyle=line_style,
                linewidth=linewidth,
                color=color,
                marker=marker_style,
                markersize=marker_size,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                label=label)
    
    #Fonts
    ax.set_title(options.NameLbl, 
                fontsize=options.TitleFontSize,
                fontfamily=options.TitleFontFamily,
                fontweight=options.TitleFontWeight,
                pad=20)
    
    # Add radial label with custom font
    ax.set_ylabel(options.ValLbl, 
                 fontsize=options.LabelFontSize,
                 fontfamily=options.LabelFontFamily,
                 fontweight=options.LabelFontWeight,
                 labelpad=30)

    ax.tick_params(axis='y', labelsize=options.TickFontSize )
    for label in ax.yaxis.get_ticklabels():
        label.set_fontweight(options.TickFontWeight)

    # Set axis limits
    if options.AxesLimit != 'auto':
        ax.set_ylim(0, options.AxesLimit)
    else:
        ax.set_ylim(0, max_val * 1.1)  # Add 10% padding
    
    # Indicate out-of-area data points with bold arc segments at the edge
    if options.IndicateOutOfAreaData and options.AxesLimit != 'auto':
        # Get the radial limit
        r_limit = options.AxesLimit
        
        # Collect all out-of-area points across all datasets
        out_of_area_angles = []
        
        for i in range(len(Val)):
            theta = np.deg2rad(Dir[i])
            r = Val[i]
            
            # Find points that exceed the limit
            if len(r) > 0:
                mask = np.abs(r) > r_limit
                if np.any(mask):
                    # Get angles where data exceeds limit
                    exceeded_angles = np.rad2deg(theta[mask]) % 360
                    out_of_area_angles.extend(exceeded_angles)
        
        # If there are out-of-area points, draw indicator arcs
        if len(out_of_area_angles) > 0:
            # Group angles into sectors based on resolution
            out_of_area_angles = np.array(out_of_area_angles)
            
            # Bin the angles into sectors
            sector_size = options.OutOfAreaSectorResolution
            sectors = np.arange(0, 360, sector_size)
            
            # Find which sectors contain out-of-area points
            active_sectors = set()
            for angle in out_of_area_angles:
                sector_index = int(angle // sector_size)
                active_sectors.add(sector_index)
            
            # Draw thick arcs for each active sector
            for sector_idx in active_sectors:
                theta_start = np.deg2rad(sector_idx * sector_size)
                theta_end = np.deg2rad((sector_idx + 1) * sector_size)
                
                # Create an arc at the outer radius
                theta_arc = np.linspace(theta_start, theta_end, 50)
                r_arc = np.full_like(theta_arc, r_limit)
                
                # Draw thick line at the edge
                ax.plot(theta_arc, r_arc,
                       color=options.OutOfAreaIndicatorColor,
                       linewidth=options.OutOfAreaIndicatorWidth,
                       alpha=options.OutOfAreaIndicatorAlpha,
                       solid_capstyle='butt',
                       zorder=10)  # High zorder to draw on top
    
    # Add event angular sectors if provided
    # IMPORTANT: This must be done AFTER setting axis limits to get correct r_lim
    if options.EventFi is not None and options.EventType is not None and options.EventColors is not None:
        # Default color map for event types if not all types are in EventColors
        default_colors = {
            0: 'yellow',    # PAUSE
            1: 'lightblue', # SAMPLE  'lightblue'
            2: 'lightgreen',# TRANSFER
            3: 'lightcoral',# FINISHED
            4: 'green',     # EXT_OK
            5: 'orange',    # EXT_PAUSE
            6: 'red',       # EXT_FAILED
            7: 'lightgray'  # UNDEF
        }
        
        # Merge user-provided colors with defaults
        event_colors = {**default_colors, **options.EventColors}
        
        # Get current radial limits for drawing sectors (now properly set)
        r_lim = ax.get_ylim()[1]
        
        # Extend radius by ~30% to fill the white ring that matplotlib adds around polar plots
        # This accounts for the space between the outermost grid circle and the plot border
        r_extended = r_lim * 1.30
        
        # Draw angular sectors for each event period
        for i in range(len(options.EventFi)):
            # Start angle of current event (in degrees)
            fi_start = options.EventFi[i]
            
            # End angle is the next event's start angle, or wrap to 360/0
            if i < len(options.EventFi) - 1:
                fi_end = options.EventFi[i + 1]
            else:
                # For the last event, wrap to the first event or 360 degrees
                if len(options.EventFi) > 1:
                    fi_end = options.EventFi[0] + 360  # Complete the circle
                else:
                    fi_end = fi_start + 360  # Full circle if only one event
            
            # Convert angles to radians
            theta_start = np.deg2rad(fi_start)
            theta_end = np.deg2rad(fi_end)
            
            # Get event type and corresponding color
            event_type = options.EventType[i]
            color = event_colors.get(event_type, 'lightgray')
            
            # Create angular sector (wedge) spanning full radius
            # Use fill_between to create a colored wedge
            theta_range = np.linspace(theta_start, theta_end, 100)
            r_range = np.full_like(theta_range, r_extended)
            
            # Fill the sector from center to edge
            ax.fill_between(theta_range, 0, r_range, 
                          alpha=0.3, color=color, zorder=0)
    
    # Customize the plot
    ax.set_theta_zero_location('N')  # Set 0 degrees to North
    ax.set_theta_direction(-1)  # Clockwise
    
    # Add title and labels  custom 
    #ax.set_title(options.NameLbl, fontsize=14, pad=20)
    
    # Add radial label -  custom 
    #ax.set_ylabel(options.ValLbl, labelpad=30)
    
    # Configure grid - UPDATED
    # First, turn off the default grid
    ax.grid(False)
    
    # Set up angular grid lines (radial lines)
    if options.ShowRadialGrid:
        angles = np.linspace(0, 2*np.pi, options.RadialGridLines, endpoint=False)
        ax.set_thetagrids(np.degrees(angles))
        # Style the radial grid lines
        for line in ax.xaxis.get_gridlines():
            line.set_color(options.GridColor)
            line.set_linewidth(options.GridLineWidth)
            line.set_alpha(options.GridAlpha)
            line.set_linestyle(options.GridLineStyle)
    
    # Set up radial grid lines (concentric circles)
    if options.ShowConcentricGrid:
        if options.AxesLimit != 'auto':
            r_ticks = np.linspace(0, options.AxesLimit, options.ConcentricGridLines + 1)[1:]
        else:
                if max_val <= 0:
                    nice_max = 1
                elif max_val <= 1:
                    nice_max = np.ceil(max_val * 10) / 10  # Round to 0.1
                elif max_val <= 10:
                    nice_max = np.ceil(max_val)  # Round to 1
                elif max_val <= 100:
                    nice_max = np.ceil(max_val / 10) * 10  # Round to 10
                else:
                    nice_max = np.ceil(max_val / 100) * 100  # Round to 100
        
                ax.set_ylim(0, nice_max)
                r_ticks = np.linspace(0, nice_max, options.ConcentricGridLines + 1)[1:]    
                #r_ticks = np.linspace(0, max_val * 1.1, options.ConcentricGridLines + 1)[1:]
        
        ax.set_rticks(r_ticks)
        # Format radial tick labels to always show 1 decimal place
        ax.set_yticklabels([f'{tick:.1f}' for tick in r_ticks])
        # Style the concentric grid lines
        for line in ax.yaxis.get_gridlines():
            line.set_color(options.GridColor)
            line.set_linewidth(options.GridLineWidth)
            line.set_alpha(options.GridAlpha)
            line.set_linestyle(options.GridLineStyle)
    
    # Set the angular position of radial tick labels
    ax.set_rlabel_position(options.TickPositionDirection)
    

    ax.grid(True)
    
    # Add legend if labels provided
    if options.PlotLegend and len(options.PlotLegend) > 0:
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Tight layout
    plt.tight_layout()
    
    # Configure tick label z-order for visibility control
    # Bring tick labels to front if requested
    if options.TickLabelZOrder != 'auto':
        for label in ax.yaxis.get_ticklabels():
            label.set_zorder(options.TickLabelZOrder)
        for label in ax.xaxis.get_ticklabels():
            label.set_zorder(options.TickLabelZOrder)
    
    plt.ion()
    return fig



def PolarPlot_XY(X, Y, options):
    """
    Convert X, Y coordinates to polar coordinates and create polar plot
    
    Args:
        X: List of X coordinate arrays
        Y: List of Y coordinate arrays  
        options: Plot configuration options
        
    Convention:
        Dir = 0° (North) corresponds to X=0, Y=+1
        Dir = 90° (East) corresponds to X=+1, Y=0
        Dir = 180° (South) corresponds to X=0, Y=-1
        Dir = 270° (West) corresponds to X=-1, Y=0
    
    Returns:
        Figure object from PolarPlot
    """
    # Validate inputs
    if len(X) != len(Y):
        print("Error: X and Y must have same number of datasets")
        return None
    
    # Initialize Val and Dir lists
    Val = []
    Dir = []
    
    # Convert each dataset from X,Y to Val,Dir
    for i in range(len(X)):
        x_data = np.array(X[i])
        y_data = np.array(Y[i])
        
        if len(x_data) != len(y_data):
            print(f"Error: X[{i}] and Y[{i}] must have same length")
            return None
        
        # Calculate magnitude (radius)
        val = np.sqrt(x_data**2 + y_data**2)
        
        # Calculate angle in radians using atan2
        # atan2(y, x) gives angle from positive X-axis
        #angle_rad = np.arctan2(y_data, x_data)
        angle_rad = np.arctan2( x_data,y_data)
        
        # Convert to degrees
        angle_deg = np.degrees(angle_rad)
        
        # Rotate coordinate system:
        # Standard atan2: 0° is at +X axis (East)
        # We want: 0° is at +Y axis (North)
        # So we rotate by -90 degrees
        dir_deg = angle_deg + 0
        
        # Normalize to 0-360 range
        dir_deg = dir_deg % 360
        
        Val.append(val)
        Dir.append(dir_deg)
    
    # Call the polar plot function
    return PolarPlot(Val, Dir, options)


def Plot_XY(X, Y, options):
    """
    Create a standard X-Y plot with multiple datasets
    
    Args:
        X: List of X coordinate arrays
        Y: List of Y coordinate arrays
        options: Object containing plot configuration (same as PolarPlot)
    
    Returns:
        Figure object that can be saved as PNG
    """
    # Validate inputs
    if len(X) != len(Y):
        print("Error: X and Y must have same number of datasets")
        return None
    
    # Check each dataset has matching lengths
    for i in range(len(X)):
        if len(X[i]) != len(Y[i]):
            print(f"Error: X[{i}] and Y[{i}] must have same length")
            return None
    
    # Set default options if not provided (same as PolarPlot)
    if not hasattr(options, 'PlotLegend') or options.PlotLegend is None:
        options.PlotLegend = [f'Data{i+1}' for i in range(len(X))]
    
    if not hasattr(options, 'LineColor') or options.LineColor is None:
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        options.LineColor = [colors[i % len(colors)] for i in range(len(X))]
    
    if not hasattr(options, 'LineStyle') or options.LineStyle is None:
        options.LineStyle = ['-' for _ in range(len(X))]
    
    if not hasattr(options, 'LineWidth') or options.LineWidth is None:
        options.LineWidth = [2.0 for _ in range(len(X))]
    
    if not hasattr(options, 'LineMarkerColor') or options.LineMarkerColor is None:
        options.LineMarkerColor = options.LineColor
    
    if not hasattr(options, 'LineMarkerStyle') or options.LineMarkerStyle is None:
        options.LineMarkerStyle = ['None' for _ in range(len(X))]
    
    if not hasattr(options, 'LineMarkerSize') or options.LineMarkerSize is None:
        options.LineMarkerSize = [6 for _ in range(len(X))]
    
    if not hasattr(options, 'AxesLimit'):
        options.AxesLimit = 'auto'
    
    if not hasattr(options, 'NameLbl'):
        options.NameLbl = 'Graph Name'
    
    if not hasattr(options, 'ValLbl'):
        options.ValLbl = 'Y Values'
    
    # Grid options
    if not hasattr(options, 'GridAlpha'):
        options.GridAlpha = 0.3
    
    if not hasattr(options, 'GridLineWidth'):
        options.GridLineWidth = 1.0
    
    if not hasattr(options, 'GridLineStyle'):
        options.GridLineStyle = '-'
    
    if not hasattr(options, 'GridColor'):
        options.GridColor = 'gray'
    
    if not hasattr(options, 'FigureSize'):
        options.FigureSize = (8, 8)  # Default size as tuple
    
    # Legend position option
    if not hasattr(options, 'LegendLocation'):
        options.LegendLocation = 'upper right'  # Default to upper right corner
    
    # Event time bars options
    if not hasattr(options, 'EventTime'):
        options.EventTime = None  # List of event timestamps
    
    if not hasattr(options, 'EventType'):
        options.EventType = None  # List of event types corresponding to timestamps
    
    if not hasattr(options, 'EventColors'):
        options.EventColors = None  # Dictionary mapping event type to color

        #Font size , etc
    if not hasattr(options, 'TitleFontSize'):
        options.TitleFontSize = 20
    if not hasattr(options, 'LabelFontSize'):
        options.LabelFontSize = 14        
    if not hasattr(options, 'TickFontSize'):
        options.TickFontSize = 10
    if not hasattr(options, 'TickFontWeight'):
        options.TickFontWeight = 'normal'

    # Custom Y-axis grid and tick options
    if not hasattr(options, 'YGridInterval'):
        options.YGridInterval = None  # Interval for horizontal grid lines (None = auto)
    
    if not hasattr(options, 'YTickInterval'):
        options.YTickInterval = None  # Interval for Y-axis tick labels (None = auto)


    # Create figure - square aspect ratio
    fig = plt.figure(figsize=options.FigureSize)
    ax = fig.add_subplot(111)
    
    # Make it square
    #ax.set_aspect('equal', adjustable='box')
    
    # Plot each dataset
    max_y = -np.inf
    min_y = np.inf
    max_x = -np.inf
    min_x = np.inf
    
    for i in range(len(X)):
        x_data = np.array(X[i])
        y_data = np.array(Y[i])
        
        # Update limits for axis scaling
        if len(x_data) > 0:
            max_x = max(max_x, np.max(x_data))
            min_x = min(min_x, np.min(x_data))
            max_y = max(max_y, np.max(y_data))
            min_y = min(min_y, np.min(y_data))
        
        # Map line style strings
        line_style = options.LineStyle[i] if i < len(options.LineStyle) else '-'
        if line_style == 'solid':
            line_style = '-'
        elif line_style == 'dash':
            line_style = '--'
        elif line_style == 'invisible':
            line_style = 'None'
        
        # Map marker style strings
        marker_style = options.LineMarkerStyle[i] if i < len(options.LineMarkerStyle) else 'None'
        if marker_style == 'round':
            marker_style = 'o'
        elif marker_style == 'square':
            marker_style = 's'
        elif marker_style == 'triangle':
            marker_style = '^'
        elif marker_style == 'invisible':
            marker_style = 'None'
        
        # Get other properties
        color = options.LineColor[i] if i < len(options.LineColor) else 'blue'
        linewidth = options.LineWidth[i] if i < len(options.LineWidth) else 2.0
        marker_color = options.LineMarkerColor[i] if i < len(options.LineMarkerColor) else color
        marker_size = options.LineMarkerSize[i] if i < len(options.LineMarkerSize) else 6
        label = options.PlotLegend[i] if i < len(options.PlotLegend) else f'Data{i+1}'
        
        # Plot the data
        ax.plot(x_data, y_data,
                linestyle=line_style,
                linewidth=linewidth,
                color=color,
                marker=marker_style,
                markersize=marker_size,
                markerfacecolor=marker_color,
                markeredgecolor=marker_color,
                label=label)
    

    # Set axis limits
    # X-axis always auto
    #x_margin = (max_x - min_x) * 0.1  # 10% margin
    #ax.set_xlim(min_x - x_margin, max_x + x_margin)

    x_margin = (max_x - min_x) * 0.05  # Still 1600 seconds
    x_min_limit = max(0, min_x - x_margin) if min_x >= 0 else min_x - x_margin
    ax.set_xlim(x_min_limit, max_x + x_margin)  # (0, 17600) - no negative time!
    
    # Y-axis uses AxesLimit if specified
    if options.AxesLimit != 'auto':
        ax.set_ylim(-options.AxesLimit, options.AxesLimit)
    else:
        y_margin = (max_y - min_y) * 0.1  # 10% margin
        ax.set_ylim(min_y - y_margin, max_y + y_margin)
    
    # Add event time bars if provided
    if options.EventTime is not None and options.EventType is not None and options.EventColors is not None:
        # Default color map for event types if not all types are in EventColors
        default_colors = {
            0: 'yellow',    # PAUSE
            1: 'lightblue', # SAMPLE
            2: 'lightgreen',# TRANSFER
            3: 'lightcoral',# FINISHED
            4: 'green',     # EXT_OK
            5: 'orange',    # EXT_PAUSE
            6: 'red',       # EXT_FAILED
            7: 'lightgray'  # UNDEF
        }
        
        # Merge user-provided colors with defaults
        event_colors = {**default_colors, **options.EventColors}
        
        # Draw vertical bars for each event period
        for i in range(len(options.EventTime)):
            # Start time of current event
            t_start = options.EventTime[i]
            
            # End time is the next event's start time, or the plot's x-axis max
            if i < len(options.EventTime) - 1:
                t_end = options.EventTime[i + 1]
            else:
                # For the last event, extend to the end of the plot
                t_end = max_x + x_margin
            
            # Get event type and corresponding color
            event_type = options.EventType[i]
            color = event_colors.get(event_type, 'lightgray')
            
            # Draw vertical span for this event period
            ax.axvspan(t_start, t_end, alpha=0.3, color=color, zorder=0)
    
    # Add title and labels
    ax.set_title(options.NameLbl, fontsize=options.TitleFontSize, pad=20)
    ax.set_xlabel('Time', fontsize=12)
    #ax.set_ylabel(options.ValLbl, fontsize=12)

    ax.set_ylabel(options.ValLbl, 
                 fontsize=options.LabelFontSize,
                 labelpad=30)

    
    # Configure grid (default or custom)
    y_min, y_max = ax.get_ylim()
    
    # Handle custom Y-axis grid and tick intervals
    if options.YGridInterval is not None or options.YTickInterval is not None:
        # Turn off default grid first
        ax.grid(False)
        
        # Determine grid and tick positions
        if options.YGridInterval is not None:
            y_grid_ticks = np.arange(
                np.floor(y_min / options.YGridInterval) * options.YGridInterval,
                np.ceil(y_max / options.YGridInterval) * options.YGridInterval + options.YGridInterval,
                options.YGridInterval
            )
        
        if options.YTickInterval is not None:
            y_tick_positions = np.arange(
                np.floor(y_min / options.YTickInterval) * options.YTickInterval,
                np.ceil(y_max / options.YTickInterval) * options.YTickInterval + options.YTickInterval,
                options.YTickInterval
            )
        
        # Case 1: Both grid and tick intervals specified
        if options.YGridInterval is not None and options.YTickInterval is not None:
            # Set major ticks (with labels) at YTickInterval
            ax.set_yticks(y_tick_positions, minor=False)
            ax.set_yticklabels([f'{tick:.1f}' if tick % 1 else f'{int(tick)}' for tick in y_tick_positions])
            
            # For grid: use ALL positions from YGridInterval
            # Draw grid at all YGridInterval positions
            ax.set_yticks(y_grid_ticks, minor=True)
            ax.yaxis.grid(True, which='minor', 
                         alpha=options.GridAlpha,
                         linewidth=options.GridLineWidth,
                         linestyle=options.GridLineStyle,
                         color=options.GridColor)
            # Also enable major grid so lines appear at label positions too
            ax.yaxis.grid(True, which='major', 
                         alpha=options.GridAlpha,
                         linewidth=options.GridLineWidth,
                         linestyle=options.GridLineStyle,
                         color=options.GridColor)
        
        # Case 2: Only grid interval specified
        elif options.YGridInterval is not None:
            ax.set_yticks(y_grid_ticks, minor=False)
            ax.set_yticklabels([f'{tick:.1f}' if tick % 1 else f'{int(tick)}' for tick in y_grid_ticks])
            ax.yaxis.grid(True, which='major', 
                         alpha=options.GridAlpha,
                         linewidth=options.GridLineWidth,
                         linestyle=options.GridLineStyle,
                         color=options.GridColor)
        
        # Case 3: Only tick interval specified
        else:  # options.YTickInterval is not None
            ax.set_yticks(y_tick_positions, minor=False)
            ax.set_yticklabels([f'{tick:.1f}' if tick % 1 else f'{int(tick)}' for tick in y_tick_positions])
            ax.yaxis.grid(True, which='major', 
                         alpha=options.GridAlpha,
                         linewidth=options.GridLineWidth,
                         linestyle=options.GridLineStyle,
                         color=options.GridColor)
        
        # IMPORTANT: Restore the original axis limits after setting ticks
        # set_yticks can sometimes change the limits, so we force them back
        ax.set_ylim(y_min, y_max)
    else:
        # Use default grid behavior
        ax.grid(True, 
                alpha=options.GridAlpha,
                linewidth=options.GridLineWidth,
                linestyle=options.GridLineStyle,
                color=options.GridColor)
    
    # Add legend if labels provided
    if options.PlotLegend and len(options.PlotLegend) > 0:
        ax.legend(loc=options.LegendLocation)
    
    # Add zero lines for reference (optional - can be controlled by option)
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    plt.ion()
    
    return fig

# Example usage
if __name__ == "__main__":

    class PlotOptions:
        pass
    
    options = PlotOptions()
    options.PlotLegend = ['Sensor 1', 'Sensor 2']
    options.LineColor = ['blue', 'red']
    options.LineStyle = ['solid', 'solid']
    options.LineWidth = [2, 3]
    options.LineMarkerColor = ['blue', 'red']
    options.LineMarkerStyle = ['round', 'square']
    options.LineMarkerSize = [8, 6]
    options.AxesLimit = 100
    options.NameLbl = 'Polar Data Plot'
    options.ValLbl = 'Magnitude'
    
    # GRID OPTIONS 
    options.GridAlpha = 0.1  # Higher value = more visible (0-1)
    options.GridLineWidth = 2.0  # Thicker lines
    options.GridLineStyle = '-'  # Solid lines (or '--' for dashed, ':' for dotted)
    options.GridColor = 'black'  # Or 'darkgray', 'navy', etc.
    options.RadialGridLines = 12  # More radial lines (every 30 degrees)
    options.ConcentricGridLines = 8  # More concentric circles
    
    # Or for a subtle grid:
    # options.GridAlpha = 0.2
    # options.GridLineWidth = 0.5
    # options.GridColor = 'lightgray'

    # Create sample data
    t = np.linspace(0, 2*np.pi, 100)
    # Circle 1
    x1 = 50 * np.cos(t)
    y1 = 50 * np.sin(t)
    # Circle 2 (offset)
    x2 = 30 * np.cos(t) + 20
    y2 = 30 * np.sin(t) + 10

    X = [x1, x2]
    Y = [y1, y2]

# This will work correctly
    fig = PolarPlot_XY(X, Y, options)
    fig.show()
    
    if fig:
        # Save as PNG
        fig.savefig('polar_plot.png', dpi=300, bbox_inches='tight')
        # Show plot
        plt.show()
