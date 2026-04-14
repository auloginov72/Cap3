import numpy as np
from Utils.ppform import pp_linear, pp_const, pp_cubic, ppval, resample
from Utils.RotateCoord import RotateCoord,TO_180_180,TO_EVEN_HDG
from Utils.PolarPlot import PolarPlot_XY, PolarPlot ,Plot_XY

class PlotOptions:
    pass

def ProcessCapPlotDataManual(DATA_SET):
    
    open_figures = []
    CommonDTime=4
    SampleTime=DATA_SET['ALRM']['SMPL'].get('Time')
    SampleType=DATA_SET['ALRM']['SMPL'].get('Type')
    SampleK=DATA_SET['ALRM']['SMPL'].get('K')
           # 'Time' 'Type'  'N'   'K'   
           # Type=0 PRGState::PAUSE
           # Type=1 PRGState::SAMPLE
           # Type=2 PRGState::TRANSFER
           # Type=3 PRGState::FINISHED
           # Type=4 EXT_OK,       // All fine in DP 
           # Type=5 EXT_PAUSE,     // Wait env change command from DP  
	       # Type=6 EXT_FAILED      // Control Reports Failed     


    TimeGGA0=  DATA_SET['POS']['GGA0'].get('Time')
    Xg0=      DATA_SET['POS']['GGA0'].get('Xg[0]')
    Yg0=      DATA_SET['POS']['GGA0'].get('Yg[0]')

    Time_Dez_XYg=    DATA_SET['CTRL']['c_reg_XY'].get('Time')
    Des_Xg      =   DATA_SET['CTRL']['c_reg_XY'].get('DesXg')
    Des_Yg      =   DATA_SET['CTRL']['c_reg_XY'].get('DesYg')

    Time_Dez_HDG=    DATA_SET['CTRL']['c_reg_K'].get('Time')
    Des_Dez_HDG =    DATA_SET['CTRL']['c_reg_K'].get('curDesK')

    TimeGyro0 =  DATA_SET['HDG']['gyro0'].get('Time')
    Course0 =    DATA_SET['HDG']['gyro0'].get('Course[0]')

    TimeWind0 = DATA_SET['SNS']['wind0'].get('Time')
    WindDir0 = DATA_SET['SNS']['wind0'].get('RawDir0')

    T_Power=DATA_SET['CTRL']['pms_in'].get('Time')
    Gen0=DATA_SET['CTRL']['pms_in'].get('PwrG0')
    Gen1=DATA_SET['CTRL']['pms_in'].get('PwrG1')
    Gen2=DATA_SET['CTRL']['pms_in'].get('PwrG2')
    Gen3=DATA_SET['CTRL']['pms_in'].get('PwrG3')


    T_new = np.arange(TimeGyro0[0], TimeGyro0[-1] , CommonDTime)
    Xg0=resample(TimeGGA0,Xg0 , T_new)
    Yg0=resample(TimeGGA0,Yg0 , T_new)
    Des_Xg=resample(Time_Dez_XYg, Des_Xg,T_new)
    Des_Yg=resample(Time_Dez_XYg, Des_Yg,T_new)
    dx=Des_Xg[1]
    dy=Des_Yg[1]
    Des_Xg=Des_Xg-dx ; Des_Yg=Des_Yg-dy
    Xg0=Xg0-dx ;  Yg0=Yg0-dy

    Course0=resample(TimeGyro0,Course0, T_new, 'linear', True)
    Des_Dez_HDG=resample(Time_Dez_HDG,Des_Dez_HDG ,T_new, 'linear', True)
    Course0-=(Course0[0]-Des_Dez_HDG[0])-TO_180_180(Course0[0]-Des_Dez_HDG[0])
    D_Course=TO_180_180(Course0-Des_Dez_HDG)
    WindDir0=resample(TimeWind0,WindDir0 ,T_new, 'linear', True)
    WindDir0TRue= TO_EVEN_HDG(  TO_180_180( Course0-WindDir0))

    Gen0=resample(T_Power, Gen0,T_new)
    Gen1=resample(T_Power, Gen1,T_new)
    Gen2=resample(T_Power, Gen2,T_new)
    Gen3=resample(T_Power, Gen3,T_new)
    
    Rg_90_Sample, K_90_Sample =sample_percentile(np.hypot((Xg0-Des_Xg),(Yg0-Des_Yg)), T_new, SampleTime, SampleType, SampleK, 
                      [1], 90)

    Rg_90_Transf, K_90_Transf =sample_percentile(np.hypot((Xg0-Des_Xg),(Yg0-Des_Yg)), T_new, SampleTime, SampleType, SampleK, 
                      [2], 90)


    Rg_100_Sample, K_100_Sample =sample_percentile(np.hypot((Xg0-Des_Xg),(Yg0-Des_Yg)), T_new, SampleTime, SampleType, SampleK, 
                      [1], 100)

    Rg_100_SampleDev, K_100_Sample =sample_percentile(np.hypot((Xg0-Des_Xg),(Yg0-Des_Yg)), T_new, SampleTime, SampleType, SampleK, 
                      [1], 100)


    Dev_K_100_Sample, K_100_Sample =sample_percentile(TO_180_180( abs(D_Course)  ), T_new, SampleTime, SampleType, SampleK, 
                      [1], 100)

    #Make a tabulated max deviation 
    T_SampleK_sample, SampleK_sample= GetPhaseData(SampleTime, SampleK, SampleTime, SampleType, 1)
    print("Isample   Kdes   MaxK_Dev")
    for i in range(len(Rg_100_Sample)):
        print(f"{i:2.0f}, {SampleK_sample[i]:4.1f}, {Dev_K_100_Sample[i]:10.2f}")
    print("===================")
    print("Isample   Kdes   MaxPos_Dev")
    for i in range(len(Rg_100_Sample)):
        print(f"{i:2.0f}, {SampleK_sample[i]:4.1f}, {Rg_100_Sample[i]:10.2f}")

    T_XYg0SAMPL,Xg0SAMPL=          GetPhaseData(T_new, Xg0, SampleTime, SampleType, 1) # samples
    T_XYg0SAMPL,Yg0SAMPL=          GetPhaseData(T_new, Yg0, SampleTime, SampleType, 1) # samples
    T_Des_XYgSAMPL,Des_XgSAMPL=    GetPhaseData(T_new, Des_Xg, SampleTime, SampleType, 1) # samples
    T_Des_XYgSAMPL,Des_YgSAMPL=    GetPhaseData(T_new, Des_Yg, SampleTime, SampleType, 1) # samples
    T_Dez_HDG_SAMPL,Dez_HDG_SAMPL= GetPhaseData(T_new, Des_Dez_HDG, SampleTime, SampleType, 1) # samples
    T_HDG_SAMPL,HDG_SAMPL= GetPhaseData(T_new, Course0, SampleTime, SampleType, 1) # samples

    T_XYg0TRANSF,Xg0TRANSF=          GetPhaseData(T_new, Xg0, SampleTime, SampleType, 2) # transfer
    T_XYg0TRANSF,Yg0TRANSF=          GetPhaseData(T_new, Yg0, SampleTime, SampleType, 2) # transfer
    T_Des_XYgTRANSF,Des_XgTRANSF=    GetPhaseData(T_new, Des_Xg, SampleTime, SampleType, 2) # transfer
    T_Des_XYgTRANSF,Des_YgTRANSF=    GetPhaseData(T_new, Des_Yg, SampleTime, SampleType, 2) # transfer
    T_Dez_HDG_TRANSF,Dez_HDG_TRANSF= GetPhaseData(T_new, Des_Dez_HDG, SampleTime, SampleType, 2) # transfer
    T_HDG_TRANSF,HDG_TRANSF= GetPhaseData(T_new, Course0, SampleTime, SampleType, 2) 

#===============PLOTS =================================================
#  POLAR deviations (Footprint)
    options = PlotOptions()
    options=PlotOptionsDef(2)
    options.NameLbl = 'FootPrint'
    options.PlotLegend = ['SAMPLE', 'Transfer']
    options.LineColor = ['blue' , 'red']
    options.LineStyle = ['None', 'None']                #['-']       #['None']
    options.LineMarkerSize = [2.5 , 2.5]
    options.FigureSize = (10, 8)
    
    options.AxesLimit=3
    options.ConcentricGridLines = 3
    options.TickPositionDirection = 45
    options.TickFontSize = 10

    options.TickLabelZOrder = 10  # Labels appear above data

    options.IndicateOutOfAreaData = True
    options.OutOfAreaIndicatorColor = 'magenta'
    options.OutOfAreaIndicatorWidth = 5.0
    options.OutOfAreaSectorResolution = 1  # 15-degree sectors
    


    fig = PolarPlot_XY([ (Yg0SAMPL-Des_YgSAMPL),(Yg0TRANSF-Des_YgTRANSF)   ] , [(Xg0SAMPL-Des_XgSAMPL) , (Xg0TRANSF-Des_XgTRANSF) ], options)
    fig.savefig('__polar_plot.png', dpi=300, bbox_inches='tight')
    fig.show()
    open_figures.append(fig)


#  POLAR exceedance 90
    SkipExeedance=True 
    if not SkipExeedance:
        options = PlotOptions()
        options=PlotOptionsDef(2)
        options.NameLbl = '90% Non-Exceedance Position Deviation vs. Environmental Direction'  #
        options.PlotLegend = ['SAMPLE', 'Transfer']
        options.LineColor = ['blue' , 'red']
        options.LineStyle = ['None', 'None']                #['-']       #['None']
        options.LineMarkerSize = [8.5 , 2.5]
        options.FigureSize = (10, 8)
        options.AxesLimit=10
        options.ConcentricGridLines = 4
        options.TickPositionDirection = 45        
        options.EventColors = {
                0: 'none',    # PAUSE
                1: 'lightgreen', # SAMPLE lightblue
                2: 'lightgreen',# TRANSFER
                3: 'none',# FINISHED
                4: 'green',     # EXT_OK
                5: 'orange',    # EXT_PAUSE
                6: 'red',       # EXT_FAILED
                7: 'lightgray'  # UNDEF
        }
        """
        SampleK_=np.array(SampleK)
        SampleType_=np.array(SampleType)
        ind = np.argsort(SampleK_ )  
        SampleK_=SampleK_[ind]
        SampleType_=SampleType_[ind]
        SampleK_=np.append(SampleK_, [180] )
        SampleType_ =np.append(SampleType_ , [0] )
        """
        options.EventFi=np.array(   [ 0, 20,     160 , 180 ] )  # just manual input to color correct events
        options.EventType =np.array([ 1,  6,     1,    0]  )    #[ 1,  6,    1,    0] 

        options.EventFi=np.array(   [ 0,  180 ] )  # just manual input to color correct events
        options.EventType =np.array([ 1,    0]  )    #[ 1,  6,    1,    0] 


        Valarr =[ Rg_90_Sample, Rg_90_Transf]
        DirArr =[K_90_Sample, K_90_Transf ]  #np.nan
        fig = PolarPlot( Valarr, DirArr, options)
        fig.savefig('__polar_plot_Ecxeed90.png', dpi=300, bbox_inches='tight')
        fig.show()
        open_figures.append(fig)


#  Xg, Yg
    options=PlotOptionsDef(2)
    options.NameLbl = 'Position Deviation (N)'
    options.PlotLegend = ['Position [m] ', 'Des Position [m]']
    options.LineStyle =  ['-', '-'] 
    options.LineWidth =  [1, 2]
    options.LineMarkerStyle = [ 'round' , 'invisible']
    options.FigureSize = (10, 4)
    options.EventTime = SampleTime  
    options.EventType = SampleType  
    
    TimeArr=[T_new, T_new]
    DataArr=[Xg0, Des_Xg]
    fig = Plot_XY(TimeArr , DataArr , options)
    fig.savefig('__Xg.png', dpi=300, bbox_inches='tight')
    fig.show()
    open_figures.append(fig)

    options.NameLbl = 'Position Deviation (E)'
    DataArr=[Yg0, Des_Yg]
    fig = Plot_XY(TimeArr , DataArr , options)
    fig.savefig('__Yg.png', dpi=300, bbox_inches='tight')
    fig.show()
    open_figures.append(fig)
#  Heading
    options.NameLbl = 'Heading Deviation'
    options.PlotLegend = ['Heading', 'Des Heading']
    DataArr=[Course0, Des_Dez_HDG]
    fig = Plot_XY(TimeArr , DataArr , options)
    fig.savefig('__HDG.png', dpi=300, bbox_inches='tight')
    fig.show()
    open_figures.append(fig)

#  Wind    
    options=PlotOptionsDef(1)
    options.NameLbl = 'True Wind direction'
    options.PlotLegend = [ 'True Wind Dir ']
    options.LineStyle =  ['-'] 
    options.LineWidth =  [  2]
    options.LineMarkerStyle = [  'invisible']
    options.FigureSize = (10, 4)
    options.EventTime = SampleTime  
    options.EventType = SampleType  
    options.YGridInterval = 2   #None
    options.YTickInterval = 10
    
    TimeArr=[ SampleTime]
    DataArr=[ SampleK]  
    fig = Plot_XY(TimeArr , DataArr , options)
    fig.savefig('__Wind.png', dpi=300, bbox_inches='tight')
    fig.show()
    open_figures.append(fig)

   
# PMS
    options=PlotOptionsDef(2)
    options.NameLbl = 'Power [kW]'
    options.PlotLegend = ['BUS#1' , 'BUS#2']
    options.LineStyle =  ['-']*2 
    options.LineWidth =  [ 1 , 1]
    options.LineMarkerStyle = [  'invisible']*2
    options.FigureSize = (10, 4)
    options.EventTime = SampleTime  
    options.EventType = SampleType  
    TimeArr=[T_new, T_new]
    DataArr=[Gen0+Gen2, Gen1+Gen3]
    fig = Plot_XY(TimeArr , DataArr , options)
    fig.savefig('__PMS.png', dpi=300, bbox_inches='tight')
    fig.show()
    open_figures.append(fig)


    #PP_AXg=pp_linear(time, AXg)
    #AXg=ppval(PP_AXg, T_new)
    #Ax, Ay = RotateCoord(AXg, AYg, np.deg2rad(Hdg0), "glob2loc")   #"loc2glob"

   
    return  open_figures


def PlotOptionsDef(NCurves=2):
    options = PlotOptions()
    options.FigureSize = (10, 8)
    options.PlotLegend = [f'Sample{i+1}' for i in range(NCurves)]
    color_palette = ['blue', 'red', 'green', 'orange', 'purple', 
                     'brown', 'pink', 'gray', 'olive', 'cyan']
    # Cycle through colors if more than 10 curves
    options.LineColor = [color_palette[i % len(color_palette)] for i in range(NCurves)]
    options.LineMarkerColor = options.LineColor.copy()
    
    options.LineStyle = ['None'] * NCurves
    
    # Line width - can vary slightly for distinction
    options.LineWidth = [2  for i in range(NCurves)]  # 2, 2.5, 3, 2, 2.5...
    
    # Marker styles - cycle through different markers
    marker_styles = ['round', 'square', 'triangle', 'star', 'pentagon', 
                     'hexagon', 'plus', 'x', 'diamond', 'thin_diamond']
    options.LineMarkerStyle = [marker_styles[0] for i in range(NCurves)]
    
    # Marker sizes - slight variation
    options.LineMarkerSize = [2.5  for i in range(NCurves)]  # 6, 8, 10, 6, 8...
    
    # Axes and labels
    options.AxesLimit = 'auto'
    options.NameLbl = 'Title Name'
    options.ValLbl = ''
    
    # Grid options
    options.GridAlpha = 0.1
    options.GridLineWidth = 2.0
    options.GridLineStyle = '-'
    options.GridColor = 'black'
    options.RadialGridLines = 12
    options.ConcentricGridLines = 8
    
    # Font options
    options.TitleFontSize = 12
    options.LabelFontSize = 10
    options.TickFontSize = 8
    options.TickFontWeight = '900'  # 900 might be too bold for ticks 'normal'
    options.LegendLocation='best'
    """
    'upper right' (default)
    'upper left'
    'lower right'
    'lower left'
    'upper center'
    'lower center'
    'center left'
    'center right'
    'center'
    'best' (matplotlib will choose the best location)
    """
    options.EventTime = None
    options.EventType = None  # Your list of event types
    options.EventColors = {
            0: 'none',    # PAUSE
            1: 'lightblue', # SAMPLE lightblue
            2: 'lightgreen',# TRANSFER
            3: 'none',# FINISHED
            4: 'green',     # EXT_OK
            5: 'orange',    # EXT_PAUSE
            6: 'red',       # EXT_FAILED
            7: 'lightgray'  # UNDEF
    }

    options.YGridInterval = None
    options.YTickInterval = None

    return options


def GetPhaseData(T, X, SampleTime, SampleType, phase_type):
    """
    Generic function to extract data for any phase type
    
    Args:
        phase_type: 0=PAUSE, 1=SAMPLE, 2=TRANSFER, 3=FINISHED
    """
    T = np.array(T)
    X = np.array(X)
    SampleTime = np.array(SampleTime)
    SampleType = np.array(SampleType)
    
    T_phase = []
    X_phase = []
    
    # Find all starts of the requested phase
    phase_starts = np.where(SampleType == phase_type)[0]
    
    for start_idx in phase_starts:
        start_time = SampleTime[start_idx]
        
        # Find the end of this phase (next event that's different)
        end_idx = start_idx + 1
        while end_idx < len(SampleType) and SampleType[end_idx] == phase_type:
            end_idx += 1
        
        if end_idx < len(SampleTime):
            end_time = SampleTime[end_idx]
        else:
            end_time = T[-1]
        
        # Extract data points in this time range
        mask = (T >= start_time) & (T < end_time)
        T_phase.extend(T[mask])
        X_phase.extend(X[mask])
    
    return np.array(T_phase), np.array(X_phase)


def sample_percentile(X, T, sample_times, sample_types, sample_K, 
                      types_to_include, percent_exceedance):
    """
    Calculate percentile exceedance of X within specified sample types.
    
    Parameters:
    -----------
    X : array-like
        Time series data values
    T : array-like
        Time points corresponding to X
    sample_times : array-like
        Start times of each sample period
    sample_types : array-like
        Type of each sample (0=PAUSE, 1=SAMPLE, 2=TRANSFER, 3=FINISHED)
    sample_K : array-like
        K values for each sample period
    types_to_include : list of int
        Sample types to include in calculation (e.g., [1, 2])
    percent_exceedance : float
        Percentile exceedance threshold (e.g., 95 for 95th percentile)
    
    Returns:
    --------
    X_exceed : array-like
        Exceedance values at each sample time
    K_exceed : array-like
        Average K values for each sample period
    """
    n_samples = len(sample_times)
    X_exceed = np.zeros(n_samples)
    K_exceed = np.zeros(n_samples)
    
    for i in range(n_samples):
        # Determine time window for this sample
        t_start = sample_times[i]
        t_end = sample_times[i + 1] if i < n_samples - 1 else T[-1]
        
        # Check if this sample type should be included
        if sample_types[i] in types_to_include:
            # Find X values within this time window
            mask = (T >= t_start) & (T < t_end)
            X_window = X[mask]
            
            #print(f"Sample {i}: t_start={t_start:.2f}, t_end={t_end:.2f}")
            #print(f"  Found {len(X_window)} points")
            
            # Calculate percentile exceedance
            if len(X_window) > 0:
                #print(f"  X range: [{np.min(X_window):.2f}, {np.max(X_window):.2f}]")
                X_exceed[i] = np.percentile(X_window, percent_exceedance)
                #print(f"  {percent_exceedance}th percentile: {X_exceed[i]:.2f}")
                #print(f"  K_exceed: {sample_K[i]:.2f}")
            else:
                X_exceed[i] = np.nan
        else:
            X_exceed[i] = np.nan
        
        # Average K between current and next sample (or use current for last)
        if i < n_samples - 1:
            K_exceed[i] = (sample_K[i] + sample_K[i + 1]) / 2
        else:
            K_exceed[i] = sample_K[i]

    indnan = ~np.isnan(X_exceed)
    X_exceed=X_exceed[indnan]
    K_exceed=K_exceed[indnan]    
    return X_exceed, K_exceed

"""
def GetSamples( Val ,Dir, options):
    #Dir[i] and Val[i] represents data set  
    if len(Time) != len(Val) 
        return None  # error
    options.PlotLegend = None # ['Data1' , 'Data2'  ] #   len(options.PlotLegend) should be 0 or =  len(Val)
    options.LineColor = None # ['???' , '??'  ] #    len(options.LineColor) should be 0 or =  len(Val)
    options.LineStyle = .... # solid, dash. invisible
    options.LineWidth = .... # value???
    options.LineMarkerColor = None # ['???' , '??'  ] #  
    options.LineMarkerStyle = .... # round, square, triangle, invisible
    options.LineMarkerSize = .... # value????

    options.AxesLimit = 100 # either a value or 'auto' for automatic scale
    options.NameLbl ='Graph Name'
    options.ValLbl ='Name of Values'

    # we need to make a polar plot which consits of all pairs of Val and Dir with legend and lables and appropriate drawing colors and style
    # Essential that further PLOT handle must be used for making a png image of plot figure 

    PLOT =None
    return
"""