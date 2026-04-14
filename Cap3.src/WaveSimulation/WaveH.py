#!/usr/bin/env python3
"""
Simple wave height simulator based on P_WAVE.CPP and WAVE.CPP
Implements stochastic (irregular) wave generation for Sea State 5
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
G_ACC = 9.81  # m/s^2
M_PI = np.pi
DEG_RAD = M_PI / 180.0
RAD_DEG = 180.0 / M_PI

# =============================================================================
# SEA STATE 5 PARAMETERS (from GUMS-53 table)
# =============================================================================
# From P_WAVE.CPP lines 48-49 for type_wave_calc=2 (not ==2, just =2)
h1_3 = 4.      # Significant wave height [m]
T=7.5
#a_w_ = 1.0472    # Frequency [rad/s]
a_w_=2*M_PI/T
a_w_max = a_w_ * 0.77  # Peak frequency
h3 = h1_3 / 0.7575     # Mean wave height

# =============================================================================
# SPECTRUM CALCULATION (from P_WAVE.CPP lines 120-143)
# =============================================================================
def calculate_spectrum_parameters(h3, sigma_max, angle=45.0*DEG_RAD):
    """
    Calculate spectrum and extract filter parameters
    Based on prp_stoch_wave() function
    """
    # Spectral moments (line 123)
    m_01 = 0.143 * (0.5 * h3)**2
    
    # Peak frequencies for two-peak spectrum (lines 124-130)
    sigma_m1 = sigma_max
    tau_max = 2.0 * M_PI / sigma_max
    tau_ = 0.77 * tau_max
    sigma_1_ = 2.0 * M_PI / tau_
    m_02 = 0.12 * m_01
    sigma_2_ = 0.82 * sigma_m1
    sigma_m2 = 0.7 * sigma_m1
    
    # Build frequency array (line 105)
    MAX_SIGMA = 100
    a_sigma = np.linspace(0.02, 4.0, MAX_SIGMA)
    
    # Calculate two-peak spectrum for each frequency (lines 136-143)
    Sr = np.zeros(MAX_SIGMA)
    for i, sigma in enumerate(a_sigma):
        # First peak
        x = sigma_m1 / sigma
        coeff = 9.43 * m_01 / sigma_1_
        Sr1 = coeff * x**6 * np.exp(-1.5 * x**4)
        
        # Second peak
        x = sigma_m2 / sigma
        coeff = 17.29 * m_02 / sigma_2_
        Sr2 = coeff * x**8 * np.exp(-2.0 * x**4)
        
        Sr[i] = Sr1 + Sr2
    
    # Convert to wave slope spectrum (line 148)
    k = a_sigma**2 / G_ACC
    Sgamma = k**2 * Sr
    
    # Ship-specific reduction factors (simplified for demo)
    # Lines 150-186: kappa_T, kappa1_kr, kappa_x
    # For simplicity, we'll use constant reduction of 0.7
    kappa_combined = 0.7
    Sgamma_teta = Sgamma * kappa_combined**2
    
    # Extract filter parameters using "triple method" (lines 190-249)
    # Find spectrum maximum
    i_max = np.argmax(Sgamma_teta)
    S_max = Sgamma_teta[i_max]
    S_max_2 = S_max / 2.0
    
    # Find half-power points
    i = 0
    while i < MAX_SIGMA and Sgamma_teta[i] < S_max_2:
        i += 1
    # Linear interpolation for sigma1
    if i > 0 and i < MAX_SIGMA:
        sigma1 = a_sigma[i-1] + (a_sigma[i] - a_sigma[i-1]) * \
                 (S_max_2 - Sgamma_teta[i-1]) / (Sgamma_teta[i] - Sgamma_teta[i-1])
    else:
        sigma1 = a_sigma[i] if i < MAX_SIGMA else a_sigma[i-1]
    
    i = i_max
    while i < MAX_SIGMA and Sgamma_teta[i] > S_max_2:
        i += 1
    # Linear interpolation for sigma2
    if i > 0 and i < MAX_SIGMA:
        sigma2 = a_sigma[i-1] + (a_sigma[i] - a_sigma[i-1]) * \
                 (S_max_2 - Sgamma_teta[i-1]) / (Sgamma_teta[i] - Sgamma_teta[i-1])
    else:
        sigma2 = a_sigma[i-1] if i > 0 else a_sigma[i]
    
    # Calculate filter parameters (lines 215-217)
    alfa_teta = (sigma2 - sigma1) / 2.0
    beta_teta = sigma1 + alfa_teta
    Variance_teta = S_max * alfa_teta
    
    return alfa_teta, beta_teta, Variance_teta, a_sigma, Sr, Sgamma_teta


# =============================================================================
# DISCRETE FILTER (from WAVE.CPP lines 430-480)
# =============================================================================
def build_discrete_filter(alfa, beta, step):
    """
    Build discrete state-space filter matrices
    Based on stoch_calc() function
    """
    # State matrix (lines 431-440)
    exp_alfa_h = np.exp(-alfa * step)
    cos_beta_h = np.cos(beta * step)
    sin_beta_h = np.sin(beta * step)
    alfa_beta = alfa / beta
    
    a11 = exp_alfa_h * (cos_beta_h + alfa_beta * sin_beta_h)
    a12 = exp_alfa_h * sin_beta_h / beta
    a21 = exp_alfa_h * (-alfa * alfa_beta - beta) * sin_beta_h
    a22 = exp_alfa_h * (cos_beta_h - alfa_beta * sin_beta_h)
    
    A = np.array([[a11, a12],
                  [a21, a22]])
    
    # Covariance matrix (lines 443-467)
    exp_2alfa_h = exp_alfa_h * exp_alfa_h
    cos_2beta_h = np.cos(2.0 * beta * step)
    sin_2beta_h = np.sin(2.0 * beta * step)
    Sigma_Max_2 = alfa**2 + beta**2
    
    q11 = (-alfa * Sigma_Max_2 / (beta**2)) * \
          (exp_2alfa_h / alfa + 
           exp_2alfa_h * (-alfa * cos_2beta_h + beta * sin_2beta_h) / Sigma_Max_2 -
           1.0 / alfa + alfa / Sigma_Max_2)
    
    q_ = (-alfa / beta) * \
         (exp_2alfa_h * (alfa * sin_2beta_h + beta * cos_2beta_h) - beta)
    
    q__ = -2.0 * Sigma_Max_2 * (exp_2alfa_h - 1.0) - beta**2 * q11
    
    q12 = -alfa * q11 + q_
    q22 = alfa**2 * q11 - 2.0 * alfa * q_ + q__
    
    # Input matrix via Cholesky decomposition (lines 461-467)
    l22 = np.sqrt(q22)
    l12 = q12 / l22
    l11_sq = q11 - l12**2
    l11 = np.sqrt(max(0.0, l11_sq))  # Ensure non-negative
    
    L = np.array([[l11, 0.0],
                  [l12, l22]])
    
    return A, L


# =============================================================================
# WAVE SIMULATION
# =============================================================================
def simulate_waves(duration=120.0, dt=0.1, vessel_speed=0.0, wave_angle=45.0*DEG_RAD):
    """
    Simulate irregular waves over time
    
    Parameters:
    -----------
    duration : float
        Simulation duration [seconds]
    dt : float
        Timestep [seconds]
    vessel_speed : float
        Vessel speed [m/s] (for Doppler effect)
    wave_angle : float
        Wave encounter angle [radians]
    """
    print(f"Simulating Sea State 5:")
    print(f"  h1/3 = {h1_3:.2f} m")
    print(f"  Peak period = {2*M_PI/a_w_max:.2f} s")
    print(f"  Duration = {duration:.0f} s\n")
    
    # Calculate spectrum parameters
    alfa, beta, Variance, a_sigma, Sr, Sgamma = calculate_spectrum_parameters(h3, a_w_max, wave_angle)
    print(f"Filter parameters:")
    print(f"  alfa (damping) = {alfa:.4f} rad/s")
    print(f"  beta (frequency) = {beta:.4f} rad/s")
    print(f"  Period = {2*M_PI/beta:.2f} s")
    print(f"  Variance = {Variance:.4f} m^2\n")
    
    # Doppler effect (line 426-428)
    cos_wave_angle = np.cos(wave_angle)
    k_V = np.abs(1.0 + vessel_speed * beta * cos_wave_angle / G_ACC)
    alfa *= k_V
    beta *= k_V
    
    # Build discrete filter
    A, L = build_discrete_filter(alfa, beta, dt)
    
    # Initialize state
    x = np.zeros(2)  # x_transversal[0], x_transversal[1]
    
    # Time array
    n_steps = int(duration / dt)
    time = np.linspace(0, duration, n_steps)
    
    # Storage
    wave_height = np.zeros(n_steps)
    wave_slope = np.zeros(n_steps)
    
    Sqrt_Variance = np.sqrt(Variance)
    
    # Main simulation loop (lines 473-479)
    print("Running simulation...")
    
    # Scaling factor to match h1/3
    # The filter output variance should produce h1/3 ~ 4*RMS
    # We need to find the right scaling
    scale_factor = h1_3 / (4.0 * Sqrt_Variance)
    
    for i in range(n_steps):
        # Generate white noise (line 402-403)
        noice_1 = np.random.randn()
        noice_2 = np.random.randn()
        eta = np.array([noice_1, noice_2])
        
        # Update filter state (lines 473-479)
        x_new = A @ x + Sqrt_Variance * L @ eta
        
        # Store results
        # x[0] is the filter output representing wave-related quantity
        # Scale to match target h1/3
        wave_slope[i] = x_new[0]
        wave_height[i] = x_new[0] * scale_factor
        
        # Update state
        x = x_new
    
    # Calculate statistics
    h1_3_simulated = 4.0 * np.sqrt(np.mean(wave_height**2))  # Approximate h1/3
    print(f"\nSimulation complete!")
    print(f"  Target h1/3 = {h1_3:.2f} m")
    print(f"  Simulated h1/3 = {h1_3_simulated:.2f} m")
    print(f"  RMS wave height = {np.sqrt(np.mean(wave_height**2)):.2f} m")
    
    return time, wave_height, wave_slope


# =============================================================================
# PLOTTING
# =============================================================================
def plot_waves(time, wave_height):
    """
    Plot wave height vs time
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(time, wave_height, 'b-', linewidth=0.8, label='Wave Height')
    ax.axhline(y=h1_3, color='r', linestyle='--', linewidth=2, 
               label=f'Target H1/3 = {h1_3:.2f} m')
    ax.axhline(y=-h1_3, color='r', linestyle='--', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Wave Height H [m]', fontsize=12)
    ax.set_title('Stochastic Wave Simulation - Sea State 5 (h1/3 = 2.65 m)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add statistics text box
    h1_3_sim = 4.0 * np.sqrt(np.mean(wave_height**2))
    max_height = np.max(np.abs(wave_height))
    stats_text = f'Simulated Statistics:\n'
    stats_text += f'H1/3 = {h1_3_sim:.2f} m\n'
    stats_text += f'Max |H| = {max_height:.2f} m\n'
    stats_text += f'RMS = {np.sqrt(np.mean(wave_height**2)):.2f} m'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_spectrum_comparison(time, wave_height):
    """
    Plot spectrum comparison (optional detailed analysis)
    """
    # Calculate filter parameters
    alfa, beta, Variance, a_sigma, Sr, Sgamma = \
        calculate_spectrum_parameters(h3, a_w_max)
    
    # Calculate FFT of simulated waves
    from scipy import signal
    dt = time[1] - time[0]
    f, Pxx = signal.welch(wave_height, fs=1/dt, nperseg=min(512, len(wave_height)//4))
    omega = 2 * np.pi * f
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot theoretical spectrum
    ax1.plot(a_sigma, Sgamma, 'b-', linewidth=2, label='Target Spectrum')
    ax1.axvline(x=beta, color='r', linestyle='--', linewidth=1.5, 
                label=f'Peak w = {beta:.3f} rad/s')
    ax1.set_xlabel('Frequency w [rad/s]', fontsize=11)
    ax1.set_ylabel('Spectrum S(w) [m^2*s/rad]', fontsize=11)
    ax1.set_title('Target Wave Spectrum (from P_WAVE.CPP)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim([0, 3])
    
    # Plot simulated spectrum
    ax2.plot(omega, Pxx, 'g-', linewidth=1.5, label='Simulated Spectrum')
    ax2.axvline(x=beta, color='r', linestyle='--', linewidth=1.5, 
                label=f'Peak w = {beta:.3f} rad/s')
    ax2.set_xlabel('Frequency w [rad/s]', fontsize=11)
    ax2.set_ylabel('Power Spectral Density', fontsize=11)
    ax2.set_title('Simulated Wave Spectrum', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim([0, 3])
    
    plt.tight_layout()
    return fig


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Simulation parameters
    duration = 500.0  # seconds
    dt = 0.1          # timestep
    
    # Run simulation
    time, wave_height, wave_slope = simulate_waves(
        duration=duration,
        dt=dt,
        vessel_speed=0.0,        # Stationary vessel
        wave_angle=45.0*DEG_RAD  # 45 degree wave approach
    )
    
    # Create plots
    fig1 = plot_waves(time, wave_height)
    plt.savefig('wave_height_time.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: wave_height_time.png")
    
    # Optional: detailed spectrum comparison
    fig2 = plot_spectrum_comparison(time, wave_height)
    plt.savefig('wave_spectrum_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Plot saved: wave_spectrum_comparison.png")
    
    plt.show()
