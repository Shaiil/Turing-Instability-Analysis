import numpy as np
import matplotlib.pyplot as plt

def turing_instability_comparison():
    """
    Analyzes and compares Turing instability on a bounded 2D domain for
    both Dirichlet and Periodic boundary conditions.
    
    It plots the fundamental instability threshold (continuous dispersion
    relation) and overlays the discrete, allowed eigenmodes for each
    boundary condition type to show which ones become unstable.
    """
    # --- 1. System and Domain Parameters ---
    # Reaction-diffusion parameters
    a = 2.0
    d = -3.0
    b = 3.0
    c = -4.0
    sigma_u = 0.02
    sigma_v = 1.0

    # Bounded Domain Parameters
    L = 64.0  # Domain size (e.g., grid_size * dx from simulation)
    max_mode_number = 15 # Max mode number (m or n) to check

    # --- 2. Calculate the Fundamental Instability Threshold ---
    # This continuous dispersion curve is independent of boundary conditions.
    k_continuous = np.linspace(0, 0.8, 500)
    lambda_k_continuous = k_continuous**2
    
    trace_T = a + d
    det_D = a * d - b * c
    
    # Function to calculate growth rate for any given lambda_k
    def get_growth_rate(lambda_k):
        B = (sigma_u + sigma_v) * lambda_k - trace_T
        C = (sigma_u * sigma_v) * lambda_k**2 - (d * sigma_u + a * sigma_v) * lambda_k + det_D
        discriminant = B**2 - 4 * C
        sqrt_discriminant = np.sqrt(discriminant.astype(np.complex128))
        lmbda1 = (-B + sqrt_discriminant) / 2
        lmbda2 = (-B - sqrt_discriminant) / 2
        return np.maximum(np.real(lmbda1), np.real(lmbda2))

    # Calculate the continuous curve
    growth_rate_continuous = get_growth_rate(lambda_k_continuous)
    
    # Find the instability threshold from the continuous curve
    unstable_k_indices = np.where(growth_rate_continuous > 0)[0]
    if len(unstable_k_indices) > 0:
        k_min_threshold = k_continuous[unstable_k_indices[0]]
        k_max_threshold = k_continuous[unstable_k_indices[-1]]
        print("--- Fundamental Instability Threshold ---")
        print(f"Instability occurs for wavenumbers k in the range: ({k_min_threshold:.4f}, {k_max_threshold:.4f})\n")
    else:
        print("No instability threshold found for these parameters.")

    # --- 3. Analyze Modes for Dirichlet Boundary Conditions ---
    dirichlet_modes = []
    for m in range(1, max_mode_number + 1):
        for n in range(1, max_mode_number + 1):
            # Formula for Dirichlet eigenvalues
            lambda_mn = (np.pi / L)**2 * (m**2 + n**2)
            k_mn = np.sqrt(lambda_mn)
            growth_rate = get_growth_rate(lambda_mn)
            dirichlet_modes.append({'m': m, 'n': n, 'k': k_mn, 'growth_rate': growth_rate})
            
    print("--- Analysis for Dirichlet Boundary Conditions ---")
    unstable_dirichlet = [m for m in dirichlet_modes if m['growth_rate'] > 0]
    if not unstable_dirichlet:
        print("No unstable modes found.")
    for mode in unstable_dirichlet:
        print(f"Mode (m={mode['m']}, n={mode['n']}) is UNSTABLE. k={mode['k']:.4f}, Growth Rate={mode['growth_rate']:.4f}")
    
    # --- 4. Analyze Modes for Periodic Boundary Conditions ---
    periodic_modes = []
    # Note: m,n can be 0, but we skip (0,0) as it has k=0 (no spatial variation)
    for m in range(max_mode_number + 1):
        for n in range(max_mode_number + 1):
            if m == 0 and n == 0:
                continue
            # Formula for Periodic eigenvalues
            lambda_mn = (2 * np.pi / L)**2 * (m**2 + n**2)
            k_mn = np.sqrt(lambda_mn)
            growth_rate = get_growth_rate(lambda_mn)
            periodic_modes.append({'m': m, 'n': n, 'k': k_mn, 'growth_rate': growth_rate})

    print("\n--- Analysis for Periodic Boundary Conditions ---")
    unstable_periodic = [m for m in periodic_modes if m['growth_rate'] > 0]
    if not unstable_periodic:
        print("No unstable modes found.")
    for mode in unstable_periodic:
        print(f"Mode (m={mode['m']}, n={mode['n']}) is UNSTABLE. k={mode['k']:.4f}, Growth Rate={mode['growth_rate']:.4f}")

    # --- 5. Plot the Comprehensive Results ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot the fundamental instability threshold (continuous curve)
    ax.plot(k_continuous, growth_rate_continuous, lw=2.5, color='black',
            label='Instability Threshold (Continuous)')
    ax.fill_between(k_continuous, 0, growth_rate_continuous, where=(growth_rate_continuous > 0),
                    color='gray', alpha=0.3, label='Unstable Region')
    ax.axhline(0, color='gray', linestyle='--')

    # Plot Dirichlet modes
    stable_d = [m for m in dirichlet_modes if m['growth_rate'] <= 0]
    ax.scatter([m['k'] for m in stable_d], [m['growth_rate'] for m in stable_d],
               facecolors='none', edgecolors='royalblue', s=60, label='Stable Dirichlet Modes')
    ax.scatter([m['k'] for m in unstable_dirichlet], [m['growth_rate'] for m in unstable_dirichlet],
               color='royalblue', s=80, zorder=5, marker='o', label='Unstable Dirichlet Modes')

    # Plot Periodic modes
    stable_p = [m for m in periodic_modes if m['growth_rate'] <= 0]
    ax.scatter([m['k'] for m in stable_p], [m['growth_rate'] for m in stable_p],
               color='firebrick', s=80, marker='x', label='Stable Periodic Modes')
    ax.scatter([m['k'] for m in unstable_periodic], [m['growth_rate'] for m in unstable_periodic],
               color='red', s=100, zorder=5, marker='x', lw=2, label='Unstable Periodic Modes')

    ax.set_title('Turing Instability: Dirichlet vs. Periodic Boundary Conditions', fontsize=16)
    ax.set_xlabel('Wavenumber, $k$', fontsize=12)
    ax.set_ylabel('Growth Rate, Re($\\lambda$)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, k_continuous[-1])
    ax.set_ylim(min(growth_rate_continuous) - 0.005, max(growth_rate_continuous) + 0.005)
    ax.grid(True)

    plt.show()

# Run the comparison
turing_instability_comparison()
