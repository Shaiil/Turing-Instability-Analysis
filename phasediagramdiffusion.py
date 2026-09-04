import numpy as np
import matplotlib.pyplot as plt

def generate_phase_diagram(ax, D):
    """
    Calculates and plots the Turing instability boundaries for the Gray-Scott model.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        D (float): The ratio of diffusion coefficients, D = Du/Dv.
    """
    # Create a grid of F and k values
    k = np.linspace(0, 0.08, 200)
    F = np.linspace(0.001, 0.25, 200)
    k_grid, F_grid = np.meshgrid(k, F)

    F_sn = np.linspace(1e-6, 0.25, 400)
    k_sn = 0.5 * (np.sqrt(F_sn) - 2 * F_sn)
    # --- Calculate Turing Conditions on the grid ---

    # Condition for the "blue" steady state to exist
    # X > 0 is required for real solutions for the steady state
    with np.errstate(divide='ignore', invalid='ignore'):
        X = 1 - 4 * (F_grid + k_grid)**2 / F_grid
    
    # Set invalid regions (where X < 0) to NaN
    X[X < 0] = np.nan

    # Calculate the steady state (u-, v+) where it exists (eq SM1.25)
    u_minus = 0.5 * (1 - np.sqrt(X))
    v_plus = (F_grid / (2 * (F_grid + k_grid))) * (1 + np.sqrt(X))

    # Calculate Jacobian elements evaluated at the steady state
    f_u = -v_plus**2 - F_grid
    g_u = v_plus**2
    g_v = 2 * u_minus * v_plus - (F_grid + k_grid)

    # Condition I & II: Stability without diffusion
    tr_J = f_u + g_v
    det_J = f_u * g_v - (-2 * u_minus * v_plus) * g_u

    # Condition III
    cond_III = 1 + np.sqrt(X) - 2 * D * (F_grid + k_grid)**3 / F_grid**2

    # Condition IV 
    # (f_u + D*g_v)^2 - 4*D*|J| > 0
    cond_IV = (f_u + D * g_v)**2 - 4 * D * det_J

    # --- Plot the boundaries (where conditions equal zero) ---
    turing_mask = (X > 0) & (tr_J < 0) & (det_J > 0) & (cond_III < 0) & (cond_IV > 0)
    # Black curves: Saddle-node (|J|=0) and Hopf (tr(J)=0) bifurcations
    ax.contour(k_grid, F_grid, det_J, levels=[0], colors=['black'], linewidths=[1.5])
    ax.contour(k_grid, F_grid, tr_J, levels=[0], colors=['black'], linestyles='dashed', linewidths=[1.5])
    
    ax.plot(k_sn, F_sn, 'k-', linewidth=2, label='Saddle-Node Bifurcation')
    # Blue curve: Condition III = 0
    ax.contour(k_grid, F_grid, cond_III, levels=[0], colors=['#0000FF'], linewidths=[1.5]) # Blue

    # Red curves: Condition IV = 0
    ax.contour(k_grid, F_grid, cond_IV, levels=[0], colors=['#FF0000'], linewidths=[1.5]) # Red

    ax.contourf(k_grid, F_grid, turing_mask, levels=[0.5, 1.5], colors=['yellow'])

    # --- Formatting ---
    ax.set_title(f'$D_u / D_v  = {D}$')
    ax.set_xlabel('$k$')
    ax.set_ylabel('$F$')
    ax.set_xlim(0, 0.065)
    ax.set_ylim(0, 0.25)
    ax.grid(False)
    if D == 2:
    # Use annotate to create text with an arrow
        ax.annotate('T',
                    xy=(0.06, 0.059),          # The point the arrow points to
                    xytext=(0.045, 0.07),       # The position of the text 'T'
                    arrowprops=dict(arrowstyle="->", lw=1.5),
                    fontsize=14,
                    ha='center')
    else:
        # For all other plots (e.g., D=6), use the original text label
        ax.text(0.055, 0.07, 'T', fontsize=14, ha='center')
    
# Create the figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# Generate the plot for D = 2
generate_phase_diagram(ax1, D=2)

# Generate the plot for D = 6
generate_phase_diagram(ax2, D=6)


plt.suptitle('Turing Instability Regions for the Gray-Scott Model', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

