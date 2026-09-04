import numpy as np
import matplotlib.pyplot as plt

# --- 1. Calculate the Saddle-Node Bifurcation Curve (Solid Line) ---
# This is the outer boundary where non-trivial solutions appear.
# The formula is k = 0.5 * (sqrt(F) - 2F).
# It's easier to generate F and calculate k.
# The curve exists for 0 < F <= 0.25.
F_sn = np.linspace(1e-6, 0.25, 400)
k_sn = 0.5 * (np.sqrt(F_sn) - 2 * F_sn)


# --- 2. Calculate the Hopf Bifurcation Curve (Dotted Line) ---
# This is the inner boundary where stability changes.
# The formula is F^2 + (2k - sqrt(k))F + k^2 = 0.
# It's easier to solve for F given k.
# Solution: F = 0.5 * (sqrt(k) - 2k ± sqrt(k - 4k^1.5))
# This is valid for k <= 1/16 (or 0.0625).
k_hopf = np.linspace(1e-6, 1/16, 400)
sqrt_term = np.sqrt(k_hopf - 4 * k_hopf**1.5)

# The dotted line in the diagram corresponds to the lower branch (minus sign).
F_hopf = 0.5 * (np.sqrt(k_hopf) - 2 * k_hopf - sqrt_term)


# --- 3. Plot the Diagram using Matplotlib ---
plt.style.use('default') # Use a clean, standard plot style
fig, ax = plt.subplots(figsize=(7, 6))

# Plot the two curves
ax.plot(k_sn, F_sn, 'k-', linewidth=2, label='Saddle-Node Bifurcation')
ax.plot(k_hopf, F_hopf, 'k:', linewidth=2, label='Hopf Bifurcation')

# Add region labels
ax.text(0.05, 0.2, 'I', fontsize=16, ha='center')
ax.text(0.04, 0.1, 'II', fontsize=16, ha='center')
ax.text(0.065, 0.025, 'III', fontsize=16, ha='center', va='center')

# Set labels, title, and limits to match the original figure
ax.set_xlabel('k', fontsize=14)
ax.set_ylabel('F', fontsize=14)
ax.set_title('Bifurcation Diagram for the Gray-Scott Model', fontsize=15, pad=10)
ax.set_xlim(0, 0.07)
ax.set_ylim(0, 0.25)
ax.tick_params(axis='both', which='major', labelsize=12)

# Optional: Add a legend to clarify the lines
# ax.legend()

plt.tight_layout()
plt.show()