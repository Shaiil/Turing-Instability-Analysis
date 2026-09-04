import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags, identity, kron, bmat
from scipy.sparse.linalg import spsolve

# Parameters
N = 100               # Grid size (N x N)
dx = 1.0              # Grid spacing
dt = 0.1              # Time step
steps = 100          # Total number of steps
plot_interval = 1   # Plot every this many steps
name = 'turing_patterns_fully_implicit_unstable_mp4_test'

# Diffusion coefficients
Du, Dv = 0.01, 1

# Reaction parameters
a, b, c, d = 2, 3, -4, -3

# Laplacian with Neumann (zero-flux) boundary conditions
def laplacian_matrix(N, dx):
    Ix = identity(N)
    Iy = identity(N)

    main = -2 * np.ones(N)
    off = np.ones(N - 1)
    Lx = diags([off, main, off], [-1, 0, 1], shape=(N, N)).tolil()
    Lx[0, 1] = 2
    Lx[-1, -2] = 2
    Lx /= dx**2

    Ly = diags([off, main, off], [-1, 0, 1], shape=(N, N)).tolil()
    Ly[0, 1] = 2
    Ly[-1, -2] = 2
    Ly /= dx**2

    return kron(Iy, Lx) + kron(Ly, Ix)

# Initial conditions
u = 0.4 * np.random.rand(N, N)
v = 0.4 * np.random.rand(N, N)


# Flattened identity and Laplacian
I_big = identity(N * N)
L = laplacian_matrix(N, dx).tocsr()

# Construct the full system matrix for implicit update
A11 = I_big - dt * Du * L - dt * a * I_big
A12 = -dt * b * I_big
A21 = -dt * c * I_big
A22 = I_big - dt * Dv * L - dt * d * I_big
A_full = bmat([[A11, A12], [A21, A22]], format='csr')

# Animation update function
def update(frame):
    global u, v
    step = frame * plot_interval

    if step % (steps // 20) == 0:
        print(f"Progress: {100 * step // steps:.0f}%")

    for _ in range(plot_interval):
        rhs = np.concatenate([u.flatten(), v.flatten()])
        sol = spsolve(A_full, rhs)
        u = sol[:N * N].reshape(N, N)
        v = sol[N * N:].reshape(N, N)
        

    img.set_data(u)
    return [img]


# Set up the plot
fig, ax = plt.subplots()
img = ax.imshow(u, cmap='plasma', vmin=0, vmax=1)
plt.colorbar(img, ax=ax)
ax.set_title("$\sigma_u = {s}, \sigma_v = {v}$".format(s = Du,v = Dv))

# Create animation
ani = animation.FuncAnimation(fig, update, frames=steps // plot_interval, blit=True)

# Save animation
ani.save(name + ".mp4", writer='ffmpeg', fps=10)
# ani.save(name + ".gif", writer='pillow', fps=10)  # Alternative: save as GIF

plt.close()  # Prevent static plot from displaying
