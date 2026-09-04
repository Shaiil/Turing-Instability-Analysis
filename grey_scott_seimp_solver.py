"""
Gray-Scott model with implicit diffusion and explicit reaction (operator splitting).
Saves animation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags, identity, kron, csc_matrix
from scipy.sparse.linalg import spsolve, factorized

def build_laplacian_2d(n, h):
    """Construct 2D Laplacian with periodic boundary conditions using Kronecker product."""
    e = np.ones(n)
    L1D = diags([e, -2 * e, e], [-1, 0, 1], shape=(n, n)).tolil()
    L1D[0, -1] = 1
    L1D[-1, 0] = 1
    L = kron(identity(n), L1D) + kron(L1D, identity(n))
    return csc_matrix(L) / (h * h)


def main():

    print('starting....')

    
    Du = 0.2
    Dv = 0.1
    F = 0.03
    k = 0.06

    name = f"shail_movie_fkuv{F}_{k}_{Du}_{Dv}"
    n = 200
    h = 1
    h2 = h * h

    nt = 10000
    dt = 0.5

    initial_condition_arr = ["rect","C",'blob','rand']
    initial_condition = initial_condition_arr[3]

    # Initial conditions

    # a rectangular patch
    if initial_condition == "rect":
        U = np.ones((n, n))
        V = np.zeros((n, n))
        low = (n // 2) - 9
        high = (n // 2) + 10
        U[low:high, low:high] = 0.5 + np.random.uniform(0, 0.1, (19, 19))
        V[low:high, low:high] = 0.25 + np.random.uniform(0, 0.1, (19, 19))

    # a C-shaped patch
    if initial_condition == "C":
        U = np.ones((n, n))
        V = np.zeros((n, n))

        low = (n // 2) - 9
        high = (n // 2) + 10

        # Create blank patches
        patch_U = np.ones((19, 19))
        patch_V = np.zeros((19, 19))

        # Thicker C-shape: 3 rows/cols thick
        for i in range(19):
            for j in range(19):
                if (i in [0, 1, 2, 16, 17, 18]) and (3 <= j <= 15):  
                    patch_U[i, j] = 0.5 + np.random.uniform(0, 0.1)
                    patch_V[i, j] = 0.25 + np.random.uniform(0, 0.1)
                elif (3 <= i <= 15) and (j in [2, 3, 4]):  
                    patch_U[i, j] = 0.5 + np.random.uniform(0, 0.1)
                    patch_V[i, j] = 0.25 + np.random.uniform(0, 0.1)

        # Apply to center
        U[low:high, low:high] = patch_U
        V[low:high, low:high] = patch_V
        
    if initial_condition == "blob":
        U = np.ones((n, n))
        V = np.zeros((n, n))

        cx, cy = n // 2, n // 2  # center
        radius = 9

        for i in range(n):
            for j in range(n):
                if (i - cx)**2 + (j - cy)**2 < radius**2:
                    U[i, j] = 0.5 + np.random.uniform(0, 0.1)
                    V[i, j] = 0.25 + np.random.uniform(0, 0.1)

    if initial_condition == "rand":
        U = np.ones((n, n))
        V = np.zeros((n, n))

        num_blobs = 20
        blob_radius = 5

        for _ in range(num_blobs):
            cx = np.random.randint(blob_radius, n - blob_radius)
            cy = np.random.randint(blob_radius, n - blob_radius)

            for i in range(cx - blob_radius, cx + blob_radius):
                for j in range(cy - blob_radius, cy + blob_radius):
                    if 0 <= i < n and 0 <= j < n:
                        dx, dy = i - cx, j - cy
                        dist2 = dx**2 + dy**2
                        if dist2 < blob_radius**2:
                            # Randomize only some points to make it look more natural
                            if np.random.rand() < 0.8:
                                U[i, j] = 0.5 + np.random.uniform(0, 0.1)
                                V[i, j] = 0.25 + np.random.uniform(0, 0.1)

    # Flatten for sparse solve
    U = U.reshape(-1)
    V = V.reshape(-1)

    # Sparse matrix setup
    L = build_laplacian_2d(n, h)
    I = identity(n * n)
    LU = (I - dt * Du * L).tocsc()
    LV = (I - dt * Dv * L).tocsc()

    # Plot setup
    fig, ax = plt.subplots(tight_layout=True)
    ims = []
    solve_U = factorized(LU)
    solve_V = factorized(LV)

    for i in range(nt):
        print(f"Running {i + 1:,}/{nt:,}", end="\r")

        # Step 1: Implicit diffusion
        U_star = solve_U(U)
        V_star = solve_V(V)

        # Step 2: Explicit reaction
        U_star_2D = U_star.reshape((n, n))
        V_2D = V.reshape((n, n))
        UVV = U_star_2D * V_2D**2
        U_new = U_star + dt * (-UVV.reshape(-1) + F * (1 - U_star))

        U_2D = U_new.reshape((n, n))
        V_star_2D = V_star.reshape((n, n))
        UVV2 = U_2D * V_star_2D**2
        V_new = V_star + dt * (UVV2.reshape(-1) - (F + k) * V_star)

        U = U_new
        V = V_new

        if i % 100 == 0:
            im = ax.imshow(U.reshape((n, n)), interpolation='bicubic', cmap=plt.cm.jet, animated=True)
            ims.append([im])

    # Save animation
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True)
    ani.save(f'/Users/subhasdh/Desktop/Sem III - Uni resources/Practical/Code/{name}.mp4')

    # Plot final state
    _, ax = plt.subplots(tight_layout=True)
    ax.imshow(U.reshape((n, n)), interpolation='bicubic', cmap=plt.cm.jet)
    # plt.show()


if __name__ == '__main__':
    main()
