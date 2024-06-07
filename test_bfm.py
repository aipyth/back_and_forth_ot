from back_and_forth import back_and_forth, map_T, pushforward
import numpy as np
from matplotlib import pyplot as plt
import ot
import time

plt.rcParams['image.cmap'] = 'Greys'


def generate_gaussian_distribution(
        num_points, mean, std, range_min, range_max
):
    grid = np.linspace(range_min, range_max, num_points)
    pdf_values = 1/(std * np.sqrt(2 * np.pi)) * np.exp(
        -1/2 * np.power((grid - mean)/std, 2))
    pdf_values /= pdf_values.sum()
    return grid, pdf_values


# n = 30 + 1
# n = 100 + 1   # Number of points
n = 1000
std = 0.08
range_min = 0
range_max = 1

x = np.linspace(range_min, range_max, n)
y = np.linspace(range_min, range_max, n)
X, Y = np.meshgrid(x, y, indexing='ij')

grid1, pdf1 = generate_gaussian_distribution(n, 0.2, std, range_min, range_max)
grid2, pdf2 = generate_gaussian_distribution(n, 0.6, std, range_min, range_max)

# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].scatter(grid1, pdf1)
# axs[1].scatter(grid2, pdf2)
# fig.show()


c = X * Y
# c = np.power(X - Y, 2) / 2
c_name = 'quadratic'

# phi, psi = back_and_forth(
#     x, y, pdf1, pdf2, c, c_name, sigma=0.01, iterations=10_000)

sigma_init = 8 * np.minimum(pdf1.max(), pdf2.max())
# sigma_init = 0.01
phi, psi, err1, duals = back_and_forth(
    x, y, pdf1, pdf2, c, c_name, sigma=sigma_init, iterations=500)

rho1 = pushforward(pdf1, map_T(phi, c_name, x, C=c), x)

rho2 = pushforward(pdf2, map_T(psi, c_name, y, C=c), y)

T_psi = map_T(psi, c_name, x, C=c)
T_phi = map_T(phi, c_name, x, C=c)
T = np.zeros((n, n))
for i, val in enumerate(T_psi):
    j = np.argmin(np.abs(y - val))
    if rho1[i] != 0 and rho2[j] != 0:
        T[i, j] = 1

cost = np.power(X - Y, 2) / 2

wasserstein_dist1 = ot.emd2(rho1, pdf1, cost)
wasserstein_dist2 = ot.emd2(rho2, pdf2, cost)
print(f"W1 T_phi mu: {wasserstein_dist1}")
print(f"W1 T_psi nu: {wasserstein_dist2}")

wasserstein_dist_mu_nu = ot.emd2(pdf1, pdf2, cost)
print(f"W1 mu nu: {wasserstein_dist_mu_nu}")

wasserstein_dist_phi_psi = ot.emd2(rho1, rho2, cost)
print(f"W1 T_phimu T_psinu: {wasserstein_dist_phi_psi}")

exact_start = time.time()
exact_sol = ot.emd(pdf1, pdf2, cost)
print(f"EMD is done in {time.time() - exact_start}.")

fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0][0].plot(grid2, rho1, 'r', label='$T_{\\psi \\#}\\mu$')
axs[0][0].plot(grid2, rho2, 'y', label='$T_{\\phi \\#}\\nu$')
# axs[0].plot(grid1, T_psi, 'y', label='$T_{\\psi}$')
axs[0][0].scatter(grid1, pdf1, s=2, alpha=0.7, c='b', label='$\\mu$')
axs[0][0].scatter(grid2, pdf2, s=2, alpha=0.7, c='g', label='$\\nu$')
axs[0][0].legend()

err_start = 100
axs[0][1].plot(np.arange(err_start, err1.size), err1[err_start:])

axs[1][0].imshow(T, vmax=1)
axs[1][0].set_title('Back-and-Forth Solution Map')
# axs[1][1].imshow(exact_sol, vmax=1)
# axs[1][1].set_title('Exact Solution Map')
axs[1][1].plot(np.arange(0, duals.size), duals)
axs[1][1].set_title('Dual values')
# fig.show()

plt.show()
