import numpy as np
from matplotlib import pyplot as plt
from scipy.fft import dctn, idctn, dct, idct
from scipy.interpolate import interp1d

import ot

from tqdm import tqdm

from typing import Union


def _direct_c_transform(phi: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """
    Perform a direct c-transform operation on the given cost matrix adjusted by vector phi.

    This function calculates the c-transform of a cost matrix and a vector phi.
    It computes the minimum value across rows of the element-wise subtraction between
    cost matrix and phi vector, treating phi as a bias.

    Parameters:
    - phi (np.ndarray): A vector that represents potential values.
    - cost (np.ndarray): A cost matrix.

    Returns:
    - np.ndarray: The transformed vector resulting from the direct c-transform computation.
    """
    # transform = (cost - phi).min(axis=1)
    transform = (cost - phi).max(axis=1)
    return transform


def ctransform(phi: np.ndarray, cost: Union[np.ndarray, str]) -> np.ndarray:
    """
    Computes the c-transform of a vector phi with respect to a cost matrix or handles
    alternative string-based methods (not implemented).

    Parameters:
    - phi (np.ndarray): A vector representing potential values.
    - cost (Union[np.ndarray, str]): Either a numpy array representing the cost matrix or a
      string indicating the method to be used for an optimized c-transform.

    Returns:
    - np.ndarray: The result of the c-transform calculation.

    Raises:
    - NotImplementedError: If the cost is specified as a string indicating an unimplemented method.
    - TypeError: If the cost parameter is neither a numpy array nor a string.
    """
    if isinstance(cost, np.ndarray):
        return _direct_c_transform(phi, cost)
    elif isinstance(cost, str):
        raise NotImplementedError(
            'optimized ctransform has not been implemented')
    else:
        raise TypeError('cost must be either a numpy array or a string')


def _map_T_quadratic_cost(phi: np.ndarray, x: np.ndarray, C: np.ndarray) -> np.ndarray:
    # Note: currently, this works only for quadratic cost with given matrix C
    # this is due to unfinished ctransform function
    dx = x[1] - x[0]
    return np.gradient(
        ctransform(phi, cost=C),
        dx,  # data is evenly spaced. TODO: change later to a better way?
    )


def map_T(phi: np.ndarray, cost: Union[np.ndarray, str], X: np.ndarray, C: np.ndarray = None) -> np.ndarray:
    """Calculates the map T of a function phi given a cost type, using domains X and Y."""

    # Handling string type cost function
    if isinstance(cost, str):
        if cost == 'quadratic':
            return _map_T_quadratic_cost(phi, X, C)
        else:
            raise NotImplementedError(
                f"Not implemented for cost type '{cost}'")

    # Handling numpy array type cost function (currently not implemented)
    elif isinstance(cost, np.ndarray):
        raise NotImplementedError("Matrix cost types are not implemented")

    # Handling invalid cost type
    else:
        raise TypeError("Cost must be either a string or a numpy array")


def sampling_pushforward_1d(mu, xMap, xs):
    rho = np.zeros_like(mu)
    n = xs.shape[0]

    for j in range(n):
        mass = mu[j]
        if mass > 0:
            xStretch = abs(xMap[min(j + 1, n - 1)] - xMap[j])
            xSamples = max(int(n * xStretch), 1)
            factor = 1.0 / xSamples

            for k in range(xSamples):
                a = (k + 0.5) / xSamples
                xPoint = (1 - a) * xMap[j] + a * xMap[min(j + 1, n - 1)]
                X = xPoint * n - 0.5

                xIndex = int(X)
                xFrac = X - xIndex

                # Boundary conditions
                xIndex = min(max(xIndex, 0), n - 1)
                xOther = min(max(xIndex + 1, 0), n - 1)

                # Distribute mass to the nearest indices
                rho[xIndex] += (1 - xFrac) * mass * factor
                rho[xOther] += xFrac * mass * factor

    # Normalize the resulting density to maintain total mass
    rho /= np.sum(rho)

    return rho


def pushforward(mu: np.ndarray, T: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Transforms the measure mu using the multidimensional transport map T.

    :param mu: numpy array representing the original measure (discrete densities or probabilities)
    :param T: numpy array where each row represents the new multidimensional position of each corresponding element in mu
    :return: numpy array of the transformed measure
    """
    # nu = T.T @ mu
    # nu /= nu.sum()
    nu = sampling_pushforward_1d(mu, T, x)
    return nu


def solve_pde(f, L, N):
    """
    Solves the Poisson equation -∆phi = f with Neumann boundary conditions
    using the Discrete Cosine Transform.

    Parameters:
    f : ndarray
        The source term of the Poisson equation.
    L : float
        The length of the domain.
    N : int
        The number of discretization points.

    Returns:
    x : ndarray
        The discretized domain.
    phi : ndarray
        The solution of the Poisson equation.
    """
    # Compute the discrete cosine transform of f
    f_hat = dct(f, norm='ortho')
    # Create an array of wave numbers (avoid division by zero for k=0)
    k = np.pi * np.arange(N) / L
    k[0] = 1  # avoid division by zero; will set phi_hat[0] to zero later
    # Compute the solution in the spectral domain
    phi_hat = f_hat / (k**2)
    # Set the zero-frequency component to zero
    phi_hat[0] = 0
    # Compute the inverse discrete cosine transform to get the spatial solution
    phi = idct(phi_hat, norm='ortho')
    return phi


def gradient(nu: np.ndarray, t_mu: np.ndarray, x: np.ndarray):
    # f = nu - t_mu
    f = t_mu - nu
    L = x[-1]
    phi = solve_pde(f, L, x.size)
    return phi


def compute_dual(phi: np.ndarray, psi: np.ndarray, mu: np.ndarray,
                 nu: np.ndarray, x: np.ndarray, y: np.ndarray):
    n = phi.size
    return np.sum(0.5 * (x*x+y*y) * (mu + nu) - nu*phi - mu*psi)/(n)


def stepsize_update(sigma, value, oldValue, gradSq):
    diff = value - oldValue
    scaleDown = 0.95
    scaleUp = 1/scaleDown
    upper = 0.75
    lower = 0.25

    if diff > gradSq * sigma * upper:
        return sigma * scaleUp
    elif diff < gradSq * sigma * lower:
        return sigma * scaleDown
    return sigma


def back_and_forth(
        x: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        nu: np.ndarray,
        cost: np.ndarray,
        cost_name: str,
        sigma: float = 0.01,
        iterations: int = 20,
):
    phi = np.zeros_like(x)
    psi = np.zeros_like(y)
    n, m = x.size, y.size

    rho = np.copy(mu)
    oldValue = compute_dual(phi, psi, mu, nu, x, y)
    h1err_history = np.empty((1, 1))
    dual_history = np.empty((1, 1))

    for i in tqdm(range(iterations)):

        rho = pushforward(
            mu,
            map_T(phi, cost_name, y, C=cost),
            x
        )
        phi_grad = gradient(
            nu,
            rho,
            x,
        )
        phi += sigma * phi_grad

        h1_error = np.sum(phi_grad * rho) / n

        psi = ctransform(phi, cost)
        phi = ctransform(psi, cost)

        value = compute_dual(phi, psi, mu, nu, x, y)
        sigma = stepsize_update(sigma, value, oldValue, h1_error)
        oldValue = value

        rho = pushforward(
            nu,
            map_T(psi, cost_name, x, C=cost),
            y
        )
        psi_grad = gradient(
            mu,
            rho,
            y,
        )
        psi += sigma * psi_grad

        h1_error = np.sum(psi_grad * rho) / n

        phi = ctransform(psi, cost.T)
        psi = ctransform(phi, cost)

        value = compute_dual(phi, psi, mu, nu, x, y)
        sigma = stepsize_update(sigma, value, oldValue, h1_error)
        oldValue = value

        h1err_history = np.append(h1err_history, h1_error)
        dual_history = np.append(dual_history, value)

    return phi, psi, h1err_history, dual_history
