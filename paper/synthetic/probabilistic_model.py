import numpy as np
from pathlib import Path
from numba import njit


output_dir = Path(__file__).parent / 'output'

num_dim = 100
num_dim_y = 50
prior_mean = 0
prior_var = 1
prior_std = np.sqrt(prior_var)

likelihood_std = 5
likelihood_var = likelihood_std ** 2

designs = np.arange(1, 11)

default_rng = np.random.default_rng(2)
random_matrices = [default_rng.random((num_dim_y, num_dim)) * np.cos(xi) for xi in designs]



def expected_information_gain(xi):
    a = random_matrices[xi-1]
    return 0.5 * np.log(np.linalg.det(np.eye(num_dim) + prior_var / likelihood_var * a.T @ a))


def draw_samples(num_draws, rng):
    return np.exp(prior_mean + prior_std * rng.normal(size=(num_draws, num_dim)))


@njit
def noise_model(model_output, theta, xi, rng):
    return np.exp(model_output + likelihood_std * rng.normal(size=model_output.shape))


LOGPDF_CONST = -0.5 * num_dim * np.log(2*np.pi*likelihood_var)


@njit
def log_likelihood(model_outputs, observations, theta, xi):
    log_obs = np.log(observations)
    return LOGPDF_CONST - np.sum(log_obs) - np.sum((model_outputs - log_obs) ** 2, axis=1) / (2 * likelihood_var)


def forward_model(theta, xi):
    return np.log(theta) @ random_matrices[xi-1].T



true_eig = np.array([expected_information_gain(xi) for xi in designs])
optimal_design = designs[np.argmax(true_eig)]
