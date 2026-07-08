import numpy as np
from pathlib import Path
from numba import njit
from scipy.stats import beta
import jax
import jax.numpy as jnp


output_dir = Path(__file__).parent / 'output'

num_dim = 3
prior_mean = np.log([1, 0.1, 20])
prior_var = np.ones(num_dim) * 0.05

prior_std = np.sqrt(prior_var)

std_a = np.sqrt(0.1)
std_m = 0.1


def geometric_design(delta, gamma, n=15):
    assert delta > 1.0
    assert gamma >= 0.2

    d1 = gamma / (delta - 1)
    s = np.arange(n)
    return d1 * (delta ** s)

geometric_designs = np.array([
    geometric_design(1.1, 0.2),
    geometric_design(1.15, 0.2),
    geometric_design(1.15, 0.3),
    geometric_design(1.15, 0.5),
    geometric_design(1.05, 0.5),
])


def even_design(d1, delta, n=15):
    s = np.arange(n)
    return d1 + delta * s

even_designs = np.array([
    even_design(0.0, 0.5),
    even_design(0.0, 1.0),
    even_design(0.0, 24 / 14),
    even_design(10.0, 1.0),
    even_design(17.0, 0.5),

])


def beta_design(a, b, n=15, T=24.0):
    u = np.linspace(0.01, 0.99, n)
    return T * beta.ppf(u, a, b)

beta_designs = np.array([

    beta_design(1.0, 5.0),
    beta_design(5.0, 1),
    beta_design(5.0, 5.0),
    beta_design(2.0, 1.0),
    beta_design(1.0, 2.0),

])


design_times = np.concatenate([geometric_designs, even_designs, beta_designs])


designs = np.arange(len(design_times))
optimal_design = np.array(14)


dose = 400


def draw_samples(num_draws, rng):
    samples = np.empty((num_draws, num_dim))
    num_filled = 0

    while num_filled < num_draws:
        new_samples = np.exp(prior_mean + prior_std * rng.normal(size=(num_draws, num_dim)))

        valid = new_samples[:, 0] > new_samples[:, 1]
        num_valid = valid.sum()

        num_take = min(num_valid, num_draws - num_filled)
        samples[num_filled:num_filled + num_take] = new_samples[valid][:num_take]
        num_filled += num_take

    return samples



@njit
def noise_model(model_output, theta, xi, rng):
    shape = model_output.shape
    return std_a * rng.normal(size=shape) + model_output * (1 + std_m * rng.normal(size=shape))



def get_log_likelihood_fun(np_lib):
    def log_likelihood_fun(model_outputs, observations, theta, xi):
        squared_distances = (model_outputs - observations) ** 2
        noise_variance = std_a ** 2 + (std_m * model_outputs) ** 2
        return (
            - 0.5 * np_lib.sum(squared_distances / noise_variance, axis=1)
            - 0.5 * np_lib.sum(np_lib.log(2*np_lib.pi*noise_variance), axis=1)
        )
    return log_likelihood_fun



log_likelihood = njit(get_log_likelihood_fun(np_lib=np))
log_likelihood_jax = jax.jit(get_log_likelihood_fun(np_lib=jnp))


@njit
def forward_model(theta, xi):
    k_a, k_e, V = theta[:, 0:1], theta[:, 1:2], theta[:, 2:3]
    times = design_times[xi].reshape(1, -1)
    delta = k_a - k_e

    return (dose / V) * (k_a / delta) * np.exp(-k_e * times) * -np.expm1(-delta * times)
