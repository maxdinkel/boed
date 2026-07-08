from pathlib import Path
import numpy as np
import pickle
from numba import njit
import jax
import jax.numpy as jnp

from ..helper_functions import PrecomputedModel


output_dir = Path(__file__).parent / 'output'


with open(output_dir.parents[1] / "data" / "lung_model_output_samples.pickle", 'rb') as f:
    precomputed_model_outputs = np.array(pickle.load(f))


forward_model = PrecomputedModel(precomputed_model_outputs)
draw_samples = forward_model.draw_samples


std_a = 100
std_m = 0.05


designs = np.arange(4)
optimal_design = np.array(0)


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