from pathlib import Path
from math import lgamma
import numpy as np
import pickle
from numba import njit

from ..helper_functions import PrecomputedModel


output_dir = Path(__file__).parent / 'output'



with open(output_dir.parents[1] / "data" / "viscoplastic_model_output_samples.pickle", 'rb') as f:
    data = pickle.load(f)

num_samples = len(data[0])
precomputed_model_outputs = [outputs.reshape(num_samples, -1) for outputs in data]

forward_model = PrecomputedModel(precomputed_model_outputs)
draw_samples = forward_model.draw_samples


scale = 0.002
designs = np.arange(12)
optimal_design = np.array(8)
nu=10

def noise_model(model_output, theta, xi, rng):
    return model_output + scale * rng.standard_t(df=nu, size=model_output.shape)


LOGPDF_CONST = (
        lgamma((nu + 1.0) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * np.log(nu * np.pi)
        - np.log(scale)
    )



@njit
def log_likelihood(model_outputs, observations, theta, xi):
    squared_distances = (model_outputs - observations) ** 2

    n_obs = model_outputs.shape[1]

    scaled_sq = squared_distances / (nu * scale**2)

    return (
        n_obs * LOGPDF_CONST
        - 0.5 * (nu + 1.0)
        * np.sum(np.log1p(scaled_sq), axis=1)
    )

