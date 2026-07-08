import jax.numpy as jnp
import jax
import numpy as np
from functools import partial

from .probabilistic_model import designs, log_likelihood_jax, noise_model, output_dir, precomputed_model_outputs
from ..helper_functions import write_pickle_file, write_latex_table
from ..nmc_jax import compute_single_pmi_jax



if __name__ == '__main__':
    estimators = {"rnmc": False, "lrnmc": True}
    results = {}
    for estimator, leave_out in estimators.items():
        compute_single_pmi = jax.jit(partial(compute_single_pmi_jax, log_likelihood_jax=log_likelihood_jax, leave_out=leave_out))
        all_eig = np.zeros(len(designs))
        rng = np.random.default_rng(0)
        for i, xi in enumerate(designs):
            outputs = precomputed_model_outputs[i]
            y_obs = jnp.array(noise_model(outputs, None, xi, rng=rng))
            partial_compute_pmi = partial(compute_single_pmi, outputs=outputs, y_obs=y_obs, xi=xi)
            all_eig[i] = np.mean(jax.lax.map(partial_compute_pmi, np.arange(len(y_obs)), batch_size=100))
            print(all_eig[i])

        results[estimator] = all_eig

    write_latex_table(output_dir, results)
    write_pickle_file(output_dir / "reference_eig.pickle", results)
