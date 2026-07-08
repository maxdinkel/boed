import numpy as np

from .probabilistic_model import designs, log_likelihood, noise_model, output_dir, precomputed_model_outputs
from ..helper_functions import write_pickle_file, write_latex_table, get_estimators, PrecomputedModel
from boed.ade import SDC




if __name__ == '__main__':
    _, rnmc, _, anmc = get_estimators(log_likelihood, noise_model)
    estimators = {"rnmc": rnmc, "anmc": anmc}
    results = {}
    for name, estimator in estimators.items():
        model = PrecomputedModel(precomputed_model_outputs)
        rng = np.random.default_rng(0)
        results[name] = SDC(model, designs, model.draw_samples, estimator, 1000, True).compute_eig(rng)[0]
        print(results[name])
        print(np.argmax(results[name]))

    write_latex_table(output_dir, results)
    write_pickle_file(output_dir / "reference_eig.pickle", results)
