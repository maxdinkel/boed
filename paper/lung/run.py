from ..parallel_evaluation import repeat_parallel, num_cpus

import numpy as np

from .probabilistic_model import designs, log_likelihood, noise_model, output_dir, precomputed_model_outputs, optimal_design
from boed.ade import SDC, ADE
from functools import partial
from multiprocessing import shared_memory
from ..helper_functions import get_estimators, write_pickle_file, PrecomputedModel


def run_function(rng, algorithm, shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    model_outputs = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    forward_model = PrecomputedModel(model_outputs, rng.permutation(shape[1]))
    algorithm.model = forward_model
    algorithm.draw_samples = forward_model.draw_samples

    result = algorithm.run(rng)
    shm.close()

    return result


def repeat_algorithm(algorithm):
    shm = shared_memory.SharedMemory(create=True, size=precomputed_model_outputs.nbytes)

    shm_array = np.ndarray(precomputed_model_outputs.shape, dtype=precomputed_model_outputs.dtype, buffer=shm.buf)
    shm_array[:] = precomputed_model_outputs[:]  # one copy here only
    shm_array.flags.writeable = False

    partial_function = partial(run_function, algorithm=algorithm, shm_name=shm.name, shape=precomputed_model_outputs.shape, dtype=precomputed_model_outputs.dtype)

    optimized_design_list, num_eval_list = zip(*repeat_parallel(partial_function))

    shm.close()
    shm.unlink()

    print("Accuracy: ", np.mean(np.array(optimized_design_list) == optimal_design) * 100, "%")
    print("Average evalutations: ", np.mean(np.sum(num_eval_list, axis=1)))

    return {"design": optimized_design_list, "num_eval": num_eval_list}


if __name__ == '__main__':
    print(f"Starting evaluation using {num_cpus} cores.")

    nmc, rnmc, lrnmc, anmc = get_estimators(log_likelihood, noise_model)

    results = {"nmc": [], "rnmc": [], "lrnmc": [], "anmc": []}
    SDC_ = partial(SDC, model=None, designs=designs, draw_samples=None, crn=False)
    for i in range(4):
        results["nmc"].append(repeat_algorithm(SDC_(estimator=nmc, num_eval=nmc.compute_num_total([32, 40, 50, 62][i]))))
        results["rnmc"].append(repeat_algorithm(SDC_(estimator=rnmc, num_eval=500 * 2 ** i)))
        results["lrnmc"].append(repeat_algorithm(SDC_(estimator=lrnmc, num_eval=500 * 2 ** i)))
        results["anmc"].append(repeat_algorithm(SDC_(estimator=anmc, num_eval=500 * 2 ** i)))

        write_pickle_file(output_dir / "sdc.pickle", results)


    results = {"nmc": [], "rnmc": [], "lrnmc": [], "anmc": []}
    SDC_ = partial(SDC, model=None, designs=designs, draw_samples=None, crn=True)
    for i in range(4):
        results["nmc"].append(repeat_algorithm(SDC_(estimator=nmc, num_eval=nmc.compute_num_total([3, 4, 6, 9][i]))))
        results["rnmc"].append(repeat_algorithm(SDC_(estimator=rnmc, num_eval=250 * 2 ** i)))
        results["lrnmc"].append(repeat_algorithm(SDC_(estimator=lrnmc, num_eval=30 * 2 ** i)))
        results["anmc"].append(repeat_algorithm(SDC_(estimator=anmc, num_eval=10 * 2 ** i)))

        write_pickle_file(output_dir / "sdc_crn.pickle", results)


    results = {}
    for num_criterion in [2, 3]:
        for num_new_eval in [8, 16]:
            for q in [0.95, 0.99, 0.999]:
                ADE_ = partial(ADE, model=None, designs=designs, draw_samples=None, quantile=q, num_criterion=num_criterion, num_new_eval=num_new_eval)
                results[f'nc{num_criterion}_q{str(q).replace("0.", "")}_n{num_new_eval}'] = {
                    "rnmc": repeat_algorithm(ADE_(estimator=rnmc)),
                    "lrnmc": repeat_algorithm(ADE_(estimator=lrnmc)),
                    "anmc": repeat_algorithm(ADE_(estimator=anmc)),
                }
                write_pickle_file(output_dir / 'ade.pickle', results)
