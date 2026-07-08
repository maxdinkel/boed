from ..parallel_evaluation import repeat_parallel, num_cpus

import numpy as np

from .probabilistic_model import forward_model, designs, log_likelihood, draw_samples, noise_model, output_dir, optimal_design
from boed.ade import SDC, ADE
from functools import partial
from ..helper_functions import get_estimators, write_pickle_file



def repeat_algorithm(algorithm):
    optimized_design_list, num_eval_list = zip(*repeat_parallel(algorithm.run))

    print("Accuracy: ", np.mean(np.array(optimized_design_list) == optimal_design) * 100, "%")
    print("Average evalutations: ", np.mean(np.sum(num_eval_list, axis=1)))

    return {"design": optimized_design_list, "num_eval": num_eval_list}


if __name__ == '__main__':
    print(f"Starting evaluation using {num_cpus} cores.")

    nmc, rnmc, lrnmc, anmc = get_estimators(log_likelihood, noise_model)


    results = {"nmc": [], "rnmc": [], "lrnmc": [], "anmc": []}
    SDC_ = partial(SDC, model=forward_model, designs=designs, draw_samples=draw_samples, crn=False)
    for i in range(4):
        results["nmc"].append(repeat_algorithm(SDC_(estimator=nmc, num_eval=nmc.compute_num_total([16, 20, 25, 31][i]))))
        results["rnmc"].append(repeat_algorithm(SDC_(estimator=rnmc, num_eval=400 * 2 ** i)))
        results["lrnmc"].append(repeat_algorithm(SDC_(estimator=lrnmc, num_eval=200  * 2 ** i)))
        results["anmc"].append(repeat_algorithm(SDC_(estimator=anmc, num_eval=50 * 2 ** i)))

        write_pickle_file(output_dir / "sdc.pickle", results)


    results = {"nmc": [], "rnmc": [], "lrnmc": [], "anmc": []}
    SDC_ = partial(SDC, model=forward_model, designs=designs, draw_samples=draw_samples, crn=True)
    for i in range(4):
        results["nmc"].append(repeat_algorithm(SDC_(estimator=nmc, num_eval=nmc.compute_num_total([13, 16, 20, 25][i]))))
        results["rnmc"].append(repeat_algorithm(SDC_(estimator=rnmc, num_eval=400 * 2 ** i)))
        results["lrnmc"].append(repeat_algorithm(SDC_(estimator=lrnmc, num_eval=200 * 2 ** i)))
        results["anmc"].append(repeat_algorithm(SDC_(estimator=anmc, num_eval=25 * 2 ** i)))

        write_pickle_file(output_dir / "sdc_crn.pickle", results)


    results = {}
    for num_criterion in [2, 3]:
        for num_new_eval in [8, 16]:
            for q in [0.95, 0.99, 0.999]:
                ADE_ = partial(ADE, model=forward_model, designs=designs, draw_samples=draw_samples, quantile=q, num_criterion=num_criterion, num_new_eval=num_new_eval)
                results[f'nc{num_criterion}_q{str(q).replace("0.", "")}_n{num_new_eval}'] = {
                    "rnmc": repeat_algorithm(ADE_(estimator=rnmc)),
                    "lrnmc": repeat_algorithm(ADE_(estimator=lrnmc)),
                    "anmc": repeat_algorithm(ADE_(estimator=anmc)),
                }
                write_pickle_file(output_dir / 'ade.pickle', results)


    def compute_eig(estimator, num_eval):
        algorithm = SDC(model=forward_model, designs=designs, draw_samples=draw_samples, estimator=estimator, num_eval=num_eval, crn=True)
        return list(zip(*repeat_parallel(algorithm.compute_eig, num_repeat=10_000)))[0]


    num_samples_list = [8, 16, 32]
    results = {"rnmc": [], "anmc": [], "N": num_samples_list}
    for num_samples in num_samples_list:
        results["rnmc"].append(compute_eig(rnmc, num_samples))
        results["anmc"].append(compute_eig(lrnmc, num_samples))

    write_pickle_file(output_dir / "eig_fixed_N.pickle", results)