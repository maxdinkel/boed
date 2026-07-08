import numpy as np


def bootstrap_mean(samples, num_bootstrap_samples, rng):
    n = len(samples)
    return samples[rng.integers(0, n, size=(num_bootstrap_samples, n))].mean(axis=1)


def append(arr_1, arr_2):
    return np.concatenate([arr_1.reshape(-1, arr_2.shape[1]), arr_2], axis=0)


class ADE:
    def __init__(self, model, designs, draw_samples, estimator, num_new_eval=8, num_criterion=3, quantile=0.99, num_bootstrap=100_000):
        self.model = model
        self.designs = designs
        self.draw_samples = draw_samples
        self.estimator = estimator
        self.num_new_eval = num_new_eval

        self.num_criterion = num_criterion
        self.quantile = quantile
        self.num_bootstrap = num_bootstrap

        self.num_designs = len(self.designs)


    def pre_run(self):
        candidate_indices = [i for i in range(self.num_designs)]
        all_outputs = [np.empty((0, 0)) for _ in range(self.num_designs)]
        num_evaluations = np.zeros(self.num_designs, dtype=int)
        counter = np.zeros(self.num_designs, dtype=int)
        return candidate_indices, all_outputs, num_evaluations, counter

    def run(self, rng):
        candidate_indices, all_outputs, num_evaluations, counter = self.pre_run()
        all_theta = np.empty((0, 0))

        while len(candidate_indices) > 1:
            new_theta = self.draw_samples(self.num_new_eval, rng=rng)
            all_theta = append(all_theta, new_theta)

            pmi = []
            for i in candidate_indices:
                new_outputs = self.model(new_theta, self.designs[i])
                num_evaluations[i] += self.num_new_eval
                all_outputs[i] = append(all_outputs[i], new_outputs)

                pmi.append(self.estimator.compute_pmi(all_outputs[i], all_theta, self.designs[i], rng=rng))
            eig_mean = np.mean(pmi, axis=1)

            j_max = np.argmax(eig_mean)
            for j, i in enumerate(candidate_indices.copy()):
                if j != j_max:
                    pmi_diff = pmi[j_max] - pmi[j]
                    bootstrap_means = bootstrap_mean(pmi_diff, self.num_bootstrap, rng=rng)

                    if np.mean(bootstrap_means > 0) > self.quantile:
                        counter[i] += 1
                    else:
                        counter[i] = 0
                    if counter[i] >= self.num_criterion:
                        candidate_indices.remove(i)

        optimal_design = self.designs[candidate_indices[0]]
        return optimal_design, num_evaluations


class SDC:
    def __init__(self, model, designs, draw_samples, estimator, num_eval, crn):
        self.model = model
        self.designs = designs
        self.draw_samples = draw_samples
        self.estimator = estimator
        self.num_eval = num_eval
        self.num_designs = len(self.designs)
        self.crn = crn

    def run(self, rng):
        eig, num_evaluations = self.compute_eig(rng)
        optimal_design = self.designs[np.argmax(eig)]

        return optimal_design, num_evaluations

    def compute_eig(self, rng):
        eig = []
        if self.crn:
            theta = self.draw_samples(self.num_eval, rng=rng)
        for xi in self.designs:
            if not self.crn:
                theta = self.draw_samples(self.num_eval, rng=rng)
            outputs = self.model(theta, xi)
            eig.append(np.mean(self.estimator.compute_pmi(outputs, theta, xi, rng=rng)))

        num_evaluations = np.full(self.num_designs, self.num_eval)

        return np.array(eig), num_evaluations
