import numpy as np


def log_mean_exp(arr):
    max_value = np.max(arr)
    return np.log(np.sum(np.exp(arr - max_value))) + max_value - np.log(len(arr))


class ANMC:
    def __init__(self, log_likelihood, noise_model, num_noise, leave_out):
        self.log_likelihood = log_likelihood
        self.noise_model = noise_model
        self.num_noise = num_noise
        self.leave_out = leave_out

    def compute_pmi(self, outputs, theta, xi, rng):
        pmi = np.zeros(len(outputs))
        for _ in range(self.num_noise):
            y_obs = self.noise_model(outputs, theta, xi, rng=rng)

            for i in range(len(y_obs)):
                pmi[i] += self.compute_single_pmi(i, outputs, y_obs, theta, xi)

        pmi /= self.num_noise
        return pmi

    def compute_single_pmi(self, i, outputs, y_obs, theta, xi):
        log_likelihood = self.log_likelihood(outputs, y_obs[i], theta, xi)
        log_likelihood_gt = log_likelihood[i]
        if self.leave_out:
            log_likelihood = np.delete(log_likelihood, i)

        log_evidence = log_mean_exp(log_likelihood)

        return log_likelihood_gt - log_evidence


class NMC:
    def __init__(self, log_likelihood, noise_model):
        self.log_likelihood = log_likelihood
        self.noise_model = noise_model

    def compute_pmi(self, outputs, theta, xi, rng):
        """C = N * (M+1) = M^3 + M^2"""
        num_inner, num_outer = self.compute_num_inner_num_outer(len(outputs))

        y_obs = self.noise_model(outputs[:num_outer], theta, xi, rng=rng)
        pmi = np.zeros(num_outer)
        for i in range(num_outer):
            start_index = num_outer+i*num_inner
            outputs_inner = outputs[start_index:start_index+num_inner]
            theta_inner = theta[start_index:start_index+num_inner]
            log_likelihood = self.log_likelihood(outputs_inner, y_obs[i], theta_inner, xi)
            log_likelihood_gt = self.log_likelihood(outputs[[i]], y_obs[i], theta[[i]], xi)[0]
            log_evidence = log_mean_exp(log_likelihood)
            pmi[i] = (log_likelihood_gt - log_evidence)

        return pmi

    def compute_num_inner_num_outer(self, num_outputs):
        num_inner = int(num_outputs ** (1 / 3))
        while self.compute_num_total(num_inner) > num_outputs:
            num_inner -= 1

        num_outer = num_outputs // (num_inner + 1)
        if num_outer * (num_inner + 1) != num_outputs:
          raise Warning("Wasting resources")

        return num_inner, num_outer

    @staticmethod
    def compute_num_total(num_inner):
        return num_inner ** 2 + num_inner ** 3