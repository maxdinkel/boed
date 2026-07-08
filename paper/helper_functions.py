import pickle
from pathlib import Path

import numpy as np

from boed.nmc import ANMC, NMC


def write_pickle_file(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle_file(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def get_estimators(log_likelihood, noise_model):
    nmc = NMC(log_likelihood, noise_model)
    rnmc = ANMC(log_likelihood, noise_model, num_noise=1, leave_out=False)
    lrnmc = ANMC(log_likelihood, noise_model, num_noise=1, leave_out=True)
    anmc = ANMC(log_likelihood, noise_model, num_noise=100, leave_out=True)
    return nmc, rnmc, lrnmc, anmc


class PrecomputedModel:
    def __init__(self, outputs, permutation=None):
        self.outputs = outputs
        self.start_index = 0
        if permutation is None:
            permutation = np.arange(len(self.outputs[0]))
        self.permutation = permutation


    def __call__(self, theta, xi):
        if len(self.outputs[0]) < max(theta):
            raise ValueError("Too few output samples left!")
        return self.outputs[xi][self.permutation[theta[:, 0]]]

    def draw_samples(self, num_draws, rng):
        samples = np.arange(self.start_index, self.start_index+num_draws)[:, np.newaxis]
        self.start_index += num_draws

        return samples


def write_latex_table(output_dir, results, precision=3, color="cyellow"):
    keys = list(results.keys())
    n_designs = len(results[keys[0]])

    row_best = {key: int(np.argmax(results[key])) for key in keys}
    header_highlight = set(row_best.values())

    def hl(s, active):
        return rf"\cellcolor{{{color}}}{s}" if active else s

    lines = []
    lines.append(r"\begin{tabular}{" + "|" + "|".join(["c"] * (n_designs + 1)) + "|}")
    lines.append(r"\hline")

    # Header row
    lines.append(
        " & ".join(
            [r"$\xib$"]
            + [hl(rf"$\xib_{{{j + 1}}}$", j in header_highlight) for j in range(n_designs)]
        )
        + r" \\"
    )
    lines.append(r"\hline")

    # Data rows
    for key in keys:
        vals = results[key]
        best_j = row_best[key]

        lines.append(
            " & ".join(
                [rf"$\widehat{{\mathrm{{EIG}}}}_{{\mathrm{{{key.upper()}}}}}(\xib)$"]
                + [hl(f"{vals[j]:.{precision}f}", j == best_j) for j in range(n_designs)]
            )
            + r" \\"
        )
        lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    tex = "\n".join(lines)
    (output_dir.parents[1] / f"tables/reference_eig_{output_dir.parent.name}.tex").write_text(tex, encoding="utf-8")
