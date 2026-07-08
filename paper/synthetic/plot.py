from .probabilistic_model import output_dir, optimal_design, true_eig, designs
from ..plot_functions import plot_ade, plot_sdc_ade, textwidth, load_pickle_file, plt, colors3, fontsize, figure_dir


if __name__ == "__main__":
    plot_sdc_ade(output_dir, optimal_design, (1e2, 1e6), sdc_without_crn=True, save=True, show=False)
    plot_ade(output_dir, optimal_design, (2e2, 5e3), (92, 101), save=True, show=False)

    import numpy as np
    from pathlib import Path


    def hl(s, active):
        return rf"\cellcolor{{cyellow}}{s}" if active else s

    lines = []
    lines.append(r"\begin{tabular}{" + "|" + "|".join(["c"] * (len(designs) + 1)) + "|}")
    lines.append(r"\hline")

    header = [r"$\xi$"] + [hl(f"{design}", design == optimal_design) for design in designs]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\hline")

    row = [r"$\mathrm{EIG}(\xi)$"] + [
        hl(f"{eig:.3f}", design == optimal_design) for design, eig in zip(designs, true_eig)
    ]
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")

    lines.append(r"\end{tabular}")
    tex = "\n".join(lines)

    (output_dir.parents[1] / "tables/true_eig_synthetic.tex").write_text(tex, encoding="utf-8")


    fig, axes = plt.subplots(1, 2, figsize=(textwidth, 2.2))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.32, wspace=0.6, hspace=0.4)

    sorted_indices = np.argsort(true_eig)

    results = load_pickle_file(output_dir / 'eig_fixed_N.pickle')
    num_samples = results.pop("N")


    for k, (method, method_results) in enumerate(results.items()):
        labels = []

        for i, n in enumerate(num_samples):
            eig_expectation = np.mean(method_results[i], axis=0)
            bias = eig_expectation - true_eig
            labels.append(rf"$N={n}$")
            axes[k].plot(true_eig[sorted_indices], bias[sorted_indices], marker='.', color=colors3[i])


        axes[k].axhline(y=0, linestyle='--', color="k", alpha=0.2, zorder=0)

        axes[k].set_title(method.upper(), size=fontsize + 1)
        axes[k].set_xlim(0, 9)
        axes[k].set_ylabel(r"$\mathbb{E}\big[\widehat{\mathrm{EIG}}\big] - \mathrm{EIG}$", size=fontsize)
        axes[k].set_xlabel(r"$\mathrm{EIG}$", size=fontsize)


    fig.legend(labels, fontsize=fontsize, ncols=3, loc="lower center")

    fig.savefig(figure_dir / f"bias_{output_dir.parent.name}.pdf", transparent=True)
    # plt.show()