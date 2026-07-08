import numpy as np

from .probabilistic_model import output_dir, optimal_design, precomputed_model_outputs, noise_model

from ..plot_functions import plot_ade, plot_sdc_ade, textwidth, fontsize, colors3, plt, load_pickle_file, figure_dir


if __name__ == "__main__":
    plot_sdc_ade(output_dir, optimal_design, (10, 1e6), sdc_without_crn=True, save=True, show=False)
    plot_ade(output_dir, optimal_design, (7e1, 4e3), (93, 101), save=True, show=False)


    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(textwidth, 2.5))
    fig.subplots_adjust(left=0.09, right=0.99, top=0.93, bottom=0.15, wspace=0.7, hspace=0.7)

    maneuvers = load_pickle_file(output_dir / "../../data/maneuvers.pickle")
    time = maneuvers["time"]
    pressure_list = maneuvers["pressure"]

    titles = [r"$\boldsymbol{\xi}_1$", r"$\boldsymbol{\xi}_2$", r"$\boldsymbol{\xi}_3$", r"$\boldsymbol{\xi}_4$"]

    for i in range(4):
        axes[0, i].plot(time, pressure_list[i], color=colors3[0])

        axes[0, i].set_title(titles[i], fontsize=fontsize + 1)
        axes[0, i].set_xlabel("$t$ [s]")
        axes[0, i].set_ylabel(r"$P_{\mathrm{in}}^{(1)}$ [Pa]")
        axes[0, i].set_ylim([700, 2600])
        axes[0, i].set_xlim([-1, 43])

        flow = precomputed_model_outputs[i, 0]
        axes[1, i].plot(time, noise_model(flow, None, None, np.random.default_rng(0)) / 1000, color=colors3[1])

        axes[1, i].set_xlabel("$t$ [s]")
        axes[1, i].set_ylabel(r"$\boldsymbol{y}$ [ml/s]")
        axes[1, i].set_ylim([-35, 35])
        axes[1, i].set_xlim([-2, 43])

    plt.savefig(figure_dir / "lung_maneuvers.pdf", transparent=True)
    # plt.show()
