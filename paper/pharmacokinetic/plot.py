from .probabilistic_model import output_dir, optimal_design, design_times
from ..plot_functions import plot_ade, plot_sdc_ade, plt, colors3, textwidth, np, figure_dir


if __name__ == "__main__":
    plot_sdc_ade(output_dir, optimal_design, (5e2, 5e6), sdc_without_crn=True, save=True, show=False)
    plot_ade(output_dir, optimal_design, (4e2, 3e3), (95.5, 100.5), save=True, show=False)

    fig, ax = plt.subplots(figsize=(textwidth, 2.5))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.99, bottom=0.25)

    ax.axvline(0, color='grey', linestyle='dashed')
    ax.axvline(24, color='grey', linestyle='dashed')

    for i, approach in enumerate(["geometric", "even", "beta"]):
        for j in range(5):
            index = 5*i+j
            ax.plot(design_times[index, :], -np.ones(len(design_times))*index, '.', color=colors3[i], label=approach)

    ax.set_xlabel("t")

    labels = [rf"$\boldsymbol{{\xi}}_{{{i}}}$" for i in range(1, 16)]
    ax.set_yticks(range(0, -15, -1), labels=labels)
    ax.set_xticks(range(0, 28, 4))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles=handles[::5], labels=labels[::5], loc='lower center', ncol=3)
    plt.savefig(figure_dir / "pharmacokinetic_desings.pdf", transparent=True)
    # plt.show()