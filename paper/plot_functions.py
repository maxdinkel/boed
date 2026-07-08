import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import NullFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

from .helper_functions import load_pickle_file

cmap = colormaps.get_cmap('plasma')
plt.rcParams.update(plt.rcParamsDefault)
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')
plt.rcParams["font.family"] = "Modern"
fontsize = 8
textwidth = 6.5
plt.rcParams.update({'font.size': fontsize})

colors = [cmap(0), cmap(0.266), cmap(0.533), cmap(0.8)]
colors3 = [cmap(0), cmap(0.4), cmap(0.8)]

figure_dir = Path(__file__).parent / 'figures'



def compute_accuracy_and_eval(designs, num_eval, optimal_design):
    accuracy = np.sum(designs == optimal_design) / len(designs) * 100
    mean_eval = np.mean(np.sum(num_eval, axis=1))
    return accuracy, mean_eval


def configure_ax(ax):
    ax.set(xscale="log", ylabel=r"Accuracy in \%")
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid()



def plot_ade(output_dir, optimal_design, xlim, ylim, save=True, show=False):
    fig, axes = plt.subplots(3, 1, figsize=(textwidth, 7.5))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.12, hspace=0.5)

    q_colors = {'95': colors3[0], '99': colors3[1], '999': colors3[2]}
    nc_markers = {2: {"marker": "o", "markersize": 5.}, 3: {"marker": "D", "markersize": 4}}
    offset = (ylim[1] - ylim[0]) / 10
    all_results = load_pickle_file(output_dir / "ade.pickle")

    for nc, marker in nc_markers.items():
        for q, color in q_colors.items():
            n_fill = {8: color, 16: 'white'}
            for n, fill in n_fill.items():
                results = all_results[f"nc{nc}_q{q}_n{n}"]

                for m, (method, res) in enumerate(results.items()):
                    accuracy, mean_eval = compute_accuracy_and_eval(res["design"], res["num_eval"], optimal_design)
                    axes[m].plot(mean_eval, accuracy, color=fill, linestyle="", markeredgecolor=color, **marker)
                    axes[m].set_title(method.upper())

                    if nc == 3 and q == '99' and n == 8:
                        axes[m].plot(mean_eval, accuracy - offset, marker=r"$\uparrow$", color="k", markersize=8)

    for i, ax in enumerate(axes):
        configure_ax(ax)
        ax.set(xlim=xlim, ylim=ylim, xlabel="Average total number of model evaluations")

    handles = (
        [mpatches.Patch(color=color, label=rf"$q=0.{q}$") for q, color in q_colors.items()] +
        [Line2D([0], [0], linestyle="", label=rf"$N_C={nc}$", color="k", **marker) for nc, marker in nc_markers.items()]
    )

    fig.legend(handles=handles, loc="lower center", ncol=7)

    if save:
        fig.savefig(figure_dir / f"ade_{output_dir.parent.name}.pdf", transparent=True)
    if show:
        plt.show()




def plot_sdc_ade(output_dir, optimal_design, xlim, sdc_without_crn, save=True, show=False):
    fig, ax = plt.subplots(figsize=(textwidth, 2.1))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.99, bottom=0.18)


    res_styles = {"sdc_crn.pickle" : {"linestyle": "-", "marker": "x", "label": "SDC w/ CRN"}}
    if sdc_without_crn:
        res_styles["sdc.pickle"] = {"linestyle": ":", "marker": ".", "label": "SDC w/o CRN"}
    color_map = {"nmc": colors[0], "rnmc": colors[1], "lrnmc": colors[2], "anmc": colors[3]}

    for file, style in res_styles.items():
        results = load_pickle_file(output_dir / file)

        for i, (method, method_results) in enumerate(results.items()):
            accuracy = []
            mean_eval = []

            for row in method_results:
                acc, me = compute_accuracy_and_eval(row["design"], row["num_eval"], optimal_design)
                accuracy.append(acc)
                mean_eval.append(me)
            print(method_results[0]["num_eval"][0][0])
            print(accuracy)
            ax.plot(mean_eval, accuracy, color=color_map[method], **style)


    results = load_pickle_file(output_dir / "ade.pickle")["nc3_q99_n8"]
    ade_style = {"marker": "d", "markersize": 5}
    for i, (method, res) in enumerate(results.items()):
        acc, me = compute_accuracy_and_eval(res["design"], res["num_eval"], optimal_design)
        print(me, acc)
        ax.plot(me, acc, color=color_map[method], **ade_style)

    handles = (
        [mpatches.Patch(color=color, label=method.upper()) for method, color in color_map.items()]
        + [Line2D([0], [0], color="k", **style) for style in res_styles.values()]
        + [Line2D([0], [0], linestyle="", label="ADE", color="k", **ade_style)]
    )

    configure_ax(ax)
    ax.set(xlim=xlim, xlabel="(Average) total number of model evaluations")
    ax.legend(handles=handles, loc="best", framealpha=1)

    if save:
        fig.savefig(figure_dir / f"sdc_ade_{output_dir.parent.name}.pdf", transparent=True)
    if show:
        plt.show()
