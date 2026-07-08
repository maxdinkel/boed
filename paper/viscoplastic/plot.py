from .probabilistic_model import output_dir, optimal_design
from ..plot_functions import plot_ade, plot_sdc_ade


if __name__ == "__main__":
    plot_sdc_ade(output_dir, optimal_design, (60, 1e4), sdc_without_crn=False, save=True, show=False)
    plot_ade(output_dir, optimal_design, (2e2, 1500), (93, 101), save=True, show=False)
