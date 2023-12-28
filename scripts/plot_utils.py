import logging
import matplotlib.pyplot as plt


FIG_SIZE = (12, 10)
LINE_WIDTH = 5.0
GIRD_WIDTH = 2.0
FONT_SIZE = 50
LABEL_SIZE = 40


def configure_matplotlib_logging():
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def simple_plot(steps, values, y_label, save_path):
    """
    Generate a simple line plot and save it to a specified path.

    Parameters:
    - steps (list): X-axis values (e.g., rounds).
    - values (list): Y-axis values to be plotted.
    - y_label (str): Label for the Y-axis.
    - save_path (str): File path to save the plot.

    Returns:
    - fig (matplotlib.figure.Figure): The generated figure.
    - ax (matplotlib.axes._axes.Axes): The axes of the plot.
    """
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    ax.plot(steps, values, linewidth=LINE_WIDTH)

    ax.grid(True, linewidth=GIRD_WIDTH)

    ax.set_ylabel(f"{y_label}", fontsize=FONT_SIZE)
    ax.set_xlabel("Rounds", fontsize=FONT_SIZE)
    ax.tick_params(axis='both', labelsize=LABEL_SIZE)

    fig.savefig(save_path, bbox_inches='tight')

    return fig, ax


def close_fig(fig):
    plt.close(fig)

