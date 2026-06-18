import numpy as np
import matplotlib.pyplot as plt

def plot_res(filename="res.txt",
             columns=None,
             figsize=(10, 6),
             title=None,
             show=True):
    """
    Plot data from a results file.

    Parameters
    ----------
    filename : str
        Input file.
    columns : list[str] or None
        Columns to plot. If None, plots all columns except t.
    figsize : tuple
        Figure size.
    title : str or None
        Plot title.
    show : bool
        Whether to display the figure.

    Returns
    -------
    fig, ax
    """

    # Read header
    with open(filename, encoding="utf-8") as f:
        header = f.readline().split()

    # Load numeric data
    data = np.loadtxt(filename, skiprows=1)

    # First column is always time
    t = data[:, 0]

    # If no columns specified, plot everything except t
    if columns is None:
        columns = header[1:]

    fig, ax = plt.subplots(figsize=figsize)

    for col in columns:
        if col not in header:
            print(f"Warning: column '{col}' not found")
            continue

        idx = header.index(col)
        ax.plot(t, data[:, idx], label=col)

    ax.set_xlabel(header[0])
    ax.grid(True)

    if len(columns) > 1:
        ax.legend()

    if title:
        ax.set_title(title)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax