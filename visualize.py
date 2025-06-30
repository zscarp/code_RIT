"""
################################################################################
CSCI 633: Biologically-Inspired Intelligent Systems
Version taught by Alexander G. Ororbia II

Visualization of various benchmark functions. Plots 2D and 3D plots.

Note, all implementations assume row-oriented vectors.
################################################################################
"""

import sys
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import cm

from inspect import getmembers, isfunction

import functions

def visualize(name, f, min_bound, max_bound, fig, rows=1, cols=1, idx=1, colorbar=False):
    
    # fig = plot.figure(name)
    ax = fig.add_subplot(rows, cols, idx, projection='3d', title=name)

    step = 0.05
    x = y = np.arange(min_bound, max_bound, step)
    
    X, Y = np.meshgrid(x, y)

    xs = [np.array([x]) for x in zip(np.ravel(X), np.ravel(Y))]
    zs = np.array([f(x) for x in xs])
    Z = zs.reshape(X.shape)

    surface = ax.plot_surface(X, Y, Z, cmap=cm.RdGy)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x)')

    if colorbar:
        # Add a color bar which maps values to colors.
        fig.colorbar(surface, shrink=0.5, aspect=5)


def all_in_one():
    """
    Plots all functions on the same figure.
    """
    funcs = getmembers(functions, isfunction)

    grid_rows = 2
    grid_cols = int(np.ceil(float(len(funcs)) / float(grid_rows)))
    
    HEIGHT_SCALE = 3 # The scale factor for the height
    WIDTH_SCALE = 2


    fig = plot.figure("Function Visualizations", figsize=(grid_cols * WIDTH_SCALE, grid_rows * HEIGHT_SCALE))

    index = 1
    for (name, f) in funcs:
        name = name.replace("_", " ")
        name = name.title()

        metadata = f.metadata if hasattr(f, "metadata") else {}
        min = metadata["min"] if "min" in metadata else -1
        max = metadata["max"] if "max" in metadata else 1

        # print(f"{name}: x \u220A ( {min}, {max} )")
        visualize(name, f, min, max, fig, grid_rows, grid_cols, index)

        index += 1
    
    plot.show()

def single(save=False, root_path="./"):
    """
    Visualize one function at a time.

    # Arguments
    * `save` - Boolean flag. If true, will save the image.
    * `root_path` - Path to the directory to save images in.

    # Notes
    If save is set to true, will save the plot as a png in the directory specified by `root_path`.
    """

    funcs = getmembers(functions, isfunction)

    for (name, f) in funcs:
        name = name.replace("_", " ")
        name = name.title()

        fig = plot.figure("Function Visualization")

        metadata = f.metadata if hasattr(f, "metadata") else {}
        min = metadata["min"] if "min" in metadata else -1
        max = metadata["max"] if "max" in metadata else 1

        visualize(name, f, min, max, fig, colorbar=True)

        name = name.replace(" ", "_")
        if save:
            plot.savefig(f"{root_path}{name}.png", bbox_inches="tight")

        plot.show()

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        save = True if sys.argv[1].lower() == "true" else False
        root = sys.argv[2] if len(sys.argv) > 2 else "./"
        single(save, root)
    else:
        all_in_one()
