"""
---------------------------------------------------
---------------- MODULE: Formatter ----------------
---------------------------------------------------
In this module, the user will find some useful
functions for formatting matplotlib and seaborn
plots, like customizing borders and putting labels
on data

Table of Contents
---------------------------------------------------
1. Initial setup
    1.1 Importing libraries
2. Customizing axis and labels
    2.1 Modifying borders in a figure
    2.2 Putting labels on graphics
---------------------------------------------------
"""

# Author: Thiago Panini
# Data: 29/04/2021


"""
---------------------------------------------------
---------------- 1. INITIAL SETUP -----------------
             1.1 Importing libraries
---------------------------------------------------
"""

# Standard libraries
import matplotlib
from matplotlib.patches import Patch
from matplotlib.axes import Axes

# AnnotateBars class (reference on the class definition)
from dataclasses import dataclass
from typing import *


"""
---------------------------------------------------
---------- 2. FORMATTING AXIS AND LABELS ----------
        2.1 Modifying borders in a figure
---------------------------------------------------
"""

# Formatting spines in a matplotlib plot
def format_spines(ax, right_border=False):
    """
    Modify borders and axis colors of matplotlib figures

    Parameters
    ----------
    :param ax: figura axis created using matplotlib [type: matplotlib.pyplot.axes]
    :param right_border: boolean flag for hiding right border [type: bool, default=False]

    Return
    ------
    This functions has no return besides the customization of matplotlib axis

    Example
    -------
    fig, ax = plt.subplots()
    format_spines(ax=ax, right_border=False)
    """

    # Setting colors on the axis
    ax.spines['bottom'].set_color('#CCCCCC')
    ax.spines['left'].set_color('#CCCCCC')
    ax.spines['top'].set_visible(False)

    # Right border formatting
    if right_border:
        ax.spines['right'].set_color('#CCCCCC')
    else:
        ax.spines['right'].set_color('#FFFFFF')
    ax.patch.set_facecolor('#FFFFFF')


"""
---------------------------------------------------
---------- 2. FORMATTING AXIS AND LABELS ----------
           2.2 Putting labels on graphics
---------------------------------------------------
"""

# Reference: https://towardsdatascience.com/annotating-bar-charts-and-other-matplolib-techniques-cecb54315015
# Creating allias
#Patch = matplotlib.patches.Patch
PosVal = Tuple[float, Tuple[float, float]]
#Axis = matplotlib.axes.Axes
Axis = Axes
PosValFunc = Callable[[Patch], PosVal]

@dataclass
class AnnotateBars:
    font_size: int = 10
    color: str = "black"
    n_dec: int = 2
    def horizontal(self, ax: Axis, centered=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_width()
            div = 2 if centered else 1
            pos = (
                p.get_x() + p.get_width() / div,
                p.get_y() + p.get_height() / 2,
            )
            return value, pos
        ha = "center" if centered else  "left"
        self._annotate(ax, get_vals, ha=ha, va="center")
    def vertical(self, ax: Axis, centered:bool=False):
        def get_vals(p: Patch) -> PosVal:
            value = p.get_height()
            div = 2 if centered else 1
            pos = (p.get_x() + p.get_width() / 2,
                   p.get_y() + p.get_height() / div
            )
            return value, pos
        va = "center" if centered else "bottom"
        self._annotate(ax, get_vals, ha="center", va=va)
    def _annotate(self, ax, func: PosValFunc, **kwargs):
        cfg = {"color": self.color,
               "fontsize": self.font_size, **kwargs}
        for p in ax.patches:
            value, pos = func(p)
            ax.annotate(f"{value:.{self.n_dec}f}", pos, **cfg)


# Putting labels on a pie/donut chart
def make_autopct(values):
    """
    Setting data labels on pie/donut chart

    Parameters
    ----------
    :param values: values of data label [type: np.array]

    Return
    -------
    :return my_autopct: formatted string for putting on chart label
    """

    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct * total / 100.0))

        return '{p:.1f}%\n({v:d})'.format(p=pct, v=val)

    return my_autopct
