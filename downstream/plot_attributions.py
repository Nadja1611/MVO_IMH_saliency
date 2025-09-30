"""Plot signals and corresponding attributions derived from a deep learning model."""

from typing import Any, Union, List, Tuple, Dict

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np


def add_channel_label(ax: matplotlib.axes.Axes, channel: str) -> None:
    """Add a label in the upper left corner of the axis."""
    txt = ax.text(
        0,
        1,
        channel,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize = 18,
        bbox=dict(
            facecolor="white",
            edgecolor="lightgrey",
            alpha=0.9,
            boxstyle="square,pad=0.3",
        ),
    )

    # here we need to adjust for the padding so that the box does not overshoot.
    # we need to draw first to get actual figure coordinates.
    ax.get_figure().canvas.draw()  # type: ignore
    bb = txt.get_bbox_patch().get_window_extent()  # type: ignore
    tb = txt.get_window_extent()
    inv = ax.transAxes.inverted()
    dx, dy = inv.transform([tb.x0 + 1, tb.y0 + 1]) - inv.transform([bb.x0, bb.y0])
    txt.set_position((dx, 1 - dy))


def get_color_range(attribution: np.ndarray, abs: bool = False) -> Tuple[float, float]:
    """Define color range for plotting."""
    if abs:
        # attribution = attribution / np.max(attribution)
        return 0, np.max(attribution)

    maxval = np.max(np.abs(attribution))
    return -maxval, maxval


def plot_attribution(
    signal: np.ndarray,
    attribution: np.ndarray,
    ncols: int = 2,
    channels_last=False,
    fs: float = 1,
    names: Union[List[str], None] = None,
    cmap: Any = None,
    abs: bool = False,
    line_kws: Union[Dict[str, Any], None] = None,
    vscale: Union[Tuple[float, float], None] = None,
    **kwargs,
) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
    """Plot signal and corresponding attributions as colored background."""
    line_kws = {"color": ".05", "lw": 1.5} | (line_kws or {})
    if channels_last:
        signal = signal.T
        attribution = attribution.T

    if signal.ndim == 1:
        signal = np.expand_dims(signal, axis=0)

    if attribution.ndim == 1:
        attribution = np.expand_dims(attribution, axis=0)
        if signal.shape[0] > 1:
            attribution = np.repeat(attribution, signal.shape[0], axis=0)

    t = np.arange(signal.shape[1]) / fs
    nrows = int(np.ceil(signal.shape[0] / ncols))
    fig, axarr = plt.subplots(nrows, ncols, squeeze=False, **kwargs)
    for i, y in enumerate(signal):
        ax = axarr[i % nrows, i // nrows]
        ax.plot(t, y, **line_kws)
        if names is not None:
            add_channel_label(ax, names[i])
        ax.set_xlim(t[0], t[-1])

    if vscale is not None:
        vmin, vmax = vscale
    else:
        vmin, vmax = get_color_range(attribution, abs=abs)

    for i, attrib in enumerate(attribution):
        ax = axarr[i % nrows, i // nrows]
        ylim = ax.get_ylim()
        ax.imshow(
            np.expand_dims(attrib, 0),
            aspect="auto",
            extent=(t[0], t[-1], ylim[0], ylim[1]),
            interpolation="none",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    return fig, axarr
