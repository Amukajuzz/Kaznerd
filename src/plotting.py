"""Matplotlib styling and reusable chart helpers (static PNGs for the report)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
TEXT_CMAP = LinearSegmentedColormap.from_list("text_blue", BLUE_RAMP[3:])  # readable on light surface


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


def barh(labels, values, title, path, xlabel="", color=SERIES[0], fmt="{:,.0f}", figsize=(8, 6)):
    """Horizontal bar chart, largest on top, with value labels at bar ends."""
    setup_style()
    fig, ax = plt.subplots(figsize=figsize)
    labels, values = list(labels)[::-1], list(values)[::-1]
    bars = ax.barh(labels, values, color=color, height=0.7)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    vmax = max(values) if values else 1
    for b, v in zip(bars, values):
        ax.text(b.get_width() + vmax * 0.01, b.get_y() + b.get_height() / 2, fmt.format(v),
                va="center", fontsize=8, color=INK_2)
    ax.set_xlim(0, vmax * 1.12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def grouped_bars(categories, series, title, path, ylabel="", ylim=None, fmt="{:.3f}"):
    """Grouped vertical bars. `series` is {name: [values per category]} (<= 8 series)."""
    setup_style()
    n = len(series)
    width = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.4), 5))
    for i, (name, vals) in enumerate(series.items()):
        xs = [j + (i - (n - 1) / 2) * width for j in range(len(categories))]
        bars = ax.bar(xs, vals, width=width * 0.92, color=SERIES[i % len(SERIES)], label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v),
                    ha="center", va="bottom", fontsize=7, color=INK_2)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=15, ha="right")
    ax.grid(axis="x", visible=False)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=min(n, 3), loc="upper left", bbox_to_anchor=(0, -0.22))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
