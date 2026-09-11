"""Explicit, scoped plot output; no global Matplotlib monkey patches."""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

_destination = ContextVar("figure_destination", default=None)
_date_range = ContextVar("dataset_date_range", default="")
_counter = ContextVar("figure_counter", default=0)


def init_dataset_context(df, date_col="gameDateTimeEst"):
    dates = pd.to_datetime(df[date_col])
    _date_range.set(f"{dates.min().date()} to {dates.max().date()}")


def dataset_title(title):
    suffix = _date_range.get()
    return f"{title}\n{suffix}" if suffix and suffix not in str(title) else title


@contextmanager
def figure_output(path):
    tokens = (_destination.set(Path(path)), _date_range.set(""), _counter.set(0))
    with plt.rc_context({"figure.figsize": (14, 8)}):
        try:
            yield
        finally:
            plt.close("all")
            for var, token in zip((_destination, _date_range, _counter), tokens):
                var.reset(token)


def show_figures():
    """Export previously interactive figures and release their memory immediately."""
    destination = _destination.get()
    if destination is None:
        plt.show()
        return
    for number in plt.get_fignums():
        fig = plt.figure(number)
        index = _counter.get() + 1
        _counter.set(index)
        fig.savefig(destination / f"descriptive_{index:03d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
