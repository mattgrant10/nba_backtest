"""One stream for tables and diagnostics, restored even when a stage fails."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import logging
import sys


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def research_logging(path, verbose=False):
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    with open(path, "w", encoding="utf-8", buffering=1) as stream:
        target = Tee(sys.stdout, stream) if verbose else stream
        handler = logging.StreamHandler(target)
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        root.handlers = [handler]
        root.setLevel(logging.INFO)
        try:
            with redirect_stdout(target), redirect_stderr(target):
                yield
        finally:
            root.handlers = handlers
            root.setLevel(level)
            handler.close()
