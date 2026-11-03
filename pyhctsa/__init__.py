import logging
from importlib.metadata import version, PackageNotFoundError

# silent unless the application configures logging
logging.getLogger("pyhctsa").addHandler(logging.NullHandler())

try:
    __version__ = version("pyhctsa")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .quality import Quality, classify, quality_labels, quality_summary  # noqa: E402
from .results import Results, load_results, save_results  # noqa: E402
