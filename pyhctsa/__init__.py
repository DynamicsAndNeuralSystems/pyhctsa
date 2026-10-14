import logging
from importlib.metadata import version, PackageNotFoundError

# silent unless the application configures logging
logging.getLogger("pyhctsa").addHandler(logging.NullHandler())

try:
    __version__ = version("pyhctsa")
except PackageNotFoundError:
    __version__ = "0.0.0"