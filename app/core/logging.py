"""Logging configuration for RegGraph."""
import logging
import sys


def setup_logging(debug: bool = False) -> logging.Logger:
    """Configure and return root application logger."""
    log_level = logging.DEBUG if debug else logging.INFO

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger("reggraph")
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if not root_logger.handlers:
        root_logger.addHandler(handler)

    return root_logger


logger = setup_logging()
