import logging
import os


def setup_logger(output_folder, logger_name="compute_logger"):
    """
    Create and configure a logger.
    """

    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers if called multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    log_folder = os.path.join(output_folder, "logs")
    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, "interpolation.log")

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Terminal handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger