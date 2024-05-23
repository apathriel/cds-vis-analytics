import logging


import logging

def get_logger(name: str) -> logging.Logger:
    """
    Create and configure a logger with the specified name.

    Parameters:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The configured logger object.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] - %(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger
