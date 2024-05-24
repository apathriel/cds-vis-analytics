import logging
import timeit


import logging

def get_logger(name: str) -> logging.Logger:
    """
    Creates and configures a logger with the specified name.

    Parameters:
        name (str): The name of the logger. Convention is to use __name__.

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


def timing_decorator(func, logger=None):
    """
    A decorator function that measures the execution time of a given function.

    Parameters:
    - func: The function to be decorated.
    - logger: (optional) A logger object for logging the execution time.

    Returns:
    - The decorated function.

    Example usage:
    @timing_decorator
    def my_function():
        # code goes here
    """

    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        if logger:
            logger.debug(f"{func.__name__} took {end - start} seconds to run")
        print(f"{func.__name__} took {end - start} seconds to run")
        return result

    return wrapper
