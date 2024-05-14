import logging
import timeit


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(levelname)s] - %(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger


def timing_decorator(func, logger=None):
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        end = timeit.default_timer()
        if logger:
            logger.debug(f"{func.__name__} took {end - start} seconds to run")
        print(f"{func.__name__} took {end - start} seconds to run")
        return result

    return wrapper
