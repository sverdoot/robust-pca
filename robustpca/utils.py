import time
from functools import wraps


def time_printer(func):
    @wraps(func)
    def with_time(*args, **kwargs):
        s = time.time()
        result = func(*args, **kwargs)
        e = time.time()
        print(f"Elapsed: {e - s:.2f}")
        return result

    return with_time
