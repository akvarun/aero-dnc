import random
import time

def generate_points(n, x_range=(0, 10000), y_range=(0, 10000)):
    return [(random.uniform(*x_range), random.uniform(*y_range)) for _ in range(n)]


def time_function(func, *args):
    start = time.time()
    result = func(*args)
    elapsed = time.time() - start
    return result, elapsed

