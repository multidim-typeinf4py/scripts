from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial, update_wrapper

from scripts import utils


def wrapped_partial(func, *args, **kwargs):
    """http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/"""
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    partial_func.__name__ = f"{partial_func.__name__}{kwargs['topn']}"
    return partial_func


def cpu_executor() -> ProcessPoolExecutor:
    return ProcessPoolExecutor(max_workers=utils.worker_count())


def model_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=1)
