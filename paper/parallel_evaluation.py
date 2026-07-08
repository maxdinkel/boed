import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
num_cpus = os.cpu_count()
import numpy as np

from tqdm.contrib.concurrent import process_map


def repeat_parallel(function, num_repeat=1000, num_worker=num_cpus):
    rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(0).spawn(num_repeat)]
    return process_map(function, rngs, max_workers=num_worker, chunksize=1)