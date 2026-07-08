import logging
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys
import time

from . import decors as dc
from . import utils as u
from . import process_utils as pu
from .pbar import PBar


logger = logging.getLogger(__name__)
                

@dc.embellish(logger,
              message="Applying percentile minmax normalisation...",
              appx=f"{'-'*os.get_terminal_size().columns}")
@dc.timeit(logger, separate=True)
def normalise(sig: ndarray, visualise=False) -> ndarray:

    norm_sig = pu.percMinmaxNorm(sig, 2000)

    if visualise:
        plt.hist(norm_sig, bins=50)
        plt.show()
    
    return norm_sig


@dc.embellish(logger,
              message="Splitting squiggle with optimal splits...",
              appx=f"{'-'*os.get_terminal_size().columns}")
@dc.timeit(logger, separate=True)
def split(sig: ndarray, visualise=False) -> ndarray:
    
    opts_1, gains_1 = pu.optSplit(sig, 24)
    opts_2, gains_2 = pu.optSplit(sig, 12)
    opts = np.concatenate([opts_1, opts_2], axis=0)
    gains = np.concatenate([gains_1, gains_2], axis=0)
    splits, vals = pu.consensus(opts, gains, 4)
    squigs = np.split(sig, splits)
    features = featLoader(squigs)

    if visualise:
        plt.plot(sig)
        plt.scatter(splits + 0.5, pu.minmax(vals), color='red')
        visual = np.concatenate([
            [
                np.full(f[-1].astype('int'), f[0]),
                np.full(f[-1].astype('int'), f[0] + f[1]),
                np.full(f[-1].astype('int'), f[0] - f[1]),
            ] for f in features
        ], axis=1)
        plt.plot(visual[0], color='orange')
        plt.plot(visual[1], color='orange', linestyle='--')
        plt.plot(visual[2], color='orange', linestyle='--')
        plt.show()

    return features


@dc.timeit(logger)
def featLoader(squigs: list(ndarray)) -> ndarray:
    lengths = np.array([len(s) for s in squigs])
    max_len = np.percentile(lengths, q=95).astype('int')
    padded = np.full((len(squigs), max_len), np.nan)
    for i, s in enumerate(squigs):
        if len(s) > max_len:
            s = s[:max_len]
        padded[i, :len(s)] = s
    features = np.stack([
        medians := np.nanmedian(padded, axis=1),
        mads := np.nanmedian(np.abs(padded - medians[:, None]), axis=1),
        lengths,
    ], axis=1)

    return features
