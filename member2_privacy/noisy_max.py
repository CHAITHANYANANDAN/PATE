import numpy as np


def noisy_argmax(counts, scale, rng):
    """counts: (Q, C) or (C,) vote counts. Adds independent Laplace(0, scale) noise to every
    class count of every query and returns the argmax."""
    if scale <= 0:
        raise ValueError("scale must be > 0")
    counts = np.asarray(counts, dtype=float)
    return (counts + rng.laplace(0.0, scale, size=counts.shape)).argmax(axis=-1)
