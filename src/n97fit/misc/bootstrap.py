import numpy as np


def make_rng(seed: int = None):
    """Random generator PCG64DXSM

    Args:
        seed (int): seed. If None, 128-bit entropy is drawn from the OS.

    Returns:
        (np.random.Generator, int): generator, and the seed to reproduce it
    """

    ss = np.random.SeedSequence(seed)
    return np.random.Generator(np.random.PCG64DXSM(ss)), ss.entropy


def jackknife_scale(frac: float) -> float:
    """Stdev factor of the delete-d jackknife (Shao and Wu, 1989)

    The variance over random subsets without d = frac*n of n data is
    multiplied by (n-d)/d = (1-frac)/frac. The factor is 1 at frac=0.5.

    Args:
        frac (float): fraction of the excluded data

    Returns:
        float: sqrt((1-frac)/frac)
    """

    return np.sqrt((1.0 - frac) / frac)


def check_exclude_frac(exclude_frac: float, nunit: int) -> float:
    """Check the fraction of the excluded data

    Args:
        exclude_frac (float): 0 < exclude_frac < 1
        nunit (int): number of data units

    Returns:
        float: exclude_frac
    """

    try:
        f = float(exclude_frac)
    except (TypeError, ValueError):
        f = np.nan

    if not 0.0 < f < 1.0:
        raise ValueError(f"exclude_frac must be 0 < f < 1: {exclude_frac!r}")
    if not 0 < round(f * nunit) < nunit:
        raise ValueError(
            f"exclude_frac={exclude_frac} excludes no data or all data of {nunit}"
        )
    return f


def exclude_units(
    rng: np.random.Generator, nunit: int, exclude_frac: float
) -> np.ndarray:
    """Draw the data units excluded from a bootstrap member

    Args:
        rng (np.random.Generator): random generator
        nunit (int): number of data units
        exclude_frac (float): fraction of the excluded units

    Returns:
        np.ndarray: indices of round(exclude_frac*nunit) units (sorted)
    """

    nexcl = round(exclude_frac * nunit)
    return np.sort(rng.choice(nunit, nexcl, replace=False))
