import numpy as np
from scipy import signal


class FiltFilt:
    """Zero-phase filter, same as signal.sosfiltfilt(sos, x, padtype=None)

    sosfiltfilt computes the initial conditions (sosfilt_zi) on every call.
    They depend only on the filter, so they are computed once here.
    """

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi = signal.sosfilt_zi(sos)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        # forward, starting from the steady state for x[0]
        y, _ = signal.sosfilt(self.sos, x, zi=self.zi * x[:1])
        # backward, starting from the steady state for the last value
        y, _ = signal.sosfilt(self.sos, y[::-1], zi=self.zi * y[-1:])
        return y[::-1]
