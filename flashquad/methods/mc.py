"""Monte Carlo integration."""

import numpy as np


def mc(xp, dtype, func, intervals, num_points, *, params=None, boundary=None):
    """Integrate using Monte Carlo sampling."""
    vector_length = params.shape[0] if params is not None else 1

    samples = []
    volume = 1.0
    for bound in intervals:
        a, b = bound[0], bound[1]
        raw = np.random.rand(vector_length, num_points)
        sample = xp.asarray(raw, dtype=dtype) * (b - a) + a
        samples.append(sample[..., None])
        volume *= b - a

    if params is not None:
        params_expanded = [
            xp.reshape(params[:, i], (vector_length, 1, 1))
            for i in range(params.shape[1])
        ]
        Y = func(*samples, params_expanded)
    else:
        Y = func(*samples)

    if boundary is not None:
        Y = Y * boundary(*samples).astype(Y.dtype)

    return volume * xp.mean(Y, axis=1).squeeze()
