"""Monte Carlo integration."""

import numpy as np

from flashquad.methods.rand import _rand
from flashquad.utils import _to_device


def mc(
    xp, dtype, func, intervals, num_points, *, params=None, boundary=None, device=None
):
    """Integrate using Monte Carlo sampling."""
    if not isinstance(num_points, int):
        num_points = int(np.prod(num_points))
    vector_length = params.shape[0] if params is not None else 1

    samples = []
    volume = 1.0
    for bound in intervals:
        a, b = bound[0], bound[1]
        sample = _rand(xp, (vector_length, num_points), dtype, device) * (b - a) + a
        samples.append(sample[..., None])
        volume *= b - a

    if params is not None:
        params = _to_device(params, device)
        params_expanded = [
            xp.reshape(params[:, i], (vector_length, 1, 1))
            for i in range(params.shape[1])
        ]
        Y = func(*samples, *params_expanded)
    else:
        Y = func(*samples)

    if boundary is not None:
        Y = Y * boundary(*samples)

    return volume * xp.mean(Y, axis=1).squeeze()
