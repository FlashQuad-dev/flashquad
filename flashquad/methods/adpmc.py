"""Iterative (adaptive) Monte Carlo integration."""

import numpy as np

from flashquad.methods.rand import _rand
from flashquad.utils import _to_device


def adpmc(
    xp,
    dtype,
    func,
    intervals,
    num_points,
    *,
    params=None,
    boundary=None,
    num_iterations=10,
    device=None,
):
    """Integrate using iterative Monte Carlo sampling."""
    if not isinstance(num_points, int):
        num_points = int(np.prod(num_points))
    if params is not None:
        params = _to_device(params, device)
    vector_length = params.shape[0] if params is not None else 1

    volume = 1.0
    for bound in intervals:
        volume *= bound[1] - bound[0]

    for _ in range(num_iterations):
        samples = []
        for bound in intervals:
            a, b = bound[0], bound[1]
            sample = _rand(xp, (vector_length, num_points), dtype, device) * (b - a) + a
            samples.append(sample[..., None])

        if params is not None:
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
