# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Facilities for reporting and collecting training statistics across
multiple processes and devices. The interface is designed to minimize
synchronization overhead as well as the amount of boilerplate in user
code."""

import re
import numpy as np
import torch
import dnnlib

from . import misc

#----------------------------------------------------------------------------

_num_moments    = 3             # [num_scalars, sum_of_scalars, sum_of_squares]
_reduce_dtype   = torch.float32 # Data type to use for initial per-tensor reduction.
_counter_dtype  = torch.float64 # Data type to use for the internal counters.
_counters       = dict()        # Running counters on each device, updated by report(): name => device => torch.Tensor
_cumulative     = dict()        # Cumulative counters on the CPU, updated by _sync(): name => torch.Tensor

#----------------------------------------------------------------------------

@misc.profiled_function
def report0(name, value):
    r"""Broadcasts the given set of scalars to all interested instances of
    `Collector`, across device and process boundaries.

    This function is expected to be extremely cheap and can be safely
    called from anywhere in the training loop, loss function, or inside a
    `torch.nn.Module`.

    Warning: The current implementation expects the set of unique names to
    be consistent across processes. Please make sure that `report()` is
    called at least once for each unique name by each process, and in the
    same order. If a given process has no scalars to broadcast, it can do
    `report(name, [])` (empty list).

    Args:
        name:   Arbitrary string specifying the name of the statistic.
                Averages are accumulated separately for each unique name.
        value:  Arbitrary set of scalars. Can be a list, tuple,
                NumPy array, PyTorch tensor, or Python scalar.

    Returns:
        The same `value` that was passed in.
    """

    elems = torch.as_tensor(value)
    if elems.numel() == 0:
        return value

    elems = elems.detach().flatten().to(_reduce_dtype)
    moments = torch.stack([
        torch.ones_like(elems).sum(),
        elems.sum(),
        elems.square().sum(),
    ])
    assert moments.ndim == 1 and moments.shape[0] == _num_moments
    moments = moments.to(_counter_dtype)

    if name not in _counters:
        _counters[name] = torch.zeros_like(moments)
    _counters[name].add_(moments)
    return value

#----------------------------------------------------------------------------

class Collector:
    r"""Collects the scalars broadcasted by `report()` and `report0()` and
    computes their long-term averages (mean and standard deviation) over
    user-defined periods of time.

    The averages are first collected into internal counters that are not
    directly visible to the user. They are then copied to the user-visible
    state as a result of calling `update()` and can then be queried using
    `mean()`, `std()`, `as_dict()`, etc. Calling `update()` also resets the
    internal counters for the next round, so that the user-visible state
    effectively reflects averages collected between the last two calls to
    `update()`.

    Args:
        regex:          Regular expression defining which statistics to
                        collect. The default is to collect everything.
        keep_previous:  Whether to retain the previous averages if no
                        scalars were collected on a given round
                        (default: True).
    """
    def __init__(self, regex='.*', keep_previous=True):
        self._regex = re.compile(regex)
        self._keep_previous = keep_previous
        self._cumulative = dict()
        self._moments = dict()
        self.update()
        self._moments.clear()

    def names(self):
        r"""Returns the names of all statistics broadcasted so far that
        match the regular expression specified at construction time.
        """
        return [name for name in _counters if self._regex.fullmatch(name)]
    
    def update(self):
        r"""Copies current values of the internal counters to the
        user-visible state and resets them for the next round.

        If `keep_previous=True` was specified at construction time, the
        operation is skipped for statistics that have received no scalars
        since the last update, retaining their previous averages.

        This method performs a number of GPU-to-CPU transfers and one
        `torch.distributed.all_reduce()`. It is intended to be called
        periodically in the main training loop, typically once every
        N training steps.
        """
        if not self._keep_previous:
            self._moments.clear()
        for name, cumulative in _get_cumulative(self.names()):
            if name not in self._cumulative:
                self._cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
            delta = cumulative - self._cumulative[name]
            self._cumulative[name].copy_(cumulative)
            if float(delta[0]) != 0:
                self._moments[name] = delta
    
    def reset(self, ex='Eval/.*'):
        r"""
        Clear the variables that contains name.
        It is used to save a variable in certain intervals (e.g. evaluation)
        """
        regex = re.compile(ex)
        names = [name  for name in _counters if regex.fullmatch(name)]
        for name in names:
            _counters[name] = torch.zeros_like(_counters[name])

    def _get_delta(self, name):
        r"""Returns the raw moments that were accumulated for the given
        statistic between the last two calls to `update()`, or zero if
        no scalars were collected.
        """
        assert self._regex.fullmatch(name)
        if name not in self._moments:
            self._moments[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        return self._moments[name]

    def num(self, name):
        r"""Returns the number of scalars that were accumulated for the given
        statistic between the last two calls to `update()`, or zero if
        no scalars were collected.
        """
        delta = self._get_delta(name)
        return int(delta[0])

    def mean(self, name):
        r"""Returns the mean of the scalars that were accumulated for the
        given statistic between the last two calls to `update()`, or NaN if
        no scalars were collected.
        """
        delta = self._get_delta(name)
        if int(delta[0]) == 0:
            return float('nan')
        return float(delta[1] / delta[0])

    def std(self, name):
        r"""Returns the standard deviation of the scalars that were
        accumulated for the given statistic between the last two calls to
        `update()`, or NaN if no scalars were collected.
        """
        delta = self._get_delta(name)
        if int(delta[0]) == 0 or not np.isfinite(float(delta[1])):
            return float('nan')
        if int(delta[0]) == 1:
            return float(0)
        mean = float(delta[1] / delta[0])
        raw_var = float(delta[2] / delta[0])
        return np.sqrt(max(raw_var - np.square(mean), 0))

    def as_dict(self):
        r"""Returns the averages accumulated between the last two calls to
        `update()` as an `dnnlib.EasyDict`. The contents are as follows:

            dnnlib.EasyDict(
                NAME = dnnlib.EasyDict(num=FLOAT, mean=FLOAT, std=FLOAT),
                ...
            )
        """
        stats = dnnlib.EasyDict()
        for name in self.names():
            stats[name] = dnnlib.EasyDict(num=self.num(name), mean=self.mean(name), std=self.std(name))
        return stats

    def __getitem__(self, name):
        r"""Convenience getter.
        `collector[name]` is a synonym for `collector.mean(name)`.
        """
        return self.mean(name)

#----------------------------------------------------------------------------

def _get_cumulative(names):
    r"""Synchronize the global cumulative counters across devices and
    processes. Called internally by `Collector.update()`.
    """

    if len(names) == 0:
        return []

    # Collect deltas within current rank.
    deltas = []
    for name in names:
        delta = _counters[name]
        deltas.append(delta)
    deltas = torch.stack(deltas)

    # Update cumulative values.
    deltas = deltas.cpu()
    for idx, name in enumerate(names):
        if name not in _cumulative:
            _cumulative[name] = torch.zeros([_num_moments], dtype=_counter_dtype)
        _cumulative[name].add_(deltas[idx])

    # Return name-value pairs.
    return [(name, _cumulative[name]) for name in names]

#----------------------------------------------------------------------------
# Convenience.

default_collector = Collector()

#----------------------------------------------------------------------------
