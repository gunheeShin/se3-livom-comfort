"""Parameter-sweep benchmark layer for SE(3)-LIO.

`evaluate` is pure numpy/stdlib (no compiled binding). `run` holds the sweep
runner, dataset registry, and grid search, pulling in the LIO core lazily (only
when a sweep actually executes).
"""
