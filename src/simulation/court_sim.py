"""Stretch goal: toy discrete-event court simulator (SimPy).

A court is a queue: cases arrive (Poisson), `slots` are the effective hearing capacity, and each case needs
a random amount of 'bench time'. It is an ILLUSTRATIVE capacity-planning toy, not a calibrated model:
fit `service_mean_months` / `slots` to real data before reading anything into the numbers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import simpy


@dataclass
class SimParams:
    arrivals_per_month: float = 60.0
    slots: int = 20
    service_mean_months: float = 0.30      # bench time needed per case
    service_sigma: float = 0.8
    extra_capacity_pct: float = 0.0        # +x% slots
    fast_track_share: float = 0.0          # share of cases given priority and half the bench time
    horizon_months: int = 240
    warmup_months: int = 48
    seed: int = 7


def simulate(p: SimParams) -> dict:
    rng = random.Random(p.seed)
    env = simpy.Environment()
    court = simpy.PriorityResource(env, capacity=max(1, round(p.slots * (1 + p.extra_capacity_pct / 100))))
    done: list[float] = []

    def case(arrived: float, fast: bool):
        mu = np.log(p.service_mean_months) - p.service_sigma ** 2 / 2
        service = rng.lognormvariate(mu, p.service_sigma) * (0.5 if fast else 1.0)
        with court.request(priority=0 if fast else 1) as req:
            yield req
            yield env.timeout(service)
        if arrived >= p.warmup_months:
            done.append(env.now - arrived)

    def arrivals():
        while True:
            yield env.timeout(rng.expovariate(p.arrivals_per_month))
            env.process(case(env.now, rng.random() < p.fast_track_share))

    env.process(arrivals())
    env.run(until=p.horizon_months)
    d = np.array(done) if done else np.array([np.nan])
    t = np.arange(0, 61, 3.0)
    return {"n_completed": int(len(done)), "median_months": float(np.nanmedian(d)),
            "p90_months": float(np.nanpercentile(d, 90)), "share_over_24m": float(np.mean(d > 24)),
            "times_months": t.tolist(), "survival": [float(np.mean(d > x)) for x in t],
            "utilisation_note": "offered load = arrivals * service / slots = %.2f" % (
                p.arrivals_per_month * p.service_mean_months / max(1, round(p.slots * (1 + p.extra_capacity_pct / 100))))}
