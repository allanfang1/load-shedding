"""
Async producer that replays a dataset edge file, inserting a configurable
random delay between successive edges.

File format (space-separated):
    <src>  <dst>  <type>  <timestamp>

Usage
-----
    from producer_sim import produce, Edge

    async for edge in produce(path):
        src, dst, etype, ts = edge
        ...

Delay control
-------------
Each inter-edge sleep is drawn from ``Uniform(speed - width, speed + width)``
seconds.  Set ``width=0`` (default) for a constant delay of ``speed`` seconds.
Set ``speed=0, width=0`` to replay as fast as possible with no sleeping.
"""

import asyncio
import random
import time
from typing import AsyncIterator


async def produce(
    filepath: str,
    idle_rate: float = 0.001,
    spike_start: float = 0.0,
    spike_duration: float = 0.0,
    spike_rate: float = 0.0,
) -> AsyncIterator[str]:

    edge_counter = 0
    with open(filepath, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            edge_counter += 1
            
            now = time.perf_counter()
            if spike_duration != 0 and now >= spike_start and now < spike_start + spike_duration:
                await asyncio.sleep(spike_rate)
            else:
                await asyncio.sleep(idle_rate)

            yield raw