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
from typing import AsyncIterator, NamedTuple


class Edge(NamedTuple):
    src: int
    dst: int
    etype: int
    ts: int


async def produce(
    filepath: str,
    speed: float = 1.0,
    width: int = 0,
) -> AsyncIterator[Edge]:
    """
    Async-generate :class:`Edge` objects from *filepath*, sleeping a random
    amount between each edge.

    Parameters
    ----------
    filepath:
        Path to the edge-list file.
    speed:
        Centre of the sleep interval in seconds.
    width:
        Half-range of the random jitter around ``speed``.
        The actual sleep is ``Uniform(speed - width, speed + width)``.
    """

    with open(filepath, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue

            parts = raw.split()
            edge = Edge(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
            
            gap = random.uniform(speed - width, speed + width)  # seconds to sleep
            print(f"Sleeping for {gap:.2f} seconds")
            await asyncio.sleep(gap)

            yield edge
