"""
Async producer that replays a dataset edge file, honouring the real
timestamp gaps between events.

File format (space-separated):
    <src>  <dst>  <type>  <timestamp>

Usage
-----
    from producer_sim import produce, Edge

    async for edge in produce(path):
        src, dst, etype, ts = edge
        ...

Speed control
-------------
Pass ``speed`` > 1 to compress time (e.g. speed=60 → 1 real second per
simulated minute).  Pass ``speed=0`` to replay as fast as possible with no
sleeping.
"""

import asyncio
from typing import AsyncIterator, NamedTuple


class Edge(NamedTuple):
    src: int
    dst: int
    etype: int
    ts: int


async def produce(
    filepath: str,
    speed: float = 1.0,
) -> AsyncIterator[Edge]:
    """
    Async-generate :class:`Edge` objects from *filepath*, sleeping between
    events to honour the original timestamp gaps.

    Parameters
    ----------
    filepath:
        Path to the edge-list file.
    speed:
        Time-compression factor.  ``speed=1`` → real-time replay.
        ``speed=10`` → 10× faster.  ``speed=0`` → no sleeping.
    """
    prev_ts: int | None = None

    with open(filepath, "r") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue

            parts = raw.split()
            edge = Edge(int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))

            if prev_ts is not None and speed > 0:
                gap = (edge.ts - prev_ts) / speed  # seconds to sleep
                print(f"Sleeping for {gap:.2f} seconds (simulated gap: {edge.ts - prev_ts} seconds)")
                if gap < 0:
                    gap = 0  # tolerate out-of-order timestamps
                if gap > 0:
                    await asyncio.sleep(gap)

            prev_ts = edge.ts
            yield edge
