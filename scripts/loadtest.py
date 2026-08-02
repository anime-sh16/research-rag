"""Concurrent-load smoke test for the /query endpoint.

Fires N concurrent POST /query requests and reports wall-clock, p50/p95
latency, and throughput. Used to capture a real before/after number for the
sync -> async migration — not meant as a load-testing framework.

Usage:
    uv run python scripts/loadtest.py --url http://localhost:8000 --concurrency 10 --question "What is attention?"
"""

import argparse
import asyncio
import statistics
import time

import httpx


async def _one_request(client: httpx.AsyncClient, url: str, question: str) -> float:
    start = time.perf_counter()
    response = await client.post(f"{url}/query", json={"question": question})
    response.raise_for_status()
    return time.perf_counter() - start


async def run(url: str, concurrency: int, question: str) -> None:
    async with httpx.AsyncClient(timeout=60) as client:
        start = time.perf_counter()
        durations = await asyncio.gather(
            *[_one_request(client, url, question) for _ in range(concurrency)]
        )
        total = time.perf_counter() - start

    durations_sorted = sorted(durations)
    p50 = statistics.median(durations_sorted)
    p95 = durations_sorted[max(0, int(len(durations_sorted) * 0.95) - 1)]

    print(f"Concurrency : {concurrency}")
    print(f"Total time  : {total:.2f}s")
    print(f"Throughput  : {concurrency / total:.2f} req/s")
    print(f"p50 latency : {p50:.2f}s")
    print(f"p95 latency : {p95:.2f}s")
    print(f"min / max   : {min(durations_sorted):.2f}s / {max(durations_sorted):.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--question", default="What is attention?")
    args = parser.parse_args()

    asyncio.run(run(args.url, args.concurrency, args.question))


if __name__ == "__main__":
    main()
