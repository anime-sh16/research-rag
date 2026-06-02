from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class QueryResult:
    answer: str
    sources: list[dict[str, Any]]


class APIClient:
    def __init__(self, base_url: str, timeout: int = 60) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def query(self, question: str) -> QueryResult:
        response = requests.post(
            f"{self.base_url}/query",
            json={"question": question},
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return QueryResult(answer=payload["answer"], sources=payload["sources"])

    def health(self) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/health", timeout=self.timeout)
        response.raise_for_status()
        return response.json()
