"""Read-only mock order database, loaded from a JSON file."""

from __future__ import annotations

import json
from pathlib import Path

from support_agent.models import Order


class OrderRepository:
    def __init__(self, orders: list[Order]):
        self._orders = {order.order_number: order for order in orders}

    @classmethod
    def from_json(cls, path: Path) -> OrderRepository:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls([Order.model_validate(item) for item in data])

    def get(self, order_number: str) -> Order | None:
        return self._orders.get(order_number)
