from pathlib import Path

import pytest

from support_agent.orders import OrderRepository
from support_agent.state import TicketSession

ORDERS_PATH = Path(__file__).resolve().parents[1] / "data" / "orders.json"


@pytest.fixture
def orders() -> OrderRepository:
    return OrderRepository.from_json(ORDERS_PATH)


@pytest.fixture
def session(orders: OrderRepository) -> TicketSession:
    return TicketSession(orders, max_attempts=3)
