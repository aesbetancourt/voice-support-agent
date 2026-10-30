from pathlib import Path

import pytest

from support_agent.adapters.outbound.json_orders import JsonOrderRepository
from support_agent.domain.state import TicketSession

ORDERS_PATH = Path(__file__).resolve().parents[1] / "data" / "orders.json"


@pytest.fixture
def orders() -> JsonOrderRepository:
    return JsonOrderRepository.from_file(ORDERS_PATH)


@pytest.fixture
def session(orders: JsonOrderRepository) -> TicketSession:
    return TicketSession(orders.get, max_attempts=3)
