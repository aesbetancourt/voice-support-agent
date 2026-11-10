"""``InterviewCalendar`` adapter: a mock calendar with free times on the next working days.

The times are made from today's date, so the demo always offers future times.
Bookings stay in memory for the life of the process.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta

from screening_agent.domain.models import InterviewSlot

# The warehouse manager interviews between the morning and afternoon peaks.
INTERVIEW_TIMES = (time(10, 0), time(12, 30), time(16, 0))
WORKING_DAYS_AHEAD = 3


class MockInterviewCalendar:
    def __init__(self, today: Callable[[], date] = date.today) -> None:
        self._today = today
        self._booked: set[str] = set()

    def free_slots(self, limit: int) -> list[InterviewSlot]:
        slots = [slot for slot in self._all_slots() if slot.id not in self._booked]
        return slots[:limit]

    def book(self, slot_id: str) -> None:
        self._booked.add(slot_id)

    def _all_slots(self) -> list[InterviewSlot]:
        slots: list[InterviewSlot] = []
        day = self._today()
        while len({slot.starts_at.date() for slot in slots}) < WORKING_DAYS_AHEAD:
            day += timedelta(days=1)
            if day.weekday() >= 5:  # No interviews on weekends.
                continue
            for at in INTERVIEW_TIMES:
                starts_at = datetime.combine(day, at)
                slots.append(InterviewSlot(id=f"{starts_at:%Y%m%d-%H%M}", starts_at=starts_at))
        return slots
