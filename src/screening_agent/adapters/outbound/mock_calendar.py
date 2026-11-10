"""``InterviewCalendar`` adapter: a mock calendar with free times on the next working days.

The times are made from today's date, so the demo always offers future times.
Bookings stay in memory for the life of the process.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta

from screening_agent.domain.models import InterviewSlot

# Monday is 0: days from Saturday (5) on are weekend days, with no interviews.
_FIRST_WEEKEND_DAY = 5
_SLOT_ID_FORMAT = "%Y%m%d-%H%M"


class MockInterviewCalendar:
    def __init__(
        self,
        interview_times: Sequence[time],
        working_days_ahead: int,
        today: Callable[[], date] = date.today,
    ) -> None:
        self._interview_times = tuple(interview_times)
        self._working_days_ahead = working_days_ahead
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
        while len({slot.starts_at.date() for slot in slots}) < self._working_days_ahead:
            day += timedelta(days=1)
            if day.weekday() >= _FIRST_WEEKEND_DAY:
                continue
            for at in self._interview_times:
                starts_at = datetime.combine(day, at)
                slots.append(
                    InterviewSlot(id=starts_at.strftime(_SLOT_ID_FORMAT), starts_at=starts_at)
                )
        return slots
