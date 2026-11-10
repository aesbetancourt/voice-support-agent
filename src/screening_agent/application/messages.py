"""Fixed candidate-facing messages, one set per language.

Code says these messages itself (see ADR-0001), so the LLM cannot change them:
the booking time and reference, and the reason an application stops, come from code.
Add a language by adding one entry to each dictionary;
``tests/application/test_messages.py`` checks that none is missing.
"""

from __future__ import annotations

from datetime import datetime

from screening_agent.domain.models import Language, Requirement, Shift, Stage
from screening_agent.domain.state import ScreeningSession

LANGUAGE_NAMES = {
    Language.EN: "English",
    Language.ES: "Spanish",
}

_WEEKDAYS = {
    Language.EN: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    Language.ES: ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
}
_MONTHS = {
    Language.EN: ["January", "February", "March", "April", "May", "June", "July", "August",
                  "September", "October", "November", "December"],
    Language.ES: ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                  "septiembre", "octubre", "noviembre", "diciembre"],
}  # fmt: skip
_WHEN = {
    Language.EN: "{weekday} {day} {month} at {time}",
    Language.ES: "el {weekday} {day} de {month} a las {time}",
}
_SHIFT_NAMES = {
    Language.EN: {Shift.MORNING: "morning", Shift.AFTERNOON: "afternoon", Shift.NIGHT: "night"},
    Language.ES: {Shift.MORNING: "mañana", Shift.AFTERNOON: "tarde", Shift.NIGHT: "noche"},
}
_AND = {Language.EN: " and ", Language.ES: " y "}

_REQUIREMENT_REASONS = {
    Language.EN: {
        Requirement.WORK_PERMIT: "a valid permit to work in Spain",
        Requirement.SHIFTS: "availability for {shifts} shifts",
        Requirement.START_DATE: "a start date within the next {days} days",
        Requirement.COMMUTE: "a reliable way to get to the warehouse for every shift",
    },
    Language.ES: {
        Requirement.WORK_PERMIT: "un permiso válido para trabajar en España",
        Requirement.SHIFTS: "disponibilidad para los turnos de {shifts}",
        Requirement.START_DATE: "poder empezar en los próximos {days} días",
        Requirement.COMMUTE: "una forma fiable de llegar al almacén en cada turno",
    },
}

_CLOSING_MESSAGES = {
    Language.EN: {
        Stage.BOOKED: (
            "Perfect. Your video interview with the warehouse manager is booked for {when}. "
            "Your reference is {booking_id}. You will get the details by email. "
            "Thank you for your time, and good luck!"
        ),
        Stage.NOT_ELIGIBLE: (
            "Thank you for your time. This role needs {reason}, so we cannot continue with "
            "this application. You are welcome to apply again if your situation changes. Goodbye!"
        ),
        Stage.ESCALATED: (
            "I will pass your application to a recruiter. They will call you back within one "
            "working day. Thank you, and goodbye!"
        ),
        Stage.ENDED: "Thank you for your time. Goodbye!",
    },
    Language.ES: {
        Stage.BOOKED: (
            "Perfecto. Su entrevista por videollamada con el responsable del almacén queda "
            "reservada para {when}. Su referencia es {booking_id}. Recibirá los detalles por "
            "correo electrónico. Gracias por su tiempo, ¡y mucha suerte!"
        ),
        Stage.NOT_ELIGIBLE: (
            "Gracias por su tiempo. Este puesto requiere {reason}, así que no podemos continuar "
            "con esta candidatura. Puede volver a presentarse si su situación cambia. "
            "¡Hasta luego!"
        ),
        Stage.ESCALATED: (
            "Paso su candidatura a una persona del equipo de selección. Le llamará en un día "
            "laborable. Gracias, ¡hasta luego!"
        ),
        Stage.ENDED: "Gracias por su tiempo. ¡Hasta luego!",
    },
}

_RETRY_MESSAGES = {
    Language.EN: "Sorry, I did not catch that. Could you say it again?",
    Language.ES: "Disculpe, no le entendí. ¿Podría repetirlo?",
}

_TECHNICAL_ERROR_MESSAGES = {
    Language.EN: "Sorry, I have a technical problem and must stop. Your answers are saved.",
    Language.ES: (
        "Disculpe, tengo un problema técnico y debo terminar. Sus respuestas están guardadas."
    ),
}


def retry_message(language: Language) -> str:
    return _RETRY_MESSAGES[language]


def technical_error_message(language: Language) -> str:
    return _TECHNICAL_ERROR_MESSAGES[language]


def spoken_time(language: Language, when: datetime) -> str:
    """A date and time as a person says it, for example "Thursday 12 November at 10:00"."""
    return _WHEN[language].format(
        weekday=_WEEKDAYS[language][when.weekday()],
        day=when.day,
        month=_MONTHS[language][when.month - 1],
        time=f"{when:%H:%M}",
    )


def requirement_reason(language: Language, session: ScreeningSession) -> str:
    requirement = session.failed_requirement
    if requirement is None:
        return ""
    shifts = _AND[language].join(_SHIFT_NAMES[language][s] for s in session.job.required_shifts)
    template = _REQUIREMENT_REASONS[language][requirement]
    return template.format(shifts=shifts, days=session.job.max_start_days)


def closing_message(language: Language, session: ScreeningSession) -> str | None:
    template = _CLOSING_MESSAGES[language].get(session.stage)
    if template is None:
        return None
    interview = session.scorecard.interview
    return template.format(
        when=spoken_time(language, interview.starts_at) if interview else "",
        booking_id=session.scorecard.booking_id or "",
        reason=requirement_reason(language, session),
    )


def closing_stages(language: Language) -> set[Stage]:
    """The stages that have a closing message in one language."""
    return set(_CLOSING_MESSAGES[language])
