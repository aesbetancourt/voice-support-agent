"""Loads the job posting that the screening is for, from a JSON file."""

from __future__ import annotations

from pathlib import Path

from screening_agent.domain.models import JobPosting


def load_job(path: Path) -> JobPosting:
    return JobPosting.model_validate_json(path.read_text(encoding="utf-8"))
