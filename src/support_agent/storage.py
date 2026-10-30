"""JSON file storage: one file per conversation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from support_agent.models import ConversationRecord


class ConversationStore:
    def __init__(self, directory: Path):
        self.directory = directory

    def path_for(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.json"

    def save(self, record: ConversationRecord) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.path_for(record.id)
        # Write to a temporary file, then rename it, so a crash never leaves half a file.
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(record.model_dump_json(indent=2))
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return target

    def load(self, conversation_id: str) -> ConversationRecord:
        return ConversationRecord.model_validate_json(
            self.path_for(conversation_id).read_text(encoding="utf-8")
        )
