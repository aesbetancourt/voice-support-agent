"""``ConversationStore`` adapter: one JSON file per conversation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from support_agent.domain.models import ConversationRecord


class JsonConversationStore:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def path_for(self, conversation_id: str) -> Path:
        return self.directory / f"{conversation_id}.json"

    def location(self, conversation_id: str) -> str:
        return str(self.path_for(conversation_id))

    def save(self, record: ConversationRecord) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file, then rename it, so a crash never leaves half a file.
        fd, tmp_name = tempfile.mkstemp(dir=self.directory, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(record.model_dump_json(indent=2))
            os.replace(tmp_name, self.path_for(record.id))
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
