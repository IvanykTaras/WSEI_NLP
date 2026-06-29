import json
import os
from datetime import datetime, timezone
from typing import Any


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB4_RESULTS_DIR = os.path.join(BASE_DIR, "lab4results")


class Lab4ArtifactProvider:
    def __init__(self, results_dir: str = LAB4_RESULTS_DIR):
        self.results_dir = results_dir

    def save_json(self, command: str, payload: Any) -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, f"latest_{command}.json")
        document = {
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "command": command,
            "result": payload,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
        return path

    def save_text(self, command: str, text: str, metadata: dict) -> str:
        os.makedirs(self.results_dir, exist_ok=True)
        path = os.path.join(self.results_dir, f"latest_{command}.txt")
        header = "\n".join(f"{key}: {value}" for key, value in metadata.items())
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{header}\n\n{text.strip()}\n")
        return path
