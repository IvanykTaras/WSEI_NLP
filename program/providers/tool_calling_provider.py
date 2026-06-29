import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

from providers.lab5_tools_provider import Lab5ToolsProvider


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB5_HISTORY_FILE = os.path.join(BASE_DIR, "lab5results", "tool_history.jsonl")
MAX_PROMPT_LENGTH = 3000
MAX_TOOL_RESULT_LENGTH = 6000


@dataclass
class AgentResult:
    answer: str
    tool_calls: list
    model: str
    rounds: int
    duration_seconds: float
    history_path: str


class ToolCallingProvider:
    SYSTEM_PROMPT = (
        "Jesteś lokalnym agentem bota Telegram dla laboratorium NLP. Odpowiadaj po polsku. "
        "Sam zdecyduj, czy potrzebujesz narzędzia. Używaj narzędzi dla aktualnych faktów, "
        "pogody, obliczeń, lokalnej bazy i obrazów; nie wymyślaj ich wyników. Możesz wywołać "
        "kilka narzędzi w jednej rundzie lub wykonać kilka rund. Po otrzymaniu wyników podaj "
        "krótką, zrozumiałą odpowiedź i zaznacz niepewność albo błąd źródła."
    )

    def __init__(
        self,
        tools_provider: Optional[Lab5ToolsProvider] = None,
        ollama_base_url: Optional[str] = None,
        model: Optional[str] = None,
        history_file: str = LAB5_HISTORY_FILE,
        session: Optional[requests.Session] = None,
        max_rounds: int = 4,
    ):
        self.tools_provider = tools_provider or Lab5ToolsProvider()
        self.ollama_base_url = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_TOOL_MODEL", "qwen3:1.7b")
        self.vision_model = self.tools_provider.vision_model
        self.history_file = history_file
        self.session = session or requests.Session()
        self.max_rounds = max(1, min(int(max_rounds), 8))
        self._history_lock = threading.Lock()
        self._functions = {
            "web_search": self.tools_provider.web_search,
            "analyze_image": self.tools_provider.analyze_image,
            "simple_calculator": self.tools_provider.simple_calculator,
            "local_knowledge": self.tools_provider.local_knowledge,
            "get_weather": self.tools_provider.get_weather,
        }

    @property
    def tool_schemas(self) -> list:
        return [
            self._schema(
                "web_search",
                "Wyszukuje aktualne informacje w internecie (Wikipedia).",
                {"query": ("string", "Zapytanie wyszukiwania")},
                ["query"],
            ),
            self._schema(
                "analyze_image",
                "Opisuje załączony obraz lokalnym modelem Vision.",
                {
                    "image_path": ("string", "Lokalna ścieżka załączonego obrazu"),
                    "question": ("string", "Pytanie dotyczące obrazu"),
                },
                ["image_path"],
            ),
            self._schema(
                "simple_calculator",
                "Bezpiecznie oblicza wyrażenie matematyczne.",
                {"expression": ("string", "Wyrażenie, np. (17 + 3) * 2")},
                ["expression"],
            ),
            self._schema(
                "local_knowledge",
                "Przeszukuje lokalną bazę wiedzy Lab4 bez dostępu do internetu.",
                {"query": ("string", "Osoba, organizacja lub pojęcie")},
                ["query"],
            ),
            self._schema(
                "get_weather",
                "Pobiera aktualną pogodę dla jednego miasta.",
                {"city": ("string", "Nazwa miasta")},
                ["city"],
            ),
        ]

    def run(self, prompt: str, image_path: Optional[str] = None) -> AgentResult:
        prompt = (prompt or "").strip()
        if not prompt and not image_path:
            raise ValueError("Podaj pytanie po komendzie /agent lub dołącz zdjęcie.")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"Pytanie może mieć maksymalnie {MAX_PROMPT_LENGTH} znaków.")
        if image_path and not prompt:
            prompt = "Opisz obraz i wskaż jego najważniejsze elementy."

        started = time.monotonic()
        tool_events = []
        rounds = 0
        answer = ""
        error = None
        user_content = prompt
        if image_path:
            user_content += (
                "\n\nDo wiadomości dołączono obraz zapisany pod ścieżką: "
                f"{image_path}. Jeśli pytanie dotyczy obrazu, użyj analyze_image."
            )
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        try:
            for round_number in range(1, self.max_rounds + 1):
                rounds = round_number
                assistant = self._chat(messages, include_tools=True)
                messages.append(assistant)
                calls = assistant.get("tool_calls") or []
                if not calls:
                    answer = (assistant.get("content") or "").strip()
                    if not answer:
                        raise RuntimeError("Model nie zwrócił odpowiedzi ani wywołania narzędzia.")
                    break

                for call in calls:
                    event, tool_content = self._execute_tool(
                        call, round_number, image_path=image_path
                    )
                    tool_events.append(event)
                    messages.append({
                        "role": "tool",
                        "tool_name": event["name"],
                        "content": tool_content,
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": (
                        "Osiągnięto limit wywołań narzędzi. Nie wywołuj kolejnych; "
                        "podsumuj dostępne wyniki i odpowiedz użytkownikowi."
                    ),
                })
                rounds = self.max_rounds
                answer = (self._chat(messages, include_tools=False).get("content") or "").strip()
                if not answer:
                    raise RuntimeError("Model nie utworzył odpowiedzi końcowej po limicie rund.")
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            duration = time.monotonic() - started
            history_path = self._save_history({
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "image_path": os.path.relpath(image_path, BASE_DIR) if image_path else None,
                "tool_model": self.model,
                "vision_model": self.vision_model,
                "rounds": rounds,
                "tool_calls": tool_events,
                "answer": answer or None,
                "error": error,
                "duration_seconds": round(duration, 3),
            })

        return AgentResult(
            answer=answer,
            tool_calls=tool_events,
            model=self.model,
            rounds=rounds,
            duration_seconds=duration,
            history_path=history_path,
        )

    def _chat(self, messages: list, include_tools: bool) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": 500},
        }
        if include_tools:
            payload["tools"] = self.tool_schemas
        try:
            response = self.session.post(
                f"{self.ollama_base_url}/api/chat", json=payload, timeout=120
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Ollama nie jest dostępna. Uruchom usługę poleceniem `ollama serve`."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError("Odpowiedź Ollama przekroczyła limit 120 sekund.") from exc
        except requests.RequestException as exc:
            detail = self._response_error(exc)
            if "not found" in detail.lower():
                raise RuntimeError(
                    f"Brak modelu `{self.model}`. Uruchom `ollama pull {self.model}`."
                ) from exc
            raise RuntimeError(f"Błąd Ollama: {detail}") from exc
        message = response.json().get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Ollama zwróciła odpowiedź w nieoczekiwanym formacie.")
        return message

    def _execute_tool(self, call: dict, round_number: int, image_path: Optional[str]):
        function = call.get("function") or {}
        name = function.get("name") or "unknown"
        arguments = function.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if name == "analyze_image" and image_path:
            arguments["image_path"] = image_path

        started = time.monotonic()
        result = None
        error = None
        try:
            function_handler = self._functions.get(name)
            if function_handler is None:
                raise ValueError(f"Model wybrał nieznane narzędzie: {name}.")
            result = str(function_handler(**arguments))
            content = result[:MAX_TOOL_RESULT_LENGTH]
            if len(result) > MAX_TOOL_RESULT_LENGTH:
                content += "\n[wynik skrócony]"
        except Exception as exc:
            error = str(exc)
            content = f"Błąd narzędzia {name}: {error}"
        event = {
            "round": round_number,
            "name": name,
            "arguments": arguments,
            "result": result,
            "error": error,
            "duration_seconds": round(time.monotonic() - started, 3),
        }
        return event, content

    def _save_history(self, document: dict) -> str:
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        line = json.dumps(document, ensure_ascii=False, default=str)
        with self._history_lock:
            with open(self.history_file, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        return self.history_file

    @staticmethod
    def _schema(name: str, description: str, properties: dict, required: list) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        key: {"type": value[0], "description": value[1]}
                        for key, value in properties.items()
                    },
                    "required": required,
                },
            },
        }

    @staticmethod
    def _response_error(exc: requests.RequestException) -> str:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                return response.json().get("error") or f"HTTP {response.status_code}"
            except (ValueError, AttributeError):
                return f"HTTP {response.status_code}"
        return str(exc)
