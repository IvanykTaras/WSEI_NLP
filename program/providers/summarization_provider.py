import os
import time
from typing import Optional

import requests


SUPPORTED_SUMMARY_TYPES = ("extractive", "abstractive", "bullets", "custom")
SUPPORTED_SUMMARY_LENGTHS = ("short", "medium", "long")
SUMMARY_TOKEN_LIMITS = {"short": 120, "medium": 250, "long": 450}
MAX_SUMMARY_TEXT_LENGTH = 3000


class SummarizationProvider:
    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 120,
        session: Optional[requests.Session] = None,
    ):
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "gemma3:1b")
        self.timeout = timeout
        self.session = session or requests.Session()

    def summarize(
        self,
        text: str,
        summary_type: str = "abstractive",
        length: str = "medium",
        custom_prompt: str = "",
    ) -> dict:
        text = self._validate_text(text)
        summary_type = (summary_type or "").strip().lower()
        length = (length or "").strip().lower()
        if summary_type not in SUPPORTED_SUMMARY_TYPES:
            raise ValueError(
                f"Typ podsumowania musi być jednym z: {', '.join(SUPPORTED_SUMMARY_TYPES)}."
            )
        if length not in SUPPORTED_SUMMARY_LENGTHS:
            raise ValueError(
                f"Długość musi być jedną z: {', '.join(SUPPORTED_SUMMARY_LENGTHS)}."
            )
        if summary_type == "custom" and not custom_prompt.strip():
            raise ValueError("Dla summary_type=custom wymagany jest parametr prompt.")

        self._check_model_available()
        prompt = self._build_prompt(text, summary_type, length, custom_prompt)
        started = time.monotonic()
        try:
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": SUMMARY_TOKEN_LIMITS[length],
                    },
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise TimeoutError(
                f"Ollama nie odpowiedziała w ciągu {self.timeout} sekund. Skróć tekst lub wybierz length=short."
            ) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Nie można połączyć się z Ollama. Uruchom usługę poleceniem `ollama serve`."
            ) from exc
        except requests.HTTPError as exc:
            raise RuntimeError(f"Ollama zwróciła błąd HTTP: {exc}.") from exc

        payload = response.json()
        summary = (payload.get("response") or "").strip()
        if not summary:
            raise RuntimeError("Ollama zwróciła pustą odpowiedź.")
        return {
            "model": self.model,
            "summary_type": summary_type,
            "length": length,
            "source_characters": len(text),
            "summary": summary,
            "generation_seconds": round(time.monotonic() - started, 2),
        }

    def _check_model_available(self) -> None:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Nie można połączyć się z Ollama. Uruchom usługę poleceniem `ollama serve`."
            ) from exc
        except requests.Timeout as exc:
            raise TimeoutError("Ollama nie odpowiedziała podczas sprawdzania modeli.") from exc
        except requests.HTTPError as exc:
            raise RuntimeError(f"Nie udało się sprawdzić modeli Ollama: {exc}.") from exc
        names = {row.get("name") for row in response.json().get("models", [])}
        if self.model not in names:
            raise FileNotFoundError(
                f"Brak modelu Ollama `{self.model}`. Uruchom `ollama pull {self.model}`."
            )

    @staticmethod
    def _build_prompt(text: str, summary_type: str, length: str, custom_prompt: str) -> str:
        length_instruction = {
            "short": "bardzo krótkie (2-3 zdania lub maksymalnie 3 punkty)",
            "medium": "średniej długości (około 1/3 tekstu źródłowego)",
            "long": "szczegółowe, ale wyraźnie krótsze od tekstu źródłowego",
        }[length]
        type_instruction = {
            "extractive": "Wybierz wyłącznie najważniejsze zdania z tekstu, bez parafrazowania.",
            "abstractive": "Napisz własnymi słowami spójne streszczenie najważniejszych informacji.",
            "bullets": "Przedstaw najważniejsze informacje w punktach zaczynających się od '-'.",
            "custom": custom_prompt.strip(),
        }[summary_type]
        return (
            "Odpowiedz w tym samym języku co tekst źródłowy. Nie dodawaj faktów, których nie ma w tekście.\n"
            f"Typ zadania: {type_instruction}\n"
            f"Długość: {length_instruction}.\n\n"
            f"TEKST ŹRÓDŁOWY:\n{text}\n\nPODSUMOWANIE:"
        )

    @staticmethod
    def _validate_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Tekst nie może być pusty.")
        if len(text) > MAX_SUMMARY_TEXT_LENGTH:
            raise ValueError(f"Tekst może mieć maksymalnie {MAX_SUMMARY_TEXT_LENGTH} znaków.")
        return text
