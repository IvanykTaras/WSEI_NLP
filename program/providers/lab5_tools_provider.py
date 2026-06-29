import ast
import base64
import json
import math
import operator
import os
import re
from typing import Optional

import requests


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_FILE = os.path.join(BASE_DIR, "lab4_knowledge_base.json")
LAB5_UPLOADS_DIR = os.path.join(BASE_DIR, "lab5uploads")
WIKIPEDIA_API_URL = "https://pl.wikipedia.org/w/api.php"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
MAX_TOOL_TEXT_LENGTH = 300
MAX_IMAGE_BYTES = 10 * 1024 * 1024

WEATHER_CODES = {
    0: "bezchmurnie",
    1: "przeważnie bezchmurnie",
    2: "częściowe zachmurzenie",
    3: "pochmurno",
    45: "mgła",
    48: "mgła osadzająca szadź",
    51: "lekka mżawka",
    53: "umiarkowana mżawka",
    55: "silna mżawka",
    56: "lekka marznąca mżawka",
    57: "silna marznąca mżawka",
    61: "lekki deszcz",
    63: "umiarkowany deszcz",
    65: "silny deszcz",
    66: "lekki marznący deszcz",
    67: "silny marznący deszcz",
    71: "lekkie opady śniegu",
    73: "umiarkowane opady śniegu",
    75: "silne opady śniegu",
    77: "ziarna śnieżne",
    80: "lekkie przelotne opady",
    81: "umiarkowane przelotne opady",
    82: "gwałtowne przelotne opady",
    85: "lekkie przelotne opady śniegu",
    86: "silne przelotne opady śniegu",
    95: "burza",
    96: "burza z lekkim gradem",
    99: "burza z silnym gradem",
}


class Lab5ToolsProvider:
    def __init__(
        self,
        knowledge_base_file: str = KNOWLEDGE_BASE_FILE,
        uploads_dir: str = LAB5_UPLOADS_DIR,
        ollama_base_url: Optional[str] = None,
        vision_model: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ):
        self.knowledge_base_file = knowledge_base_file
        self.uploads_dir = os.path.realpath(uploads_dir)
        self.ollama_base_url = (
            ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        ).rstrip("/")
        self.vision_model = vision_model or os.getenv(
            "OLLAMA_VISION_MODEL", "gemma4:latest"
        )
        self.session = session or requests.Session()
        self._knowledge_base = self._load_knowledge_base()

    def web_search(self, query: str) -> str:
        """Wyszukaj aktualne informacje w polskiej Wikipedii."""
        query = self._validate_short_text(query, "Zapytanie")
        try:
            response = self.session.get(
                WIKIPEDIA_API_URL,
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": query,
                    "gsrlimit": 3,
                    "prop": "extracts|info",
                    "exintro": 1,
                    "explaintext": 1,
                    "exsentences": 3,
                    "inprop": "url",
                    "format": "json",
                    "formatversion": 2,
                    "utf8": 1,
                },
                headers={"User-Agent": "WSEI-NLP-Lab5/1.0"},
                timeout=10,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError("Nie udało się połączyć z Wikipedią.") from exc

        pages = response.json().get("query", {}).get("pages", [])
        pages.sort(key=lambda row: row.get("index", 999))
        results = [
            {
                "title": row.get("title", ""),
                "summary": (row.get("extract") or "Brak krótkiego opisu.").strip(),
                "url": row.get("fullurl"),
            }
            for row in pages[:3]
            if not row.get("missing")
        ]
        if not results:
            return json.dumps(
                {"query": query, "results": [], "message": "Nie znaleziono wyników."},
                ensure_ascii=False,
            )
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)

    def analyze_image(self, image_path: str, question: str = "Opisz obraz.") -> str:
        """Przeanalizuj bezpiecznie zapisany obraz lokalnym modelem Vision."""
        path = self._validate_image_path(image_path)
        question = self._validate_short_text(question or "Opisz obraz.", "Pytanie")
        with open(path, "rb") as handle:
            encoded_image = base64.b64encode(handle.read()).decode("ascii")

        try:
            response = self.session.post(
                f"{self.ollama_base_url}/api/chat",
                json={
                    "model": self.vision_model,
                    "messages": [{
                        "role": "user",
                        "content": (
                            "Odpowiedz po polsku. Opisz wyłącznie to, co rzeczywiście "
                            f"widać na obrazie. Pytanie użytkownika: {question}"
                        ),
                        "images": [encoded_image],
                    }],
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 350},
                },
                timeout=120,
            )
            response.raise_for_status()
        except requests.ConnectionError as exc:
            raise RuntimeError(
                "Ollama nie jest dostępna. Uruchom usługę poleceniem `ollama serve`."
            ) from exc
        except requests.Timeout as exc:
            raise RuntimeError("Analiza obrazu przekroczyła limit 120 sekund.") from exc
        except requests.RequestException as exc:
            detail = self._response_error(exc)
            raise RuntimeError(f"Błąd modelu Vision: {detail}") from exc

        content = response.json().get("message", {}).get("content", "").strip()
        if not content:
            raise RuntimeError("Model Vision nie zwrócił opisu obrazu.")
        return content

    def simple_calculator(self, expression: str) -> str:
        """Oblicz bezpieczne wyrażenie matematyczne bez używania eval."""
        expression = self._validate_short_text(expression, "Wyrażenie", max_length=120)
        try:
            tree = ast.parse(expression, mode="eval")
            result = self._evaluate_node(tree.body)
        except (SyntaxError, TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
            raise ValueError(f"Niepoprawne wyrażenie matematyczne: {exc}") from exc
        if not math.isfinite(float(result)) or abs(result) > 1e15:
            raise ValueError("Wynik jest zbyt duży.")
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)

    def local_knowledge(self, query: str) -> str:
        """Wyszukaj informacje wyłącznie w lokalnej bazie JSON z Lab4."""
        query = self._validate_short_text(query, "Zapytanie")
        query_normalized = query.casefold()
        query_tokens = set(self._tokenize(query))
        scored = []
        for entity in self._knowledge_base.get("entities", []):
            labels = list(entity.get("labels", {}).values())
            aliases = entity.get("aliases", [])
            descriptions = list(entity.get("descriptions", {}).values())
            searchable = " ".join(labels + aliases + descriptions).casefold()
            searchable_tokens = set(self._tokenize(searchable))
            overlap = len(query_tokens & searchable_tokens) / max(len(query_tokens), 1)
            exact = 1.0 if query_normalized in searchable else 0.0
            score = exact * 2.0 + overlap
            if score > 0:
                scored.append((score, entity))
        scored.sort(key=lambda row: row[0], reverse=True)

        results = []
        for score, entity in scored[:3]:
            labels = entity.get("labels", {})
            descriptions = entity.get("descriptions", {})
            wikipedia = entity.get("wikipedia", {})
            results.append({
                "id": entity.get("id"),
                "label": labels.get("pl") or labels.get("en"),
                "description": descriptions.get("pl") or descriptions.get("en", ""),
                "wikipedia_url": wikipedia.get("pl") or wikipedia.get("en"),
                "score": round(score, 4),
            })
        return json.dumps(
            {
                "query": query,
                "results": results,
                "message": None if results else "Brak informacji w lokalnej bazie.",
            },
            ensure_ascii=False,
        )

    def get_weather(self, city: str) -> str:
        """Pobierz bieżącą pogodę dla miasta z Open-Meteo."""
        city = self._validate_short_text(city, "Miasto", max_length=100)
        try:
            geocoding = self.session.get(
                OPEN_METEO_GEOCODING_URL,
                params={"name": city, "count": 1, "language": "pl", "format": "json"},
                timeout=10,
            )
            geocoding.raise_for_status()
            locations = geocoding.json().get("results", [])
            if not locations:
                raise ValueError(f"Nie znaleziono miasta: {city}.")
            location = locations[0]

            forecast = self.session.get(
                OPEN_METEO_FORECAST_URL,
                params={
                    "latitude": location["latitude"],
                    "longitude": location["longitude"],
                    "current": (
                        "temperature_2m,apparent_temperature,weather_code,wind_speed_10m"
                    ),
                    "timezone": "auto",
                },
                timeout=10,
            )
            forecast.raise_for_status()
        except ValueError:
            raise
        except requests.RequestException as exc:
            raise RuntimeError("Nie udało się pobrać pogody z Open-Meteo.") from exc

        current = forecast.json().get("current", {})
        if "temperature_2m" not in current:
            raise RuntimeError("Open-Meteo zwróciło niepełne dane pogodowe.")
        weather_code = current.get("weather_code")
        result = {
            "city": location.get("name", city),
            "country": location.get("country", ""),
            "time": current.get("time"),
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": weather_code,
            "conditions": WEATHER_CODES.get(weather_code, "nieznane warunki"),
            "source": "Open-Meteo",
        }
        return json.dumps(result, ensure_ascii=False)

    def _validate_image_path(self, image_path: str) -> str:
        if not image_path:
            raise ValueError("Nie przekazano obrazu do analizy.")
        path = os.path.realpath(image_path)
        try:
            common = os.path.commonpath([path, self.uploads_dir])
        except ValueError as exc:
            raise ValueError("Ścieżka obrazu jest niedozwolona.") from exc
        if common != self.uploads_dir:
            raise ValueError("Ścieżka obrazu jest niedozwolona.")
        if not os.path.isfile(path):
            raise ValueError("Plik obrazu nie istnieje.")
        if os.path.getsize(path) > MAX_IMAGE_BYTES:
            raise ValueError("Obraz może mieć maksymalnie 10 MB.")
        if os.path.splitext(path)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            raise ValueError("Obsługiwane formaty obrazu: JPG, PNG i WEBP.")
        return path

    def _load_knowledge_base(self) -> dict:
        try:
            with open(self.knowledge_base_file, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Brak lokalnej bazy Lab4: program/lab4_knowledge_base.json."
            ) from exc
        except json.JSONDecodeError as exc:
            raise ValueError("Lokalna baza Lab4 nie jest poprawnym plikiem JSON.") from exc
        return payload

    @classmethod
    def _evaluate_node(cls, node):
        binary_operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
        }
        unary_operators = {ast.UAdd: operator.pos, ast.USub: operator.neg}
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_operators:
            left = cls._evaluate_node(node.left)
            right = cls._evaluate_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Wykładnik może mieć wartość od -12 do 12.")
            result = binary_operators[type(node.op)](left, right)
            if abs(result) > 1e15:
                raise ValueError("Wynik pośredni jest zbyt duży.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operators:
            return unary_operators[type(node.op)](cls._evaluate_node(node.operand))
        raise ValueError("Dozwolone są wyłącznie liczby i podstawowe operatory matematyczne.")

    @staticmethod
    def _validate_short_text(value: str, name: str, max_length: int = MAX_TOOL_TEXT_LENGTH) -> str:
        value = (value or "").strip()
        if not value:
            raise ValueError(f"{name} nie może być puste.")
        if len(value) > max_length:
            raise ValueError(f"{name} może mieć maksymalnie {max_length} znaków.")
        return value

    @staticmethod
    def _tokenize(text: str) -> list:
        return re.findall(r"\w+", text.casefold(), flags=re.UNICODE)

    @staticmethod
    def _response_error(exc: requests.RequestException) -> str:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                return response.json().get("error") or f"HTTP {response.status_code}"
            except (ValueError, AttributeError):
                return f"HTTP {response.status_code}"
        return str(exc)
