# Laboratorium 5

## Temat

Function Calling i Tool Calling w lokalnych modelach LLM (Ollama) z wykorzystaniem narzędzi: Web Search, Vision, Weather oraz własnych funkcji użytkownika.

Laboratorium rozszerza funkcjonalność bota z **Laboratorium 1**, **Laboratorium 2**, **Laboratorium 3** oraz **Laboratorium 4** o:

* wykorzystanie mechanizmu **function calling / tool calling** w Ollama,
* integrację wielu narzędzi sterowanych przez model (LLM jako kontroler wykonania),
* użycie modeli multimodalnych (Vision) do analizy obrazów,
* dynamiczne pobieranie informacji z internetu (Web Search),
* wykorzystanie zewnętrznych danych kontekstowych (np. Weather API),
* budowę i integrację prostych narzędzi własnych (custom tools),
* realizację scenariuszy **multi-tool reasoning** (łączenie wyników wielu narzędzi),
* poprawę architektury systemu poprzez separację logiki LLM i logiki narzędzi.

---

## Cel laboratorium

Celem zadania jest rozbudowanie istniejącego bota (np. Telegram lub CLI) tak, aby:

* automatycznie decydował, **czy i kiedy użyć narzędzia (tool)**,
* wybierał odpowiednie narzędzie na podstawie zapytania użytkownika,
* przekazywał parametry do funkcji w sposób ustrukturyzowany (JSON),
* integrował wyniki z wielu źródeł (np. web + weather + vision),
* interpretował wyniki narzędzi zamiast tylko je zwracać,
* obsługiwał różne typy danych (tekst, obraz, dane z API),
* umożliwiał łatwe rozszerzanie o nowe narzędzia,
* działał w oparciu o lokalny model LLM (bez zależności od zewnętrznych API LLM),
* realizował bardziej złożone scenariusze analizy (multi-step reasoning),
* zapisywał historię interakcji oraz wyniki wywołań narzędzi.

---


## 🔧 Wymagane narzędzia (tools)

Student musi zaimplementować co najmniej **5 narzędzi**:

---

### 1. 🌐 Web Search Tool

Opis:

* pobiera informacje z internetu (np. Wikipedia / API / scraping)

Funkcja:

```python
def web_search(query: str) -> str:
    """Search the web and return summary results."""
```

Przykład użycia:

```
User: Kto jest CEO Tesli?
→ model wywołuje web_search()
```

---

### 2. 🖼 Vision Tool

Opis:

* analizuje obraz (np. podpis, klasyfikacja)

Funkcja:

```python
def analyze_image(image_path: str) -> str:
    """Describe image content."""
```

Wymagania:

* użycie modelu multimodalnego w Ollama (vision)

---

### 3. 🧮 Custom Tool (prosty)

Opis:

* prosta funkcja lokalna

Przykłady (wybierz min. 1):

* kalkulator
* analiza sentymentu (prosta heurystyka)
* ekstrakcja liczb

```python
def simple_calculator(expression: str) -> str:
    """Evaluate math expression."""
```

---

### 4. 🧾 Local Knowledge Tool

Opis:

* działa na lokalnych danych (JSON/CSV)

Przykład:

```python
def local_knowledge(query: str) -> str:
    """Search in local knowledge base."""
```

---

### 5. 🌦 Weather Tool

Opis:

* pobiera aktualną pogodę dla miasta,
* testuje poprawność decyzji modelu (czy użyć toola),
* umożliwia łączenie z innymi narzędziami (multi-tool reasoning).

---

#### Funkcja:

```python
def get_weather(city: str) -> str:
    """Return current weather for a given city."""
```


---

#### Implementacja (wariant API – opcjonalnie):

```python
import requests

def get_weather(city: str) -> str:
    # uproszczenie: Warszawa
    url = "https://api.open-meteo.com/v1/forecast?latitude=52.23&longitude=21.01&current_weather=true"
    res = requests.get(url).json()
    
    weather = res["current_weather"]
    return f"{weather['temperature']}°C, wind {weather['windspeed']} km/h"
```

---

#### Definicja toola (JSON schema)

```json
{
  "name": "get_weather",
  "description": "Get current weather for a city",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "City name"
      }
    },
    "required": ["city"]
  }
}
```

---

## 💬 Rozszerzone scenariusze użycia

---

### 5. Weather

```
User: Jaka jest pogoda w Warszawie?

→ tool: get_weather("Warsaw")
→ odpowiedź: 15°C, cloudy
```

---

### 6. Weather + reasoning

```
User: Czy dziś jest dobra pogoda na spacer w Warszawie?

→ get_weather("Warsaw")
→ model interpretuje warunki
```

---

### 7. Multi-tool (ważne!)

```
User: Porównaj pogodę w Warszawie i Paryżu

→ get_weather("Warsaw")
→ get_weather("Paris")
→ model generuje porównanie
```

---

### 8. Multi-tool advanced

```
User: Czy pogoda w Warszawie jest typowa dla maja?

→ get_weather()
→ web_search("average weather Warsaw May")
→ analiza porównawcza
```

---

## ⚙️ Integracja toola z modelem

W konfiguracji requestu do Qwen3.5 (np. 0.8B):

```python
{
  "model": "qwen3.5",
  "messages": [...],
  "tools": [
    web_search_tool,
    vision_tool,
    calculator_tool,
    knowledge_tool,
    weather_tool
  ]
}
```
