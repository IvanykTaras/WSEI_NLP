import json
import os
from difflib import SequenceMatcher
from typing import Optional
from urllib.parse import quote

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_FILE = os.path.join(BASE_DIR, "lab4_knowledge_base.json")
WIKIDATA_CACHE_FILE = os.path.join(BASE_DIR, "lab4results", "wikidata_cache.json")
SUPPORTED_NER_METHODS = ("spacy", "stanza")
SUPPORTED_ENTITY_LANGUAGES = ("pl", "en")
MAX_ENTITY_TEXT_LENGTH = 3000
WIKIDATA_URL = "https://www.wikidata.org/w/api.php"

ENTITY_LABELS = {
    "persName": "PERSON",
    "orgName": "ORG",
    "placeName": "LOCATION",
    "geogName": "GPE",
    "date": "DATE",
    "time": "TIME",
    "PER": "PERSON",
    "LOC": "LOCATION",
}


class EntityProvider:
    def __init__(
        self,
        knowledge_base_file: str = KNOWLEDGE_BASE_FILE,
        cache_file: str = WIKIDATA_CACHE_FILE,
        session: Optional[requests.Session] = None,
    ):
        self.knowledge_base_file = knowledge_base_file
        self.cache_file = cache_file
        self.session = session or requests.Session()
        if hasattr(self.session, "headers"):
            self.session.headers.update({
                "User-Agent": (
                    "WSEI-NLP-Lab4/1.0 "
                    "(educational project; https://github.com/IvanykTaras/WSEI_NLP)"
                )
            })
        self._spacy_pipeline = None
        self._stanza_pipeline = None
        self._knowledge_base = self._load_json(knowledge_base_file, default={"entities": []})
        self._cache = self._load_json(cache_file, default={})

    def recognize(self, method: str, text: str) -> dict:
        method = method.strip().lower()
        text = self._validate_text(text)
        if method not in SUPPORTED_NER_METHODS:
            raise ValueError(f"Metoda NER musi być jedną z: {', '.join(SUPPORTED_NER_METHODS)}.")

        if method == "spacy":
            entities = self._recognize_spacy(text)
        else:
            entities = self._recognize_stanza(text)
        return {"method": method, "language": "pl", "text": text, "entities": entities}

    def link_entity(self, text: str, language: str = "pl", limit: int = 5) -> dict:
        text = self._validate_text(text)
        language = self._validate_language(language)
        limit = max(1, min(int(limit), 5))
        cache_key = f"{language}:{text.casefold()}"

        local_candidates = self._local_candidates(text, language)
        remote_error = None
        if cache_key in self._cache:
            remote_candidates = self._cache[cache_key]
        else:
            try:
                remote_candidates = self._wikidata_candidates(text, language, limit)
                self._cache[cache_key] = remote_candidates
                self._save_json(self.cache_file, self._cache)
            except requests.RequestException as exc:
                remote_error = str(exc)
                remote_candidates = []

        candidates = self._merge_candidates(local_candidates, remote_candidates)
        for index, candidate in enumerate(candidates):
            label_score = self._label_similarity(text, candidate["label"])
            rank_score = 1.0 / (index + 1)
            candidate["confidence"] = round(0.7 * label_score + 0.3 * rank_score, 4)
        candidates.sort(key=lambda row: row["confidence"], reverse=True)

        if not candidates and remote_error:
            raise RuntimeError(
                "Nie udało się połączyć z Wikidata i brak pasujących wpisów w lokalnej bazie."
            )
        return {
            "entity": text,
            "language": language,
            "candidates": candidates[:limit],
            "source": "wikidata+local" if remote_candidates and local_candidates else (
                "wikidata" if remote_candidates else "local"
            ),
        }

    def disambiguate(self, entity: str, context: str, language: str = "pl") -> dict:
        entity = self._validate_text(entity)
        context = self._validate_text(context)
        linked = self.link_entity(entity, language=language, limit=5)
        candidates = linked["candidates"]
        if not candidates:
            return {"entity": entity, "context": context, "selected": None, "candidates": []}

        descriptions = [candidate.get("description", "") or "" for candidate in candidates]
        similarities = self._context_similarities(context, descriptions)
        for index, (candidate, context_score) in enumerate(zip(candidates, similarities)):
            label_score = self._label_similarity(entity, candidate["label"])
            rank_score = 1.0 / (index + 1)
            candidate["context_similarity"] = round(context_score, 4)
            candidate["ned_score"] = round(
                0.6 * context_score + 0.35 * label_score + 0.05 * rank_score, 4
            )
        candidates.sort(key=lambda row: row["ned_score"], reverse=True)
        selected = candidates[0] if candidates[0]["ned_score"] >= 0.25 else None
        return {
            "entity": entity,
            "context": context,
            "language": language,
            "selected": selected,
            "candidates": candidates,
        }

    def analyze_entities(self, text: str, link: bool = False, method: str = "spacy") -> dict:
        result = self.recognize(method, text)
        if link:
            for entity in result["entities"]:
                try:
                    linked = self.link_entity(entity["text"], language="pl", limit=3)
                    entity["linking"] = linked["candidates"]
                except RuntimeError as exc:
                    entity["linking"] = []
                    entity["linking_error"] = str(exc)
        result["link"] = link
        return result

    def _recognize_spacy(self, text: str) -> list:
        try:
            import spacy
        except ImportError as exc:
            raise ImportError(
                "Brak spaCy. Uruchom `python3 -m pip install -r program/requirements.txt`."
            ) from exc
        if self._spacy_pipeline is None:
            try:
                self._spacy_pipeline = spacy.load("pl_core_news_sm")
            except OSError as exc:
                raise FileNotFoundError(
                    "Brak modelu pl_core_news_sm. Uruchom `python3 -m spacy download pl_core_news_sm`."
                ) from exc
        doc = self._spacy_pipeline(text)
        return [
            {
                "text": ent.text,
                "type": ENTITY_LABELS.get(ent.label_, ent.label_.upper()),
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    def _recognize_stanza(self, text: str) -> list:
        try:
            import stanza
        except ImportError as exc:
            raise ImportError(
                "Brak Stanza. Uruchom `python3 -m pip install -r program/requirements.txt`."
            ) from exc
        if self._stanza_pipeline is None:
            try:
                self._stanza_pipeline = stanza.Pipeline(
                    lang="pl", processors="tokenize,ner", download_method=None, verbose=False
                )
            except Exception as exc:
                raise FileNotFoundError(
                    "Brak polskiego modelu Stanza NER. Uruchom `python3 -c \"import stanza; stanza.download('pl', processors='tokenize,ner')\"`."
                ) from exc
        doc = self._stanza_pipeline(text)
        return [
            {
                "text": ent.text,
                "type": ENTITY_LABELS.get(ent.type, ent.type),
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
        ]

    def _wikidata_candidates(self, text: str, language: str, limit: int) -> list:
        response = self.session.get(
            WIKIDATA_URL,
            params={
                "action": "wbsearchentities",
                "search": text,
                "language": language,
                "uselang": language,
                "type": "item",
                "limit": limit,
                "format": "json",
                "origin": "*",
            },
            timeout=10,
        )
        response.raise_for_status()
        search_rows = response.json().get("search", [])
        ids = [row["id"] for row in search_rows if row.get("id")]
        details = self._wikidata_details(ids, language) if ids else {}
        candidates = []
        for row in search_rows:
            entity_id = row.get("id")
            detail = details.get(entity_id, {})
            candidates.append({
                "id": entity_id,
                "label": row.get("label") or detail.get("label") or text,
                "description": row.get("description") or detail.get("description") or "",
                "wikipedia_url": detail.get("wikipedia_url"),
                "wikidata_url": f"https://www.wikidata.org/wiki/{entity_id}",
                "source": "wikidata",
            })
        return candidates

    def _wikidata_details(self, ids: list, language: str) -> dict:
        response = self.session.get(
            WIKIDATA_URL,
            params={
                "action": "wbgetentities",
                "ids": "|".join(ids),
                "props": "labels|descriptions|sitelinks",
                "languages": f"{language}|en",
                "sitefilter": f"{language}wiki|enwiki",
                "format": "json",
                "origin": "*",
            },
            timeout=10,
        )
        response.raise_for_status()
        output = {}
        for entity_id, row in response.json().get("entities", {}).items():
            labels = row.get("labels", {})
            descriptions = row.get("descriptions", {})
            sitelinks = row.get("sitelinks", {})
            site = sitelinks.get(f"{language}wiki") or sitelinks.get("enwiki") or {}
            site_language = language if f"{language}wiki" in sitelinks else "en"
            title = site.get("title")
            output[entity_id] = {
                "label": (labels.get(language) or labels.get("en") or {}).get("value"),
                "description": (
                    descriptions.get(language) or descriptions.get("en") or {}
                ).get("value"),
                "wikipedia_url": (
                    f"https://{site_language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"
                    if title else None
                ),
            }
        return output

    def _local_candidates(self, text: str, language: str) -> list:
        query = text.casefold()
        candidates = []
        for row in self._knowledge_base.get("entities", []):
            labels = row.get("labels", {})
            aliases = row.get("aliases", [])
            searchable = [value.casefold() for value in labels.values()] + [
                value.casefold() for value in aliases
            ]
            if query not in searchable and not any(query in value for value in searchable):
                continue
            label = labels.get(language) or labels.get("en") or next(iter(labels.values()))
            wikipedia = row.get("wikipedia", {})
            candidates.append({
                "id": row["id"],
                "label": label,
                "description": row.get("descriptions", {}).get(language)
                or row.get("descriptions", {}).get("en", ""),
                "wikipedia_url": wikipedia.get(language) or wikipedia.get("en"),
                "wikidata_url": f"https://www.wikidata.org/wiki/{row['id']}",
                "source": "local",
            })
        return candidates

    @staticmethod
    def _merge_candidates(local: list, remote: list) -> list:
        merged = []
        seen = set()
        for row in remote + local:
            if row.get("id") in seen:
                continue
            seen.add(row.get("id"))
            merged.append(dict(row))
        return merged

    @staticmethod
    def _context_similarities(context: str, descriptions: list) -> list:
        if not any(descriptions):
            return [0.0] * len(descriptions)
        try:
            matrix = TfidfVectorizer(ngram_range=(1, 2)).fit_transform([context] + descriptions)
            return cosine_similarity(matrix[0:1], matrix[1:]).flatten().tolist()
        except ValueError:
            return [0.0] * len(descriptions)

    @staticmethod
    def _label_similarity(query: str, label: str) -> float:
        query_normalized = query.casefold().strip(" .,;:()[]")
        label_normalized = label.casefold().strip(" .,;:()[]")
        if label_normalized.startswith(query_normalized) or query_normalized.startswith(label_normalized):
            return 1.0
        return SequenceMatcher(None, query_normalized, label_normalized).ratio()

    @staticmethod
    def _validate_text(text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("Tekst nie może być pusty.")
        if len(text) > MAX_ENTITY_TEXT_LENGTH:
            raise ValueError(f"Tekst może mieć maksymalnie {MAX_ENTITY_TEXT_LENGTH} znaków.")
        return text

    @staticmethod
    def _validate_language(language: str) -> str:
        language = (language or "").strip().lower()
        if language not in SUPPORTED_ENTITY_LANGUAGES:
            raise ValueError(
                f"Język musi być jednym z: {', '.join(SUPPORTED_ENTITY_LANGUAGES)}."
            )
        return language

    @staticmethod
    def _load_json(path: str, default):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError):
            return default

    @staticmethod
    def _save_json(path: str, payload) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
