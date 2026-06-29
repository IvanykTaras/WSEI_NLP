import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import requests


PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROGRAM_DIR not in sys.path:
    sys.path.insert(0, PROGRAM_DIR)

import main
from providers.entity_provider import EntityProvider
from providers.lab4_artifact_provider import Lab4ArtifactProvider
from providers.summarization_provider import SummarizationProvider
from providers.translation_provider import TranslationProvider


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload or {}
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


class OfflineSession:
    def get(self, *args, **kwargs):
        raise requests.ConnectionError("offline")


class OllamaSession:
    def __init__(self, summary="Krótkie podsumowanie."):
        self.summary = summary
        self.last_payload = None

    def get(self, url, timeout):
        return FakeResponse({"models": [{"name": "gemma3:1b"}]})

    def post(self, url, json, timeout):
        self.last_payload = json
        return FakeResponse({"response": self.summary})


class FakeTokenizer:
    src_lang = None

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}

    def get_lang_id(self, language):
        return 7

    def batch_decode(self, generated, skip_special_tokens):
        return ["Good morning"]


class FakeModel:
    def generate(self, **kwargs):
        return [[1, 2]]


class FakeMessage:
    def __init__(self):
        self.replies = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)


class FakeUpdate:
    def __init__(self):
        self.message = FakeMessage()


class FakeContext:
    def __init__(self, args):
        self.args = args


class EntityProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.kb_path = os.path.join(self.temp.name, "kb.json")
        self.cache_path = os.path.join(self.temp.name, "cache.json")
        with open(self.kb_path, "w", encoding="utf-8") as handle:
            json.dump({
                "entities": [
                    {
                        "id": "Q312",
                        "labels": {"en": "Apple Inc.", "pl": "Apple Inc."},
                        "aliases": ["Apple"],
                        "descriptions": {
                            "en": "technology company making computers and phones",
                            "pl": "firma technologiczna produkująca komputery i telefony",
                        },
                        "wikipedia": {"en": "https://en.wikipedia.org/wiki/Apple_Inc."},
                    },
                    {
                        "id": "Q89",
                        "labels": {"en": "apple", "pl": "jabłko"},
                        "aliases": ["Apple"],
                        "descriptions": {
                            "en": "fruit growing on a tree used in pies",
                            "pl": "owoc rosnący na drzewie",
                        },
                        "wikipedia": {"en": "https://en.wikipedia.org/wiki/Apple"},
                    },
                ]
            }, handle)
        self.provider = EntityProvider(
            knowledge_base_file=self.kb_path,
            cache_file=self.cache_path,
            session=OfflineSession(),
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_nel_uses_local_fallback_when_wikidata_is_offline(self):
        result = self.provider.link_entity("Apple", "en")
        self.assertEqual({row["id"] for row in result["candidates"]}, {"Q312", "Q89"})
        self.assertEqual(result["source"], "local")

    def test_ned_uses_context_to_select_company(self):
        result = self.provider.disambiguate(
            "Apple", "technology company makes computers and phones", "en"
        )
        self.assertEqual(result["selected"]["id"], "Q312")

    def test_ned_uses_context_to_select_fruit(self):
        result = self.provider.disambiguate(
            "Apple", "fruit growing on a tree and used in pies", "en"
        )
        self.assertEqual(result["selected"]["id"], "Q89")

    def test_ner_validation_happens_before_heavy_import(self):
        with self.assertRaisesRegex(ValueError, "Metoda NER"):
            self.provider.recognize("unknown", "Warszawa")

    def test_recognize_returns_normalized_shape(self):
        entities = [{"text": "Warszawa", "type": "GPE", "start": 0, "end": 8}]
        with patch.object(self.provider, "_recognize_spacy", return_value=entities):
            result = self.provider.recognize("spacy", "Warszawa")
        self.assertEqual(result["entities"], entities)


class TranslationProviderTests(unittest.TestCase):
    def test_translation_uses_cached_model_objects(self):
        provider = TranslationProvider()
        provider._tokenizer = FakeTokenizer()
        provider._model = FakeModel()
        result = provider.translate("Dzień dobry", "en", source_language="pl")
        self.assertEqual(result["translation"], "Good morning")
        self.assertEqual(result["source_language"], "pl")

    def test_same_source_and_target_is_rejected(self):
        provider = TranslationProvider()
        with self.assertRaisesRegex(ValueError, "muszą być różne"):
            provider.translate("Dzień dobry", "pl", source_language="pl")

    def test_unsupported_target_is_rejected(self):
        provider = TranslationProvider()
        with self.assertRaises(ValueError):
            provider.translate("Test", "it", source_language="pl")


class SummarizationProviderTests(unittest.TestCase):
    def test_summary_calls_non_streaming_ollama(self):
        session = OllamaSession()
        provider = SummarizationProvider(session=session)
        result = provider.summarize(
            "Pierwsze zdanie. Drugie zdanie.", "bullets", "short"
        )
        self.assertEqual(result["summary"], "Krótkie podsumowanie.")
        self.assertFalse(session.last_payload["stream"])
        self.assertEqual(session.last_payload["options"]["num_predict"], 120)

    def test_custom_summary_requires_prompt(self):
        provider = SummarizationProvider(session=OllamaSession())
        with self.assertRaisesRegex(ValueError, "wymagany jest parametr prompt"):
            provider.summarize("Tekst testowy", "custom", "short")


class ArtifactProviderTests(unittest.TestCase):
    def test_json_and_text_artifacts_use_stable_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            provider = Lab4ArtifactProvider(directory)
            json_path = provider.save_json("ner", {"entities": []})
            text_path = provider.save_text("summarize", "Podsumowanie", {"model": "test"})
            self.assertEqual(os.path.basename(json_path), "latest_ner.json")
            self.assertEqual(os.path.basename(text_path), "latest_summarize.txt")
            self.assertTrue(os.path.exists(json_path))
            self.assertTrue(os.path.exists(text_path))


class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_ner_handler_rejects_missing_params(self):
        update = FakeUpdate()
        await main.handle_ner(update, FakeContext([]))
        self.assertIn("❌", update.message.replies[-1])
        self.assertIn("/ner", update.message.replies[-1])

    async def test_translate_handler_rejects_unsupported_language(self):
        update = FakeUpdate()
        context = FakeContext(['text="Test"', "target_lang=it"])
        await main.handle_translate(update, context)
        self.assertIn("❌", update.message.replies[-1])
        self.assertIn("Język docelowy", update.message.replies[-1])

    async def test_summarize_handler_rejects_missing_custom_prompt(self):
        update = FakeUpdate()
        context = FakeContext([
            'text="Przykładowy tekst"', "summary_type=custom", "length=short"
        ])
        await main.handle_summarize(update, context)
        self.assertIn("❌", update.message.replies[-1])
        self.assertIn("prompt", update.message.replies[-1])

    async def test_ner_handler_saves_success_result(self):
        update = FakeUpdate()
        context = FakeContext(["method=spacy", 'text="Warszawa"'])
        result = {
            "method": "spacy",
            "language": "pl",
            "text": "Warszawa",
            "entities": [{"text": "Warszawa", "type": "GPE", "start": 0, "end": 8}],
        }
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(main.entity_provider, "recognize", return_value=result), \
                patch.object(main, "lab4_artifact_provider", Lab4ArtifactProvider(directory)):
            await main.handle_ner(update, context)
            self.assertTrue(os.path.exists(os.path.join(directory, "latest_ner.json")))
        self.assertTrue(any("Warszawa (GPE)" in reply for reply in update.message.replies))


if __name__ == "__main__":
    unittest.main()
